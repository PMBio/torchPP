"""
Neural Temporal Point Processes.

A neural sequence encoder maps event history to per-interval intensity
function parameters that are fed into closed-form likelihood expressions.

Class hierarchy
---------------
NeuralParametrizedTPP (IntensityBasedTPP, ABC)
    Abstract base.  Owns a :class:`~torchpp.modules.nn.encoder.SequenceEncoder`
    and implements the caching / parameter-lookup machinery shared by all
    encoder-based TPPs.  Subclasses must implement:
      - _get_params(ts)           → (B, L, ...) params from encoder context.
      - _get_prior()             → (...) prior params, shape = single row of _get_params.
      - _intensity_at_t(t, ts)
      - _interval_ci(params, t_lo, t_hi)
      - _sequence_intensities(ts) → (B, L, D)  [overrides base default for efficiency]
    Provided methods:
      - prepare_batch(ts)        → run encoder once, cache full params (B, L+1, ...)
      - clear_batch()            → free cache
      - fix_context(ts)          → context manager wrapping prepare/clear
      - _full_params(ts)          → (B, L+1, ...) tensor: prior at col 0, then L positions
      - _get_params_at_time(t, ts) → (B, ...) causal param lookup at a scalar time
      - transform_params()       → raises NotImplementedError (history-dep.)

TransformerParametrizedTPP (NeuralParametrizedTPP)
    Convenience subclass that accepts a :class:`~torchpp.modules.nn.transformer.TransformerConfig`
    and constructs a :class:`~torchpp.modules.nn.transformer.TransformerEncoder` internally.
    All existing concrete neural models (NeuralPoissonProcess, etc.) inherit from this.

Parameter layout
----------------
``_full_params(ts)`` returns a tensor of shape ``(B, L+1, ...)``:
  - Column 0        : prior parameters (from ``_get_prior()``)
  - Column 1..L     : parameters predicted from encoder context positions 0..L-1

When computing intensities at event position l, we use column l
(= the prior for l=0, context after event l-1 for l>0).

Causal conventions
------------------
Encoder context: position q encodes all events with time <= time[q],
INCLUDING simultaneous events.  This gives each position full context
from its own time point backward.

Intensity queries: ``_get_params_at_time`` counts the number of real events
with time strictly less than ``t`` in each sequence and uses that as the
column index into ``full_params``.  This gives the correct causal context
for any query time, including within simultaneous event blocks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterator, Optional

import torch

from ..utils.event_data import BatchedMVEventData, MVEventData
from .base_tpp import IntensityBasedTPP, DensityBasedTPP, CIConfig
from ..utils.math import LinearSpline, PiecewiseConstant, loglinear_cumulative_intensity
from ..modules.nn.encoder import SequenceEncoder, GRUEncoder, ContTimeLSTMEncoder, ContTimeLSTMCell
from ..modules.nn.attnhp import AttNHPEncoder
from ..modules.nn.lognormal_mixture import (
    LogNormalMixtureDistribution,
    clamp_preserve_gradients,
)

from ..modules.nn.transformer import (
    Transformer,
    TransformerConfig,
    TransformerEncoder,
    PoissonHead,
    LogLinearHead,
    SplineHead,
    PiecewiseConstantHead,
    THPHead,
    SAHPHead,
    ScaledSoftplus,
    EasyTPPTHPHead,
    EasyTPPSAHPHead,
    MLPIntensityHead,
)


# ---------------------------------------------------------------------------
# Numerical constants (IntensityFree density realization; value-preserving)
# ---------------------------------------------------------------------------

# Inter-event time floor: tau = t - t_last is clamped to this before the
# log-normal mixture log_prob / log_survival (guards log(0) as tau -> 0).
_MIN_INTER_EVENT_TIME = 1e-8

# Offset added to t_last to land strictly after the last event, selecting the
# full-history parameter column (mark head / mixture after all events).
_FULL_HISTORY_OFFSET = 1e-6

# Log-scale clamp bounds for the log-normal mixture (Shchur et al., 2020).
_LOG_SCALE_MIN = -5.0
_LOG_SCALE_MAX = 3.0

# Finite upper bound on the NHP intensity before log / cumulative-intensity.
# ScaledSoftplus can pass through an arbitrarily large linear pre-activation, so
# a pathological batch (huge weights) could otherwise emit inf and poison the
# log/CI.  Far above the healthy range (intensities are O(1e0–1e3)); healthy
# intensities are untouched, so the clamp is number-preserving.
_NHP_INTENSITY_MAX = 1e30


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class NeuralEncoderMixin(ABC):
    """Encoder plumbing shared by the intensity and density neural families.

    A standalone mixin (not a base class in its own right): it owns a
    :class:`~torchpp.modules.nn.encoder.SequenceEncoder`, the per-batch parameter
    cache, and the causal parameter-lookup machinery, and forwards construction to
    whatever concrete family base follows it in the MRO (``IntensityBasedTPP`` for
    :class:`NeuralParametrizedTPP`, ``DensityBasedTPP`` for
    :class:`IntensityFreeProcess`).  Inherits :class:`abc.ABC` only so the
    ``_get_params`` / ``_get_prior`` hooks register as abstract.

    See module docstring for parameter layout and causal conventions.

    Cache management
    ----------------
    ``prepare_batch(ts)`` runs the encoder once and caches the full params
    tensor ``(B, L+1, ...)`` in ``_cached_full_params``.  ``clear_batch()``
    releases it.  ``build_integration_grid`` and ``evaluate_at_time`` call
    prepare/clear automatically so the encoder runs only once per batch.
    """

    def __init__(
        self,
        encoder: SequenceEncoder,
        terminating: bool = False,
        seed: Optional[int] = None,
    ):
        # Forward D/seed/terminating to whichever family base follows the mixin in
        # the MRO (IntensityBasedTPP or DensityBasedTPP).
        super().__init__(
            D=encoder.num_event_types,
            seed=seed,
            terminating=terminating,
        )
        self.encoder = encoder
        # Cache filled by prepare_batch(), released by clear_batch().
        self._cached_full_params: Optional[torch.Tensor] = None  # (B, L+1, ...)
        self._cache_key: Optional[tuple] = None  # (id(ts), time_fingerprint, type_fingerprint)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Run the encoder and return ``(B, L, n_embd)`` context embeddings."""
        return self.encoder(ts)

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Project the encoder context to the head's parameters.

        The parameters are the head's own — intensity params for the intensity
        family, mixture + mark-logit params for the density family — so the mixin
        stays family-neutral.

        Args:
            ts: Batched event data, shape (B, L, ...).

        Returns:
            Parameter tensor, shape (B, L, D) / (B, L, 2, D) / (B, L, D, K).
        """

    @abstractmethod
    def _get_prior(self) -> torch.Tensor:
        """Return the head's prior parameters (the no-history column).

        Returns a tensor whose shape matches a single row of ``_get_params``,
        i.e. (D,) / (2, D) / (D, K).  Called with no batch dimension;
        ``_full_params`` expands it to (B, 1, ...).
        """

    # ------------------------------------------------------------------
    # Core: _full_params
    # ------------------------------------------------------------------

    def _full_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Build the ``(B, L+1, ...)`` parameter tensor.

        Column 0 holds the learned prior (expanded from ``_get_prior()``).
        Columns 1..L hold the parameters predicted from context positions 0..L-1.

        At training time (position l of the sequence), column l gives the
        parameters that should govern event l — i.e. the prior for l=0 and
        the context *after* event l-1 for l>0.

        Args:
            ts: Batched event data, shape (B, L).

        Returns:
            Full params, shape (B, L+1, ...).
        """
        B, L = ts.time_points.shape
        params = self._get_params(ts=ts)  # (B, L, ...)
        prior = self._get_prior()  # (...)
        # Cast prior to same dtype/device as params (head always float32; batch may be float64)
        prior = prior.to(device=params.device, dtype=params.dtype)
        prior_row = prior.unsqueeze(0).expand(B, *prior.shape).unsqueeze(1)  # (B, 1, ...)
        return torch.cat([prior_row, params], dim=1)  # (B, L+1, ...)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_cache_key(ts: BatchedMVEventData) -> tuple:
        """Compute a cheap fingerprint for *ts* to validate cache hits.

        Uses the object identity (``id``) plus scalar sums of ``time_points``
        and ``event_types``.  This catches the most common mistake — passing a
        different batch while the cache from a previous batch is still live —
        without the overhead of a full tensor comparison.
        """
        return (
            id(ts),
            ts.time_points.sum().item(),
            ts.event_types.sum().item(),
        )

    def prepare_batch(self, ts: BatchedMVEventData) -> None:
        """Run the transformer on *ts* and cache the full params tensor.

        Call once before a block of ``_intensity_at_t`` / ``_raw_ci`` calls.
        Released by ``clear_batch()``.
        """
        self._cached_full_params = self._full_params(ts)  # (B, L+1, ...)
        self._cache_key = self._batch_cache_key(ts)

    def clear_batch(self) -> None:
        """Release the cached params."""
        self._cached_full_params = None
        self._cache_key = None

    @contextmanager
    def fix_context(self, ts: BatchedMVEventData) -> Iterator[None]:
        """Context manager that caches transformer params for *ts* for the duration of the block.

        Calls ``prepare_batch(ts)`` on entry and ``clear_batch()`` on exit,
        even if an exception is raised.  Use this whenever multiple intensity
        or CI queries against the same batch are needed, to avoid running the
        transformer repeatedly.

        Example::

            with model.fix_context(batch):
                intensity = model.intensity(t, batch)
                ci        = model.cumulative_intensity(T, batch)

        Args:
            ts: Batched event data that all operations inside the block should use.
        """
        self.prepare_batch(ts)
        try:
            yield
        finally:
            self.clear_batch()

    def _get_cached_params(self, ts: BatchedMVEventData) -> Optional[torch.Tensor]:
        """Return cached params if they match *ts*, else None."""
        if self._cached_full_params is None:
            return None
        if self._cache_key != self._batch_cache_key(ts):
            raise RuntimeError(
                "NeuralParametrizedTPP: cached params were computed for a "
                "different batch. Call prepare_batch(ts) with the current batch "
                "before using the cache, or call clear_batch() to disable caching."
            )
        return self._cached_full_params

    # ------------------------------------------------------------------
    # Param lookup at a scalar query time
    # ------------------------------------------------------------------

    def _get_params_at_time(
        self,
        t: torch.Tensor,
        ts: BatchedMVEventData,
    ) -> torch.Tensor:
        """Return intensity parameters valid for query time ``t``.

        For batch item b, counts the number of real events with time strictly
        less than ``t[b]``, capped at ``seq_lengths[b]``.  Returns
        ``full_params[b, n_before, ...]`` — the context after those events.

        Column 0 = prior (no events seen), column k = context after k events.

        Uses ``_cached_full_params`` when set by ``prepare_batch``.

        Args:
            t:  Query times, shape (B,).
            ts: Batched event sequences.

        Returns:
            Parameter tensor, shape (B, ...) — one row per batch item.
        """
        cached = self._get_cached_params(ts)
        full_params = cached if cached is not None else self._full_params(ts)

        B = full_params.shape[0]

        # Count real events strictly before t[b] in each sequence.
        n_before = ts.count_events_before(t.to(ts.time_points.dtype))  # (B,)

        # full_params: (B, L+1, ...) — build index with correct trailing dims.
        trailing = full_params.shape[2:]
        idx = n_before.view(B, 1, *([1] * len(trailing))).expand(B, 1, *trailing)
        return full_params.gather(1, idx).squeeze(1)  # (B, ...)

    def _get_params_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Multi-query analogue of :meth:`_get_params_at_time`.  ``(B, Q)`` -> ``(B, Q, ...)``.

        Gathers, per query, the causal parameter column ``full_params[b, n_before(q), ...]``
        (number of real events strictly before ``q`` — the same count as
        ``count_events_before``), vectorised over the query axis.
        """
        cached = self._get_cached_params(ts)
        full_params = cached if cached is not None else self._full_params(ts)  # (B, L+1, ...)
        B, Q = query_times.shape
        trailing = full_params.shape[2:]
        n_before = self._count_events_before_queries(query_times, ts)  # (B, Q)
        idx = n_before.view(B, Q, *([1] * len(trailing))).expand(B, Q, *trailing)
        return full_params.gather(1, idx)  # (B, Q, ...)

    @staticmethod
    def _count_events_before_queries(query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Per-query count of real events strictly before each time. ``(B, Q)`` -> ``(B, Q)`` long.

        The multi-query form of ``count_events_before`` (strictly-before, real events).
        """
        tp = ts.time_points  # (B, N)
        q = query_times.to(tp.dtype)
        before = (tp.unsqueeze(1) < q.unsqueeze(2)) & ts.event_mask.unsqueeze(1)  # (B, Q, N)
        return before.sum(dim=2).long()  # (B, Q)

    def _event_time_before_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Per-query interval-start time (last real event strictly before ``q``, else 0).

        ``(B, Q)`` -> ``(B, Q)``; the multi-query form of ``event_time_before``.
        """
        tp = ts.time_points  # (B, N)
        B, N = tp.shape
        n_before = self._count_events_before_queries(query_times, ts)  # (B, Q)
        tp_with_zero = torch.cat([torch.zeros(B, 1, device=tp.device, dtype=tp.dtype), tp], dim=1)  # (B, N+1)
        return tp_with_zero.gather(1, n_before)  # (B, Q)

    # ------------------------------------------------------------------
    # TemporalPointProcess contract
    # ------------------------------------------------------------------

    def transform_params(self):
        """Not meaningful for neural TPPs (parameters depend on history)."""
        raise NotImplementedError(
            "NeuralTPP parameters are history-dependent. "
            "Use prepare_batch(ts) + _get_params_at_time(t, ts) for inference, "
            "or _full_params(x) for training."
        )


class NeuralParametrizedTPP(NeuralEncoderMixin, IntensityBasedTPP):
    """
    Abstract base class for encoder-based *intensity* temporal point processes.

    Combines the encoder plumbing (:class:`NeuralEncoderMixin`) with the intensity
    harness (:class:`IntensityBasedTPP`): the mixin maps event history to per-interval
    parameters, and this class turns those into the closed-form / numerical cumulative
    intensity the harness needs.  Concrete subclasses implement ``_get_params`` /
    ``_get_prior`` (mixin) plus an intensity hook and, for closed-form heads,
    ``_interval_ci_from_params``.

    Accepts any :class:`~torchpp.modules.nn.encoder.SequenceEncoder`, so an RNN, LSTM,
    or Transformer encoder can be swapped in freely.
    """

    def _interval_ci_from_params(
        self,
        params: torch.Tensor,  # (B, ...) or (B, L+1, ..., ...) — arbitrary leading dims
        t_lo: torch.Tensor,  # same leading shape as params, last dim = D
        t_hi: torch.Tensor,  # same shape as t_lo
    ) -> torch.Tensor:
        """Per-interval closed-form integral from pre-gathered params (closed-form heads).

        The base-class ``_interval_ci(ts, t_lo, t_hi)`` is the public per-interval
        closed-form hook; this is the closed-form-head fast-path that takes the causally
        gathered per-interval ``params`` directly (called by ``_interval_ci`` / ``_raw_ci``).
        Closed-form heads override it; intensity-numerical heads (NHP, AttNHP, EasyTPP
        THP/SAHP, MLP) leave it unimplemented and integrate via ``_interval_intensities``
        / the numerical CIConfig instead.

        Compute ∫_{t_lo_d}^{t_hi_d} λ_d(t) dt for each (b, ..., d).  Both ``t_lo`` and
        ``t_hi`` have the **same** shape as the output, last dim D (one bound per type);
        the caller expands type-agnostic bounds to ``(..., D)`` first.

        Args:
            params: Intensity parameters. Shape (B, ...,D)  or (B,L+1, ..., D)
            t_lo:   Interval start, shape ``(B,L+1,D)``
            t_hi:   Interval end,   shape ``(B,L+1,D)``

        Returns:
            Integrals with shape (B,L+1,D).
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no closed-form per-interval integral; "
            "it integrates numerically via _interval_intensities."
        )

    def _ci_interval_layout(
        self, effective_T: torch.Tensor, ts: BatchedMVEventData
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompose ``[0, effective_T]`` into the B×(L+1) inter-event intervals.

        Shared by :meth:`_raw_ci` and by per-interval-quadrature overrides
        (e.g. :class:`AttNHPProcess`).  Returns
        ``(params_all, t_lo_all, t_hi_all, needs_int)`` where ``params_all``
        is ``(B, L+1, ...)`` (causally gathered per interval), the bounds are
        ``(B, L+1, D)`` with invalid intervals clamped to zero length, and
        ``needs_int`` is the ``(B, L+1, D)`` bool mask of intervals that
        actually contribute.
        """
        cached = self._get_cached_params(ts)
        full_params = cached if cached is not None else self._full_params(ts)

        B = effective_T.shape[0]
        L = full_params.shape[1] - 1  # number of events per batch element (some might be padded)
        dtype = effective_T.dtype
        device = effective_T.device

        tp = ts.time_points.to(dtype)  # (B, L)
        real_mask = ts.event_mask  # (B, L)

        # Parameters for all L+1 intervals.
        # 0..L-1 parameters for the intervals [0.0, tp[:,0]], [tp[:,0], tp[:,1]], ..., [tp[:,L-2], tp[:,L-1]]
        # The causal index selects the parameters from the event causally preceding each position,
        # so simultaneous events do not observe each other's contribution.
        col_idx = ts.causal_indices  # (B, L)
        trailing = full_params.shape[2:]
        idx_events = col_idx.view(B, L, *([1] * len(trailing))).expand(B, L, *trailing)
        params_events = full_params.gather(1, idx_events).to(dtype=dtype)  # (B, L, ...)

        # Last index: params after the last real event for the interval [tp[:,last], effective_T]
        real_counts = ts.real_counts  # (B,)
        idx_final = real_counts.view(B, 1, *([1] * len(trailing))).expand(B, 1, *trailing)
        params_final = full_params.gather(1, idx_final).to(dtype=dtype)  # (B, 1, ...)

        # Parameters for invalid intervals (padding positions or beyond effective_T) are
        # masked out during integration below.
        params_all = torch.cat([params_events, params_final], dim=1)  # (B, L+1, ...)

        # Compute the interval bounds t_lo and t_high for each of the Bx(L+1) intervals,
        # then mask invalid intervals (t_hi <= t_lo or contain padding events).
        # The last interval is always from [tp[last], effective_T].

        # col_idx[b, l] = number of real events strictly before position l, which
        # is exactly the index into tp_with_zero of the last such event's time.
        # Prepend a zero column so col_idx == 0 (no prior events) maps to 0.0.
        tp_with_zero = torch.cat([torch.zeros(B, 1, device=device, dtype=dtype), tp], dim=1)  # (B, L+1)
        t_lo_events = tp_with_zero.gather(1, col_idx)  # (B, L)
        # t_lo for the last interval: real_counts[b] is the index in tp_with_zero of the
        # last real event's time (0 for empty sequences).
        t_lo_final = tp_with_zero.gather(1, real_counts.unsqueeze(1)).squeeze(1)  # (B,)

        t_lo_all = (
            torch.cat([t_lo_events, t_lo_final.unsqueeze(1)], dim=1).unsqueeze(-1).expand(B, L + 1, self.D)
        )  # (B, L+1, D)

        # t_hi for 0..L-1: min(tp[l], effective_T)  — per type
        t_hi_events = torch.minimum(
            tp.unsqueeze(-1).expand(B, L, self.D),
            effective_T.unsqueeze(1).expand(B, L, self.D),
        )  # (B, L, D)
        t_hi_final = effective_T.unsqueeze(1)  # (B, 1, D)

        t_hi_all = torch.cat([t_hi_events, t_hi_final], dim=1)  # (B, L+1, D)

        # Only integrate real-event slots where t_hi > t_lo.
        # Slots 0..L-1 are active only if a real event occupies that position.
        # Padded events always follow all real events, so a real event at position l
        # guarantees real events at all earlier positions.
        real_mask_ext = torch.cat(
            [real_mask, torch.ones(B, 1, dtype=torch.bool, device=device)], dim=1
        )  # (B, L+1) — final interval is always active

        # Deduplicate simultaneous events: when multiple event positions share
        # the same causal column (col_idx value), they produce identical
        # intervals [t_lo, t_hi] with the same params.  Keep only the first
        # occurrence per causal column per batch element to avoid double-counting.
        # We only deduplicate among the L event slots (0..L-1).  The final
        # slot (index L) is always kept — it covers the tail interval from the
        # last real event to effective_T and uses real_counts as its column,
        # which may collide with padding positions' causal_indices but must
        # never be suppressed.
        first_occurrence = torch.ones(B, L + 1, dtype=torch.bool, device=device)
        if L > 0:
            # Since causal_indices is non-decreasing (events are time-sorted),
            # simultaneous events always appear consecutively with the same
            # col_idx, so a simple consecutive comparison suffices.
            first_occurrence[:, 1:L] = col_idx[:, 1:] != col_idx[:, :-1]

        # Exclude intervals where effective_T lies below the interval's lower bound,
        # which can happen when effective_T is clamped by termination.
        needs_int = (
            real_mask_ext.unsqueeze(-1)  # (B, L+1, 1)
            & first_occurrence.unsqueeze(-1)  # (B, L+1, 1)
            & (t_hi_all > t_lo_all)  # (B, L+1, D)
        )  # (B, L+1, D)

        t_hi_all = torch.where(
            needs_int, t_hi_all, t_lo_all
        )  # (B, L+1, D) — clamp invalid intervals to zero length, to avoid numerical problems with negative intervals.

        return params_all, t_lo_all, t_hi_all, needs_int

    def _raw_ci(self, effective_T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Fully vectorised piecewise CI over all B×(L+1)×D intervals simultaneously.

        ``effective_T`` is ``(B, D)`` — per-type upper limit, already clamped
        to termination times by the base class.

        Layout of the L+1 intervals (one row in full_params per interval):

          slot 0      : [0,         tp[:,0]]   governed by prior  (col 0)
          slot l∈[1,L): [tp[:,l-1], tp[:,l]]   governed by col l  (causal)
          slot L      : [tp_last,   eff_T]     governed by col after last real event

        Both ``t_lo`` and ``t_hi`` are ``(B, L+1, D)``; ``_interval_ci``
        receives these directly and returns ``(B, L+1, D)`` which is summed
        over the interval dimension.
        """
        params_all, t_lo_all, t_hi_all, needs_int = self._ci_interval_layout(effective_T, ts)
        integrals = self._interval_ci_from_params(params_all, t_lo_all, t_hi_all)  # (B, L+1, D)
        ci = torch.where(needs_int, integrals, torch.zeros_like(integrals)).sum(dim=1)  # (B, D)

        return ci

    def _interval_ci(self, ts: BatchedMVEventData, t_lo: torch.Tensor, t_hi: torch.Tensor) -> torch.Tensor:
        """Per-interval closed-form CI for the intervals defined by ``(t_lo, t_hi)``.

        Gathers each interval's causal parameters — column = the number of real events
        with time ``<=`` the interval's left endpoint, i.e. the right-limit history
        ``H_i`` (the just-fired boundary event is included) — and applies the head's
        closed-form per-interval integral ``_interval_ci_from_params``.  Summing over
        intervals reproduces ``_raw_ci``; this is the closed-form dispatch's hook, so
        ``_raw_ci`` is not on the live path.  Zero-width / padding intervals integrate
        to zero.  ``t_lo``/``t_hi`` are ``(B, L+1)`` or ``(B, L+1, D)``.
        """
        B = ts.time_points.shape[0]
        cached = self._get_cached_params(ts)
        full = cached if cached is not None else self._full_params(ts)  # (B, L+1, ...)
        M = t_lo.shape[1]
        lo2 = t_lo[..., 0] if t_lo.dim() == 3 else t_lo  # (B, M) — left endpoints (type-agnostic)
        # Column for interval m = #real events with time <= its left endpoint (H_m).
        # Compare in lo2's dtype: interval_bounds casts event times to the query dtype,
        # so an exact-boundary event (lo_m == t_{m-1}) must be matched in the same dtype
        # — otherwise float32(t) < float64(t) drops it and the column is off by one.
        tp = ts.time_points.to(lo2.dtype)  # (B, L)
        col = ((tp.unsqueeze(1) <= lo2.unsqueeze(2)) & ts.event_mask.unsqueeze(1)).sum(2)  # (B, M)
        col = col.clamp(max=full.shape[1] - 1)
        trailing = full.shape[2:]
        idx = col.view(B, M, *([1] * len(trailing))).expand(B, M, *trailing)
        params = full.gather(1, idx)  # (B, M, ...)
        lo_d = t_lo if t_lo.dim() == 3 else t_lo.unsqueeze(-1).expand(B, M, self.D)
        hi_d = t_hi if t_hi.dim() == 3 else t_hi.unsqueeze(-1).expand(B, M, self.D)
        return self._interval_ci_from_params(params, lo_d, hi_d)  # (B, M, D)

    # ------------------------------------------------------------------
    # Wrap likelihood/grid/evaluation calls in prepare/clear
    # ------------------------------------------------------------------

    def likelihood(self, ts, T=None, log=True):
        """Compute likelihood with a single transformer forward pass.

        The base ``likelihood`` evaluates the positive term (``_event_intensities``)
        and the survival term (``cumulative_intensity``) inline; caching the
        transformer output for the whole block keeps both on a single ``_full_params``
        evaluation, and pins one dropout mask across them during training.
        """
        self.prepare_batch(ts)
        try:
            return super().likelihood(ts, T, log=log)
        finally:
            self.clear_batch()

    def build_integration_grid(
        self, ts, T_max=None, num_points=200, points_per_unit_time=200.0, integrate_marginal_via_grid=True
    ):
        """Build grid with a single transformer forward pass."""
        self.prepare_batch(ts)
        try:
            return super().build_integration_grid(
                ts,
                T_max=T_max,
                num_points=num_points,
                points_per_unit_time=points_per_unit_time,
                integrate_marginal_via_grid=integrate_marginal_via_grid,
            )
        finally:
            self.clear_batch()


# ---------------------------------------------------------------------------
# Transformer-specific subclass
# ---------------------------------------------------------------------------


class TransformerParametrizedTPP(NeuralParametrizedTPP):
    """Convenience subclass of :class:`NeuralParametrizedTPP` for Transformer encoders.

    Accepts a :class:`~torchpp.modules.nn.transformer.TransformerConfig`,
    constructs a :class:`~torchpp.modules.nn.transformer.TransformerEncoder`
    internally, and passes it to the parent.  All existing concrete neural
    models (e.g. :class:`NeuralPoissonProcess`) inherit from this class so
    that their public API — ``__init__(config, ...)`` — remains unchanged.

    To use a different encoder architecture (RNN, LSTM, …), subclass
    :class:`NeuralParametrizedTPP` directly and pass your own
    :class:`~torchpp.modules.nn.encoder.SequenceEncoder` instance.
    """

    def __init__(
        self,
        config: TransformerConfig,
        terminating: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(
            encoder=TransformerEncoder(config),
            terminating=terminating,
            seed=seed,
        )
        # Expose config for downstream heads that reference config.n_embd / config.num_event_types.
        self.config = config


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------


class NeuralPoissonProcess(TransformerParametrizedTPP):
    """
    History-conditioned homogeneous Poisson process.

    Params shape: (B, L+1, D)  — column l holds rates valid in interval l.
    Intensity:          λ_d(t) = rates_d   (constant between predictions)
    Cumulative intens.: Λ_d(T) = rates_d * T
    """

    def __init__(
        self,
        config: TransformerConfig,
        terminating: bool = False,
        seed: Optional[int] = None,
        link: str = "exp",
    ):
        super().__init__(config, terminating=terminating, seed=seed)
        self.head = PoissonHead(config, link=link)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        embedding = self._encode(ts)
        return self.head(embedding)  # (B, L, D)

    def _get_prior(self) -> torch.Tensor:
        return self.head.get_prior()  # (D,)

    def _intensity_at_t(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        rates = self._get_params_at_time(t, ts)  # (B, D)
        return rates

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Constant-per-interval rates gathered per query. ``(B, Q)`` -> ``(B, Q, D)``."""
        return self._get_params_at_queries(query_times, ts)  # (B, Q, D)

    def _interval_ci_from_params(self, params: torch.Tensor, t_lo: torch.Tensor, t_hi: torch.Tensor) -> torch.Tensor:
        """rates * (t_hi - t_lo).  params/t_lo/t_hi all same shape (..., D)."""
        return params * (t_hi - t_lo)

    def _sequence_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """(B, L, D) intensities in one forward pass.

        For each event at position l, uses the causal column: the context
        from all events with time strictly before time[l].  Simultaneous
        events therefore share the same parameter column.
        """
        B, L = ts.time_points.shape
        if L == 0:
            return torch.zeros(B, 0, self.D, device=ts.time_points.device)
        cached = self._get_cached_params(ts)
        full = cached if cached is not None else self._full_params(ts)  # (B, L+1, D)
        col_idx = ts.causal_indices  # (B, L), values in [0, L]
        idx = col_idx.unsqueeze(-1).expand(B, L, self.D)
        return full.gather(1, idx)  # (B, L, D)


# ---------------------------------------------------------------------------
# Log-linear (inhomogeneous Poisson)
# ---------------------------------------------------------------------------


class NeuralLogLinearProcess(TransformerParametrizedTPP):
    """
    History-conditioned inhomogeneous Poisson process with log-linear intensity.

    Params shape: (B, L+1, 2, D) — column l holds (b, s) for interval l.
      - params[:, l, 0, :] = b  (log-intercept)
      - params[:, l, 1, :] = s  (slope)
    Intensity:          λ_d(t) = exp(b_d + s_d * t)
    Cumulative intens.: Λ_d(T) = (exp(b_d) / s_d) * (exp(s_d * T) - 1)
                                  with Taylor fallback for s_d ≈ 0.
    """

    def __init__(self, config: TransformerConfig, terminating: bool = False, seed: Optional[int] = None):
        super().__init__(config, terminating=terminating, seed=seed)
        self.head = LogLinearHead(config)
        # Initialise slope weights near zero so initial predictions are near-flat.
        torch.nn.init.zeros_(self.head.linear_s.weight)
        torch.nn.init.constant_(self.head.linear_s.bias, 0.3)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        embedding = self._encode(ts)
        return self.head(embedding)  # (B, L, 2, D)

    def _get_prior(self) -> torch.Tensor:
        return self.head.get_prior()  # (2, D)

    def _intensity_at_t(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        p = self._get_params_at_time(t, ts)  # (B, 2, D)
        b, s = p[:, 0, :], p[:, 1, :]  # (B, D) each
        # Relative time within the current interval: t - t_interval_start.
        # t_interval_start is the time of the last real event strictly before t.
        t_start = ts.event_time_before(t)  # (B,)
        dt = (t - t_start).unsqueeze(1)  # (B, 1)
        return torch.exp(b + s * dt)

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """``λ_d(q) = exp(b_d + s_d·(q − t_start))`` per query. ``(B, Q)`` -> ``(B, Q, D)``."""
        p = self._get_params_at_queries(query_times, ts)  # (B, Q, 2, D)
        b, s = p[..., 0, :], p[..., 1, :]  # (B, Q, D) each
        t_start = self._event_time_before_queries(query_times, ts)  # (B, Q)
        dt = (query_times - t_start).unsqueeze(-1)  # (B, Q, 1)
        return torch.exp(b + s * dt)  # (B, Q, D)

    def _interval_ci_from_params(self, params: torch.Tensor, t_lo: torch.Tensor, t_hi: torch.Tensor) -> torch.Tensor:
        """∫₀^{dt} exp(b_d + s_d * t) dt  where dt = t_hi - t_lo.

        The transformer predicts (b, s) for each interval, so the intensity
        is exp(b + s * t_relative) with t_relative starting at 0.  Only the
        interval width matters, not the absolute position of t_lo.

        params: (..., 2, D), t_lo/t_hi: (..., D) — all same leading shape.
        """
        b, s = params[..., 0, :], params[..., 1, :]  # (..., D) each
        dt = t_hi - t_lo
        return self._log_linear_integral(b, s, dt)

    @staticmethod
    def _log_linear_integral(b: torch.Tensor, s: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Closed-form integral of exp(b + s*t) from 0 to T.

        Delegates to :func:`~torchpp.utils.math.loglinear_cumulative_intensity`,
        the shared implementation used by all log-linear intensity modules.
        """
        return loglinear_cumulative_intensity(b, s, T)

    def _sequence_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """(B, L, D) intensities in one forward pass.

        Uses causal column indices so simultaneous events share context.
        """
        B, L = ts.time_points.shape
        if L == 0:
            return torch.zeros(B, 0, self.D, device=ts.time_points.device)
        full = (
            self._cached_full_params if self._cached_full_params is not None else self._full_params(ts)
        )  # (B, L+1, 2, D)
        col_idx = ts.causal_indices  # (B, L)
        idx = col_idx.unsqueeze(-1).unsqueeze(-1).expand(B, L, 2, self.D)
        p = full.gather(1, idx)  # (B, L, 2, D)
        b, s = p[:, :, 0, :], p[:, :, 1, :]  # (B, L, D) each
        # Relative time: t - t_interval_start for each position.
        tp = ts.time_points  # (B, L)
        tp_with_zero = torch.cat([torch.zeros(B, 1, device=tp.device, dtype=tp.dtype), tp], dim=1)
        t_start = tp_with_zero.gather(1, col_idx)  # (B, L)
        dt = (tp - t_start).unsqueeze(-1)  # (B, L, 1)
        return torch.exp(b + s * dt)  # (B, L, D)


# ---------------------------------------------------------------------------
# Spline
# ---------------------------------------------------------------------------


class NeuralSplineProcess(TransformerParametrizedTPP):
    """
    History-conditioned piecewise-linear spline intensity process.

    Params shape: (B, L+1, D, K) — column l holds K heights per type for interval l.
    Intensity:          λ_d(t) = linear interpolation of h_d over knot_locs
    Cumulative intens.: Λ_d(T) = trapezoidal integral of the spline up to T

    Args:
        config:          TransformerConfig.
        num_spline_bins: Number of spline knots K.
        knot_gaps:       1-D tensor of length ``K-1`` with the inter-knot gaps
                         (distances between consecutive knots starting from 0).
                         Supports non-uniform schedules (e.g. geometric) for better
                         near-term resolution.  When ``None``, falls back to uniform
                         spacing on ``[0, spline_t_max]``.
        spline_t_max:    Total time span — only used when ``knot_gaps`` is ``None``.
        terminating:     Whether each event type can only occur once.
        dataset_stats:   Optional empirical statistics; currently unused but accepted
                         for API consistency with non-neural models and forward
                         compatibility with data-driven prior initialisation.
    """

    def __init__(
        self,
        config: TransformerConfig,
        num_spline_bins: int = 5,
        knot_gaps: Optional[torch.Tensor] = None,
        spline_t_max: float = 1.0,
        terminating: bool = False,
        seed: Optional[int] = None,
        dataset_stats=None,
    ):
        super().__init__(config, terminating=terminating, seed=seed)
        self.num_spline_bins = num_spline_bins
        self.head = SplineHead(config, num_spline_bins=num_spline_bins)
        if knot_gaps is None:
            knot_gaps = torch.full((num_spline_bins - 1,), spline_t_max / (num_spline_bins - 1))
        knot_locs = torch.cat([torch.zeros(1), knot_gaps.float().cumsum(0)])
        self.register_buffer("knot_locs", knot_locs)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        embedding = self._encode(ts)
        return self.head(embedding)  # (B, L, D, K)

    def _get_prior(self) -> torch.Tensor:
        return self.head.get_prior()  # (D, K)

    def _intensity_at_t(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        h = self._get_params_at_time(t, ts)  # (B, D, K)
        B, D, K = h.shape
        # Relative time within the current interval.
        t_start = ts.event_time_before(t)  # (B,)
        dt = (t - t_start).unsqueeze(1).expand(B, D)  # (B, D)
        intensity_flat = LinearSpline.interpolate_flat(self.knot_locs, h.reshape(B * D, K), dt.reshape(B * D))
        return intensity_flat.view(B, D)

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Spline intensity in relative time, gathered per query. ``(B, Q)`` -> ``(B, Q, D)``."""
        h = self._get_params_at_queries(query_times, ts)  # (B, Q, D, K)
        B, Q, D, K = h.shape
        t_start = self._event_time_before_queries(query_times, ts)  # (B, Q)
        dt = (query_times - t_start).unsqueeze(-1).expand(B, Q, D)  # (B, Q, D)
        flat = LinearSpline.interpolate_flat(self.knot_locs, h.reshape(B * Q * D, K), dt.reshape(B * Q * D))
        return flat.view(B, Q, D)  # (B, Q, D)

    def _interval_ci_from_params(self, params: torch.Tensor, t_lo: torch.Tensor, t_hi: torch.Tensor) -> torch.Tensor:
        """Spline interval CI using relative time.

        The transformer predicts spline heights for each interval, so the
        spline is evaluated in relative time starting at 0.  The integral
        is ∫₀^{dt} spline(t) dt  where dt = t_hi - t_lo.

        params: (..., D, K), t_lo/t_hi: (..., D) — same leading shape.
        Flattens all leading dims × D into a single N dimension for the
        batched spline helpers, then restores the original shape.
        """
        h = params  # (..., D, K)
        *_, K = h.shape
        dt = t_hi - t_lo  # (..., D)
        N_total = dt.numel()  # prod(leading) * D
        integral = LinearSpline.integrate_flat(self.knot_locs, h.reshape(N_total, K), dt.reshape(N_total))
        return integral.view(*dt.shape)

    def spline_smoothness_penalty(self, ts: BatchedMVEventData) -> torch.Tensor:
        """P-spline-style second-difference smoothness penalty on spline heights.

        Returns the mean over (batch, event, type, internal-knot) of
        ``(h_{k+1} - 2 h_k + h_{k-1})**2``. This is the standard discrete-
        second-difference penalty used for P-splines (Eilers & Marx, 1996); it
        controls how rapidly the spline coefficients can change and so acts as
        a smoothness prior.

        Knot-spacing-independent: we deliberately do NOT divide by gap-squared
        (which would correspond to slope-difference-based ∫|f''|^2 and blows up
        for our geometric kernel schedule whose smallest gap near 0 is on the
        order of T·(1/(K-1))^p ≈ 1e-8 for K=200, p=4 — making any non-trivial
        λ overwhelm the likelihood).

        Uses ``self._cached_full_params`` if populated (i.e. after a ``likelihood``
        / forward pass on the same ``ts``); otherwise computes it.
        """
        full = self._cached_full_params if self._cached_full_params is not None else self._full_params(ts)
        # full: (B, L+1, D, K)
        h_dd = full[..., 2:] - 2.0 * full[..., 1:-1] + full[..., :-2]  # (B, L+1, D, K-2)
        return (h_dd ** 2).mean()

    def _sequence_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """(B, L, D) intensities in one forward pass.

        Uses causal column indices so simultaneous events share context.
        """
        B, L = ts.time_points.shape
        if L == 0:
            return torch.zeros(B, 0, self.D, device=ts.time_points.device)
        full = (
            self._cached_full_params if self._cached_full_params is not None else self._full_params(ts)
        )  # (B, L+1, D, K)
        K = full.shape[3]
        col_idx = ts.causal_indices  # (B, L)
        idx = col_idx.unsqueeze(-1).unsqueeze(-1).expand(B, L, self.D, K)
        h = full.gather(1, idx)  # (B, L, D, K)

        # Relative time: t - t_interval_start for each position.
        tp = ts.time_points  # (B, L)
        tp_with_zero = torch.cat([torch.zeros(B, 1, device=tp.device, dtype=tp.dtype), tp], dim=1)
        t_start = tp_with_zero.gather(1, col_idx)  # (B, L)
        dt = tp - t_start  # (B, L)

        h_flat = h.reshape(B * L * self.D, K)
        dt_exp = dt.unsqueeze(-1).expand(B, L, self.D).reshape(B * L * self.D)
        intensity_flat = LinearSpline.interpolate_flat(self.knot_locs, h_flat, dt_exp)
        return intensity_flat.view(B, L, self.D)  # (B, L, D)


# ---------------------------------------------------------------------------
# Piecewise Constant
# ---------------------------------------------------------------------------


class NeuralPiecewiseConstantProcess(TransformerParametrizedTPP):
    """History-conditioned piecewise-constant intensity process.

    Like :class:`NeuralSplineProcess` but with piecewise-constant (step
    function) intensity within each inter-event interval — no slope.

    Params shape: (B, L+1, D, K) — column l holds K heights per type.
    Intensity:          λ_d(t) = h_k   for  x_k ≤ t < x_{k+1}
    Cumulative intens.: Λ_d(T) = Σ_{j<k} h_j Δx_j  +  h_k (T - x_k)

    Args:
        config:        TransformerConfig.
        num_bins:      Number of piecewise-constant bins K.
        knot_gaps:     1-D tensor of length ``K-1`` with the inter-knot gaps
                       (distances between consecutive bin edges starting from 0).
                       Supports non-uniform schedules for better near-term resolution.
                       When ``None``, falls back to uniform spacing on ``[0, bin_t_max]``.
        bin_t_max:     Total time span — only used when ``knot_gaps`` is ``None``.
        terminating:   Whether each event type can only occur once.
        dataset_stats: Optional empirical statistics; currently unused but accepted
                       for API consistency and forward compatibility.
    """

    def __init__(
        self,
        config: TransformerConfig,
        num_bins: int = 5,
        knot_gaps: Optional[torch.Tensor] = None,
        bin_t_max: float = 1.0,
        terminating: bool = False,
        seed: Optional[int] = None,
        dataset_stats=None,
    ):
        super().__init__(config, terminating=terminating, seed=seed)
        self.num_bins = num_bins
        self.head = PiecewiseConstantHead(config, num_spline_bins=num_bins)
        if knot_gaps is None:
            knot_gaps = torch.full((num_bins - 1,), bin_t_max / (num_bins - 1))
        knot_locs = torch.cat([torch.zeros(1), knot_gaps.float().cumsum(0)])
        self.register_buffer("knot_locs", knot_locs)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        embedding = self._encode(ts)
        return self.head(embedding)  # (B, L, D, K)

    def _get_prior(self) -> torch.Tensor:
        return self.head.get_prior()  # (D, K)

    def _intensity_at_t(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        h = self._get_params_at_time(t, ts)  # (B, D, K)
        B, D, K = h.shape
        t_start = ts.event_time_before(t)  # (B,)
        dt = (t - t_start).unsqueeze(1).expand(B, D)  # (B, D)
        intensity_flat = PiecewiseConstant.interpolate_flat(self.knot_locs, h.reshape(B * D, K), dt.reshape(B * D))
        return intensity_flat.view(B, D)

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Piecewise-constant intensity in relative time, per query. ``(B, Q)`` -> ``(B, Q, D)``."""
        h = self._get_params_at_queries(query_times, ts)  # (B, Q, D, K)
        B, Q, D, K = h.shape
        t_start = self._event_time_before_queries(query_times, ts)  # (B, Q)
        dt = (query_times - t_start).unsqueeze(-1).expand(B, Q, D)  # (B, Q, D)
        flat = PiecewiseConstant.interpolate_flat(self.knot_locs, h.reshape(B * Q * D, K), dt.reshape(B * Q * D))
        return flat.view(B, Q, D)  # (B, Q, D)

    def _interval_ci_from_params(self, params: torch.Tensor, t_lo: torch.Tensor, t_hi: torch.Tensor) -> torch.Tensor:
        """Piecewise-constant interval CI.

        params: (..., D, K), t_lo/t_hi: (..., D) — same leading shape.
        """
        h = params  # (..., D, K)
        *_, K = h.shape
        dt = t_hi - t_lo  # (..., D)
        N_total = dt.numel()
        integral = PiecewiseConstant.integrate_flat(self.knot_locs, h.reshape(N_total, K), dt.reshape(N_total))
        return integral.view(*dt.shape)

    def _sequence_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """(B, L, D) intensities in one forward pass."""
        B, L = ts.time_points.shape
        if L == 0:
            return torch.zeros(B, 0, self.D, device=ts.time_points.device)
        full = (
            self._cached_full_params if self._cached_full_params is not None else self._full_params(ts)
        )  # (B, L+1, D, K)
        K = full.shape[3]
        col_idx = ts.causal_indices  # (B, L)
        idx = col_idx.unsqueeze(-1).unsqueeze(-1).expand(B, L, self.D, K)
        h = full.gather(1, idx)  # (B, L, D, K)

        tp = ts.time_points  # (B, L)
        tp_with_zero = torch.cat([torch.zeros(B, 1, device=tp.device, dtype=tp.dtype), tp], dim=1)
        t_start = tp_with_zero.gather(1, col_idx)  # (B, L)
        dt = tp - t_start  # (B, L)

        h_flat = h.reshape(B * L * self.D, K)
        dt_exp = dt.unsqueeze(-1).expand(B, L, self.D).reshape(B * L * self.D)
        intensity_flat = PiecewiseConstant.interpolate_flat(self.knot_locs, h_flat, dt_exp)
        return intensity_flat.view(B, L, self.D)  # (B, L, D)


# ---------------------------------------------------------------------------
# THP (Transformer Hawkes Process)
# ---------------------------------------------------------------------------


class NeuralTHPProcess(TransformerParametrizedTPP):
    """History-conditioned THP: Transformer Hawkes Process (Zuo et al., ICML 2020).

    Intensity in interval l at relative time t_rel = t - t_start:

        λ_d(t) = exp(b_d + w_d * t_rel)

    where ``b_d`` is predicted by the transformer per event (log-intercept) and
    ``w_d`` is a single global learned parameter per type (not event-dependent).
    This matches EasyTPP's THP formulation: intensity_state = w * dtime + h + base,
    lambda = softplus(state) — but here we use exp for an exact closed-form CI.

    Cumulative intensity (closed form, identical to NeuralLogLinearProcess):

        Λ_d(T) = (exp(b_d) / w_d) * (exp(w_d * T) - 1)    for |w_d| > eps
               ≈ exp(b_d) * (T + w_d*T²/2 + w_d²*T³/6)   (Taylor) for |w_d| ≈ 0

    Params shape: (B, L+1, D) — per-interval log-intercept b.
    The global slope w is stored on the head as ``THPHead.log_decay``.

    Note: w can be positive (excitatory) or negative (inhibitory). Using
    ``exp(log_decay)`` ensures w > 0; for a free-sign version one would store
    w directly. We keep w > 0 for stability (bounded intensity growth).
    """

    def __init__(self, config: TransformerConfig, terminating: bool = False, seed: Optional[int] = None):
        super().__init__(config, terminating=terminating, seed=seed)
        self.head = THPHead(config)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        embedding = self._encode(ts)
        return self.head(embedding)  # (B, L, D) — raw log-intercepts

    def _get_prior(self) -> torch.Tensor:
        return self.head.get_prior()  # (D,) — raw prior log-intercept

    def _intensity_at_t(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        b = self._get_params_at_time(t, ts)  # (B, D) — raw log-intercept
        w = self.head.get_decay()  # (D,) — positive global slope
        t_start = ts.event_time_before(t)  # (B,)
        t_rel = (t - t_start).unsqueeze(-1)  # (B, 1)
        return torch.exp(b + w * t_rel)  # (B, D)

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """``λ_d(q) = exp(b_d + w_d·(q − t_start))`` per query. ``(B, Q)`` -> ``(B, Q, D)``."""
        b = self._get_params_at_queries(query_times, ts)  # (B, Q, D)
        w = self.head.get_decay()  # (D,)
        t_start = self._event_time_before_queries(query_times, ts)  # (B, Q)
        t_rel = (query_times - t_start).unsqueeze(-1)  # (B, Q, 1)
        return torch.exp(b + w * t_rel)  # (B, Q, D)

    def _interval_ci_from_params(self, params: torch.Tensor, t_lo: torch.Tensor, t_hi: torch.Tensor) -> torch.Tensor:
        """Closed-form CI for exp(b + w * t_rel) from 0 to dt = t_hi - t_lo.

        ∫₀^dt exp(b + w*t) dt = exp(b)/w * (exp(w*dt) - 1)
        with Taylor fallback for |w| < 1e-8.

        params: (..., D), t_lo/t_hi: (..., D) — same leading shape.
        """
        b = params  # (..., D) — raw log-intercept
        w = self.head.get_decay().to(b.device, dtype=b.dtype)  # (D,)
        w = w.expand_as(b)
        dt = t_hi - t_lo  # (..., D)
        return NeuralLogLinearProcess._log_linear_integral(b, w, dt)

    def _sequence_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """(B, L, D) intensities in one forward pass."""
        B, L = ts.time_points.shape
        if L == 0:
            return torch.zeros(B, 0, self.D, device=ts.time_points.device)
        cached = self._get_cached_params(ts)
        full = cached if cached is not None else self._full_params(ts)  # (B, L+1, D)
        col_idx = ts.causal_indices  # (B, L)
        idx = col_idx.unsqueeze(-1).expand(B, L, self.D)
        b = full.gather(1, idx)  # (B, L, D) — raw log-intercepts
        w = self.head.get_decay().to(b.device, dtype=b.dtype)  # (D,)
        tp = ts.time_points  # (B, L)
        tp_with_zero = torch.cat([torch.zeros(B, 1, device=tp.device, dtype=tp.dtype), tp], dim=1)
        t_start = tp_with_zero.gather(1, col_idx)  # (B, L)
        t_rel = (tp - t_start).unsqueeze(-1)  # (B, L, 1)
        return torch.exp(b + w * t_rel)  # (B, L, D)


# ---------------------------------------------------------------------------
# SAHP (Self-Attentive Hawkes Process)
# ---------------------------------------------------------------------------


class NeuralSAHPProcess(TransformerParametrizedTPP):
    """History-conditioned SAHP: Self-Attentive Hawkes Process (Zhang et al., ICML 2020).

    Intensity in interval l at relative time t_rel = t - t_start:

        λ_d(t) = mu_d + (eta_d - mu_d) * exp(-gamma_d * t_rel)

    where mu, eta, gamma are all predicted by the transformer per event.
    Slight modification: we enforce positivity constraints on all three parameters (mu, eta, gamma)
    instead of using softplus on the final intensity, to allow an exact closed-form CI.

    Constraints: mu ≥ 0, eta ≥ 0, gamma > 0 (enforced via exp/softplus).

    Cumulative intensity (closed form):

        Λ_d(t_rel) = mu_d * t_rel + (eta_d - mu_d)/gamma_d * (1 - exp(-gamma_d * t_rel))

    with Taylor fallback for gamma_d ≈ 0:

        Λ_d ≈ eta_d * t_rel - (eta_d - mu_d) * gamma_d * t_rel^2 / 2

    Params shape: (B, L+1, 3, D) — stacked (mu, eta, gamma) raw values.
    Constraints applied at use: mu=exp(raw_mu), eta=exp(raw_eta), gamma=softplus(raw_gamma).
    """

    def __init__(self, config: TransformerConfig, terminating: bool = False, seed: Optional[int] = None):
        super().__init__(config, terminating=terminating, seed=seed)
        self.head = SAHPHead(config)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        embedding = self._encode(ts)
        return self.head(embedding)  # (B, L, 3, D) — raw

    def _get_prior(self) -> torch.Tensor:
        return self.head.get_prior()  # (3, D) — raw

    @staticmethod
    def _constrain(params: torch.Tensor):
        """Apply positivity constraints: mu=exp, eta=exp, gamma=softplus.

        Args:
            params: (..., 3, D)
        Returns:
            mu, eta, gamma — each (..., D), all positive.
        """
        import torch.nn.functional as F

        mu = torch.exp(params[..., 0, :].clamp(max=10.0))  # (..., D)
        eta = torch.exp(params[..., 1, :].clamp(max=10.0))  # (..., D)
        gamma = F.softplus(params[..., 2, :])  # (..., D) > 0
        return mu, eta, gamma

    def _intensity_at_t(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        raw = self._get_params_at_time(t, ts)  # (B, 3, D)
        mu, eta, gamma = self._constrain(raw)  # (B, D) each
        t_start = ts.event_time_before(t)  # (B,)
        t_rel = (t - t_start).unsqueeze(1)  # (B, 1)
        return mu + (eta - mu) * torch.exp(-gamma * t_rel)  # (B, D)

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """``λ_d(q) = mu + (eta−mu)·exp(−gamma·(q − t_start))`` per query. ``(B, Q)`` -> ``(B, Q, D)``."""
        raw = self._get_params_at_queries(query_times, ts)  # (B, Q, 3, D)
        mu, eta, gamma = self._constrain(raw)  # (B, Q, D) each
        t_start = self._event_time_before_queries(query_times, ts)  # (B, Q)
        t_rel = (query_times - t_start).unsqueeze(-1)  # (B, Q, 1)
        return mu + (eta - mu) * torch.exp(-gamma * t_rel)  # (B, Q, D)

    def _interval_ci_from_params(self, params: torch.Tensor, t_lo: torch.Tensor, t_hi: torch.Tensor) -> torch.Tensor:
        """Closed-form CI for mu + (eta-mu)*exp(-gamma*t_rel) from 0 to dt.

        Λ = mu*dt + (eta-mu)/gamma * (1 - exp(-gamma*dt))
        Taylor for gamma ≈ 0: Λ ≈ eta*dt - (eta-mu)*gamma*dt^2/2

        params: (..., 3, D), t_lo/t_hi: (..., D) — same leading shape.
        """
        mu, eta, gamma = self._constrain(params)  # (..., D) each
        dt = t_hi - t_lo  # (..., D)
        return self._sahp_integral(mu, eta, gamma, dt)

    @staticmethod
    def _sahp_integral(mu: torch.Tensor, eta: torch.Tensor, gamma: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """∫₀^T [mu + (eta-mu)*exp(-gamma*t)] dt, closed form."""
        gamma_small = gamma < 1e-6
        safe_gamma = gamma.clone()
        safe_gamma[gamma_small] = 1.0
        integral_full = mu * T + (eta - mu) / safe_gamma * (1.0 - torch.exp(-safe_gamma * T))
        integral_taylor = eta * T - (eta - mu) * gamma * T**2 / 2.0
        return torch.where(gamma_small, integral_taylor, integral_full)

    def _sequence_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """(B, L, D) intensities in one forward pass."""
        B, L = ts.time_points.shape
        if L == 0:
            return torch.zeros(B, 0, self.D, device=ts.time_points.device)
        cached = self._get_cached_params(ts)
        full = cached if cached is not None else self._full_params(ts)  # (B, L+1, 3, D)
        col_idx = ts.causal_indices  # (B, L)
        idx = col_idx.unsqueeze(-1).unsqueeze(-1).expand(B, L, 3, self.D)
        raw = full.gather(1, idx)  # (B, L, 3, D)
        mu, eta, gamma = NeuralSAHPProcess._constrain(raw)  # (B, L, D) each
        tp = ts.time_points  # (B, L)
        tp_with_zero = torch.cat([torch.zeros(B, 1, device=tp.device, dtype=tp.dtype), tp], dim=1)
        t_start = tp_with_zero.gather(1, col_idx)  # (B, L)
        t_rel = (tp - t_start).unsqueeze(-1)  # (B, L, 1)
        return mu + (eta - mu) * torch.exp(-gamma * t_rel)  # (B, L, D)


# ===========================================================================
# EasyTPP-faithful replicas (ScaledSoftplus wrapping → numerical CI)
# ===========================================================================


class EasyTPPTHPProcess(TransformerParametrizedTPP):
    """Exact replica of the EasyTPP Transformer Hawkes Process (Zuo et al., ICML 2020).

    Intensity at relative time ``t_rel = t - t_start``::

        state_d(t) = w_d * t_rel + linear(h)_d + b_d
        λ_d(t)     = ScaledSoftplus(state_d(t))

    where ``w_d`` (factor_intensity_decay) and ``b_d`` (factor_intensity_base)
    are global learnable per-type parameters, and ``linear(h)`` is a projection
    of the transformer encoder output.  ``ScaledSoftplus`` uses a learnable
    per-mark beta, matching EasyTPP's implementation exactly.

    Because ScaledSoftplus has no closed-form integral, cumulative intensity
    is computed numerically (per-interval trapezoid CIConfig).

    Params shape: ``(B, L+1, D)`` — the ``linear(h)`` projections + prior.
    """

    def __init__(
        self,
        config: TransformerConfig,
        terminating: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(config, terminating=terminating, seed=seed)
        self.head = EasyTPPTHPHead(config)
        self.softplus = ScaledSoftplus(config.num_event_types)

    def _default_ci_config(self) -> CIConfig:
        """ScaledSoftplus has no closed-form integral; the intensity is smooth within an
        interval and jumps only at events → event-knotted per-interval trapezoid CI."""
        return CIConfig("numerical", "interval", "trapezoidal", n_points=100)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        embedding = self._encode(ts)
        return self.head(embedding)  # (B, L, D) — linear(h) projection

    def _get_prior(self) -> torch.Tensor:
        return self.head.get_prior()  # (D,)

    def _intensity_at_t(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        h_proj = self._get_params_at_time(t, ts)  # (B, D) — linear(h)
        w = self.head.factor_intensity_decay  # (1, D) — unconstrained
        b = self.head.factor_intensity_base  # (1, D) — unconstrained
        t_start = ts.event_time_before(t)  # (B,)
        t_rel = (t - t_start).unsqueeze(-1)  # (B, 1)
        state = w * t_rel + h_proj + b  # (B, D)
        return self.softplus(state)  # (B, D)

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """``softplus(w·t_rel + linear(h) + b)`` per query. ``(B, Q)`` -> ``(B, Q, D)``."""
        h_proj = self._get_params_at_queries(query_times, ts)  # (B, Q, D)
        w = self.head.factor_intensity_decay  # (1, D)
        b = self.head.factor_intensity_base  # (1, D)
        t_start = self._event_time_before_queries(query_times, ts)  # (B, Q)
        t_rel = (query_times - t_start).unsqueeze(-1)  # (B, Q, 1)
        return self.softplus(w * t_rel + h_proj + b)  # (B, Q, D)

    def _sequence_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """(B, L, D) intensities in one forward pass."""
        B, L = ts.time_points.shape
        if L == 0:
            return torch.zeros(B, 0, self.D, device=ts.time_points.device)
        cached = self._get_cached_params(ts)
        full = cached if cached is not None else self._full_params(ts)  # (B, L+1, D)
        col_idx = ts.causal_indices  # (B, L)
        idx = col_idx.unsqueeze(-1).expand(B, L, self.D)
        h_proj = full.gather(1, idx)  # (B, L, D) — linear(h) projections
        w = self.head.factor_intensity_decay  # (1, D)
        b = self.head.factor_intensity_base  # (1, D)
        tp = ts.time_points  # (B, L)
        tp_with_zero = torch.cat([torch.zeros(B, 1, device=tp.device, dtype=tp.dtype), tp], dim=1)
        t_start = tp_with_zero.gather(1, col_idx)  # (B, L)
        t_rel = (tp - t_start).unsqueeze(-1)  # (B, L, 1)
        state = w * t_rel + h_proj + b  # (B, L, D)
        return self.softplus(state)  # (B, L, D)


class EasyTPPSAHPProcess(TransformerParametrizedTPP):
    """Exact replica of the EasyTPP Self-Attentive Hawkes Process (Zhang et al., ICML 2020).

    From the encoder output ``h``, three quantities are predicted::

        mu_d    = GELU(W_mu @ h)           — asymptotic intensity (can be negative)
        eta_d   = GELU(W_eta @ h)          — initial intensity   (can be negative)
        gamma_d = Softplus(W_gamma @ h)    — decay rate (> 0)

    State-decay (pre-activation intensity)::

        state_d(t) = mu_d + (eta_d - mu_d) * exp(-gamma_d * t_rel)

    Final intensity::

        λ_d(t) = ScaledSoftplus(state_d(t))

    This differs from the TorchPP analytical SAHP which enforces positivity
    on (mu, eta, gamma) individually via exp/softplus, avoiding the outer
    ScaledSoftplus and enabling analytical CI.  Here we replicate EasyTPP
    exactly: GELU activations, no bias, ScaledSoftplus wrapping, numerical CI.

    Params shape: ``(B, L+1, 3, D)`` — stacked (mu, eta, gamma).
    """

    def __init__(
        self,
        config: TransformerConfig,
        terminating: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(config, terminating=terminating, seed=seed)
        self.head = EasyTPPSAHPHead(config)
        self.softplus = ScaledSoftplus(config.num_event_types)

    def _default_ci_config(self) -> CIConfig:
        """ScaledSoftplus has no closed-form integral; the intensity is smooth within an
        interval and jumps only at events → event-knotted per-interval trapezoid CI."""
        return CIConfig("numerical", "interval", "trapezoidal", n_points=100)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        embedding = self._encode(ts)
        return self.head(embedding)  # (B, L, 3, D) — (mu, eta, gamma) already activated

    def _get_prior(self) -> torch.Tensor:
        return self.head.get_prior()  # (3, D)

    def _intensity_at_t(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        params = self._get_params_at_time(t, ts)  # (B, 3, D)
        mu = params[:, 0, :]  # (B, D) — output of GELU
        eta = params[:, 1, :]  # (B, D) — output of GELU
        gamma = params[:, 2, :]  # (B, D) — output of Softplus, > 0
        t_start = ts.event_time_before(t)  # (B,)
        t_rel = (t - t_start).unsqueeze(-1)  # (B, 1)
        state = mu + (eta - mu) * torch.exp(-gamma * t_rel)  # (B, D)
        return self.softplus(state)  # (B, D)

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """``softplus(mu + (eta−mu)·exp(−gamma·t_rel))`` per query. ``(B, Q)`` -> ``(B, Q, D)``."""
        params = self._get_params_at_queries(query_times, ts)  # (B, Q, 3, D)
        mu, eta, gamma = params[..., 0, :], params[..., 1, :], params[..., 2, :]  # (B, Q, D) each
        t_start = self._event_time_before_queries(query_times, ts)  # (B, Q)
        t_rel = (query_times - t_start).unsqueeze(-1)  # (B, Q, 1)
        state = mu + (eta - mu) * torch.exp(-gamma * t_rel)  # (B, Q, D)
        return self.softplus(state)  # (B, Q, D)

    def _sequence_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """(B, L, D) intensities in one forward pass."""
        B, L = ts.time_points.shape
        if L == 0:
            return torch.zeros(B, 0, self.D, device=ts.time_points.device)
        cached = self._get_cached_params(ts)
        full = cached if cached is not None else self._full_params(ts)  # (B, L+1, 3, D)
        col_idx = ts.causal_indices  # (B, L)
        idx = col_idx.unsqueeze(-1).unsqueeze(-1).expand(B, L, 3, self.D)
        params = full.gather(1, idx)  # (B, L, 3, D)
        mu = params[:, :, 0, :]  # (B, L, D)
        eta = params[:, :, 1, :]  # (B, L, D)
        gamma = params[:, :, 2, :]  # (B, L, D)
        tp = ts.time_points  # (B, L)
        tp_with_zero = torch.cat([torch.zeros(B, 1, device=tp.device, dtype=tp.dtype), tp], dim=1)
        t_start = tp_with_zero.gather(1, col_idx)  # (B, L)
        t_rel = (tp - t_start).unsqueeze(-1)  # (B, L, 1)
        state = mu + (eta - mu) * torch.exp(-gamma * t_rel)  # (B, L, D)
        return self.softplus(state)  # (B, L, D)


class NeuralMLPProcess(TransformerParametrizedTPP):
    """Neural TPP with an MLP intensity head.

    The transformer backbone encodes event history into context vectors ``h``
    which are cached via ``prepare_batch``.  At each query time ``t``, the
    intensity is evaluated by a small MLP that ingests ``h`` concatenated with
    a sinusoidal encoding of the elapsed time ``t - t_prev``::

        λ(t) = ScaledSoftplus(MLP([h ; sin_enc(t - t_prev)]))

    This gives maximum modelling flexibility for the inter-event intensity
    shape — the MLP can represent arbitrary non-negative intensity functions.
    Because the transformer is run only once and the result cached, the
    per-query cost is just the lightweight MLP forward pass.

    Cumulative intensity is computed numerically (per-interval mc-trapezoid CIConfig).

    Params shape: ``(B, L+1, n_embd)`` — raw encoder outputs + learned prior.
    """

    def __init__(
        self,
        config: TransformerConfig,
        terminating: bool = False,
        seed: Optional[int] = None,
        n_hidden: int = 128,
        n_mlp_layers: int = 2,
        time_embd_dim: int | None = None,
    ):
        super().__init__(config, terminating=terminating, seed=seed)
        self.head = MLPIntensityHead(
            config,
            n_hidden=n_hidden,
            n_mlp_layers=n_mlp_layers,
            time_embd_dim=time_embd_dim,
        )
        self.softplus = ScaledSoftplus(config.num_event_types)

    def _default_ci_config(self) -> CIConfig:
        """Softplus intensity is smooth within an interval and jumps only at events → event-knotted
        per-interval grid.  MC-trapezoidal (random nodes each step) stops the MLP from hiding
        intensity spikes between fixed nodes to evade the survival penalty (the B1 gaming finding)."""
        return CIConfig("numerical", "interval", "mc_trapezoidal", n_points=100)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Return raw encoder outputs. Shape: ``(B, L, n_embd)``."""
        return self._encode(ts)

    def _get_prior(self) -> torch.Tensor:
        """Prior embedding vector. Shape: ``(n_embd,)``."""
        return self.head.get_prior()

    def _intensity_at_t(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        h = self._get_params_at_time(t, ts)  # (B, n_embd)
        t_start = ts.event_time_before(t)  # (B,)
        t_rel = t - t_start  # (B,)
        raw = self.head(h, t_rel)  # (B, D)
        return self.softplus(raw)  # (B, D)

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """MLP intensity ``softplus(head(h, t_rel))`` per query. ``(B, Q)`` -> ``(B, Q, D)``.

        The head accepts a leading axis (as in ``_sequence_intensities``), so the whole
        query grid runs in one batched MLP pass.
        """
        h = self._get_params_at_queries(query_times, ts)  # (B, Q, n_embd)
        t_start = self._event_time_before_queries(query_times, ts)  # (B, Q)
        t_rel = query_times - t_start  # (B, Q)
        raw = self.head(h, t_rel)  # (B, Q, D)
        return self.softplus(raw)  # (B, Q, D)

    def _sequence_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Compute intensities at all event times in one batched MLP pass. Shape: ``(B, L, D)``."""
        B, L = ts.time_points.shape
        if L == 0:
            return torch.zeros(B, 0, self.D, device=ts.time_points.device)
        cached = self._get_cached_params(ts)
        full = cached if cached is not None else self._full_params(ts)  # (B, L+1, n_embd)
        col_idx = ts.causal_indices  # (B, L)
        idx = col_idx.unsqueeze(-1).expand(B, L, full.shape[-1])
        h = full.gather(1, idx)  # (B, L, n_embd)
        tp = ts.time_points
        tp_with_zero = torch.cat([torch.zeros(B, 1, device=tp.device, dtype=tp.dtype), tp], dim=1)
        t_start = tp_with_zero.gather(1, col_idx)  # (B, L)
        t_rel = tp - t_start  # (B, L)
        raw = self.head(h, t_rel)  # (B, L, D)
        return self.softplus(raw)  # (B, L, D)


# ===========================================================================
# IntensityFree (Shchur et al., ICLR 2020)
# ===========================================================================


class IntensityFreeProcess(NeuralEncoderMixin, DensityBasedTPP):
    """Intensity-free TPP: a log-normal mixture on the inter-event time.

    Faithful replica of *Intensity-Free Learning of Temporal Point Processes*
    (Shchur et al., ICLR 2020).  The joint factorises as ``p(t, d | H) =
    f(tau) · p(d | H)`` — a log-normal mixture density ``f(tau)`` on the inter-event
    time (``tau = t - t_last``) and a categorical mark predicted from the history
    embedding only, **not** conditioned on the inter-event time (paper Appendix F.2:
    "we assume that the marks are conditionally independent of the time given the
    history").

    Realized natively as a :class:`~torchpp.tpp.base_tpp.DensityBasedTPP`: the
    contract (``likelihood`` / ``density_*`` / ``prob_*`` / ``sample`` / ``time_*``)
    is computed directly from the mixture ``log_prob`` / ``log_survival_function`` and
    the mark head, with **no derived hazard on the contract path**.  A diagnostic
    hazard ``lambda_d(t) = (f(tau)/S(tau))·p(d|H)`` is still exposed by
    :meth:`intensity` for plotting only.

    Terminating mode (restricted process)
    -------------------------------------
    When each type occurs at most once, an occurred type can no longer fire, so the
    process continues on the still-active set ``M' ⊆ M`` with mark mass
    ``A = Σ_{d∈M'} p(d | H) ∈ [0, 1]``.  Equivalently (the intensity view): the per-type
    intensities are all proportional to one shared hazard, ``λ_d(t) = (f(τ)/S(τ))·p(d)``,
    so dropping the occurred types just scales the *total* hazard by ``A``::

        λ'_total = A·f/S   ⟹   Λ'(τ) = −A·log S(τ)   ⟹   S'(τ) = S(τ)^A
        f'(τ)    = λ'_total·S' = A·f(τ)·S(τ)^{A−1}       (density of the reduced CDF 1 − S^A)

    The joint still factorises — ``p(t, d | M') = f'(τ) · [p(d)/A]`` for ``d ∈ M'`` — so
    the time law becomes the reduced density ``f'`` and the mark head is renormalised
    over the active types (``p(d)/A``).  For an observed sequence, each interval's
    survival covers only the types active when it starts, which adds
    ``Σ_i (A_i − 1)·log S(τ_i)`` to the per-event log-likelihood.  **Non-terminating is
    the ``A = 1`` special case** (``S^1 = S``, ``f·S^0 = f``, ``p/1 = p``); it is kept as
    a separate branch in each hook below so the exact ``f`` / ``S`` / ``p`` are returned
    without the ``0·(−∞)`` that ``A = 1`` would introduce once ``S → 0``.  A fully
    terminated history (``A = 0``) has no possible next event: ``S' ≡ 1``, ``f' ≡ 0``.

    Params layout: ``(B, L+1, 3K + D)`` — ``[locs(K), log_scales(K),
    log_weights(K), mark_logits(D)]``.

    The encoder is configurable: a :class:`GRUEncoder` (faithful EasyTPP/Shchur
    replica) or any other :class:`SequenceEncoder` (e.g. a shared
    :class:`TransformerEncoder` for a matched-encoder comparison).
    """

    def __init__(
        self,
        encoder: SequenceEncoder,
        num_mix_components: int = 8,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        terminating: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(encoder=encoder, terminating=terminating, seed=seed)
        self.K = int(num_mix_components)
        self.mean_log_inter_time = float(mean_log_inter_time)
        self.std_log_inter_time = float(std_log_inter_time)

        n_embd = encoder.n_embd
        self.mix_linear = torch.nn.Linear(n_embd, 3 * self.K)
        self.mark_linear = torch.nn.Linear(n_embd, self.D)
        # Priors for the no-history column (column 0 of _full_params).
        self._prior_mix = torch.nn.Parameter(torch.zeros(3 * self.K))
        self._prior_mark = torch.nn.Parameter(torch.zeros(self.D))

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        h = self._encode(ts)  # (B, L, n_embd)
        return torch.cat([self.mix_linear(h), self.mark_linear(h)], dim=-1)  # (B, L, 3K+D)

    def _get_prior(self) -> torch.Tensor:
        return torch.cat([self._prior_mix, self._prior_mark])  # (3K+D,)

    def _unpack(self, params: torch.Tensor, dtype: torch.dtype):
        """Split packed params into a mixture distribution and mark log-probs.

        ``params`` has trailing dim ``3K + D``; leading dims are arbitrary.
        Returns ``(dist, mark_log_probs)`` where ``dist`` has the leading shape
        as batch shape and ``mark_log_probs`` has shape ``(..., D)``.
        """
        K = self.K
        locs = params[..., :K].to(dtype)
        log_scales = clamp_preserve_gradients(params[..., K : 2 * K], _LOG_SCALE_MIN, _LOG_SCALE_MAX).to(dtype)
        log_weights = torch.log_softmax(params[..., 2 * K : 3 * K], dim=-1).to(dtype)
        mark_log_probs = torch.log_softmax(params[..., 3 * K :], dim=-1).to(dtype)
        dist = LogNormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_log_inter_time=self.mean_log_inter_time,
            std_log_inter_time=self.std_log_inter_time,
        )
        return dist, mark_log_probs

    # ------------------------------------------------------------------
    # DensityBasedTPP hooks — realized from the mixture + mark head
    # ------------------------------------------------------------------

    def _history_dist(self, t: torch.Tensor, ts: BatchedMVEventData):
        """Full-history mixture at the last observed event and tau = t - t_last per query.

        The predictive next-event density conditions on the WHOLE history H (the mark head's
        right-limit column after all events); the inter-event time is measured from
        ``t_last = ts.max_time``, not from the event before ``t`` (that would jump at the
        exact boundary ``t = t_last``).  Returns ``(dist, tau)`` with ``dist`` batch shape
        matching ``t`` (``(B,)`` or ``(B, Q)``).
        """
        t_last = self._last_event_time(ts)  # (B,) — padding-aware last real event
        params = self._get_params_at_time(t_last + _FULL_HISTORY_OFFSET, ts)  # (B, 3K+D) — column after all events
        if t.dim() == 2:
            params = params.unsqueeze(1).expand(params.shape[0], t.shape[1], params.shape[-1])
            tau = (t - t_last.unsqueeze(1)).clamp(min=_MIN_INTER_EVENT_TIME)  # (B, Q)
        else:
            tau = (t - t_last).clamp(min=_MIN_INTER_EVENT_TIME)  # (B,)
        dist, _ = self._unpack(params, dtype=t.dtype)
        return dist, tau

    def _log_remaining_mark_mass(self, ts: BatchedMVEventData) -> torch.Tensor:
        """log A = log Σ_{d active} p(d | H), the mark mass on still-active types.  ``(B,)``.

        ``A`` is the scale factor of the restricted terminating process (class
        "Terminating mode" derivation); ``A = 1`` when all types are still active.
        """
        dtype = ts.time_points.dtype
        t_query = self._last_event_time(ts) + _FULL_HISTORY_OFFSET  # full-history column
        _, mark_log_probs = self._unpack(self._get_params_at_time(t_query, ts), dtype=dtype)  # (B, D)
        active = self._get_termination_mask(ts)  # (B, D) bool — not yet occurred
        return torch.logsumexp(mark_log_probs.masked_fill(~active, float("-inf")), dim=-1)  # (B,)

    def _log_density_time_given_mark(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """log p(t | m, H) = log f(τ), the SAME inter-event density for every mark (IFTPP is
        mark ⟂ time), broadcast over the ``D`` mark axis → ``(B,[Q,]D)``.

        Terminating: the reduced-process density ``f'(τ) = A·f·S^{A-1}`` (class
        "Terminating mode"); non-terminating is ``f`` directly (the A=1 case).
        """
        if t.shape[0] == 0:
            return torch.zeros(*t.shape, self.D, dtype=t.dtype, device=t.device)
        dist, tau = self._history_dist(t, ts)
        log_f = dist.log_prob(tau)  # (B,[Q])
        if self.terminating:
            log_A = self._log_remaining_mark_mass(ts)  # (B,)
            log_S = dist.log_survival_function(tau)
            if log_f.dim() == 2:
                log_A = log_A.unsqueeze(1)
            A = log_A.exp()
            reduced = log_A + log_f + (A - 1.0) * log_S
            log_f = torch.where(A == 0, torch.full_like(reduced, float("-inf")), reduced)  # fully terminated -> no next event
        return log_f.unsqueeze(-1).expand(*log_f.shape, self.D)  # (B,[Q,]D) — identical across marks

    def _log_survival_time_given_mark(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """log S(t | m, H) = log S(τ), the SAME survival for every mark, broadcast over ``D`` → ``(B,[Q,]D)``.

        Terminating: the reduced survival ``S'(τ) = S^A`` (class "Terminating mode");
        a fully terminated history (``A = 0``) survives forever (``log S' = 0``).
        """
        if t.shape[0] == 0:
            return torch.zeros(*t.shape, self.D, dtype=t.dtype, device=t.device)
        dist, tau = self._history_dist(t, ts)
        log_S = dist.log_survival_function(tau)  # (B,[Q])
        if self.terminating:
            A = self._log_remaining_mark_mass(ts).exp()  # (B,)
            if log_S.dim() == 2:
                A = A.unsqueeze(1)
            log_S = torch.where(A == 0, torch.zeros_like(log_S), A * log_S)  # S^A; A=0 -> S=1 (avoid 0*-inf)
        return log_S.unsqueeze(-1).expand(*log_S.shape, self.D)  # (B,[Q,]D) — identical across marks

    def _log_prob_mark(self, ts: BatchedMVEventData) -> torch.Tensor:
        """log p(m | H) — mark head evaluated after all events (full-history column).  ``(B, D)``.

        Terminating: occurred types get ``-inf`` and the remaining mass is renormalised
        (log-softmax over the active types), so no probability is assigned to a type
        that can no longer occur.
        """
        dtype = ts.time_points.dtype
        if ts.time_points.shape[0] == 0:
            return torch.zeros(0, self.D, dtype=dtype, device=ts.time_points.device)
        t_query = self._last_event_time(ts) + _FULL_HISTORY_OFFSET  # strictly after the last event -> full-history column
        _, mark_log_probs = self._unpack(self._get_params_at_time(t_query, ts), dtype=dtype)  # (B, D)
        if not self.terminating:
            return mark_log_probs
        active = self._get_termination_mask(ts)  # (B, D) bool — not yet occurred
        masked = mark_log_probs.masked_fill(~active, float("-inf"))
        denom = torch.logsumexp(masked, dim=-1, keepdim=True)  # (B, 1) = log A
        # Renormalise over the active types; a fully terminated row (denom = -inf) stays all -inf.
        return torch.where(torch.isneginf(denom), masked, masked - denom)

    def _event_log_density_sum(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Sum_i [log f(tau_i) + log p(m_i | H_i)] over valid events.  ``(B,)``.

        Gathers per-event mixture params causally (as ``_sequence_intensities`` did),
        with the padding / empty-sequence / terminating-repeat handling that
        ``_event_log_intensity_sum`` applies for the intensity family.
        """
        B, L = ts.time_points.shape
        device = ts.time_points.device
        if L == 0:
            return torch.zeros(B, device=device)

        valid_events = ts.event_mask  # (B, L)
        cached = self._get_cached_params(ts)
        full = cached if cached is not None else self._full_params(ts)  # (B, L+1, 3K+D)
        col_idx = ts.causal_indices  # (B, L)
        idx = col_idx.unsqueeze(-1).expand(B, L, full.shape[-1])
        params = full.gather(1, idx)  # (B, L, 3K+D)

        tp = ts.time_points  # (B, L)
        tp_with_zero = torch.cat([torch.zeros(B, 1, device=device, dtype=tp.dtype), tp], dim=1)
        tau = (tp - tp_with_zero.gather(1, col_idx)).clamp(min=_MIN_INTER_EVENT_TIME)  # (B, L)

        dist, mark_log_probs = self._unpack(params, dtype=tp.dtype)  # dist batch (B, L); marks (B, L, D)
        log_f = dist.log_prob(tau)  # (B, L)
        log_pm = mark_log_probs.gather(2, ts.event_types_idx_safe.unsqueeze(-1)).squeeze(-1)  # (B, L)
        log_event = (log_f + log_pm).masked_fill(~valid_events, 0.0)  # (B, L)
        log_sum = log_event.sum(dim=1)  # (B,)

        log_sum = log_sum.masked_fill(valid_events.sum(dim=1) == 0, 0.0)  # empty -> 0
        if self.terminating:
            # Occurred types carry no survival penalty after their occurrence, so each
            # interval's survival covers only the types still active when it starts.  This
            # adds Σ_i (A_i - 1)·log S(τ_i) to the per-event term (A_i the active mark mass
            # over [t_{i-1}, ·)), reproducing the pre-refactor intensity-harness likelihood.
            log_S = dist.log_survival_function(tau)  # (B, L)
            t_start = tp_with_zero.gather(1, col_idx)  # (B, L) interval start t_{i-1}
            t_term = self._get_termination_times(ts)  # (B, D)
            active = t_term.unsqueeze(1) > t_start.unsqueeze(2)  # (B, L, D): type active over interval i
            A = (mark_log_probs.exp() * active.to(mark_log_probs.dtype)).sum(dim=-1)  # (B, L)
            correction = ((A - 1.0) * log_S).masked_fill(~valid_events, 0.0)  # (B, L)
            log_sum = log_sum + correction.sum(dim=1)
            # A repeated type is impossible under first-occurrence semantics (log p = -inf).
            log_sum = log_sum.masked_fill(self._has_repeated_event_types(ts), float("-inf"))
        return log_sum

    def _draw_next(self, ts_current: MVEventData, rng):
        """Draw the next (time, mark) directly from the mixture + mark head."""
        device = ts_current.time_points.device
        batched = BatchedMVEventData([ts_current.time_points], [ts_current.event_types])
        t_last = self._last_event_time(batched)  # (1,) — padding-aware last real event
        params = self._get_params_at_time(t_last + _FULL_HISTORY_OFFSET, batched)  # (1, 3K+D)
        dist, mark_log_probs = self._unpack(params, dtype=t_last.dtype)  # dist batch (1,); marks (1, D)
        t_next = t_last + self._sample_mixture(dist, rng, device)  # (1,)

        probs = mark_log_probs.exp()[0]  # (D,)
        if self.terminating:
            active = self._get_termination_mask(batched)[0]  # (D,) bool — types not yet occurred
            probs = probs * active.to(probs.dtype)
            total = probs.sum()
            if total <= 0:  # no mark mass on the remaining types -> uniform over them
                remaining = torch.where(active)[0]
                if remaining.numel() == 0:
                    return None
                pick = torch.randint(remaining.numel(), (1,), generator=rng, device=device)
                return t_next, remaining[pick], active.to(probs.dtype) / active.sum()
            probs = probs / total
        type_idx = torch.multinomial(probs, num_samples=1, generator=rng)  # (1,)
        return t_next, type_idx, probs

    def _sample_mixture(self, dist, rng, device) -> torch.Tensor:
        """Reproducible draw of one inter-event time per batch row from the log-normal mixture.

        ``torch.distributions.sample`` ignores a generator, so we draw the component and
        the base normal manually with ``rng`` and push them through the mixture's
        affine/exp transforms.
        """
        gmm = dist.base_dist  # MixtureSameFamily
        probs = torch.softmax(gmm.mixture_distribution.logits, dim=-1)  # (B, K)
        k = torch.multinomial(probs, num_samples=1, generator=rng)  # (B, 1)
        loc_k = gmm.component_distribution.loc.gather(-1, k).squeeze(-1)  # (B,)
        scale_k = gmm.component_distribution.scale.gather(-1, k).squeeze(-1)  # (B,)
        z = loc_k + scale_k * torch.randn(loc_k.shape, generator=rng, device=device, dtype=loc_k.dtype)  # (B,)
        for transform in dist.transforms:
            z = transform(z)
        return z  # (B,) inter-event time

    def likelihood(self, ts, T=None, log=True):
        """Native density log-likelihood with a single encoder forward pass (caches params)."""
        self.prepare_batch(ts)
        try:
            return super().likelihood(ts, T, log=log)
        finally:
            self.clear_batch()

    # ------------------------------------------------------------------
    # Diagnostic hazard (plotting only — off the contract path)
    # ------------------------------------------------------------------

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Diagnostic hazard ``lambda_d(t) = (f(tau)/S(tau))·p(d|H)`` for plotting.

        Provided so notebooks can plot the IFTPP hazard; nothing in the
        likelihood / predictive / eval / sampling path routes through it.  Handles
        ``(B,)`` and ``(B, Q)`` queries and mirrors the intensity family's
        termination masking so a diagnostic overlay stays consistent.
        """
        assert (t >= 0.0).all(), "Intensity is only defined for t >= 0."
        if t.dim() == 1:
            if t.shape[0] == 0:
                return torch.zeros(0, self.D, dtype=t.dtype, device=t.device)
            dist, mark_log_probs = self._unpack(self._get_params_at_time(t, ts), dtype=t.dtype)
            tau = (t - ts.event_time_before(t)).clamp(min=_MIN_INTER_EVENT_TIME)  # (B,)
            hazard = torch.exp(dist.log_prob(tau) - dist.log_survival_function(tau))  # (B,)
            out = hazard.unsqueeze(-1) * mark_log_probs.exp()  # (B, D)
            if self.terminating:
                out = out * self._get_termination_mask(ts, t=t).to(out.dtype)
            return out
        if t.shape[0] == 0:
            return torch.zeros(0, t.shape[1], self.D, dtype=t.dtype, device=t.device)
        dist, mark_log_probs = self._unpack(self._get_params_at_queries(t, ts), dtype=t.dtype)  # batch (B, Q)
        tau = (t - self._event_time_before_queries(t, ts)).clamp(min=_MIN_INTER_EVENT_TIME)  # (B, Q)
        hazard = torch.exp(dist.log_prob(tau) - dist.log_survival_function(tau))  # (B, Q)
        out = hazard.unsqueeze(-1) * mark_log_probs.exp()  # (B, Q, D)
        if self.terminating:
            out = out * self._terminated_mask(t, ts).to(out.dtype)
        return out

    # ------------------------------------------------------------------
    # Point predictor: closed-form analytical mean of the next event time
    # ------------------------------------------------------------------

    def time_mean(self, ts: BatchedMVEventData) -> torch.Tensor:
        """E[t_next | H] via the closed-form log-normal mixture mean."""
        return self.analytical_mean_next_event_time(ts)

    def analytical_mean_next_event_time(self, ts: BatchedMVEventData) -> torch.Tensor:
        """``t_last + E[tau]`` using the closed-form log-normal mixture mean.

        ``E[tau] = sum_k w_k · exp(sigma·loc_k + mu + 0.5·(sigma·scale_k)^2)``.
        Exposed alongside the grid-based mean / median (RMSE) estimators that the
        evaluation pipeline produces; not used by the default eval.
        """
        t_last = self._last_event_time(ts)  # (B,) padding-aware last real event
        params = self._get_params_at_time(t_last + _FULL_HISTORY_OFFSET, ts)  # (B, 3K+D)
        K, dtype = self.K, ts.time_points.dtype
        locs = params[..., :K].to(dtype)
        scales = clamp_preserve_gradients(params[..., K : 2 * K], _LOG_SCALE_MIN, _LOG_SCALE_MAX).exp().to(dtype)
        log_weights = torch.log_softmax(params[..., 2 * K : 3 * K], dim=-1).to(dtype)
        mu, sigma = self.mean_log_inter_time, self.std_log_inter_time
        log_component_means = sigma * locs + mu + 0.5 * (sigma * scales) ** 2  # (B, K)
        e_tau = torch.logsumexp(log_weights + log_component_means, dim=-1).exp()  # (B,)
        return t_last + e_tau


# ===========================================================================
# Neural Hawkes Process (Mei & Eisner, NeurIPS 2017)
# ===========================================================================


class NHPProcess(NeuralParametrizedTPP):
    """Neural Hawkes Process — a continuous-time LSTM self-modulating intensity.

    Faithful port of the EasyTPP reference.  A :class:`ContTimeLSTMEncoder`
    produces, after each event, the right-limit state ``(c, c_bar, delta, o)``.
    At a query time ``t`` in the interval after event ``l``, the state is decayed
    by ``t - t_l`` to give ``h(t)`` and the intensity is::

        lambda_d(t) = ScaledSoftplus( Linear(h(t)) )_d

    ``ScaledSoftplus`` has no closed-form integral, so the cumulative intensity is
    computed numerically (per-interval trapezoid CIConfig), matching THP/SAHP/NHP in
    EasyTPP.

    Params layout: ``(B, L+1, 4H)`` — the right-limit CT-LSTM states + the prior
    (zero) initial state.
    """

    def __init__(
        self,
        encoder: ContTimeLSTMEncoder,
        terminating: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(encoder=encoder, terminating=terminating, seed=seed)
        self.H = encoder.hidden_dim
        self.linear = torch.nn.Linear(self.H, self.D)
        self.softplus = ScaledSoftplus(self.D)
        # Prior = the CT-LSTM initial state (zeros), as in EasyTPP's get_init_state.
        self.register_buffer("_prior_state", torch.zeros(4 * self.H))

    def _default_ci_config(self) -> CIConfig:
        """ScaledSoftplus has no closed-form integral → per-interval trapezoid numerical CI.
        Both the survival term and point-CI route through this one CIConfig (no hybrid)."""
        return CIConfig("numerical", "interval", "trapezoidal", n_points=100)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        return self._encode(ts)  # (B, L, 4H) right-limit states

    def _get_prior(self) -> torch.Tensor:
        return self._prior_state  # (4H,)

    def _intensity_from_state(self, params: torch.Tensor, t_rel: torch.Tensor) -> torch.Tensor:
        """Decay a right-limit state by ``t_rel`` and apply the intensity head.

        ``params`` is ``(..., 4H)``; ``t_rel`` is ``(...)``. Returns ``(..., D)``.
        """
        c, c_bar, delta, o = torch.chunk(params, 4, dim=-1)  # each (..., H)
        _, h = ContTimeLSTMCell.decay(c, c_bar, delta, o, t_rel.unsqueeze(-1))  # (..., H)
        h = h.to(self.linear.weight.dtype)
        # Clamp to a finite max so a pathological blow-up (inf) does not poison the
        # downstream log / cumulative-intensity; healthy intensities are unaffected.
        return self.softplus(self.linear(h)).clamp(max=_NHP_INTENSITY_MAX)  # (..., D)

    def _intensity_at_t(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        if t.shape[0] == 0:
            return torch.zeros(0, self.D, dtype=t.dtype, device=t.device)
        params = self._get_params_at_time(t, ts)  # (B, 4H)
        t_start = ts.event_time_before(t)  # (B,)
        return self._intensity_from_state(params, (t - t_start))  # (B, D)

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """CT-LSTM right-limit state gathered per query and decayed. ``(B, Q)`` -> ``(B, Q, D)``.

        The flat analogue of :meth:`_interval_intensities` for arbitrary query times: each
        query uses the causal state column for the events strictly before it, decayed by the
        relative time since that governing event.
        """
        params = self._get_params_at_queries(query_times, ts)  # (B, Q, 4H)
        t_start = self._event_time_before_queries(query_times, ts)  # (B, Q)
        return self._intensity_from_state(params, query_times - t_start)  # (B, Q, D)

    def _interval_intensities(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Structured per-interval intensity via the state-decay fold. ``(B,R,P) -> (B,R,P,D)``.

        Row ``i`` is interval ``I_i``, governed by causal column ``i`` (the right-limit
        CT-LSTM state after the events strictly before the interval's left endpoint),
        decayed to each query time.  At the left endpoint the just-fired event is
        therefore included (right-limit) — the conditioning the per-interval
        survival quadrature uses, so the numerical interval CI reproduces the prior
        ``_raw_ci`` numbers.  Zero-width / padding rows are dropped by the CI weights.
        """
        B, R, P = query_times.shape
        cached = self._get_cached_params(ts)
        full = cached if cached is not None else self._full_params(ts)  # (B, L+1, 4H)
        params_rows = full[:, :R, :]  # (B, R, 4H) — column i governs interval i
        tp = ts.time_points
        tp_with_zero = torch.cat([torch.zeros(B, 1, device=tp.device, dtype=tp.dtype), tp], dim=1)  # (B, L+1)
        t_start = tp_with_zero[:, :R]  # (B, R) — interval left endpoints
        t_rel = query_times - t_start.unsqueeze(-1)  # (B, R, P)
        params_exp = params_rows.unsqueeze(2).expand(B, R, P, full.shape[-1])
        return self._intensity_from_state(params_exp, t_rel)  # (B, R, P, D)

    def _sequence_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        B, L = ts.time_points.shape
        if L == 0:
            return torch.zeros(B, 0, self.D, device=ts.time_points.device)
        cached = self._get_cached_params(ts)
        full = cached if cached is not None else self._full_params(ts)  # (B, L+1, 4H)
        col_idx = ts.causal_indices  # (B, L)
        idx = col_idx.unsqueeze(-1).expand(B, L, full.shape[-1])
        params = full.gather(1, idx)  # (B, L, 4H)
        tp = ts.time_points
        tp_with_zero = torch.cat([torch.zeros(B, 1, device=tp.device, dtype=tp.dtype), tp], dim=1)
        t_start = tp_with_zero.gather(1, col_idx)  # (B, L)
        return self._intensity_from_state(params, (tp - t_start))  # (B, L, D)


# ===========================================================================
# Attentive Neural Hawkes Process (Yang et al., ICLR 2022)
# ===========================================================================


class AttNHPProcess(NeuralParametrizedTPP):
    """Attentive Neural Hawkes Process.

    Faithful port of the EasyTPP reference.  Unlike THP/NHP there is no per-event
    state that is merely decayed: the intensity at a query time is produced by
    re-running the attention stack with a temporal embedding of that query time,
    attending to the events that strictly precede it (torchPP causal convention).

        lambda_d(t) = ScaledSoftplus( Linear( state(t) ) )_d

    The :class:`AttNHPEncoder` provides the per-event embeddings (cached) and the
    query-time state machinery.  Cumulative intensity is numerical (per-interval
    trapezoid CIConfig).  Because each query re-runs attention, eval cost scales with
    the number of CI grid points — the same cost class as EasyTPP's Monte-Carlo integration.

    Params layout: ``(B, L+1, d_model + d_time)`` — per-event embeddings + prior.
    """

    def __init__(
        self,
        encoder: AttNHPEncoder,
        terminating: bool = False,
        seed: Optional[int] = None,
        quadrature_chunk_size: Optional[int] = 10,
    ):
        super().__init__(encoder=encoder, terminating=terminating, seed=seed)
        self.attn_encoder = encoder
        self.inten_linear = torch.nn.Linear(encoder.state_dim, self.D)
        self.softplus = ScaledSoftplus(self.D)
        self.register_buffer("_prior_emb", torch.zeros(encoder.n_embd))
        # Nodes per batch-folded quadrature pass; None/0 -> all P in one pass.  Folded into
        # the CIConfig by _default_ci_config, and read directly by _interval_intensities.
        self.quadrature_chunk_size = quadrature_chunk_size

    def _default_ci_config(self) -> CIConfig:
        """Per-interval trapezoid numerical CI, carrying ``quadrature_chunk_size`` as the
        CIConfig memory knob.  Both survival and point-CI route through this one config."""
        return CIConfig("numerical", "interval", "trapezoidal", 100, chunk_size=self.quadrature_chunk_size)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        return self._encode(ts)  # (B, L, d_model + d_time)

    def _get_prior(self) -> torch.Tensor:
        return self._prior_emb  # (d_model + d_time,)

    def _event_emb(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Per-event embeddings (B, L, De), from the cache when available."""
        cached = self._get_cached_params(ts)
        if cached is not None:
            return cached[:, 1:, :]  # drop the prior column
        return self._encode(ts)

    def _lambda_at_queries(self, ts: BatchedMVEventData, query_times: torch.Tensor) -> torch.Tensor:
        """Intensity at ``query_times`` (B, Q) -> (B, Q, D)."""
        event_emb = self._event_emb(ts)  # (B, L, De)
        state = self.attn_encoder.states_at_queries(
            event_emb, ts.time_points, ts.event_mask, query_times
        )  # (B, Q, state_dim)
        return self.softplus(self.inten_linear(state))  # (B, Q, D)

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Flat multi-query intensity (batched attention over all queries).

        Each query re-runs attention over the events strictly before its own time
        (events are encoded once inside ``states_at_queries``).  ``(B, Q) -> (B, Q, D)``.
        The leaf ``_intensity_at_t`` is the base default (Q=1) over this hook.
        """
        if query_times.shape[0] == 0:
            return torch.zeros(0, query_times.shape[1], self.D, dtype=query_times.dtype, device=query_times.device)
        return self._lambda_at_queries(ts, query_times)  # (B, Q, D)

    def _interval_intensities(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Structured per-interval intensity via a batch-fold of the node axis. ``(B,R,P) -> (B,R,P,D)``.

        Events are encoded once; the P node axis is folded into the batch dim so the
        attention score matrix stays linear in P (not quadratic, which a query-fold
        would cost). ``quadrature_chunk_size`` splits P to bound peak memory.
        Conditioning is strictly-before-query (left-limit), matching the flat hook
        and the survival quadrature.
        """
        B, R, P = query_times.shape
        if B == 0:
            return torch.zeros(0, R, P, self.D, dtype=query_times.dtype, device=query_times.device)
        event_emb = self._event_emb(ts)  # (B, L, De)
        cache = self.attn_encoder.encode_events(event_emb, ts.time_points, ts.event_mask)
        chunk = self.quadrature_chunk_size or P
        parts = []
        for c0 in range(0, P, chunk):
            c1 = min(c0 + chunk, P)
            pc = c1 - c0
            q = query_times[..., c0:c1].permute(0, 2, 1).reshape(B * pc, R)  # (B*pc, R)
            state = self.attn_encoder.states_from_cache(cache, q)  # (B*pc, R, state_dim)
            lam = self.softplus(self.inten_linear(state)).reshape(B, pc, R, self.D)  # (B, pc, R, D)
            parts.append(lam.permute(0, 2, 1, 3))  # (B, R, pc, D)
        return torch.cat(parts, dim=2)  # (B, R, P, D)

    def _sequence_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        B, L = ts.time_points.shape
        if L == 0:
            return torch.zeros(B, 0, self.D, device=ts.time_points.device)
        return self._lambda_at_queries(ts, ts.time_points)  # (B, L, D)
