"""
Neural Temporal Point Processes.

A neural sequence encoder maps the event history to the parameters of the intensity
on each inter-event interval (closed-form or numerically integrated).
``IntensityFreeProcess`` instead models the inter-event-time density directly.

Class hierarchy
---------------
NeuralEncoderMixin
    Owns a :class:`~torchpp.modules.nn.encoder.SequenceEncoder`, the per-batch parameter
    cache (``prepare_batch`` / ``clear_batch`` / ``fix_context``) and the causal lookup of
    the parameters valid at a query time (``_get_params_at_time``, ``_get_params_at_queries``).

NeuralParametrizedTPP (NeuralEncoderMixin, IntensityBasedTPP)
    Encoder-based intensity models. A subclass implements four hooks:
      - ``_get_params(ts)`` → ``(B, L, ...)`` head parameters from the encoder context.
      - ``_intensities_at_queries(query_times, ts)`` → ``(B, Q, D)`` intensity at query times.
      - ``_interval_ci_from_params(params, t_lo, t_hi)`` → closed-form integral over one
        interval (closed-form heads only; numerical heads use quadrature instead).
      - ``_sequence_intensities(ts)`` → ``(B, L, D)`` intensity at every event in one pass.

TransformerParametrizedTPP (NeuralParametrizedTPP)
    Builds a :class:`~torchpp.modules.nn.transformer.TransformerEncoder` from a
    :class:`~torchpp.modules.nn.transformer.TransformerConfig` (or ``D=``). The
    transformer models (``NeuralPoissonProcess``, ``THPProcess``, …) inherit from it;
    ``NHPProcess`` and ``AttNHPProcess`` take their own encoders, and
    ``IntensityFreeProcess`` combines the mixin with ``DensityBasedTPP``.

Parameter layout
----------------
``_full_params(ts)`` returns a tensor of shape ``(B, L+1, ...)``:
  - Column 0        : a zeros no-history column (no learned prior)
  - Column 1..L     : parameters predicted from encoder context positions 0..L-1

When computing intensities at event position l, we use column l.  With an origin
(``<bos>`` at t=0) the first real event lands in column 1 = ``encoder(<bos>)`` and
column 0 spans only the zero-width ``[0, t_bos]`` interval; without an origin the
first event is conditioned on (dropped), so column 0 is never scored or integrated.

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
from typing import Iterator, Optional, Sequence

import torch

from ..utils.event_data import BatchedMVEventData, MVEventData, TokenRoles
from .base_tpp import CIConfig, DensityBasedTPP, IntensityBasedTPP, SamplingConfig
from ..utils.math import LinearSpline, PiecewiseConstant, loglinear_cumulative_intensity
from ..modules.nn.encoder import SequenceEncoder, ContTimeLSTMEncoder, ContTimeLSTMCell
from ..modules.nn.attnhp import AttNHPEncoder
from ..modules.nn.lognormal_mixture import (
    LogNormalMixtureDistribution,
    clamp_preserve_gradients,
)

from ..modules.nn.transformer import (
    TransformerConfig,
    TransformerEncoder,
    PoissonHead,
    LogLinearHead,
    SplineHead,
    ScaledSoftplus,
    THPHead,
    SAHPHead,
    MLPIntensityHead,
)


# ---------------------------------------------------------------------------
# Numerical constants (IntensityFree density realization; value-preserving)
# ---------------------------------------------------------------------------

# Inter-event time floor: tau = t - t_last is clamped to this before the
# log-normal mixture log_prob / log_survival (guards log(0) as tau -> 0).
_MIN_INTER_EVENT_TIME = 1e-8

# Log-scale clamp bounds for the log-normal mixture (Shchur et al., 2020).
_LOG_SCALE_MIN = -5.0
_LOG_SCALE_MAX = 3.0

# Finite upper bound on the NHP intensity before log / cumulative-intensity.
# ScaledSoftplus can pass through an arbitrarily large linear pre-activation, so
# a pathological batch (huge weights) could otherwise emit inf and poison the
# log/CI.  Far above the healthy range (intensities are O(1e0–1e3)); healthy
# intensities are untouched.
_NHP_INTENSITY_MAX = 1e30


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


def _resolve_transformer_config(config: Optional[TransformerConfig], D: Optional[int]) -> TransformerConfig:
    """``config`` as given, or a default ``TransformerConfig`` for ``D`` event types."""
    if config is None:
        if D is None:
            raise TypeError("Pass `config` (a TransformerConfig) or `D` (the number of event types).")
        return TransformerConfig(num_event_types=D)
    if D is not None and D != config.num_event_types:
        raise ValueError(f"D={D} does not match config.num_event_types={config.num_event_types}.")
    return config


class NeuralEncoderMixin(ABC):
    """Encoder plumbing shared by the intensity and density neural families.

    A standalone mixin (not a base class in its own right): it owns a
    :class:`~torchpp.modules.nn.encoder.SequenceEncoder`, the per-batch parameter
    cache, and the causal parameter-lookup machinery, and forwards construction to
    whatever concrete family base follows it in the MRO (``IntensityBasedTPP`` for
    :class:`NeuralParametrizedTPP`, ``DensityBasedTPP`` for
    :class:`IntensityFreeProcess`).  Inherits :class:`abc.ABC` only so the
    ``_get_params`` hook registers as abstract.

    See module docstring for parameter layout and causal conventions.

    Cache management
    ----------------
    ``prepare_batch(ts)`` runs the encoder once and caches the full params
    tensor ``(B, L+1, ...)`` in ``_cached_full_params``.  ``clear_batch()``
    releases it.  ``_build_integration_grid`` and ``evaluate_at_time`` call
    prepare/clear automatically so the encoder runs only once per batch.
    """

    def __init__(
        self,
        encoder: SequenceEncoder,
        *,
        terminating: bool = False,
        seed: Optional[int] = None,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        # Forward to whichever family base follows the mixin in the MRO
        # (IntensityBasedTPP or DensityBasedTPP); only the intensity family has a CIConfig.
        family_kw = {"ci_config": ci_config} if ci_config is not None else {}
        super().__init__(
            D=encoder.num_event_types,
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            sampling_config=sampling_config,
            **family_kw,
        )
        self.encoder = encoder
        self._check_origin_encoder()
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

    # Whether the zeros no-history column gives a usable intensity. Heads that map zero
    # parameters to a zero intensity need an origin encoder when the model has an origin.
    _zero_params_have_intensity: bool = True

    def _check_origin_encoder(self) -> None:
        """Raise if the model has an origin, its encoder has none, and the head cannot score from zeros.

        Without an origin column the first event is scored from the zeros no-history column;
        for heads that map zero parameters to zero intensity that likelihood is meaningless.
        """
        if self._zero_params_have_intensity:
            return
        if self.has_origin and not getattr(self.encoder, "has_origin", True):
            raise ValueError(
                "token_roles.has_origin=True needs an encoder built with has_origin=True; "
                f"{type(self.encoder).__name__} was built without an origin. Rebuild the model "
                "(e.g. create_model(..., token_roles=TokenRoles(has_origin=True)))."
            )

    # ------------------------------------------------------------------
    # Core: _full_params
    # ------------------------------------------------------------------

    def _full_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Build the ``(B, L+1, ...)`` parameter tensor.

        Two regimes, keyed off the encoder (``prepends_origin``). A model with an origin
        needs an origin encoder (checked by :meth:`_check_origin_encoder`):

        * **origin encoder** (``<bos>`` / initial-state device): the encoder already emits
          ``(B, L+1, ...)`` with a real position-0 (before-any-event) column, so it is
          returned directly — NO zeros prepend.
        * **plain encoder** (default): column 0 is a **zeros** no-history column (there is
          no learned per-head prior) and columns 1..L are the parameters
          from context positions 0..L-1.  Column 0 is only ever selected for a query with
          no real event strictly before it: without an origin that is the conditioned-on
          first event (neither scored nor integrated).

        Args:
            ts: Batched event data, shape (B, L).

        Returns:
            Full params, shape (B, L+1, ...).
        """
        self._check_origin_encoder()
        params = self._get_params(ts=ts)  # (B, L, ...) or (B, L+1, ...) for an origin encoder
        if getattr(self.encoder, "prepends_origin", False):
            return params  # (B, L+1, ...) with a real position-0 state
        B = params.shape[0]
        # Always a genuine (B, 1, ...) column — ``params[:, :1]`` would be size-0 when L=0,
        # collapsing the "no-events" column an empty sequence still needs.
        col0 = params.new_zeros((B, 1) + params.shape[2:])  # (B, 1, ...) — no-history column
        return torch.cat([col0, params], dim=1)  # (B, L+1, ...)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_cache_key(ts: BatchedMVEventData) -> tuple:
        """Compute a cheap CONTENT fingerprint for *ts* to validate cache hits.

        Shape plus two time moments and the type sum — deliberately object-identity
        free: the time-unit shell rebuilds a rescaled batch per public
        call, so two batches with identical (scaled) content must hit the same cache
        even though they are distinct objects.  A genuine wrong-batch query has
        different content, so it misses and recomputes (see ``_get_cached_params``).
        """
        tp = ts.time_points
        return (
            tuple(tp.shape),
            tp.sum().item(),
            (tp * tp).sum().item(),
            ts.event_types.sum().item(),
        )

    def prepare_batch(self, ts: BatchedMVEventData) -> None:
        """Run the transformer on *ts* and cache the full params tensor.

        Call once before a block of ``_intensities_at_queries`` / ``_interval_ci`` calls.
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

    @contextmanager
    def prime_for_likelihood(self, ts: BatchedMVEventData) -> Iterator[None]:
        """Prime the param cache for a base ``likelihood`` call — on the SCALED batch.

        The base ``likelihood`` runs the scaled core, so the content-keyed cache only
        hits (one ``_full_params`` pass, one pinned dropout mask across the positive and
        survival terms) when primed on the same scaled batch.  Scaling *here* rather than
        at each subclass call site removes the "subclass must prime on ``_scaled_ts``"
        footgun; priming on the native batch would miss at ``time_scale != 1`` and split
        the two terms across different dropout masks.
        """
        with self.fix_context(self._scaled_ts(ts)):
            yield

    def _get_cached_params(self, ts: BatchedMVEventData) -> Optional[torch.Tensor]:
        """Return cached params if they match *ts*, else None."""
        if self._cached_full_params is None:
            return None
        if self._cache_key != self._batch_cache_key(ts):
            # Content mismatch → treat as a miss and let the caller recompute for the
            # current batch (recomputing is always correct).
            return None
        return self._cached_full_params

    def _full_params_cached(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Cached ``(B, L+1, ...)`` params if they match *ts* (set by ``prepare_batch``),
        else a fresh ``_full_params(ts)``.

        The single read path behind every intensity/CI/mark query, so the encoder runs
        at most once per batch.  Guards on ``is not None`` (an empty batch caches a valid
        size-0 tensor, which is falsy).
        """
        cached = self._get_cached_params(ts)
        return cached if cached is not None else self._full_params(ts)

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

        Column 0 = the no-history slot (zeros dummy), column k = context after k events.

        Uses ``_cached_full_params`` when set by ``prepare_batch``.

        Args:
            t:  Query times, shape (B,).
            ts: Batched event sequences.

        Returns:
            Parameter tensor, shape (B, ...) — one row per batch item.
        """
        full_params = self._full_params_cached(ts)

        B = full_params.shape[0]

        # Count real events strictly before t[b] in each sequence.
        n_before = ts.count_events_before(t.to(ts.time_points.dtype))  # (B,)

        # full_params: (B, L+1, ...) — build index with correct trailing dims.
        trailing = full_params.shape[2:]
        idx = n_before.view(B, 1, *([1] * len(trailing))).expand(B, 1, *trailing)
        return full_params.gather(1, idx).squeeze(1)  # (B, ...)

    def _full_history_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Parameters after ALL real events of each sequence (column ``real_counts``).  ``(B, ...)``.

        Selected by index rather than by a query time just after the last event, which a
        float rounding of ``t_last + ε`` could map back onto ``t_last``.
        """
        full_params = self._full_params_cached(ts)
        B = full_params.shape[0]
        trailing = full_params.shape[2:]
        idx = ts.real_counts.view(B, 1, *([1] * len(trailing))).expand(B, 1, *trailing)
        return full_params.gather(1, idx).squeeze(1)  # (B, ...)

    def _get_params_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Multi-query analogue of :meth:`_get_params_at_time`.  ``(B, Q)`` -> ``(B, Q, ...)``.

        Gathers, per query, the causal parameter column ``full_params[b, n_before(q), ...]``
        (number of real events strictly before ``q`` — the same count as
        ``count_events_before``), vectorised over the query axis.
        """
        full_params = self._full_params_cached(ts)  # (B, L+1, ...)
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

        ``(B, Q)`` -> ``(B, Q)``; the multi-query form of ``event_time_before`` — a masked
        max over the event axis (not a positional gather, so it is correct on unsorted
        history and equal to the positional form on sorted history).
        """
        tp = ts.time_points  # (B, N)
        q = query_times.to(tp.dtype)  # (B, Q)
        if tp.shape[1] == 0:
            return torch.zeros_like(q)
        before = (tp.unsqueeze(1) < q.unsqueeze(2)) & ts.event_mask.unsqueeze(1)  # (B, Q, N)
        neg_inf = torch.full((), float("-inf"), dtype=tp.dtype, device=tp.device)
        masked = torch.where(before, tp.unsqueeze(1), neg_inf)  # (B, Q, N)
        mx = masked.max(dim=2).values  # (B, Q)
        any_before = before.any(dim=2)  # (B, Q)
        return torch.where(any_before, mx, torch.zeros_like(mx))


class NeuralParametrizedTPP(NeuralEncoderMixin, IntensityBasedTPP):
    """
    Abstract base class for encoder-based *intensity* temporal point processes.

    Combines the encoder plumbing (:class:`NeuralEncoderMixin`) with the intensity
    harness (:class:`IntensityBasedTPP`): the mixin maps event history to per-interval
    parameters, and this class turns those into the closed-form / numerical cumulative
    intensity the harness needs.  Concrete subclasses implement ``_get_params``
    (mixin) plus an intensity hook and, for closed-form heads,
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
        gathered per-interval ``params`` directly (called by ``_interval_ci``).
        Closed-form heads override it; intensity-numerical heads (NHP, AttNHP,
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

    def _interval_ci(self, ts: BatchedMVEventData, t_lo: torch.Tensor, t_hi: torch.Tensor) -> torch.Tensor:
        """Per-interval closed-form CI for the intervals defined by ``(t_lo, t_hi)``.

        Gathers each interval's causal parameters — column = the number of real events
        with time ``<=`` the interval's left endpoint, i.e. the right-limit history
        ``H_i`` (the just-fired boundary event is included) — and applies the head's
        closed-form per-interval integral ``_interval_ci_from_params``.  Summing over
        intervals telescopes to the global closed-form Λ(T); this is the closed-form
        dispatch's live hook.  Zero-width / padding intervals integrate to zero.
        ``t_lo``/``t_hi`` are ``(B, L+1)`` or ``(B, L+1, D)``.
        """
        B = ts.time_points.shape[0]
        full = self._full_params_cached(ts)  # (B, L+1, ...)
        M = t_lo.shape[1]
        lo2 = t_lo[..., 0] if t_lo.dim() == 3 else t_lo  # (B, M) — left endpoints (type-agnostic)
        # Column for interval m = #real events with time <= its left endpoint (H_m).
        # Compare in lo2's dtype: _interval_bounds casts event times to the query dtype,
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

    def likelihood(self, ts, T=None, *, log: bool = True):
        """Compute likelihood with a single transformer forward pass.

        The base ``likelihood`` evaluates the positive term (``_event_intensities``)
        and the survival term (``cumulative_intensity``) inline on the scaled core;
        ``prime_for_likelihood`` caches one ``_full_params`` evaluation on the scaled
        batch (matching the core queries) and pins one dropout mask across both terms.
        """
        with self.prime_for_likelihood(ts):
            return super().likelihood(ts, T, log=log)

    def _build_integration_grid(
        self,
        ts,
        T_max=None,
        num_points=None,
        points_per_unit_time=None,
        integrate_marginal_via_grid=True,
        ignore_indices=None,
    ):
        """Build grid with a single transformer forward pass.

        ``num_points`` / ``points_per_unit_time`` default to ``None`` → resolved from
        ``self.ci_config`` (``grid_num_points`` / ``grid_points_per_unit_time``) by the base.
        """
        self.prepare_batch(ts)
        try:
            return super()._build_integration_grid(
                ts,
                T_max=T_max,
                num_points=num_points,
                points_per_unit_time=points_per_unit_time,
                integrate_marginal_via_grid=integrate_marginal_via_grid,
                ignore_indices=ignore_indices,
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
        *,
        terminating: bool = False,
        seed: Optional[int] = None,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        super().__init__(
            encoder=TransformerEncoder(config, has_origin=token_roles is not None and token_roles.has_origin),
            terminating=terminating,
            seed=seed,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
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

    _zero_params_have_intensity = False

    def __init__(
        self,
        D: Optional[int] = None,
        *,
        config: Optional[TransformerConfig] = None,
        terminating: bool = False,
        seed: Optional[int] = None,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        config = _resolve_transformer_config(config, D)
        super().__init__(
            config,
            terminating=terminating,
            seed=seed,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )
        self.head = PoissonHead(config)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        embedding = self._encode(ts)
        return self.head(embedding)  # (B, L, D)

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
        full = self._full_params_cached(ts)  # (B, L+1, D)
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

    def __init__(
        self,
        D: Optional[int] = None,
        *,
        config: Optional[TransformerConfig] = None,
        terminating: bool = False,
        seed: Optional[int] = None,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        config = _resolve_transformer_config(config, D)
        super().__init__(
            config,
            terminating=terminating,
            seed=seed,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )
        self.head = LogLinearHead(config)
        # Initialise slope weights near zero so initial predictions are near-flat.
        torch.nn.init.zeros_(self.head.linear_s.weight)
        torch.nn.init.constant_(self.head.linear_s.bias, 0.3)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        embedding = self._encode(ts)
        return self.head(embedding)  # (B, L, 2, D)

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
        full = self._full_params_cached(ts)  # (B, L+1, 2, D)
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
    """Neural TPP whose intensity is a linear spline in the time since the last event.

    A causal transformer encodes the history up to event :math:`i`; a linear head maps the
    encoding to :math:`K` non-negative knot heights :math:`h_{d,1..K}` per event type. Between
    event :math:`i` and the next, with :math:`\\tau = t - t_i`,

    .. math:: \\lambda_d(t) = \\operatorname{interp}(\\tau;\\, x_{1..K},\\, h_{d,1..K}),

    a piecewise-linear curve through the knots :math:`x_1 = 0 < \\dots < x_K`, constant after
    :math:`x_K`. Its cumulative intensity is the exact trapezoidal integral, so the likelihood
    needs no quadrature.

    Args:
        D: Number of event types (alternative to ``config``).
        config: Transformer encoder configuration; ``config.num_event_types`` must equal ``D``.
        num_bins: Number of knots :math:`K`.
        knot_gaps: The ``K-1`` gaps between consecutive knots, starting at 0, in the internal
            time unit. Non-uniform gaps (e.g. denser near 0) give finer short-term resolution.
            ``None`` places the knots uniformly on ``[0, t_max]``.
        t_max: Extent of the knots when ``knot_gaps`` is ``None``.
        spline_smoothness_reg: Weight of a penalty on second differences of the knot heights.
        terminating: Each event type occurs at most once per sequence.
        dataset_stats: Accepted for a uniform constructor; not used by this model.
    """

    _zero_params_have_intensity = False

    def __init__(
        self,
        D: Optional[int] = None,
        *,
        config: Optional[TransformerConfig] = None,
        num_bins: int = 5,
        knot_gaps: Optional[torch.Tensor] = None,
        t_max: float = 1.0,
        terminating: bool = False,
        seed: Optional[int] = None,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        dataset_stats=None,
        spline_smoothness_reg: float = 0.0,
    ):
        config = _resolve_transformer_config(config, D)
        super().__init__(
            config,
            terminating=terminating,
            seed=seed,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )
        self.num_spline_bins = num_bins
        # Weight of the P-spline smoothness penalty (0 disables); consumed by regularization().
        self._spline_smoothness_reg = float(spline_smoothness_reg)
        self.head = SplineHead(config, num_spline_bins=num_bins)
        if knot_gaps is None:
            knot_gaps = torch.full((num_bins - 1,), t_max / (num_bins - 1))
        knot_locs = torch.cat([torch.zeros(1), knot_gaps.float().cumsum(0)])
        self.register_buffer("knot_locs", knot_locs)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        embedding = self._encode(ts)
        return self.head(embedding)  # (B, L, D, K)

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

    def regularization(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Weighted P-spline smoothness penalty (see :meth:`spline_smoothness_penalty`).

        Returns the base zero when the weight is non-positive, so no penalty is computed
        in the common (unregularised) case.
        """
        if self._spline_smoothness_reg <= 0.0:
            return super().regularization(ts)
        return self._spline_smoothness_reg * self.spline_smoothness_penalty(ts)

    def _sequence_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """(B, L, D) intensities in one forward pass.

        Uses causal column indices so simultaneous events share context.
        """
        B, L = ts.time_points.shape
        if L == 0:
            return torch.zeros(B, 0, self.D, device=ts.time_points.device)
        full = self._full_params_cached(ts)  # (B, L+1, D, K)
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
                       When ``None``, falls back to uniform spacing on ``[0, t_max]``.
        t_max:         Total time span — only used when ``knot_gaps`` is ``None``.
        terminating:   Whether each event type can only occur once.
        dataset_stats: Accepted for a uniform constructor; not used by this model.
    """

    _zero_params_have_intensity = False

    def __init__(
        self,
        D: Optional[int] = None,
        *,
        config: Optional[TransformerConfig] = None,
        num_bins: int = 5,
        knot_gaps: Optional[torch.Tensor] = None,
        t_max: float = 1.0,
        terminating: bool = False,
        seed: Optional[int] = None,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        dataset_stats=None,
    ):
        config = _resolve_transformer_config(config, D)
        super().__init__(
            config,
            terminating=terminating,
            seed=seed,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )
        self.num_bins = num_bins
        self.head = SplineHead(config, num_spline_bins=num_bins)
        if knot_gaps is None:
            knot_gaps = torch.full((num_bins - 1,), t_max / (num_bins - 1))
        knot_locs = torch.cat([torch.zeros(1), knot_gaps.float().cumsum(0)])
        self.register_buffer("knot_locs", knot_locs)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        embedding = self._encode(ts)
        return self.head(embedding)  # (B, L, D, K)

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
        full = self._full_params_cached(ts)  # (B, L+1, D, K)
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


# ---------------------------------------------------------------------------
# SAHP (Self-Attentive Hawkes Process)
# ---------------------------------------------------------------------------


# ===========================================================================
# THP and SAHP (scaled-softplus intensity → numerical CI)
# ===========================================================================


class THPProcess(TransformerParametrizedTPP):
    """Transformer Hawkes Process (Zuo et al., ICML 2020).

    The THP intensity head on the shared TorchPP
    transformer encoder (``config`` or ``D=``).

    Intensity at relative time ``t_rel = t - t_start``::

        state_d(t) = w_d * t_rel + linear(h)_d + b_d
        λ_d(t)     = ScaledSoftplus(state_d(t))

    where ``w_d`` (factor_intensity_decay) and ``b_d`` (factor_intensity_base)
    are global learnable per-type parameters, and ``linear(h)`` is a projection
    of the transformer encoder output.  ``ScaledSoftplus`` uses a learnable
    per-mark beta.

    Because ScaledSoftplus has no closed-form integral, cumulative intensity
    is computed numerically (per-interval trapezoid CIConfig).

    Params shape: ``(B, L+1, D)`` — the ``linear(h)`` projections; column 0 is the empty history.
    """

    def __init__(
        self,
        D: Optional[int] = None,
        *,
        config: Optional[TransformerConfig] = None,
        terminating: bool = False,
        seed: Optional[int] = None,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        config = _resolve_transformer_config(config, D)
        super().__init__(
            config,
            terminating=terminating,
            seed=seed,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )
        self.head = THPHead(config)
        self.softplus = ScaledSoftplus(config.num_event_types)

    def _default_ci_config(self) -> CIConfig:
        """ScaledSoftplus has no closed-form integral; the intensity is smooth within an
        interval and jumps only at events → event-knotted per-interval trapezoid CI."""
        return CIConfig("numerical", "interval", "trapezoidal", n_points=100)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        embedding = self._encode(ts)
        return self.head(embedding)  # (B, L, D) — linear(h) projection

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
        full = self._full_params_cached(ts)  # (B, L+1, D)
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


class SAHPProcess(TransformerParametrizedTPP):
    """Self-Attentive Hawkes Process (Zhang et al., ICML 2020).

    The SAHP intensity head on the shared TorchPP
    transformer encoder (``config`` or ``D=``).

    From the encoder output ``h``, three quantities are predicted::

        mu_d    = GELU(W_mu @ h)           — asymptotic intensity (can be negative)
        eta_d   = GELU(W_eta @ h)          — initial intensity   (can be negative)
        gamma_d = Softplus(W_gamma @ h)    — decay rate (> 0)

    State-decay (pre-activation intensity)::

        state_d(t) = mu_d + (eta_d - mu_d) * exp(-gamma_d * t_rel)

    Final intensity::

        λ_d(t) = ScaledSoftplus(state_d(t))

    GELU activations, no bias, scaled-softplus wrapping and a numerical
    cumulative intensity, as in the original model.

    Params shape: ``(B, L+1, 3, D)`` — stacked (mu, eta, gamma).
    """

    def __init__(
        self,
        D: Optional[int] = None,
        *,
        config: Optional[TransformerConfig] = None,
        terminating: bool = False,
        seed: Optional[int] = None,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        config = _resolve_transformer_config(config, D)
        super().__init__(
            config,
            terminating=terminating,
            seed=seed,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )
        self.head = SAHPHead(config)
        self.softplus = ScaledSoftplus(config.num_event_types)

    def _default_ci_config(self) -> CIConfig:
        """ScaledSoftplus has no closed-form integral; the intensity is smooth within an
        interval and jumps only at events → event-knotted per-interval trapezoid CI."""
        return CIConfig("numerical", "interval", "trapezoidal", n_points=100)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        embedding = self._encode(ts)
        return self.head(embedding)  # (B, L, 3, D) — (mu, eta, gamma) already activated

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
        full = self._full_params_cached(ts)  # (B, L+1, 3, D)
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

    Params shape: ``(B, L+1, n_embd)`` — a zeros no-history column + raw encoder outputs.
    """

    def __init__(
        self,
        D: Optional[int] = None,
        *,
        config: Optional[TransformerConfig] = None,
        terminating: bool = False,
        seed: Optional[int] = None,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        n_hidden: int = 128,
        n_layers: int = 2,
        time_emb_dim: int | None = None,
    ):
        config = _resolve_transformer_config(config, D)
        super().__init__(
            config,
            terminating=terminating,
            seed=seed,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )
        self.head = MLPIntensityHead(
            config,
            n_hidden=n_hidden,
            n_mlp_layers=n_layers,
            time_embd_dim=time_emb_dim,
        )
        self.softplus = ScaledSoftplus(config.num_event_types)

    def _default_ci_config(self) -> CIConfig:
        """Softplus intensity is smooth within an interval and jumps only at events → event-knotted
        per-interval grid.  MC-trapezoidal (random nodes each step) stops the MLP from hiding
        intensity spikes between fixed nodes to evade the survival penalty."""
        return CIConfig("numerical", "interval", "mc_trapezoidal", n_points=100)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Return raw encoder outputs. Shape: ``(B, L, n_embd)``."""
        return self._encode(ts)

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
        full = self._full_params_cached(ts)  # (B, L+1, n_embd)
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
    """Intensity-free TPP: a log-normal mixture density on the inter-event time.

    Shchur et al., *Intensity-Free Learning of Temporal Point Processes* (ICLR 2020). An encoder
    summarises the history :math:`H`; heads predict a :math:`K`-component log-normal mixture for
    the time since the last event, :math:`\\tau = t - t_{\\mathrm{last}}`, and a categorical
    distribution over the next event type:

    .. math:: p(t, d \\mid H) = f(\\tau \\mid H)\\; p(d \\mid H)

    The type does not depend on the time given the history, as in the original model. The
    predictive contract is computed from :math:`f`, its survival :math:`S` and :math:`p(d)`
    directly; :meth:`intensity` returns the implied hazard :math:`(f/S)\\,p(d)` for plotting.
    In terminating mode the time law is restricted to the event types still possible
    (see ``docs/models.md``).

    Params layout: ``(B, L+1, 3K + D)`` — ``[locs(K), log_scales(K), log_weights(K),
    mark_logits(D)]``.

    Args:
        encoder: History encoder: a :class:`~torchpp.modules.nn.encoder.GRUEncoder` (as in the
            original model) or another :class:`SequenceEncoder`, e.g. the shared transformer.
        num_mix_components: Number of mixture components :math:`K`.
        terminating: Each event type occurs at most once per sequence.
        seed: Seed for the parameter initialisation.
    """

    def __init__(
        self,
        encoder: SequenceEncoder,
        *,
        num_mix_components: int = 8,
        terminating: bool = False,
        seed: Optional[int] = None,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        super().__init__(
            encoder=encoder,
            terminating=terminating,
            seed=seed,
            time_scale=time_scale,
            token_roles=token_roles,
            sampling_config=sampling_config,
        )
        self.K = int(num_mix_components)
        # No per-dataset log-inter-time normalisation: time_scale already
        # normalises inter-event times to O(1).

        n_embd = encoder.n_embd
        self.mix_linear = torch.nn.Linear(n_embd, 3 * self.K)
        self.mark_linear = torch.nn.Linear(n_embd, self.D)

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        h = self._encode(ts)  # (B, L, n_embd)
        return torch.cat([self.mix_linear(h), self.mark_linear(h)], dim=-1)  # (B, L, 3K+D)

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
        )
        return dist, mark_log_probs

    def _predictable_mark_log_probs(self, mark_log_probs: torch.Tensor) -> torch.Tensor:
        """Restrict the mark categorical to predictable types (role mask).

        Masks the non-predictable (context) columns to ``-inf`` and
        renormalises (log-softmax over the predictable columns), so the mark
        distribution assigns exactly zero probability to a context type.  Trailing
        dim is ``D``; leading dims are arbitrary.  Under the default all-True
        ``predictable_mask`` this returns the input unchanged (renorm over all ``D``
        equals the ``_unpack`` log-softmax).
        """
        if bool(self.predictable_mask.all()):
            return mark_log_probs
        masked = mark_log_probs.masked_fill(~self.predictable_mask, float("-inf"))
        return masked - torch.logsumexp(masked, dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # DensityBasedTPP hooks — realized from the mixture + mark head
    # ------------------------------------------------------------------

    def _history_dist(self, t: torch.Tensor, ts: BatchedMVEventData):
        """Full-history mixture at the last observed event and tau = t - t_last per query.

        The predictive next-event density conditions on the WHOLE history H (the mark head's
        right-limit column after all events); the inter-event time is measured from
        ``t_last = ts.last_event_time``, not from the event before ``t`` (that would jump at the
        exact boundary ``t = t_last``).  Returns ``(dist, tau)`` with ``dist`` batch shape
        matching ``t`` (``(B,)`` or ``(B, Q)``).
        """
        t_last = self._last_event_time(ts)  # (B,) — padding-aware last real event
        params = self._full_history_params(ts)  # (B, 3K+D) — column after all events
        if t.dim() == 2:
            params = params.unsqueeze(1).expand(params.shape[0], t.shape[1], params.shape[-1])
            tau = (t - t_last.unsqueeze(1)).clamp(min=_MIN_INTER_EVENT_TIME)  # (B, Q)
        else:
            tau = (t - t_last).clamp(min=_MIN_INTER_EVENT_TIME)  # (B,)
        dist, _ = self._unpack(params, dtype=t.dtype)
        return dist, tau

    def _log_remaining_mark_mass(self, ts: BatchedMVEventData) -> torch.Tensor:
        """log A = log Σ_{d active} p(d | H), the mark mass on still-active *predictable*
        types.  ``(B,)``.

        ``A`` is the scale factor of the restricted terminating process (class
        "Terminating mode" derivation); ``A = 1`` when every predictable type is still
        active.  The mark law is predictable-renormalized first: context
        types are outside the competing-risks set, so they carry zero mass and never
        enter ``A``.  This makes the predictive-path ``A`` (survival/density hooks)
        consistent with the training-path ``A`` in ``_event_log_density_sum``.  Inert
        under the default all-True mask (the renorm is the identity there).
        """
        dtype = ts.time_points.dtype
        _, mark_log_probs = self._unpack(self._full_history_params(ts), dtype=dtype)  # (B, D)
        mark_log_probs = self._predictable_mark_log_probs(mark_log_probs)  # context types -> zero mass
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

        Restricted to predictable types (context types get zero mass); composed
        BEFORE the terminating active-set renorm so both exclusions share one renorm.
        Terminating: occurred types also get ``-inf`` and the remaining mass is
        renormalised (log-softmax over the still-active predictable types), so no
        probability is assigned to a type that can no longer occur.
        """
        dtype = ts.time_points.dtype
        if ts.time_points.shape[0] == 0:
            return torch.zeros(0, self.D, dtype=dtype, device=ts.time_points.device)
        _, mark_log_probs = self._unpack(self._full_history_params(ts), dtype=dtype)  # (B, D)
        mark_log_probs = self._predictable_mark_log_probs(mark_log_probs)  # zero mass on context types
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
        # Role/origin mask: score only predictable-typed events, and — under the
        # no-origin (condition-on-first) regime — drop the first event and its ties; shared with
        # the intensity family and the −log s Jacobian count via ``_scored_mask``.  The
        # mark head is predictable-renormalized below.  The mark-mixture censoring
        # survival (``_log_survival_marginal`` in ``DensityBasedTPP.likelihood``) needs no
        # masking: IFTPP's survival is mark-independent and Σ_m p(m) = 1 after renorm, so
        # Σ_m p(m)·S(τ) = S(τ) is invariant to the mark restriction.
        scored = self._scored_mask(ts)  # (B, L)
        full = self._full_params_cached(ts)  # (B, L+1, 3K+D)
        col_idx = ts.causal_indices  # (B, L)
        idx = col_idx.unsqueeze(-1).expand(B, L, full.shape[-1])
        params = full.gather(1, idx)  # (B, L, 3K+D)

        tp = ts.time_points  # (B, L)
        tp_with_zero = torch.cat([torch.zeros(B, 1, device=device, dtype=tp.dtype), tp], dim=1)
        tau = (tp - tp_with_zero.gather(1, col_idx)).clamp(min=_MIN_INTER_EVENT_TIME)  # (B, L)

        dist, mark_log_probs = self._unpack(params, dtype=tp.dtype)  # dist batch (B, L); marks (B, L, D)
        # Predictable-renormalized mark head: the scored per-event log p(m|H) uses the
        # same context-masked distribution as ``_log_prob_mark`` (context types → zero mass);
        # inert under the all-True mask.  Also feeds the terminating active-mark-mass ``A``
        # below, keeping it consistent with ``_log_prob_mark`` and the predictive-path ``A``
        # in ``_log_remaining_mark_mass`` (both predictable-renormalized): context types
        # contribute zero mass to ``A``, and the repeated-type gate is scoped to predictable
        # types (``_has_repeated_event_types``), so a repeated context type is not impossible.
        mark_log_probs = self._predictable_mark_log_probs(mark_log_probs)
        log_f = dist.log_prob(tau)  # (B, L)
        log_pm = mark_log_probs.gather(2, ts.event_types_idx_safe.unsqueeze(-1)).squeeze(-1)  # (B, L)
        log_event = (log_f + log_pm).masked_fill(~scored, 0.0)  # (B, L)
        log_sum = log_event.sum(dim=1)  # (B,)

        log_sum = log_sum.masked_fill(scored.sum(dim=1) == 0, 0.0)  # no scorable events -> 0
        if self.terminating:
            # Occurred types carry no survival penalty after their occurrence, so each
            # interval's survival covers only the types still active when it starts.  This
            # adds Σ_i (A_i - 1)·log S(τ_i) to the per-event term (A_i the active mark mass
            # over [t_{i-1}, ·)), matching the intensity-harness likelihood.
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
        params = self._full_history_params(batched)  # (1, 3K+D)
        dist, mark_log_probs = self._unpack(params, dtype=t_last.dtype)  # dist batch (1,); marks (1, D)
        mark_log_probs = self._predictable_mark_log_probs(mark_log_probs)  # context types -> zero mass
        t_next = t_last + self._sample_mixture(dist, rng, device)  # (1,)

        probs = mark_log_probs.exp()[0]  # (D,)
        if self.terminating:
            # Only predictable types compete: context types are never in the active set.
            # Inert under the default all-True mask.
            active = self._get_termination_mask(batched)[0] & self.predictable_mask  # (D,) active predictable
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

    def likelihood(self, ts, T=None, *, log: bool = True):
        """Native density log-likelihood with a single encoder forward pass (caches params).

        ``prime_for_likelihood`` primes the cache on the scaled batch — the base
        likelihood runs the core there, so the content-keyed cache only hits (one
        ``_full_params`` pass, one pinned dropout mask) when primed on that batch.
        """
        with self.prime_for_likelihood(ts):
            return super().likelihood(ts, T, log=log)

    # ------------------------------------------------------------------
    # Diagnostic hazard (plotting only — OFF the contract path)
    # ------------------------------------------------------------------

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Diagnostic hazard ``lambda_d(t) = (f(tau)/S(tau))·p(d|H)`` for plotting.

        Provided so notebooks can plot the IFTPP hazard; nothing in the
        likelihood / predictive / eval / sampling path routes through it.  Handles
        ``(B,)`` and ``(B, Q)`` queries and mirrors the intensity family's
        termination masking so a diagnostic overlay stays consistent.
        """
        if not bool((t >= 0.0).all()):
            raise ValueError("Intensity is only defined for t >= 0.")
        return self._hazard_scaled(self._to_internal(t, ts), self._scaled_ts(ts)) / self.time_scale  # per native unit

    def _hazard_scaled(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Diagnostic hazard in internal time units (see :meth:`intensity`)."""
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
        # Native shell: the mixture mean is computed in scaled time, ×s
        # back to native units.
        return self._analytical_mean_next_event_time(self._scaled_ts(ts)) * self.time_scale

    def _analytical_mean_next_event_time(self, ts: BatchedMVEventData) -> torch.Tensor:
        """``t_last + E[tau]`` using the closed-form log-normal mixture mean.

        ``E[tau] = sum_k w_k · exp(loc_k + 0.5·scale_k^2)``.  When the history was observed
        beyond its last event (``T_obs > t_last``), the mean is conditional on
        ``tau > a = T_obs − t_last`` via the log-normal partial expectation
        ``E[tau·1(tau > a)] = exp(loc + scale²/2)·Φ((loc + scale² − log a)/scale)``.
        """
        t_last = self._last_event_time(ts)  # (B,) padding-aware last real event
        params = self._full_history_params(ts)  # (B, 3K+D)
        K, dtype = self.K, ts.time_points.dtype
        locs = params[..., :K].to(dtype)
        scales = clamp_preserve_gradients(params[..., K : 2 * K], _LOG_SCALE_MIN, _LOG_SCALE_MAX).exp().to(dtype)
        log_weights = torch.log_softmax(params[..., 2 * K : 3 * K], dim=-1).to(dtype)
        # E[τ] for a log-normal mixture (no per-dataset log-normalisation):
        # E[exp(N(loc, scale²))] = exp(loc + ½ scale²).
        log_component_means = locs + 0.5 * scales**2  # (B, K)
        e_tau = torch.logsumexp(log_weights + log_component_means, dim=-1).exp()  # (B,)
        a = self._forecast_origin(ts) - t_last  # (B,) observed event-free gap
        censored = a > 0
        if not bool(censored.any()):
            mean = t_last + e_tau
        else:
            log_a = torch.log(a.clamp(min=_MIN_INTER_EVENT_TIME)).unsqueeze(-1)  # (B, 1)
            log_partial = log_weights + log_component_means + torch.special.log_ndtr((locs + scales**2 - log_a) / scales)
            log_surv = log_weights + torch.special.log_ndtr((locs - log_a) / scales)
            e_cond = (torch.logsumexp(log_partial, dim=-1) - torch.logsumexp(log_surv, dim=-1)).exp()  # E[τ | τ > a]
            mean = t_last + torch.where(censored, e_cond, e_tau)
        if not self.terminating:
            return mean
        # Terminating: the next-event survival is S(τ)^A with A the remaining mark mass.  A = 1
        # (no type has occurred yet) is the mixture law above; A = 0 has no next event; otherwise
        # E[τ | τ > a] = a + ∫_a^∞ S^A dτ / S(a)^A, integrated in log τ.
        active = self._get_termination_mask(ts) | ~self.predictable_mask  # (B, D)
        reduced = ~active.all(dim=-1)
        if not bool(reduced.any()):
            return mean
        A = self._log_remaining_mark_mass(ts).exp().to(dtype)  # (B,)
        e_reduced = self._terminating_mean_tau(locs, scales, log_weights, A, a)
        mean = torch.where(reduced, t_last + e_reduced, mean)
        return torch.where(A == 0, torch.full_like(mean, float("inf")), mean)

    @staticmethod
    def _terminating_mean_tau(
        locs: torch.Tensor,
        scales: torch.Tensor,
        log_weights: torch.Tensor,
        A: torch.Tensor,
        a: torch.Tensor,
        num_points: int = 4001,
    ) -> torch.Tensor:
        """E[τ | τ > a] under the survival S(τ)^A of a log-normal mixture.  ``(B,)``.

        ``E = a + ∫_a^∞ S^A dτ / S(a)^A``.  The integral is a trapezoid in ``x = log τ`` from
        ``max(a, e^lo)`` to ``e^hi``, with ``lo``/``hi`` 12 scales beyond the outermost components
        (S ≈ 1 below ``e^lo``, and S^A is negligible above ``e^hi``).
        """
        work = torch.float64
        locs, scales, log_weights, A, a = (v.to(work) for v in (locs, scales, log_weights, A, a))
        a = a.clamp(min=0.0)
        lo = (locs - 12.0 * scales).min(dim=-1).values  # (B,)
        hi = (locs + 12.0 * scales).max(dim=-1).values
        log_a = torch.log(a.clamp(min=torch.finfo(work).tiny))
        start = torch.maximum(lo, log_a)
        frac = torch.linspace(0.0, 1.0, num_points, dtype=work, device=locs.device)
        x = start.unsqueeze(1) + (hi - start).clamp(min=0.0).unsqueeze(1) * frac  # (B, Q)

        def log_survival(log_tau: torch.Tensor) -> torch.Tensor:  # (B, ...) -> (B, ...)
            z = (locs.view(locs.shape[0], *([1] * (log_tau.dim() - 1)), -1) - log_tau.unsqueeze(-1)) / scales.view(
                scales.shape[0], *([1] * (log_tau.dim() - 1)), -1
            )
            lw = log_weights.view(log_weights.shape[0], *([1] * (log_tau.dim() - 1)), -1)
            return torch.logsumexp(lw + torch.special.log_ndtr(z), dim=-1)

        integral = torch.trapezoid(torch.exp(A.unsqueeze(1) * log_survival(x) + x), x, dim=1)
        integral = integral + (torch.exp(start) - a).clamp(min=0.0)  # [a, e^lo]: S ≈ 1
        log_S_a = torch.where(a > 0, log_survival(log_a), torch.zeros_like(a))
        return (a + integral * torch.exp(-A * log_S_a)).to(A.dtype)


# ===========================================================================
# Neural Hawkes Process (Mei & Eisner, NeurIPS 2017)
# ===========================================================================


class NHPProcess(NeuralParametrizedTPP):
    """Neural Hawkes Process — a continuous-time LSTM self-modulating intensity.

    A :class:`ContTimeLSTMEncoder`
    produces, after each event, the right-limit state ``(c, c_bar, delta, o)``.
    At a query time ``t`` in the interval after event ``l``, the state is decayed
    by ``t - t_l`` to give ``h(t)`` and the intensity is::

        lambda_d(t) = ScaledSoftplus( Linear(h(t)) )_d

    ``ScaledSoftplus`` has no closed-form integral, so the cumulative intensity is
    computed numerically (per-interval trapezoid CIConfig), as for THP and SAHP.

    Params layout: ``(B, L+1, 4H)`` — the right-limit CT-LSTM states, with a zeros
    no-history column at column 0.
    """

    def __init__(
        self,
        encoder: ContTimeLSTMEncoder,
        *,
        terminating: bool = False,
        seed: Optional[int] = None,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        super().__init__(
            encoder=encoder,
            terminating=terminating,
            seed=seed,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )
        self.H = encoder.hidden_dim
        self.linear = torch.nn.Linear(self.H, self.D)
        self.softplus = ScaledSoftplus(self.D)

    # Interval i is governed by the CT-LSTM state after the event at its left endpoint.
    _interval_hook_includes_left_event = True

    def _default_ci_config(self) -> CIConfig:
        """ScaledSoftplus has no closed-form integral → per-interval trapezoid numerical CI.
        Both the survival term and point-CI route through this one CIConfig (no hybrid)."""
        return CIConfig("numerical", "interval", "trapezoidal", n_points=100)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        return self._encode(ts)  # (B, L, 4H) right-limit states

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
        survival quadrature uses, so the numerical interval CI matches the
        closed-form CI.  Zero-width / padding rows are dropped by the CI weights.
        """
        B, R, P = query_times.shape
        full = self._full_params_cached(ts)  # (B, L+1, 4H)
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
        full = self._full_params_cached(ts)  # (B, L+1, 4H)
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

    Unlike THP/NHP there is no per-event
    state that is merely decayed: the intensity at a query time is produced by
    re-running the attention stack with a temporal embedding of that query time,
    attending to the events that strictly precede it (torchPP causal convention).

        lambda_d(t) = ScaledSoftplus( Linear( state(t) ) )_d

    The :class:`AttNHPEncoder` provides the per-event embeddings (cached) and the
    query-time state machinery.  Cumulative intensity is numerical (per-interval
    trapezoid CIConfig).  Because each query re-runs attention, eval cost scales with
    the number of CI grid points.

    Params layout: ``(B, L+1, d_model + d_time)`` — per-event embeddings, with a zeros
    no-history column at column 0.
    """

    def __init__(
        self,
        encoder: AttNHPEncoder,
        *,
        terminating: bool = False,
        seed: Optional[int] = None,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        quadrature_chunk_size: Optional[int] = 10,
    ):
        super().__init__(
            encoder=encoder,
            terminating=terminating,
            seed=seed,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )
        self.intensity_linear = torch.nn.Linear(encoder.state_dim, self.D)
        self.softplus = ScaledSoftplus(self.D)
        # Nodes per batch-folded quadrature pass; None/0 -> all P in one pass.  Folded into
        # the CIConfig by _default_ci_config, and read directly by _interval_intensities.
        self.quadrature_chunk_size = quadrature_chunk_size

    def _time_compare_dtype(self, ts: BatchedMVEventData) -> torch.dtype:
        """The encoder compares event and query times at its parameter precision."""
        model_dtype = self.intensity_linear.weight.dtype
        data_dtype = ts.time_points.dtype
        return model_dtype if torch.finfo(model_dtype).eps > torch.finfo(data_dtype).eps else data_dtype

    def _default_ci_config(self) -> CIConfig:
        """Per-interval trapezoid numerical CI, carrying ``quadrature_chunk_size`` as the
        CIConfig memory knob.  Both survival and point-CI route through this one config."""
        return CIConfig("numerical", "interval", "trapezoidal", 100, chunk_size=self.quadrature_chunk_size)

    def _get_params(self, ts: BatchedMVEventData) -> torch.Tensor:
        return self._encode(ts)  # (B, L, d_model + d_time)

    def _event_emb(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Per-event embeddings (B, L, De), from the cache when available."""
        cached = self._get_cached_params(ts)
        if cached is not None:
            return cached[:, 1:, :]  # drop the zeros dummy column (col 0)
        return self._encode(ts)

    def _lambda_at_queries(self, ts: BatchedMVEventData, query_times: torch.Tensor) -> torch.Tensor:
        """Intensity at ``query_times`` (B, Q) -> (B, Q, D)."""
        event_emb = self._event_emb(ts)  # (B, L, De)
        state = self.encoder.states_at_queries(
            event_emb, ts.time_points, ts.event_mask, query_times
        )  # (B, Q, state_dim)
        return self.softplus(self.intensity_linear(state))  # (B, Q, D)

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Flat multi-query intensity (batched attention over all queries).

        Each query re-runs attention over the events strictly before its own time
        (events are encoded once inside ``states_at_queries``).  ``(B, Q) -> (B, Q, D)``.
        A single-query need is served by ``_intensities_at_queries(t.unsqueeze(1), ts).squeeze(1)``.
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
        cache = self.encoder.encode_events(event_emb, ts.time_points, ts.event_mask)
        chunk = self.quadrature_chunk_size or P
        parts = []
        for c0 in range(0, P, chunk):
            c1 = min(c0 + chunk, P)
            pc = c1 - c0
            q = query_times[..., c0:c1].permute(0, 2, 1).reshape(B * pc, R)  # (B*pc, R)
            state = self.encoder.states_from_cache(cache, q)  # (B*pc, R, state_dim)
            lam = self.softplus(self.intensity_linear(state)).reshape(B, pc, R, self.D)  # (B, pc, R, D)
            parts.append(lam.permute(0, 2, 1, 3))  # (B, R, pc, D)
        return torch.cat(parts, dim=2)  # (B, R, P, D)

    def _sequence_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        B, L = ts.time_points.shape
        if L == 0:
            return torch.zeros(B, 0, self.D, device=ts.time_points.device)
        return self._lambda_at_queries(ts, ts.time_points)  # (B, L, D)
