"""Abstract base classes for temporal point processes.

Defines the family-agnostic contract and the model-family bases that realize it:

* :class:`TemporalPointProcess` — the root contract every family shares: the
  predictive ``density_*`` / ``prob_*`` / ``time_*`` methods, ``likelihood`` and
  ``sample``, termination helpers, and the native/scaled time-unit shell.
* :class:`AutoregressiveTPP` — the shared next-event likelihood factorization
  ``Σ_i log p(event_i | H_i) + log S(T)``; a family supplies the per-event
  conditional and inherits ``likelihood`` / ``loss``.
* :class:`IntensityBasedTPP` — the intensity harness (λ / Λ), with
  ``_intensities_at_queries`` as the single required primitive and
  :class:`CIConfig` selecting closed-form vs numerical Λ.
* :class:`DensityBasedTPP` — the inter-event-density family (``p(t)`` + mark
  head), rooting the same contract without an intensity.

Tunable numerics live on the :class:`CIConfig` / :class:`SamplingConfig` value
objects; only stability guards remain as module constants.
"""

import functools
import inspect
import math
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple
import torch
import torch.nn as nn

from ..utils.event_data import (
    TokenRoles,
    extend_T_obs,
    MVEventData,
    BatchedMVEventData,
    termination_times as _termination_times,
    termination_mask as _termination_mask,
    terminated_query_mask as _terminated_query_mask,
    has_repeated_event_types as _has_repeated_event_types_fn,
    scored_event_mask as _scored_event_mask,
)
from ..utils.math import invert_CI, numerical_integration, IntensityGrid, right_limit
from ..utils.init_record import record_init
from ..utils.seeding import seeded_init


# Map the public quadrature-rule names (CIConfig.rule) onto the numerical_integration
# method names. "mc" is the pure Monte-Carlo estimator (utils.math "monte_carlo").
_RULE_TO_METHOD = {"trapezoidal": "trapezoidal", "mc_trapezoidal": "mc_trapezoidal", "mc": "monte_carlo"}

# ---------------------------------------------------------------------------
# Numerical-STABILITY constants (guards, not tuning knobs).  The tunable
# grid / quantile / thinning / jitter numerics live on CIConfig / SamplingConfig
# (overridable via use_ci / use_sampling); these are stability guards deliberately
# NOT exposed — changing them risks divide-by-zero / log(0) / branch breakage.
# ---------------------------------------------------------------------------
_THINNING_SCAN_CHUNK = 512  # intensity queries per call in the thinning bound scan
_THINNING_MAX_WINDOWS = 10_000  # window moves before thinning gives up (defective laws)

# Query-time snap tolerance: a predictive query within this of t_last is snapped
# onto t_last.  Densities and CDF/survival use different tolerances.
_SNAP_ATOL_DENSITY = 1e-8  # density / PDF / joint queries
_SNAP_ATOL_CDF = 1e-6  # CDF / survival queries

# log(1 - exp(x)) stable-branch switch (Mächler, 2012): log(0.5).
_LOG1MEXP_SWITCH = -0.6931471805599453

# Divisor floor guarding grid interpolation / normalisation against divide-by-zero.
_DIV_FLOOR = 1e-12

# Build the inversion grid slightly past the requested quantile (safety margin).
_GRID_TARGET_CDF_MARGIN = 0.001

# Sentinel for "argument not supplied → resolve from the config" where ``None`` already
# carries a distinct meaning (e.g. ``target_cdf=None`` disables the CDF stop condition).
_USE_CI_DEFAULT: Any = object()


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log(1 - exp(x))`` for ``x <= 0`` (Mächler, 2012)."""
    return torch.where(
        x > _LOG1MEXP_SWITCH,
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


@dataclass(frozen=True)
class CIConfig:
    """How to compute the cumulative intensity Λ = ∫λ.

    A value object, not model state: the closed-form vs numerical choice and the
    numerical sub-choices live here rather than in control flow.  Resolution order
    is call-arg > active ``use_ci`` context > instance default (no global state).

    Attributes:
        mode: ``"closed_form"`` (analytic ``_interval_ci``) or
            ``"numerical"`` (quadrature on the intensity hooks).
        placement: numerical only — ``"interval"`` (event-knotted, P points per
            inter-event interval; events are nodes, so the trapezoid stays second-order
            accurate across the intensity jumps; default) or ``"global"`` (M uniform
            points over the whole window; fixed budget, cheap for long sequences, the
            natural choice for an event-free forecast horizon — first-order accurate
            when events fall inside the window).  The first node of an interval or
            window evaluates the right limit λ(t⁺), so a window starting at an event
            integrates the post-jump intensity.
        rule: numerical only — ``"trapezoidal"`` (deterministic linspace),
            ``"mc"`` (uniform-random mean×width) or ``"mc_trapezoidal"`` (sorted
            random + trapezoid weights).  **Default ``mc_trapezoidal``.**  WARNING:
            fixed ``trapezoidal`` nodes let a flexible-intensity neural head learn to
            hide intensity spikes between the nodes and evade the survival penalty
            during training — use ``mc``/``mc_trapezoidal``
            for those.
        n_points: per interval (``interval``) or total (``global``).
        chunk_size: eval/forecast memory knob, ``no_grad`` only (relevant to AttNHP,
            whose nodes fold through an attention encoder).  ``None`` = off.

    The ``grid_*`` fields tune the pre-computed **intensity grid**
    (:meth:`IntensityBasedTPP._build_intensity_grid` and its growth variant), used by
    predictive/eval integrals and grid-based inversion.  Raise ``grid_num_points`` /
    ``grid_points_per_unit_time`` for higher precision.

        grid_num_points: fixed grid-point count when a ``T_max`` bounds the window.
        grid_points_per_unit_time: grid density (points / unit scaled-time) when the
            grid grows to a target CDF instead.
        grid_target_cdf: stop growing the grid once the CDF reaches this.
        grid_chunk_duration: scaled-time span of each grid-growth chunk.
        grid_max_chunks: safety cap on the number of growth chunks.
    """

    mode: Literal["closed_form", "numerical"] = "numerical"
    placement: Literal["interval", "global"] = "interval"
    rule: Literal["trapezoidal", "mc", "mc_trapezoidal"] = "mc_trapezoidal"
    n_points: int = 100
    chunk_size: Optional[int] = None
    # Intensity-grid construction (predictive / eval integrals + grid inversion).
    grid_num_points: int = 200
    grid_points_per_unit_time: float = 200.0
    grid_target_cdf: float = 0.999
    grid_chunk_duration: float = 0.1
    grid_max_chunks: int = 100


@dataclass(frozen=True)
class SamplingConfig:
    """How to draw events from the process.

    Value object mirroring :class:`CIConfig`: the draw scheme and its numeric knobs
    live here rather than as magic constants inside ``sample``/inversion code.
    Resolution order is call-arg > active ``use_sampling`` context > instance default.
    The default is ``inversion`` with ``inversion_method="auto"``.

    Attributes:
        mode: ``"inversion"`` (inverse-CDF sampling, the default) or
            ``"thinning"`` (Ogata).
        inversion_method: ``"auto"`` picks per model (``closed_form`` where a
            closed-form inverter is overridden, ``newton`` for closed-form CI,
            ``grid`` for numerical CI); the other values force a path.
        newton_tol / newton_max_iters: bracketed-Newton convergence controls
            (``invert_CI`` defaults).
        bracket_width: initial Newton bracket width above ``t_last`` (a wider bracket
            shifts the Newton iterate within ``newton_tol``).
        grid_density: points/unit-time for the grid inverter.  ``None`` derives
            ``ci_config.n_points / 0.1``.
        quantile_cap: upper cap on the target CDF of a quantile / inverse-CDF draw (the
            grid is built just past it, so float32 uniform draws are never clipped).
        upper_bound / upper_bound_slack / max_attempts: Ogata-thinning knobs;
            unused by ``inversion``.  The bound is the sup of the intensity over a forward
            window (scanned from the right limit, including the knots of piecewise-linear
            models) times ``upper_bound_slack``; a candidate whose intensity exceeds it raises
            the bound exactly (superposed proposal stream).  ``max_attempts`` counts proposals.
        degenerate_floor: ``λ_sum ≤ floor`` ⇒ uniform type fallback.
        inversion_jitter: post-``t_last`` jitter when an inverse-CDF draw lands ``< t_last``.
        quantile_* : the density-family inverse-CDF search
            (:meth:`DensityBasedTPP.time_quantile`) — bracket-doubling iters, bisection
            iters, and the initial bracket width above ``t_last``.
        thinning_* : Ogata sup-scan window — grid points and the forward window size (in
            expected events at the local rate) for ``interval_sup`` / ``global_sup``.
    """

    mode: Literal["inversion", "thinning"] = "inversion"
    # inversion-only
    inversion_method: Literal["auto", "closed_form", "newton", "grid"] = "auto"
    newton_tol: float = 1e-6
    newton_max_iters: int = 50
    bracket_width: float = 0.1
    grid_density: Optional[float] = None
    quantile_cap: float = 1.0 - 1e-9
    inversion_jitter: float = 0.1
    # density-family inverse-CDF search (time_quantile)
    quantile_bracket_grow_iters: int = 64
    quantile_bisection_iters: int = 60
    quantile_init_bracket_width: float = 1.0
    # thinning-only (Ogata)
    upper_bound: Literal["interval_sup", "global_sup"] = "interval_sup"
    upper_bound_slack: float = 1.5
    max_attempts: int = 10_000
    thinning_sup_points: int = 64
    thinning_window_events: float = 10.0
    thinning_global_window_events: float = 50.0
    # shared
    degenerate_floor: float = 1e-12


class Sample(NamedTuple):
    """Result of :meth:`TemporalPointProcess.sample`; unpacks as ``sequence, times, mark_probs``."""

    sequence: MVEventData
    """The history followed by the sampled events; ``T_obs`` is the last sampled time."""
    times: List[torch.Tensor]
    """Absolute time of each sampled event, native units, one ``(1,)`` tensor per event."""
    mark_probs: List[torch.Tensor]
    """Mark distribution ``(D,)`` at each sampled time."""


# Public methods whose ``ts`` / time arguments are normalised and checked on entry.
_CHECKED_ENTRY_POINTS = (
    "likelihood",
    "num_scored_events",
    "loss",
    "intensity",
    "cumulative_intensity",
    "density_joint",
    "density_time",
    "survival_time",
    "cdf_time",
    "prob_mark",
    "prob_mark_given_time",
    "time_mean",
    "time_quantile",
)
_TIME_ARGS = ("t", "T", "T_low")


def _checked_entry(fn):
    """Wrap a public method: accept a single ``MVEventData``, Python/NumPy times, and check event types."""
    sig = inspect.signature(fn)
    if "ts" not in sig.parameters:
        return fn

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        ts = bound.arguments["ts"]
        if isinstance(ts, MVEventData):
            ts = ts.to_batch()
        elif not isinstance(ts, BatchedMVEventData):
            raise TypeError(f"ts must be a BatchedMVEventData or MVEventData, got {type(ts).__name__}.")
        et = ts.event_types
        if et.numel() and int(et.max()) >= self.D:
            raise ValueError(f"Event type {int(et.max())} is out of range for a model with D={self.D} types.")
        bound.arguments["ts"] = ts
        for name in _TIME_ARGS:
            v = bound.arguments.get(name)
            if v is None or (isinstance(v, torch.Tensor) and v.dim() > 0):
                continue
            v = torch.as_tensor(v, dtype=ts.time_points.dtype, device=ts.time_points.device)
            if v.dim() == 0:
                v = v.expand(ts.time_points.shape[0])
            bound.arguments[name] = v
        return fn(*bound.args, **bound.kwargs)

    wrapper._tpp_checked = True
    return wrapper


def _wrap_entry_points(cls) -> None:
    for name in _CHECKED_ENTRY_POINTS:
        fn = cls.__dict__.get(name)
        if fn is None or getattr(fn, "_tpp_checked", False) or getattr(fn, "__isabstractmethod__", False):
            continue
        setattr(cls, name, _checked_entry(fn))


def _seeded_constructor(init):
    """Run ``__init__`` under :func:`seeded_init` with its ``seed`` argument."""
    sig = inspect.signature(init)

    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        try:
            bound = sig.bind_partial(self, *args, **kwargs).arguments
        except TypeError:
            bound = {}
        seed = bound.get("seed", kwargs.get("seed"))
        for p in sig.parameters.values():
            if p.kind is p.VAR_KEYWORD and seed is None:
                seed = bound.get(p.name, {}).get("seed")
        with seeded_init(seed):
            init(self, *args, **kwargs)

    wrapper._tpp_seeded = True
    return wrapper


class TemporalPointProcess(nn.Module, ABC):
    """Abstract root for all temporal point processes — the family-agnostic contract.

    Subclassed by the model families: ``IntensityBasedTPP`` (the intensity harness,
    below) and ``DensityBasedTPP`` (inter-event density).  It holds what every family shares — the event-type count ``D``, the
    ``seed``, the ``SamplingConfig``, the ``terminating`` flag — and
    defines the universal predictive/likelihood/sampling contract right here:
    the abstract ``likelihood`` / ``sample`` plus the ``density_*`` / ``prob_*`` /
    ``time_*`` methods, with the mark-axis identities as fallbacks each family base
    overrides for the subset it roots.  Each family *realizes* the contract (intensity
    via λ/Λ, density via the inter-event density + mark head).
    """

    _tpp_recorded = True  # constructor arguments are recorded for checkpoints

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _wrap_entry_points(cls)
        init = cls.__dict__.get("__init__")
        if init is not None and not getattr(init, "_tpp_seeded", False):
            cls.__init__ = record_init(_seeded_constructor(init))

    def __init__(
        self,
        D: int,
        *,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        super().__init__(
        )
        self.D = D
        self.seed = seed
        self.terminating = terminating
        roles = token_roles if token_roles is not None else TokenRoles()
        if not isinstance(roles, TokenRoles):
            raise TypeError(f"token_roles must be a TokenRoles, got {type(roles).__name__}.")
        self._has_origin = roles.has_origin
        self._ignored_types = roles.ignored_types
        # Time-unit scale s: the model owns time normalisation.  The public contract
        # works in native units; the computational core works in scaled units t/s.
        # Default 1.0 makes the shell a no-op.  A scalar float buffer, so it is in
        # state_dict and moves with .to(device).
        self.register_buffer("time_scale", torch.tensor(float(time_scale)))
        # True = predictable (scored and integrated), False = context type. This buffer is the
        # stored form of ``token_roles.context_types``; building it consumes no RNG.
        self.register_buffer("predictable_mask", roles.predictable_mask(D))
        # Active SamplingConfig override (set here, by ``use_sampling``, or by direct
        # assignment); ``None`` lets the ``sampling_config`` property derive the family
        # default via :meth:`_default_sampling_config`.  Sampling is universal, so the
        # override + its resolution plumbing live on the root (CIConfig, being an
        # intensity-realization knob, stays on ``IntensityBasedTPP``).
        self._sampling_config_override: Optional[SamplingConfig] = sampling_config

    @property
    def token_roles(self) -> TokenRoles:
        """The model's :class:`TokenRoles` (origin regime, context and ignored types)."""
        context = (~self.predictable_mask).nonzero(as_tuple=True)[0].tolist()
        return TokenRoles(has_origin=self._has_origin, context_types=context, ignored_types=self._ignored_types)

    @token_roles.setter
    def token_roles(self, roles: TokenRoles) -> None:
        self._has_origin = roles.has_origin
        self._ignored_types = roles.ignored_types
        self.predictable_mask.copy_(roles.predictable_mask(self.D).to(self.predictable_mask.device))

    @property
    def has_origin(self) -> bool:
        """Whether time 0 is a meaningful origin (``token_roles.has_origin``)."""
        return self._has_origin

    @has_origin.setter
    def has_origin(self, value: bool) -> None:
        self._has_origin = bool(value)

    # =========================================================================
    # Time-unit shell — scaled core, native shell
    # =========================================================================
    # The public contract methods accept/return native-unit times; the entire
    # computational machinery operates in scaled units t/s.  These helpers do the
    # normalise-in / de-normalise-out at the boundary; with s = 1 they are no-ops.

    def _scaled_ts(self, ts: BatchedMVEventData) -> BatchedMVEventData:
        """The event batch in internal (scaled) time units: every time field ÷ s.

        Deterministic: the same ``ts`` always yields the same scaled content.  The
        neural / kernel per-batch caches are keyed on scaled CONTENT (not object
        identity), so priming them on ``_scaled_ts(ts)`` matches the scaled queries
        the core issues here — even though each call builds a distinct scaled object.
        """
        return ts.rescale_time(1.0 / float(self.time_scale))

    def _to_internal_time(self, ts: MVEventData) -> MVEventData:
        """Single-sequence history in scaled units (for the samplers)."""
        return ts.rescale_time(1.0 / float(self.time_scale))

    def _to_internal(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Native time(s) ``t`` (queries, ``T``, ``T_low``) in internal units.

        Rounded exactly as :meth:`_scaled_ts` rounds the event times of ``ts`` (same
        factor, same dtype), so a native time equal to an event time maps to the same
        internal value.  Otherwise a tied event can look strictly earlier than its target.
        The result keeps ``t``'s dtype; its resolution is that of the data.
        """
        factor = 1.0 / float(self.time_scale)
        if factor == 1.0:
            return t
        return (t.to(ts.time_points.dtype) * factor).to(t.dtype)

    def _to_native_sample(self, out_ts: MVEventData, time_samples: list):
        """De-scale a sampler's returned sequence and absolute sample times back to native units."""
        s = self.time_scale
        if float(s) == 1.0:
            return out_ts, time_samples
        T_obs = getattr(out_ts, "T_obs", None)
        out_ts = MVEventData(
            out_ts.time_points * s,
            out_ts.event_types,
            covariates=out_ts.covariates,
            T_obs=None if T_obs is None else float(T_obs) * float(s),
        )
        return out_ts, [t * s for t in time_samples]

    # =========================================================================
    # SamplingConfig plumbing (resolution order: call-arg > use_sampling > default)
    # =========================================================================

    def _default_sampling_config(self) -> SamplingConfig:
        """Per-family default SamplingConfig.

        The default is the ``inversion`` sampler; a family whose
        natural sampler is thinning may override this."""
        return SamplingConfig()

    @property
    def sampling_config(self) -> SamplingConfig:
        """Active SamplingConfig: explicit override if set, else the per-family default."""
        return self._sampling_config_override if self._sampling_config_override is not None else self._default_sampling_config()

    @sampling_config.setter
    def sampling_config(self, cfg: Optional[SamplingConfig]) -> None:
        self._sampling_config_override = cfg

    @contextmanager
    def use_sampling(self, cfg: SamplingConfig) -> Iterator[None]:
        """Temporarily force a SamplingConfig for a block (the sampling twin of ``use_ci``).

        Resolution is call-arg > this context > instance default, so wrapping a draw
        loop in ``with model.use_sampling(cfg):`` redirects every ``sample`` /
        inversion call inside without touching model internals.
        """
        prev = self._sampling_config_override
        self._sampling_config_override = cfg
        try:
            yield
        finally:
            self._sampling_config_override = prev

    # =========================================================================
    # Universal predictive / likelihood / sampling contract
    # =========================================================================
    #
    # The next-event decomposition, named uniformly: ``density_*`` for the
    # continuous time axis, ``prob_*`` for the discrete mark axis.  Every method
    # takes ``log: bool = True`` (log-space is the implemented primitive; ``log=False``
    # returns the ``exp``) and accepts time queries of shape ``(B,)`` or ``(B, Q)``.
    #
    # Only ``likelihood`` and ``sample`` are hard-universal (abstract).  The rest are
    # fallbacks expressing the decomposition identities; each *family base* overrides
    # the subset it roots and inherits the others.  ``density_joint`` is the
    # hub: rooted by the intensity family (λ_m·S) or derived exactly (p(t)·p(m|t)).

    @abstractmethod
    def likelihood(
        self, ts: BatchedMVEventData, T: Optional[torch.Tensor] = None, *, log: bool = True
    ) -> torch.Tensor:
        """Log-likelihood of each sequence over its observation window.

        Args:
            ts: Batched event sequences.
            T: End of the observation window per sequence, ``(B,)``. Defaults to ``ts.T_obs``.
            log: Return the log-likelihood (default) or its ``exp``.

        Returns:
            ``(B,)`` tensor: the summed log-likelihood of each sequence, in native time units.
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Per-sequence training objective ``(B,)`` that the trainer minimises — abstract.

        The root makes no assumption about *how* a model trains.
        :class:`AutoregressiveTPP` defines it as ``-likelihood`` (the intensity/density
        families); models with another training objective override it directly.  The trainer sums/normalises this per event
        and adds :meth:`regularization`.
        """
        raise NotImplementedError

    def regularization(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Scalar training penalty added to the reduced :meth:`loss` (``0`` by default).

        Receives ``ts`` so an override can scale the penalty to the batch (e.g. by
        sequence length or event count).  Kept separate from :meth:`loss` so it is not
        normalised per event.  Models with a regulariser (e.g. spline smoothness)
        override this.
        """
        return self.time_scale.new_zeros(())

    @abstractmethod
    def sample(self, ts=None, num_steps: int = 1, rng=None, sampling_config=None) -> "Sample":
        """Generate events by sampling from the process; returns a :class:`Sample`.

        ``ts`` is the conditioning history to roll forward from.  ``None`` (or an empty
        history) means *unconditional* generation, which is only defined for a model that
        learned the first-event distribution (``has_origin=True``); a condition-on-first
        model (``has_origin=False``) must be seeded with a non-empty history — see
        :meth:`_resolve_sample_history`.
        """
        raise NotImplementedError

    def _resolve_sample_history(self, ts: Optional[MVEventData]) -> MVEventData:
        """Validate + normalise the sampling seed history.

        Unconditional generation (``ts`` is ``None`` or empty) draws the first event from
        the model's learned "before any event" state, which exists only for
        ``has_origin=True``.  For a ``has_origin=False`` (condition-on-first) model that
        state is untrained — sampling from it would silently emit an uncalibrated first
        event — so an empty seed is rejected.  To sample such a model unconditionally, draw
        a seed (e.g. from the empirical first-event distribution) and pass it in as ``ts``.

        Returns a concrete (possibly empty) :class:`MVEventData` on the model's device to
        roll forward from.
        """
        if isinstance(ts, BatchedMVEventData):
            if ts.time_points.shape[0] != 1:
                raise ValueError(
                    f"sample() rolls one sequence forward; got a batch of {ts.time_points.shape[0]}. "
                    "Pass an MVEventData or a one-row batch."
                )
            ts = ts.to_sequences()[0]
        is_empty = ts is None or ts.time_points.numel() == 0
        if is_empty and not self.has_origin:
            raise ValueError(
                "Unconditional sampling (ts=None or an empty history) is undefined for a "
                "condition-on-first model (has_origin=False): it never learned p(first event). "
                "Pass a non-empty history to condition on — e.g. draw a seed from the "
                "empirical first-event distribution and pass it in as ts."
            )
        if ts is None:
            dev = self.time_scale.device
            ts = MVEventData(
                torch.zeros(0, dtype=torch.float64, device=dev),
                torch.zeros(0, dtype=torch.long, device=dev),
            )
        return ts

    # -- density_joint hub + exact mark-axis fallbacks ----------------------

    def density_joint(self, t: torch.Tensor, ts: BatchedMVEventData, *, log: bool = True) -> torch.Tensor:
        """p(t, m | H) — the joint next-event density over (time, mark).  ``(B,[Q,]D)``.

        Fallback (for families rooting the pair): ``p(t)·p(m|t)``.  The intensity
        family roots this directly as ``λ_m(t)·S(t)``.
        """
        lt = self.density_time(t, ts, log=True)  # (B,[Q])
        lc = self.prob_mark_given_time(t, ts, log=True)  # (B,[Q,]D)
        lj = lt.unsqueeze(-1) + lc
        return lj if log else lj.exp()

    def density_time(self, t: torch.Tensor, ts: BatchedMVEventData, *, log: bool = True) -> torch.Tensor:
        """p(t | H) = Σ_m p(t, m | H) — the time-marginal density.  ``(B,[Q])``. Exact."""
        if log:
            return torch.logsumexp(self.density_joint(t, ts, log=True), dim=-1)
        return self.density_joint(t, ts, log=False).sum(dim=-1)

    def prob_mark_given_time(self, t: torch.Tensor, ts: BatchedMVEventData, *, log: bool = True) -> torch.Tensor:
        """p(m | t, H) = p(t, m) / p(t) — the conditional mark distribution.  ``(B,[Q,]D)``. Exact."""
        lr = self.density_joint(t, ts, log=True) - self.density_time(t, ts, log=True).unsqueeze(-1)
        return lr if log else lr.exp()

    def density_time_given_mark(self, t: torch.Tensor, ts: BatchedMVEventData, *, log: bool = True) -> torch.Tensor:
        """p(t | m, H) = p(t, m) / p(m) — the mark-conditional time density.  ``(B,[Q,]D)``."""
        lj = self.density_joint(t, ts, log=True)  # (B,[Q,]D)
        lpm = self.prob_mark(ts, log=True)  # (B, D)
        lr = lj - (lpm.unsqueeze(1) if lj.dim() == 3 else lpm)
        return lr if log else lr.exp()

    # -- integrated quantities (family-specific; overridden by every family) --

    def prob_mark(self, ts: BatchedMVEventData, *, log: bool = True) -> torch.Tensor:
        """p(m | H) — the mark marginal, integrated over the next-event time.  ``(B, D)``."""
        raise NotImplementedError(f"{type(self).__name__} must implement prob_mark.")

    def survival_time(self, t: torch.Tensor, ts: BatchedMVEventData, *, log: bool = True) -> torch.Tensor:
        """S(t) = P(t_next ≥ t | H) — the survival function.  ``(B,[Q])``."""
        raise NotImplementedError(f"{type(self).__name__} must implement survival_time.")

    def cdf_time(self, t: torch.Tensor, ts: BatchedMVEventData, *, log: bool = True) -> torch.Tensor:
        """P(t_next < t | H) = 1 − S(t) — the time CDF.  ``(B,[Q])``."""
        raise NotImplementedError(f"{type(self).__name__} must implement cdf_time.")

    def time_mean(self, ts: BatchedMVEventData) -> torch.Tensor:
        """E[t_next | H] — the expected next-event time.  ``(B,)``."""
        raise NotImplementedError(f"{type(self).__name__} must implement time_mean.")

    def time_quantile(self, u, ts: BatchedMVEventData) -> torch.Tensor:
        """Inverse time CDF: t such that P(t_next < t | H) = u (median = ``u=0.5``).  ``(B,)``."""
        raise NotImplementedError(f"{type(self).__name__} must implement time_quantile.")

    # =========================================================================
    # Termination helpers (delegate to data-level free functions)
    # =========================================================================
    # The model carries the ``terminating``/``D`` config; the computation is a pure
    # function of the event data (in ``utils/event_data.py``).  Kept as methods so
    # every family and the existing call-sites reach them uniformly via ``self``.

    def _get_termination_mask(
        self, ts: BatchedMVEventData, t: Optional[torch.Tensor] = None, full: bool = False
    ) -> torch.Tensor:
        """Boolean ``(B, D)``: True where a type has NOT occurred (before ``t`` if given)."""
        return _termination_mask(ts, self.D, self.terminating, t=t, full=full)

    def _terminated_mask(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Per-query termination mask ``(B, Q, D)`` — the multi-query analogue."""
        return _terminated_query_mask(query_times, ts, self.D, self.terminating)

    def _get_termination_times(self, ts: BatchedMVEventData) -> torch.Tensor:
        """First-occurrence time per type (``inf`` where a type never occurred).  ``(B, D)``."""
        return _termination_times(ts, self.D, self.terminating)

    def _has_repeated_event_types(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Boolean ``(B,)``: True where a sequence repeats a PREDICTABLE event type.

        Context/``<bos>`` types are outside the competing-risks set, so a repeated
        context type is not the impossible-event case and never trips the terminating
        ``-inf`` gate.  Under the default all-True ``predictable_mask`` this is every
        type (inert).
        """
        return _has_repeated_event_types_fn(ts, self.D, self.predictable_mask)

    # =========================================================================
    # Role/origin mask
    # =========================================================================

    def _scored_mask(self, ts: BatchedMVEventData) -> torch.Tensor:
        """``(B, L)`` bool: the scored events (see :func:`~torchpp.utils.event_data.scored_event_mask`)."""
        return _scored_event_mask(ts, self.predictable_mask, self.has_origin)


class AutoregressiveTPP(TemporalPointProcess):
    """Autoregressive TPP whose likelihood factorizes over events.

        log L = Σ_i log p(event_i | H_i) + log S(T)

    Both model families are autoregressive in this sense — the intensity family via
    ``λ_m·S`` and the density family via ``p·S`` — so they share one likelihood
    skeleton here.  A subclass supplies the per-event conditional through
    :meth:`_log_prob_events` (in scaled units); :meth:`likelihood` and the default
    :meth:`loss` (= ``−likelihood``) are derived from it.
    """

    def __init_subclass__(cls, **kwargs):
        """Fail-fast root-sufficiency guard for the AR next-event decomposition.

        The joint ``p(t, m)`` factors three equivalent ways; a concrete autoregressive
        family must root one of them, else the fallbacks recurse forever.  All three are
        accepted (privileging one would be arbitrary):

        * ``density_joint`` directly;
        * time-first ``p(t)·p(m | t)`` — ``{density_time, prob_mark_given_time}``;
        * mark-first ``p(m)·p(t | m)`` — ``{prob_mark, density_time_given_mark}``.

        Rejected at class-definition time (mirrors the intensity-hook guard on
        ``IntensityBasedTPP``).  Lives here, not on the root, so a non-autoregressive model
        that provides its own ``likelihood`` is not forced into any decomposition.  Abstract intermediate bases are skipped.
        """
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        root = TemporalPointProcess
        roots_joint = cls.density_joint is not root.density_joint
        roots_time_first = (
            cls.density_time is not root.density_time
            and cls.prob_mark_given_time is not root.prob_mark_given_time
        )
        roots_mark_first = (
            cls.prob_mark is not root.prob_mark
            and cls.density_time_given_mark is not root.density_time_given_mark
        )
        if not (roots_joint or roots_time_first or roots_mark_first):
            raise TypeError(
                f"{cls.__name__} must root the next-event joint density: either "
                "density_joint, or a full factor pair — {density_time, prob_mark_given_time} "
                "(time-first) or {prob_mark, density_time_given_mark} (mark-first)."
            )

    def _last_event_time(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Padding-aware time of the last observed (real) event per sequence.  ``(B,)`` (0 for empty).

        Unlike ``ts.last_event_time`` this uses ``real_counts`` (the event mask), so a batch padded
        with dummy timestamps still measures the inter-event time from the true last event —
        keeping the density contract padding-invariant.
        """
        tp = ts.time_points  # (B, L)
        B = tp.shape[0]
        if tp.shape[1] == 0:
            return torch.zeros(B, dtype=tp.dtype, device=tp.device)
        tp0 = torch.cat([torch.zeros(B, 1, dtype=tp.dtype, device=tp.device), tp], dim=1)  # (B, L+1)
        return tp0.gather(1, ts.real_counts.unsqueeze(1)).squeeze(1)  # (B,) = t at index real_counts-1

    def _forecast_origin(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Start of the forecast window: the end of observation ``T_obs``.  ``(B,)``.

        Equals the last event time unless the history was observed beyond it; the
        predictive contract and the samplers then condition on no event in
        ``(t_last, T_obs]``.
        """
        return torch.maximum(ts.T_obs.to(ts.time_points.dtype), self._last_event_time(ts))

    @abstractmethod
    def _log_prob_events(self, ts_scaled: BatchedMVEventData, T_scaled: torch.Tensor) -> torch.Tensor:
        """Scaled-unit autoregressive log-density ``Σ_i log p(event_i | H_i) + log S(T | H_last)`` ``(B,)``, in SCALED units (the native-unit −log s Jacobian is applied by ``likelihood``)."""
        raise NotImplementedError

    def num_scored_events(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Number of scored events per sequence ``(B,)``.

        These are the events whose density enters :meth:`likelihood`: real events of
        predictable types, excluding the first event (and events tied with it) when the
        model has no origin. Divide a summed log-likelihood by this count to get the
        per-event log-likelihood that training, validation and evaluation report.
        """
        return self._scored_mask(ts).sum(dim=1)

    def likelihood(
        self, ts: BatchedMVEventData, T: Optional[torch.Tensor] = None, *, log: bool = True
    ) -> torch.Tensor:
        """Log-likelihood of each sequence over its observation window.

        For each sequence this is the sum over scored events of the log-density of the next
        event given its history, plus the log-survival of the censored tail after the last
        event:

        ``Σ_{i scored} log p(t_i, m_i | H_i) + log S(T | H_N)``.

        Scored events are the real events of predictable types; without an origin the first
        event (and events tied with it) is conditioned on instead (:meth:`num_scored_events`).
        Values are in native time units. Divide by :meth:`num_scored_events` for the
        per-event log-likelihood that training and evaluation report.

        Args:
            ts: Batched event sequences, padded to ``(B, L)``.
            T: End of the observation window per sequence, ``(B,)``; must be at or after the last
                event. Defaults to ``ts.T_obs``.
            log: Return the log-likelihood (default) or its ``exp``, which underflows for long
                sequences.

        Returns:
            ``(B,)`` tensor of summed log-likelihoods.

        Raises:
            ValueError: If ``T`` lies before a sequence's last event.
        """
        if T is None:
            T = ts.T_obs
        if not bool((T >= self._last_event_time(ts)).all()):
            raise ValueError("Likelihood is only defined for T >= the last event time.")
        s = self.time_scale
        ts_s = self._scaled_ts(ts)
        log_lik = self._log_prob_events(ts_s, self._to_internal(T, ts))  # (B,)
        n_events = self._scored_mask(ts).sum(dim=1).to(log_lik.dtype)  # (B,)
        log_lik = log_lik - n_events * torch.log(s)
        return log_lik if log else torch.exp(log_lik)

    def loss(self, ts: BatchedMVEventData) -> torch.Tensor:
        """AR models train by maximum likelihood, so the objective is ``−likelihood``."""
        return -self.likelihood(ts)


    def _finite_queries(self, t: torch.Tensor, ts: BatchedMVEventData):
        """Replace ``+inf`` queries by the forecast origin; returns ``(t, mask of +inf)`` (mask ``None`` if none)."""
        at_inf = torch.isposinf(t)
        if not bool(at_inf.any()):
            return t, None
        origin = self._forecast_origin(ts).to(t.dtype)
        origin = origin.unsqueeze(1).expand_as(t) if t.dim() == 2 else origin
        return torch.where(at_inf, origin, t), at_inf

    def _no_next_event(self, ts: BatchedMVEventData) -> torch.Tensor:
        """``(B,)`` bool: terminating rows in which every predictable type has occurred."""
        if not self.terminating:
            return torch.zeros(ts.time_points.shape[0], dtype=torch.bool, device=ts.time_points.device)
        active = self._get_termination_mask(ts) & self.predictable_mask  # (B, D)
        return ~active.any(dim=-1)


class IntensityBasedTPP(AutoregressiveTPP):
    """Intensity-based TPP family: the likelihood is realized through λ and Λ.

    A model of this family is defined by its intensity λ_d(t) — the instantaneous
    rate of a type-``d`` event given the history.  The base turns that into the
    autoregressive likelihood (log-λ at events minus the survival integral Λ),
    sampling, and the predictive contract.

    Required hook:
        ``_intensities_at_queries(query_times, ts)`` — the raw
        ``(B, Q) → (B, Q, D)`` intensity primitive (before termination masking).
        Enforced at subclass creation.

    Cumulative intensity Λ = ∫λ is selected by the instance :class:`CIConfig`:
        * ``mode="closed_form"`` sums the per-interval analytic integral
          (override ``_interval_ci`` or ``_closed_form_ci``);
        * ``mode="numerical"`` integrates the primitive by quadrature — the
          default for models with no closed form.

    Optional overrides: ``_event_intensities`` (a cheaper positive-term path) and
    ``_inverse_marginal_cumulative_intensity`` (a closed-form CDF inverse; else the
    sampler falls back to bracketed Newton or a grid inverse per
    :class:`SamplingConfig`).  ``intensity`` applies termination masking on top of
    the primitive automatically.
    """

    def _time_compare_dtype(self, ts: BatchedMVEventData) -> torch.dtype:
        """Precision at which the model compares query and event times.

        Right limits (:func:`right_limit`) are taken at this precision so that a query
        just after an event is still after it once the model casts the times.
        """
        return ts.time_points.dtype

    #: True when ``_interval_intensities`` already conditions interval ``i`` on the history
    #: including the event at its left endpoint (e.g. a decayed per-interval state), so the
    #: first quadrature node needs no right-limit shift.
    _interval_hook_includes_left_event: bool = False

    def __init_subclass__(cls, **kwargs):
        """Require every concrete model to implement the intensity primitive.

        ``_intensities_at_queries`` is the single intensity hook; the base
        default only raises.  This guard rejects a concrete subclass that never
        overrides it, surfacing the omission at class-creation time instead of a
        bare ``NotImplementedError`` at first use.  Abstract intermediate bases
        (e.g. ``NeuralParametrizedTPP``) are skipped — ``inspect.isabstract``
        scans manually during class creation, before ``__abstractmethods__`` is set.
        """
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        if cls._intensities_at_queries is IntensityBasedTPP._intensities_at_queries:
            raise TypeError(
                f"{cls.__name__} must implement _intensities_at_queries "
                "(the raw (B, Q) -> (B, Q, D) intensity primitive)."
            )

    def __init__(
        self,
        D: int,
        *,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        """
        Initialize the temporal point process.

        Args:
            D: Number of event types (dimensions)
            seed: Random seed for reproducibility
            terminating: If True, each event type can only occur once (first occurrence data).
                After an event type occurs, its intensity becomes zero and it cannot occur again.
            time_scale: Time-unit scale s; the core computes in scaled units t/s. Default 1.0 (no-op).
            token_roles: :class:`~torchpp.utils.event_data.TokenRoles` of the data (origin
                regime, context and ignored types). ``None`` means no origin, every type
                predictable.
            ci_config: Cumulative-intensity scheme.  ``None`` (default) falls back to the
                family default via :meth:`_default_ci_config`; an explicit value overrides it.
            sampling_config: Draw scheme.  ``None`` (default) falls back to the family
                default via :meth:`_default_sampling_config`; an explicit value overrides it.
        """
        super().__init__(
            D=D,
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            sampling_config=sampling_config,
        )
        # Active CIConfig override (set here, by ``use_ci``, or by direct assignment).
        # ``None`` lets the ``ci_config`` property derive the per-family default from
        # :meth:`_default_ci_config`.  (The SamplingConfig override + plumbing live on the
        # root ``TemporalPointProcess``; only ``ci_config`` is intensity-specific.)
        self._ci_config_override: Optional[CIConfig] = ci_config

    # =========================================================================
    # Intensity Functions (Must be implemented by subclasses)
    # =========================================================================

    # -- Intensity access (the multi-query primitive) ------------------------
    #
    # Two hooks of increasing structure; a model overrides whichever fits:
    #   _intensities_at_queries (B,Q)   -> (B,Q,D)       flat — arbitrary times (required)
    #   _interval_intensities  (B,R,P)  -> (B,R,P,D)     structured — per interval
    #
    # The flat hook is the single required primitive; the __init_subclass__ guard
    # rejects a concrete model that does not override it.  The structured hook
    # always terminates by flattening into the flat hook, so it is a pure
    # fast-path seam, never required.  Both return RAW intensities (no termination
    # masking) — masking is applied by the public ``intensity``.  A single-query
    # need is served by ``_intensities_at_queries(t.unsqueeze(1), ts).squeeze(1)``.
    #
    # Conditioning contract: the tie-break at an exact event time is
    # strictly-before / left-limit — an event never contributes to its own
    # intensity (the standard TPP convention).  Every model's flat override
    # must honour this; the classical and neural causal lookups already do.

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Raw intensity at Q arbitrary query times per sequence (the intensity primitive).

        The single required hook.  A single-query evaluation is served by
        ``_intensities_at_queries(t.unsqueeze(1), ts).squeeze(1)``.

        Args:
            query_times: Absolute query times. Shape: (B, Q)
            ts: Historical events. Shape: (B, L)

        Returns:
            Raw intensities at each query. Shape: (B, Q, D)
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _intensities_at_queries."
        )

    def _interval_intensities(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Raw intensity at P query times within each of R intervals (the structured primitive).

        ``query_times[b, i, :]`` are the P **absolute** times in interval ``i`` for
        sequence ``b``; the model reads event times from ``ts`` to know the history
        ``H_i`` conditioning each row.  Default: flatten the (R, P) grid into
        the flat hook and reshape back — always correct (each query conditions on
        its own strictly-before history), so the structured hook is a pure
        speed/seam optimisation that neural models override with interval-state
        reuse / batch-folding.

        Args:
            query_times: Absolute times per interval. Shape: (B, R, P)
            ts: Historical events. Shape: (B, L)

        Returns:
            Raw intensities. Shape: (B, R, P, D)
        """
        B, R, P = query_times.shape
        lam = self._intensities_at_queries(query_times.reshape(B, R * P), ts)  # (B, R*P, D)
        return lam.reshape(B, R, P, self.D)  # (B, R, P, D)

    def _sequence_intensities(
        self,
        ts: BatchedMVEventData,
    ) -> torch.Tensor:
        """
        Compute intensities for **all** event types at every observed event position.

        This is the P=1 structured case: position ``l`` is the right endpoint
        ``t_l`` of interval ``I_l``, conditioned on ``H_l`` (events strictly before
        ``t_l``) — i.e. ``λ(t_l⁻ | H_l)``, the left-limit, for free.  The default
        routes through ``_interval_intensities`` so a model's structured fast-path
        is reused here; it equals evaluating ``_intensities_at_queries``
        at the event times.  Override directly for a cheaper all-positions pass.

        This method is the building block for ``_event_intensities``.  It returns
        raw intensities for every type without any gathering or masking.

        Args:
            ts: Batched event sequences. Shape: (B, L)

        Returns:
            Intensities for all event types at each position.
            Shape: (B, L, D).  Padding positions may contain arbitrary values.
        """
        B, L = ts.time_points.shape
        device = ts.time_points.device

        if L == 0:
            return torch.zeros(B, 0, self.D, device=device)

        # Right endpoints of intervals I_0..I_{L-1} are the event times themselves.
        query = ts.time_points.unsqueeze(-1)  # (B, L, 1) — P=1
        return self._interval_intensities(query, ts).squeeze(2)  # (B, L, D)

    def _event_intensities(
        self,
        ts: BatchedMVEventData,
    ) -> torch.Tensor:
        """
        Compute the intensity of the *observed* event type at each position.

        Default implementation calls ``_sequence_intensities`` to get all-type
        intensities ``(B, L, D)`` and then gathers the correct event type.
        Override this method directly for models that can compute the ``(B, L)``
        result more efficiently without the ``(B, L, D)`` intermediate.

        This is the hook feeding the positive likelihood term
        (``_event_log_intensity_sum``), which applies ``log`` and ``clamp`` itself.
        Subclasses must return raw (linear-space) intensities — no logging or clamping here.

        Args:
            ts: Batched event sequences. Shape: (B, L)

        Returns:
            Intensity of the observed event type at each position.
            Shape: (B, L).  Padding positions may contain arbitrary values.
        """
        B, L = ts.time_points.shape
        device = ts.time_points.device

        if L == 0:
            return torch.zeros(B, 0, device=device)

        lam_all = self._sequence_intensities(ts)  # (B, L, D)
        et = ts.event_types_idx_safe.unsqueeze(-1)  # (B, L, 1)
        return lam_all.gather(2, et).squeeze(-1)  # (B, L)

    def _intensity_scaled(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Scaled-core intensity: the leaf/flat hook dispatch + termination masking.

        Operates on already-scaled inputs (``t`` and ``ts`` in internal units) and
        returns the scaled rate.  The public :meth:`intensity` shell wraps this with
        the unit conversion; every internal caller (numerical CI, grids, inversion,
        sampling) uses this core directly to avoid double-converting.
        """
        if not bool((t >= 0.0).all()):
            raise ValueError("Intensity is only defined for t >= 0.")

        if t.dim() == 1:
            # (B,) single query — route through the flat primitive at Q=1, then mask.
            intensities = self._intensities_at_queries(t.unsqueeze(1), ts).squeeze(1)  # (B, D)
            if self.terminating:
                mask = self._get_termination_mask(ts, t=t)  # (B, D)
                intensities = intensities.masked_fill(~mask, 0.0)  # an overflow (inf) masks to 0, not NaN
            return intensities

        # (B, Q) multi-query — route through the flat primitive and mask per query.
        intensities = self._intensities_at_queries(t, ts)  # (B, Q, D)
        if self.terminating:
            mask = self._terminated_mask(t, ts)  # (B, Q, D)
            intensities = intensities.masked_fill(~mask, 0.0)
        return intensities

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute the intensity function λ_d(t) for each event type d.

        The intensity function represents the instantaneous rate of events of each
        type conditional on the history of events up to time t.

        For terminating processes, the intensity is zero for event types that have
        already occurred (strictly before time t).

        Subclasses should override _intensities_at_queries() instead of this method to
        ensure termination is properly handled.

        Args:
            t: Query times. Shape ``(B,)`` for one time per sequence, or ``(B, Q)`` for
                Q times per sequence (routed through the flat multi-query primitive).
            ts: Historical events (batched, padded sequences). Shape: (batch_size, seq_len)

        Returns:
            Intensity values. Shape ``(B, D)`` for ``(B,)`` input, ``(B, Q, D)`` for
            ``(B, Q)`` input.  For terminating processes, intensities are zero for
            already-occurred types (per query time).

        Mathematical definition:
            λ_d(t) = lim_{dt→0} P(event of type d in [t, t+dt] | H_t) / dt
            where H_t is the history of events up to time t.
            For terminating processes: λ_d(t) = 0 if d has occurred at time t' < t.
        """
        # Native shell: normalise the query and history to internal units, run the
        # scaled core, then de-normalise the rate (λ is per-unit-time → ÷ s).
        return self._intensity_scaled(self._to_internal(t, ts), self._scaled_ts(ts)) / self.time_scale

    # =========================================================================
    # Cumulative Intensity Functions. If analytical solution is not provided, numerical integration will be used
    # =========================================================================

    def _effective_T(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Per-type upper integration limit ``min(T, t_term_d)``. Shape ``(B, D)``.

        For terminating processes a type stops accruing intensity at its termination
        time; non-terminating reduces to ``T`` broadcast across all types.
        """
        if self.terminating:
            termination_times = self._get_termination_times(ts)  # (B, D)
            return torch.minimum(T.unsqueeze(1).expand_as(termination_times), termination_times)  # (B, D)
        return T.unsqueeze(1).expand(T.shape[0], self.D)  # (B, D)

    def _closed_form_ci(
        self, ts: BatchedMVEventData, T_low: Optional[torch.Tensor], T: torch.Tensor
    ) -> torch.Tensor:
        """Closed-form Λ over ``[T_low or 0, T]`` — the ``mode="closed_form"`` path.

        The full ``[0, T]`` integral is ``Σ _interval_ci`` (intervals start at events, so
        the per-interval heads see the correct relative origin; ``_active_interval_mask``
        zeroes post-termination intervals per type; ``Σ`` telescopes to the global Λ(T)).
        Windowing recurses for the lower limit — ``Λ(T) − Λ(T_low)`` — rather than clipping
        the interval grid: a mid-interval ``T_low`` would shift a relative-time head's
        origin.  The subtraction is exact for an analytic closed form (no quadrature error),
        so there is no cancellation concern.  Efficient for neural closed-form heads
        (``_interval_ci`` is ``O(L)``).  Hawkes/Baseline override this with the ``O(N·D)``
        global module CI (they provide no per-interval closed form).
        """
        lo, hi = self._interval_bounds(ts, None, T)  # (B, L+1), intervals start at events
        ci = self._interval_ci(ts, lo, hi)  # (B, L+1, D), history H_i
        if self.terminating:
            ci = ci * self._active_interval_mask(ts, lo)
        ci = ci.sum(dim=1)  # (B, D)
        if T_low is not None:
            ci = ci - self._closed_form_ci(ts, None, T_low)
        return ci  # (B, D)

    def _numerical_cumulative_intensity(
        self,
        T: torch.Tensor,
        ts: BatchedMVEventData,
        T_low: Optional[torch.Tensor] = None,
        method: Optional[str] = None,
        num_points: Optional[int] = None,
    ):
        """Global (whole-window) numerical CI Λ_d(T) — the ``placement="global"`` path.

        Internal: reached via ``cumulative_intensity`` dispatch (the model's ``CIConfig``
        selects the path); not part of the public API.

        Integrates ``self.intensity`` over ``[T_low or 0, T]`` directly (no
        ``CI(T) − CI(T_low)`` subtraction).  ``method``/``num_points`` default to the
        instance ``ci_config``'s rule/n_points; the dispatcher passes them explicitly.

        For terminating processes the finite per-type termination times are inserted
        as exact grid knots (``extra_knots``) and ``effective_t_end`` carries the
        per-type upper limits ``min(T, t_term_d)``; the trapezoid uses cumulative
        prefix sums to read each type's integral from one shared evaluation pass.

        Args:
            T: End times. Shape: (B,)
            ts: Historical events. Shape: (B, L)
            T_low: Optional start times. Shape: (B,)
            method: One of ``"trapezoidal"``, ``"mc_trapezoidal"``, ``"monte_carlo"``.
            num_points: Number of grid points over the window.
        """
        method = method if method is not None else _RULE_TO_METHOD[self.ci_config.rule]
        num_points = num_points if num_points is not None else self.ci_config.n_points
        t_start = torch.zeros_like(T) if T_low is None else T_low

        extra_knots = None
        effective_t_end = None

        if self.terminating:
            termination_times = self._get_termination_times(ts)  # (B, D), inf for unterminated
            T_exp = T.unsqueeze(1).expand_as(termination_times)  # (B, D)

            # Per-type upper limits: min(T, t_term_d).  For unterminated types
            # (inf) this reduces to T, matching the standard behaviour.
            effective_t_end = torch.minimum(T_exp, termination_times)  # (B, D)

            # Only pass finite termination times that fall inside (t_start, T)
            # as extra knots; out-of-range ones are irrelevant for accuracy.
            t_start_exp = t_start.unsqueeze(1).expand_as(termination_times)
            in_range = (
                torch.isfinite(termination_times) & (termination_times > t_start_exp) & (termination_times < T_exp)
            )
            # Replace out-of-range entries with t_end so they merge harmlessly.
            extra_knots = torch.where(in_range, termination_times, T_exp)  # (B, D)

        intensity_func = self._intensity_scaled
        if method != "monte_carlo":
            # The first node is the window start; evaluate it at the right limit so a window
            # starting at an event integrates the post-jump intensity.
            event_dtype = self._time_compare_dtype(ts)

            def intensity_func(t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
                t = torch.cat([right_limit(t[:, :1], event_dtype), t[:, 1:]], dim=1)
                return self._intensity_scaled(t, ts)

        return numerical_integration(
            intensity_func,
            t_start=t_start,
            t_end=T,
            ts=ts,
            method=method,
            num_points=num_points,
            rng=None,
            extra_knots=extra_knots,
            effective_t_end=effective_t_end,
        )

    # =========================================================================
    # Cumulative-intensity construction
    # =========================================================================

    def _interval_ci(self, ts: BatchedMVEventData, t_lo: torch.Tensor, t_hi: torch.Tensor) -> torch.Tensor:
        """Closed-form ∫λ_d over ``[t_lo_i, t_hi_i]`` within interval ``i``.

        Parallel in shape to ``_interval_intensities``: the per-interval analytic
        integral, history held constant at ``H_i`` within interval ``i``.  The default
        derives it from the model's global analytic CI as ``Λ(hi_i) − Λ(lo_i)``: no
        event lies strictly inside an inter-event interval, so the history seen on
        ``[lo_i, hi_i]`` is exactly ``H_i`` and the difference is the genuine
        per-interval integral.  Summing over ``i`` telescopes to the global closed-form
        Λ(T) — the trust-chain anchor (gate test ``Σ _interval_ci ≡ global closed form``).
        A model without an analytic closed form routes through numerical quadrature
        instead (``CIConfig`` ``mode="numerical"``).

        Closed-form families override this (neural: causal param-gather; classical:
        baseline+kernel module per-interval CI).  A model without a closed form leaves
        it unimplemented and uses a numerical ``CIConfig`` instead.

        Args:
            ts: Historical events. Shape: (B, L)
            t_lo: Per-interval lower bounds. Shape: (B, L+1) or (B, L+1, D)
            t_hi: Per-interval upper bounds. Same shape as ``t_lo``.

        Returns:
            Per-interval integrals. Shape: (B, L+1, D).
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no closed-form _interval_ci; use a numerical CIConfig."
        )

    def _default_ci_config(self) -> CIConfig:
        """Per-family default CIConfig.

        The safe majority is closed-form; families that integrate numerically override this
        to declare their placement/rule/n_points explicitly (fail-loud: a numerical model
        that forgets hits ``closed_form`` and its missing ``_interval_ci`` raises).
        :func:`~torchpp.training.create_model` applies ``arch.ci`` (and its default
        ``n_points`` for numerical models) by assigning ``self.ci_config`` after construction."""
        return CIConfig(mode="closed_form")

    @property
    def ci_config(self) -> CIConfig:
        """Active CIConfig: explicit override if set, else the per-family default."""
        return self._ci_config_override if self._ci_config_override is not None else self._default_ci_config()

    @ci_config.setter
    def ci_config(self, cfg: Optional[CIConfig]) -> None:
        self._ci_config_override = cfg

    @contextmanager
    def use_ci(self, cfg: CIConfig) -> Iterator[None]:
        """Temporarily force a CIConfig for a block (eval / forecast / diagnostic sweep).

        Resolution is call-arg > this context > instance default, so wrapping an
        evaluation loop in ``with model.use_ci(cfg):`` redirects every CI call inside
        without touching model internals.
        """
        prev = self._ci_config_override
        self._ci_config_override = cfg
        try:
            yield
        finally:
            self._ci_config_override = prev

    def _resolve_inversion_method(self, cfg: SamplingConfig) -> str:
        """Resolve ``SamplingConfig.inversion_method`` to a concrete path.

        ``"auto"``: ``grid`` for a
        numerical CI, else ``closed_form`` when the model overrides
        :meth:`_inverse_marginal_cumulative_intensity`, else ``newton``.
        """
        if cfg.inversion_method != "auto":
            return cfg.inversion_method
        if self.ci_config.mode != "closed_form":
            return "grid"
        overridden = type(self)._inverse_marginal_cumulative_intensity is not IntensityBasedTPP._inverse_marginal_cumulative_intensity
        return "closed_form" if overridden else "newton"

    def _grid_inversion_density(self, cfg: SamplingConfig) -> float:
        """Points/unit-time for the grid inverter (default ``ci_config.n_points / 0.1``)."""
        if cfg.grid_density is not None:
            return float(cfg.grid_density)
        return float(self.ci_config.n_points) / 0.1

    def _interval_bounds(
        self, ts: BatchedMVEventData, T_low: Optional[torch.Tensor], T: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The L+1 inter-event interval bounds, clipped to ``[T_low or 0, T]``.

        Returns ``(lo, hi)`` of shape ``(B, L+1)``.  Interval ``i`` is ``[t_{i-1}, t_i]``
        (with ``t_{-1}=0``, ``t_L=T``); padding positions push to ``+inf`` so they clip
        to zero width, so intervals outside the window contribute 0.
        """
        B, L = ts.time_points.shape
        dtype, device = T.dtype, T.device
        a = torch.zeros_like(T) if T_low is None else T_low.to(dtype)  # (B,)
        tp = ts.time_points.to(dtype)  # (B, L) padded
        inf = torch.full_like(tp, float("inf"))
        tp_real = torch.where(ts.event_mask, tp, inf)  # padding -> +inf (clips out)

        zeros = torch.zeros(B, 1, dtype=dtype, device=device)
        lo = torch.cat([zeros, tp_real], dim=1)  # (B, L+1): [0, t_0, ..., t_{L-1}]
        hi = torch.cat([tp_real, T.unsqueeze(1)], dim=1)  # (B, L+1): [t_0, ..., t_{L-1}, T]

        a_e, T_e = a.unsqueeze(1), T.unsqueeze(1)
        lo = lo.clamp(min=a_e).clamp(max=T_e)
        hi = hi.clamp(min=a_e).clamp(max=T_e)
        hi = torch.maximum(hi, lo)  # no negative width after clipping
        return lo, hi

    def _interval_quadrature_grid(
        self, lo: torch.Tensor, hi: torch.Tensor, cfg: CIConfig
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-interval quadrature nodes + weights for the structured numerical path.

        ``lo``/``hi`` are ``(B, L+1)``; returns ``nodes`` ``(B, L+1, P)`` absolute times
        and ``weights`` ``(B, L+1, P)`` (already × interval width).  Zero-width intervals
        produce zero weight, so clipped/padding intervals contribute nothing.
        """
        B, R = lo.shape
        device, dtype = lo.device, lo.dtype
        P = max(int(cfg.n_points), 2)
        width = (hi - lo).unsqueeze(-1)  # (B, R, 1)

        if cfg.rule == "trapezoidal":
            frac = torch.linspace(0.0, 1.0, P, device=device, dtype=dtype).expand(B, R, P)
            wbase = torch.ones(P, device=device, dtype=dtype)
            wbase[0] = wbase[-1] = 0.5
            weights = wbase * (width / (P - 1))  # (B, R, P)
        elif cfg.rule == "mc":
            frac = torch.rand(B, R, P, device=device, dtype=dtype)
            weights = (width / P).expand(B, R, P)
        elif cfg.rule == "mc_trapezoidal":
            interior = torch.rand(B, R, P - 2, device=device, dtype=dtype)
            ends = torch.zeros(B, R, 1, device=device, dtype=dtype)
            frac = torch.cat([ends, interior, ends + 1.0], dim=-1)
            frac, _ = frac.sort(dim=-1)  # sorted random nodes incl. endpoints
            seg = (frac[..., 1:] - frac[..., :-1]) * width  # (B, R, P-1) sub-widths
            weights = torch.zeros(B, R, P, device=device, dtype=dtype)
            weights[..., :-1] = weights[..., :-1] + 0.5 * seg
            weights[..., 1:] = weights[..., 1:] + 0.5 * seg
        else:
            raise ValueError(f"Unknown CIConfig.rule {cfg.rule!r}.")

        nodes = lo.unsqueeze(-1) + width * frac  # (B, R, P)
        if cfg.rule != "mc":
            nodes[..., -1] = hi  # exact right endpoint: lo + width may round past an event at hi
        return nodes, weights

    def _active_interval_mask(self, ts: BatchedMVEventData, lo: torch.Tensor) -> torch.Tensor:
        """``(B, L+1, D)`` bool: interval ``i`` is active for type ``d`` iff it starts
        before type ``d``'s termination.

        A termination time is the first occurrence of that type — itself an event time,
        i.e. an interval *boundary* — so a terminating type contributes over whole
        intervals (those starting before its terminating event) and is zero thereafter.
        No partial-interval handling is needed.  Non-terminating ⇒ ``t_term = inf`` ⇒
        all-active.
        """
        t_term = self._get_termination_times(ts)  # (B, D); inf where not (yet) terminated
        return lo.unsqueeze(-1) < t_term.unsqueeze(1)  # (B, L+1, D)

    def _numerical_interval_ci(
        self, T: torch.Tensor, ts: BatchedMVEventData, T_low: Optional[torch.Tensor], cfg: CIConfig
    ) -> torch.Tensor:
        """Event-knotted per-interval numerical CI (the ``placement="interval"`` path).

        One self-contained path: build the L+1 inter-event interval grid (P nodes per
        interval, ``cfg.rule``), evaluate the structured intensity hook with its
        per-interval conditioning, integrate ``λ·w`` within each interval, and
        sum.  Termination is whole-interval masking (see ``_active_interval_mask``) —
        no cross-branch to the global path.  ``chunk_size`` bounds peak memory by
        accumulating over the node axis (eval/forecast, ``no_grad``).
        """
        lo, hi = self._interval_bounds(ts, T_low, T)  # (B, L+1) each
        nodes, weights = self._interval_quadrature_grid(lo, hi, cfg)  # (B, L+1, P) each
        if cfg.rule != "mc" and not self._interval_hook_includes_left_event:
            # The first node sits on the interval's left endpoint (an event): evaluate the
            # post-jump right limit, so the trapezoid integrates the intensity on the interval.
            nodes[..., 0] = right_limit(nodes[..., 0], self._time_compare_dtype(ts))
        # A forecast window [T_low, T] with T_low at or after the last event covers only the
        # final interval. The default hook conditions each query on its own history, so the
        # empty intervals are skipped; overridden hooks index interval state by position and
        # always see all of them.
        keep = None
        full_shape = (*lo.shape, self.D)
        if (
            T_low is not None
            and type(self)._interval_intensities is IntensityBasedTPP._interval_intensities
            and bool((T_low >= self._last_event_time(ts).to(T_low.dtype)).all())
        ):
            active = (hi > lo).any(dim=0)  # (L+1,)
            if not bool(active.any()):
                return torch.zeros(full_shape[0], self.D, dtype=weights.dtype, device=weights.device)
            keep = active.nonzero().squeeze(1)
            nodes, weights = nodes[:, keep], weights[:, keep]
        chunk = cfg.chunk_size
        P = nodes.shape[2]
        if not chunk:
            lam = self._interval_intensities(nodes, ts)  # (B, L+1, P, D)
            contrib = (lam * weights.unsqueeze(-1)).sum(dim=2)  # (B, L+1, D)
        else:
            contrib = None
            for c0 in range(0, P, chunk):
                c1 = min(c0 + chunk, P)
                lam = self._interval_intensities(nodes[..., c0:c1], ts)  # (B, L+1, pc, D)
                part = (lam * weights[..., c0:c1].unsqueeze(-1)).sum(dim=2)  # (B, L+1, D)
                contrib = part if contrib is None else contrib + part
        if keep is not None:  # empty intervals contribute exactly zero
            contrib = torch.zeros(full_shape, dtype=contrib.dtype, device=contrib.device).index_copy(1, keep, contrib)
        if self.terminating:
            contrib = contrib * self._active_interval_mask(ts, lo)  # zero post-termination intervals
        return contrib.sum(dim=1)  # (B, D)

    def _cumulative_intensity_scaled(
        self,
        T: torch.Tensor,
        ts: BatchedMVEventData,
        T_low: Optional[torch.Tensor] = None,
        ci_config: Optional[CIConfig] = None,
    ):
        """Scaled-core cumulative intensity: the CIConfig dispatch on scaled inputs.

        Returns Λ over ``[T_low or 0, T]`` in internal units.  Λ is dimensionless, so
        the public :meth:`cumulative_intensity` shell only normalises the inputs and
        adds no output conversion.  Internal callers (likelihood, grids, inversion)
        use this core directly.
        """
        # Check legality of T:
        if T.shape[0] != ts.time_points.shape[0]:
            raise ValueError(f"T must have one entry per sequence: got {T.shape[0]} for a batch of {ts.time_points.shape[0]}.")
        if not bool((T >= 0.0).all()):
            raise ValueError("Cumulative intensity is only defined for T >= 0.")

        if T_low is not None:
            if T_low.shape[0] != ts.time_points.shape[0]:
                raise ValueError("T_low must have one entry per sequence.")
            if not bool((T_low >= 0.0).all()) or not bool((T >= T_low).all()):
                raise ValueError("Cumulative intensity requires 0 <= T_low <= T.")

        cfg = ci_config if ci_config is not None else self.ci_config

        if cfg.mode == "closed_form":
            return self._closed_form_ci(ts, T_low, T)

        if cfg.placement == "global":
            return self._numerical_cumulative_intensity(
                T, ts, T_low=T_low, method=_RULE_TO_METHOD[cfg.rule], num_points=cfg.n_points
            )
        return self._numerical_interval_ci(T, ts, T_low, cfg)

    def cumulative_intensity(
        self,
        T: torch.Tensor,
        ts: BatchedMVEventData,
        T_low: Optional[torch.Tensor] = None,
        ci_config: Optional[CIConfig] = None,
    ):
        """Cumulative intensity Λ_d(T) (or over ``[T_low, T]``), dispatched by CIConfig.

        Single entry for both the survival term and point-CI queries.  The scheme is
        resolved call-arg > active ``use_ci`` context > instance default.

        Args:
            T: End times. Shape: (B,)
            ts: Historical events. Shape: (B, L)
            T_low: Optional start times. Shape: (B,)
            ci_config: Force a specific CIConfig for this call (else the instance/context one).
        """
        # Native shell: normalise the integration limits and history to internal units
        # and run the scaled core.  Λ = ∫λ dt is dimensionless (λ ~ 1/time, dt ~ time),
        # so there is NO output conversion.
        # T and T_low are mapped exactly like the event times, so the T >= T_low and
        # T >= t_last invariants of the native inputs survive the conversion.
        T_low_s = None if T_low is None else self._to_internal(T_low.to(T.dtype), ts)
        return self._cumulative_intensity_scaled(
            self._to_internal(T, ts), self._scaled_ts(ts), T_low=T_low_s, ci_config=ci_config
        )

    # ========================================================================
    # Intensity Grid (Pre-computed intensity with fast CI lookup)
    # ========================================================================

    def _grid_funcs(self, integrate_marginal_via_grid: bool, type_mask: Optional[torch.Tensor]):
        """Scaled intensity (and exact CI, if used) for a grid, restricted to ``type_mask``."""
        ci_func = None
        if self.ci_config.mode == "closed_form" and not integrate_marginal_via_grid:

            def ci_func(t, ts):
                return self._cumulative_intensity_scaled(t, ts)

        if type_mask is None:
            return self._intensity_scaled, ci_func
        drop = ~type_mask

        def intensity_func(t, ts):
            return self._intensity_scaled(t, ts).masked_fill(drop, 0.0)

        masked_ci_func = None
        if ci_func is not None:

            def masked_ci_func(t, ts):
                return ci_func(t, ts).masked_fill(drop, 0.0)

        return intensity_func, masked_ci_func

    def _retained_types(self, ignore_indices: Optional[Sequence[int]] = None) -> Optional[torch.Tensor]:
        """``(D,)`` bool of the types that compete for the next event: predictable and not
        in ``ignore_indices``.  ``None`` when every type is retained."""
        keep = self.predictable_mask.clone()
        if ignore_indices:
            keep[list(ignore_indices)] = False
        return None if bool(keep.all()) else keep

    def _build_intensity_grid(
        self,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        ts: BatchedMVEventData,
        num_points: Optional[int] = None,
        integrate_marginal_via_grid: bool = True,
        type_mask: Optional[torch.Tensor] = None,
    ) -> IntensityGrid:
        """
        Build a pre-computed intensity grid over [t_start, t_end].

        This evaluates the intensity function once on a uniform grid and computes
        cumulative intensity. For analytical CI models, exact CI values are used
        via the analytical cumulative_intensity function. For numerical CI models,
        CI is approximated via trapezoidal prefix sums over the intensity grid.

        When ``integrate_marginal_via_grid=True``, the trapezoidal path is used
        even for analytical CI models, avoiding the expensive per-grid-point CI
        evaluations.  With ≥200 grid points the approximation error is negligible.

        The resulting IntensityGrid can be queried for intensity and CI at any
        grid point in O(1), or at arbitrary points via interpolation.

        Args:
            t_start: Start times. Shape: (B,)
            t_end: End times. Shape: (B,)
            ts: Batched event data.
            num_points: Number of grid points.
            integrate_marginal_via_grid: When True, always use trapezoidal CI
                even if the model has analytical CI.
            type_mask: Optional ``(D,)`` bool; types where it is False are removed from
                the process (zero intensity and CI).

        Returns:
            IntensityGrid with pre-computed values.
        """
        # Grids operate in scaled units (scaled ts + scaled cores) so grid density is
        # scale-covariant; the public time_* / prob_* shells feed a scaled batch and
        # convert the result.
        if num_points is None:
            num_points = self.ci_config.grid_num_points
        intensity_func, ci_func = self._grid_funcs(integrate_marginal_via_grid, type_mask)
        return IntensityGrid.build(
            intensity_func=intensity_func,
            t_start=t_start,
            t_end=t_end,
            ts=ts,
            num_points=num_points,
            ci_func=ci_func,
            right_limit_dtype=self._time_compare_dtype(ts),
        )

    def _build_intensity_grid_until(
        self,
        t_start: torch.Tensor,
        ts: BatchedMVEventData,
        points_per_unit_time: Optional[float] = None,
        target_cdf: Optional[float] = _USE_CI_DEFAULT,
        T_max: Optional[torch.Tensor] = None,
        chunk_duration: Optional[float] = None,
        max_chunks: Optional[int] = None,
        integrate_marginal_via_grid: bool = True,
        type_mask: Optional[torch.Tensor] = None,
    ) -> IntensityGrid:
        """
        Build an IntensityGrid incrementally until a stopping condition is met.

        Grows the grid in chunks, stopping when CDF >= target_cdf and/or T_max
        is reached. Grid density is controlled by ``points_per_unit_time``.

        This avoids the guess-and-rebuild pattern: the grid starts small and
        extends only as far as needed, reusing all intermediate evaluations.
        The resulting grid can be directly used for marginals, mean time,
        median time, and CDF inversion without any additional intensity evaluations.

        Args:
            t_start: Start times. Shape: (B,)
            ts: Batched event data.
            points_per_unit_time: Number of grid points per unit of scaled time.
                Higher = more accurate integration but more intensity evaluations.
            target_cdf: Stop when CDF >= this value for all batch elements.
                Set to None to disable (must then provide T_max).
            T_max: Hard upper time limit. Shape: (B,) or None.
            chunk_duration: Duration of each extension chunk in scaled time.
            max_chunks: Maximum number of chunks (safety limit).
            type_mask: Optional ``(D,)`` bool; types where it is False are removed from
                the process, so the CDF stopping rule applies to the retained types.

        Returns:
            IntensityGrid covering [t_start, t_stop].
        """
        # Scaled units (see _build_intensity_grid): scaled cores + scaled batch.
        cfg = self.ci_config
        if points_per_unit_time is None:
            points_per_unit_time = cfg.grid_points_per_unit_time
        if target_cdf is _USE_CI_DEFAULT:  # None stays None (disables the CDF stop)
            target_cdf = cfg.grid_target_cdf
        if chunk_duration is None:
            chunk_duration = cfg.grid_chunk_duration
        if max_chunks is None:
            max_chunks = cfg.grid_max_chunks
        intensity_func, ci_func = self._grid_funcs(integrate_marginal_via_grid, type_mask)
        return IntensityGrid.build_until(
            intensity_func=intensity_func,
            t_start=t_start,
            ts=ts,
            points_per_unit_time=points_per_unit_time,
            target_cdf=target_cdf,
            T_max=T_max,
            ci_func=ci_func,
            chunk_duration=chunk_duration,
            max_chunks=max_chunks,
            right_limit_dtype=self._time_compare_dtype(ts),
        )

    # ========================================================================
    # CDF/CI Inversion (Override for analytical solution)
    # ========================================================================
    # These run in scaled units: they take a scaled ``ts`` (fed by the ``time_*``
    # shells or the samplers), use the scaled cores, and return a scaled time.  The
    # public ``time_quantile`` shell multiplies the result by ``s``.

    def _inverse_marginal_cumulative_intensity(
        self,
        u: torch.Tensor,
        ts: BatchedMVEventData,
    ) -> torch.Tensor:
        """
        Invert the marginal cumulative intensity to find t such that Λ(t) = u.
        t is here in absolute (scaled) time.

        Default implementation: Uses Bracketed-Newton's method from utils.py.
        Override for analytical solutions or more efficient implementations.

        Args:
            u: Target cumulative intensity values. Shape: (batch_size,)
            ts: Historical events (batched, padded sequences). Shape: (batch_size, seq_len)

        Returns:
            Time values t such that cumulative_intensity(t) ≈ u. Shape: same as u

        Useful for sampling and other applications requiring CI inversion.
        """
        if u.shape[0] != ts.time_points.shape[0]:
            raise ValueError("u must have one entry per sequence.")

        device = ts.time_points.device
        cfg = self.sampling_config

        # For numerical CI models, use grid-based inversion to avoid repeated
        # numerical integration during rootfinding (each Newton step would
        # otherwise trigger a full numerical CI computation).
        if self.ci_config.mode != "closed_form":
            t_start = torch.zeros(u.shape[0], device=device, dtype=torch.float64)

            # Convert target CI to equivalent CDF for the grid stopping condition
            target_cdf = float((1.0 - torch.exp(-u.max())).item())
            target_cdf = min(target_cdf + _GRID_TARGET_CDF_MARGIN, cfg.quantile_cap)

            intensity_func, _ = self._grid_funcs(True, self._retained_types())
            grid = IntensityGrid.build_until(
                intensity_func=intensity_func,
                t_start=t_start,
                ts=ts,
                points_per_unit_time=self._grid_inversion_density(cfg),
                target_cdf=target_cdf,
                T_max=None,
                right_limit_dtype=self._time_compare_dtype(ts),
            )

            # Check the grid covers the target CI for all batch elements
            ci_at_end = grid.total_ci().sum(dim=1)  # (B,)
            if (ci_at_end >= u).all():
                # Grid-based inversion: find t where total_ci = u
                total_ci_grid = grid.ci_prefix.sum(dim=2)  # (B, num_points)
                k = torch.searchsorted(total_ci_grid, u.unsqueeze(1), right=False).squeeze(1)
                k = k.clamp(min=1, max=grid.num_points - 1)

                ci_lo = torch.gather(total_ci_grid, 1, (k - 1).unsqueeze(1)).squeeze(1)
                ci_hi = torch.gather(total_ci_grid, 1, k.unsqueeze(1)).squeeze(1)
                t_lo = torch.gather(grid.t_grid, 1, (k - 1).unsqueeze(1)).squeeze(1)
                t_hi = torch.gather(grid.t_grid, 1, k.unsqueeze(1)).squeeze(1)

                dci = (ci_hi - ci_lo).clamp(min=_DIV_FLOOR)
                frac = ((u - ci_lo) / dci).clamp(0.0, 1.0)

                return t_lo + frac * (t_hi - t_lo)
            # else: fall through to Newton's method

        # Analytical CI or grid fallback: use bracketed Newton's method.  Both sums drop
        # context (non-predictable) columns so the inverted total matches the role-masked
        # survival; a no-op under the default all-True mask.
        def ci_func(t: torch.Tensor) -> torch.Tensor:
            ci = self._cumulative_intensity_scaled(t, ts)  # (B,D)
            return self._predictable_sum(ci)  # (B,)

        def intensity_func(t: torch.Tensor) -> torch.Tensor:
            lam = self._intensity_scaled(t, ts)  # (B,D)
            return self._predictable_sum(lam)  # (B,)

        origin = self._forecast_origin(ts)
        t_star = invert_CI(
            cumulative_intensity_func=ci_func,
            intensity_func=intensity_func,
            target=u,
            bracket_init=(origin, origin + cfg.bracket_width),
            tol=cfg.newton_tol,
            max_newton_iters=cfg.newton_max_iters,
        )
        return t_star

    def _inverse_CDF(self, u: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Invert the CDF to find T such that P( t_next < T) = u.
        Also known as the quantile function. It is only defined for T >= last event time.
        If you want to sample inbetween events, use the _inverse_marginal_cumulative_intensity function,
        or cut of the history ts.

        For analytical CI models, transforms u → target CI and delegates to
        _inverse_marginal_cumulative_intensity (Newton's method).

        For numerical CI models, builds an IntensityGrid from t_last until
        CDF >= max(u), then uses grid-based interpolation for O(1) lookup.
        This avoids repeated numerical integration during rootfinding.

        Args:
            u: Target CDF values. Shape: (batch_size,)
            ts: Historical events. Shape: (batch_size, seq_len)

        Returns:
            Time values T such that CDF(T) ≈ u. Shape: same as u

        Useful for sampling and other applications requiring CDF inversion.
        """
        cfg = self.sampling_config
        # Grid-based CDF inversion (numerical CI by default; forceable via
        # SamplingConfig.inversion_method).
        u = u.to(torch.float64).clamp(max=cfg.quantile_cap)  # float32 would round the cap to 1
        if self._resolve_inversion_method(cfg) == "grid":
            # Build the grid from the forecast origin until the CDF passes the largest quantile.
            target_cdf = min(float(u.max().item()) + _GRID_TARGET_CDF_MARGIN, 1.0 - 0.1 * (1.0 - cfg.quantile_cap))
            grid = self._build_intensity_grid_until(
                t_start=self._forecast_origin(ts),
                ts=ts,
                points_per_unit_time=self._grid_inversion_density(cfg),
                target_cdf=target_cdf,
                T_max=None,
                type_mask=self._retained_types(),
            )
            return self._inverse_cdf_from_grid(grid, u)

        # Analytical CI: transform u to target cumulative intensity
        device = ts.time_points.device
        target_ci = -torch.log1p(-u).to(device)  # (B,)

        # The CDF starts at the forecast origin.  Offset by the predictable-only Λ(origin) so
        # it matches the role-masked total the inversion targets (context types are excluded
        # from the survival, mirroring _windowed_total_ci).
        target_ci += self._predictable_sum(self._cumulative_intensity_scaled(self._forecast_origin(ts), ts))  # (B,)

        t_star = self._inverse_marginal_cumulative_intensity(target_ci, ts)  # (B,)

        return t_star

    # =========================================================================
    # Likelihood
    # =========================================================================

    def _event_log_intensity_sum(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Σ_i log λ_{m_i}(t_i) over valid events — the positive log-likelihood term.  ``(B,)``.

        Delegates the raw per-event intensity to ``_event_intensities`` (override *that*
        to speed up the positive term).  Padding, empty sequences, and the
        terminating-repeat impossibility (log p = −inf) are handled here so subclasses
        supply only raw λ.
        """
        B = ts.time_points.shape[0]
        L = ts.time_points.shape[1]
        device = ts.time_points.device

        if L == 0:
            return torch.zeros(B, device=device)

        valid_events = ts.event_mask  # (B, L)
        # Role/origin mask: score only predictable-typed real events, and — under the
        # no-origin (condition-on-first) regime — drop the earliest real event.  Shared
        # with the −log s Jacobian count (``_scored_mask``).
        scored = self._scored_mask(ts)  # (B, L)

        # --- core: get λ_{m_i}(t_i) for every i, from the subclass hook ---
        lam = self._event_intensities(ts)  # (B, L) — raw, before masking
        # Catch a broken parameterization (NaN / negative λ) at the source rather
        # than silently clamping it; an actual structurally-zero λ is fine and is
        # floored to finfo.tiny inside log() below. Guard is debug-only: a
        # per-element .all() check on a GPU tensor forces a host sync every step,
        # which is too costly during training.
        if __debug__ and not lam.is_cuda:
            valid_lam = lam[valid_events]
            assert torch.isfinite(valid_lam).all() and (valid_lam >= 0).all(), (
                "intensity at observed events must be finite and non-negative"
            )
        # Floor only inside log(): finfo.tiny keeps log() finite for a literally
        # zero intensity (log(tiny) ≈ −87 f32 / −708 f64). Clamp the log() argument
        # only — `lam` itself stays unclamped so the survival integral Λ=∫λ is not
        # fed a spurious positive rate.  Mirrors utils/evaluation.py.
        log_lam = torch.log(lam.clamp(min=torch.finfo(lam.dtype).tiny))  # (B, L)

        # Zero out padding and context (non-predictable) positions
        log_lam = log_lam.masked_fill(~scored, 0.0)  # (B, L)
        log_positive = log_lam.sum(dim=1)  # (B,)

        # Sequences with no scorable (predictable) events contribute 0 to the positive term.
        no_events = scored.sum(dim=1) == 0  # (B,)
        log_positive = log_positive.masked_fill(no_events, 0.0)

        # Terminating: a repeated type makes the event impossible (log p = -inf).
        if self.terminating:
            has_repeated = self._has_repeated_event_types(ts)  # (B,)
            log_positive = log_positive.masked_fill(has_repeated, float("-inf"))

        return log_positive

    def _likelihood_scaled(self, ts: BatchedMVEventData, T: torch.Tensor) -> torch.Tensor:
        """Scaled-core log-likelihood on already-scaled inputs.  ``(B,)``.

            log L_scaled = ∑ᵢ log λ_{m_i}(τᵢ) − ∑_d Λ_d(T)   (all in internal units)

        The positive term reads raw λ at the observed events (a leaf bypass, not the
        public ``intensity``); the negative term is the dimensionless Λ.  The public
        :meth:`likelihood` shell converts to native units by subtracting ``n·log s``
        (one density-factor Jacobian per event).
        """
        log_positive = self._event_log_intensity_sum(ts)  # (B,)  Σ_i log λ_{m_i}(t_i)
        # No-origin (condition-on-first): survival starts at the first event t₁ (same scaled
        # units as the CI), so the no-history zeros column never integrates the [0, t₁] gap
        # (a head whose transform maps 0 → nonzero rate would otherwise inject spurious mass).
        # With an origin (<bos> at t=0) the [0, t₁] gap is real and integrated from 0.
        T_low = None
        if not self.has_origin and ts.time_points.shape[1] > 0:
            # Per-sequence: the first real event t₁ where the row HAS one (real events are
            # front-loaded + time-sorted), else 0: an all-padding (empty) row then carries only
            # the survival over [0, T] (zero for T = 0) and never T_low > T (the padding
            # sentinel would trip the CI's T >= T_low assertion in a mixed batch).
            first = ts.time_points[:, 0]
            T_low = torch.where(ts.event_mask[:, 0], first, torch.zeros_like(first))
        ci = self._cumulative_intensity_scaled(T=T, ts=ts, T_low=T_low)  # (B, D)
        # Survival over predictable type-columns only: context types do not compete
        # in Σ_d Λ_d.  ``_predictable_sum`` zeroes (not multiplies) the excluded columns
        # so a spurious inf/nan cannot leak in; a no-op under the default all-True mask.
        log_negative = -self._predictable_sum(ci)  # (B,)  −Σ_{d predictable} Λ_d(T)
        return log_positive + log_negative

    def _log_prob_events(self, ts_scaled: BatchedMVEventData, T_scaled: torch.Tensor) -> torch.Tensor:
        """Autoregressive per-event conditional: the scaled λ/Λ log-likelihood ``(B,)``."""
        return self._likelihood_scaled(ts_scaled, T_scaled)

    # =========================================================================
    # Predictive contract realization — via λ/Λ
    # =========================================================================

    def _prep_query_time(self, t: torch.Tensor, ts: BatchedMVEventData, atol: float) -> torch.Tensor:
        """Cast to double, snap near-``t_last`` queries onto ``t_last``, check ``t ≥ t_last``.

        Shared by the density/survival methods; ``atol`` is the near-boundary snap
        tolerance (densities and CDF/survival use different values).  Queries start at
        the forecast origin (``T_obs``, at least the last event).  Handles ``(B,)`` and
        ``(B, Q)`` queries.
        """
        t = t.double()
        mx = self._forecast_origin(ts).double()  # (B,)
        mx_b = mx.unsqueeze(1) if t.dim() == 2 else mx
        close = torch.abs(t - mx_b) < atol
        t = torch.where(close, mx_b.expand_as(t), t)
        if not bool((t >= mx_b).all()):
            raise ValueError("Predictive densities are only defined for t >= T_obs (the last event by default).")
        return t

    def _predictive_ci_config(self) -> CIConfig:
        """The active CIConfig with a deterministic rule, so predictive queries are reproducible.

        A Monte-Carlo training rule (``mc`` / ``mc_trapezoidal``) is replaced by the
        trapezoid; closed-form and trapezoidal configs are returned unchanged.
        """
        cfg = self.ci_config
        if cfg.mode == "numerical" and cfg.rule != "trapezoidal":
            return replace(cfg, rule="trapezoidal")
        return cfg

    def _windowed_total_ci(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Σ_{d predictable} Λ_d over ``[origin, t]``.  ``(B,)`` → ``(B,)``; ``(B, Q)`` → ``(B, Q)``.

        The survival total feeding ``survival_time`` / ``cdf_time`` / ``density_joint``:
        context (non-predictable) type-columns are zeroed before the sum, mirroring the
        likelihood's ``_likelihood_scaled`` (context types do not compete in Σ_d Λ_d).
        The default all-True ``predictable_mask`` zeroes nothing.

        The window starts at the forecast origin (``T_obs``, at least the last event) and
        uses :meth:`_predictive_ci_config`.  The ``(B,)`` path is one validated
        ``cumulative_intensity`` call; the ``(B, Q)`` path loops it over the query axis.
        """
        origin = self._forecast_origin(ts)  # (B,)
        cfg = self._predictive_ci_config()
        if t.dim() == 1:
            ci = self.cumulative_intensity(t, ts, T_low=origin, ci_config=cfg)  # (B, D)
            return self._predictable_sum(ci)  # (B,)
        Q = t.shape[1]
        cols = [
            self._predictable_sum(self.cumulative_intensity(t[:, q], ts, T_low=origin, ci_config=cfg))
            for q in range(Q)
        ]
        return torch.stack(cols, dim=1)  # (B, Q)

    def density_joint(self, t: torch.Tensor, ts: BatchedMVEventData, *, log: bool = True) -> torch.Tensor:
        """p(t, m | H) = λ_m(t)·S(t) — the intensity family's rooted joint density.

        ``(B,[Q,]D)``.  ``log=False`` returns the raw λ·exp(−ΣΛ) (as ``PDF``);
        ``log=True`` floors λ at ``finfo.tiny`` inside the log (matching the likelihood /
        eval log-accumulation), so ``exp(log)`` and the linear form agree except at a
        structurally-zero intensity.
        """
        t, at_inf = self._finite_queries(t, ts)
        t = self._prep_query_time(t, ts, atol=_SNAP_ATOL_DENSITY)  # density snap tolerance
        lam = self.intensity(t, ts)  # (B,[Q,]D)
        total_ci = self._windowed_total_ci(t, ts)  # (B,[Q])
        if log:
            tiny = torch.finfo(lam.dtype).tiny
            out = torch.log(lam.clamp(min=tiny)) - total_ci.unsqueeze(-1)
            return out.masked_fill(at_inf.unsqueeze(-1), float("-inf")) if at_inf is not None else out
        out = lam * torch.exp(-total_ci).unsqueeze(-1)
        return out.masked_fill(at_inf.unsqueeze(-1), 0.0) if at_inf is not None else out

    def survival_time(self, t: torch.Tensor, ts: BatchedMVEventData, *, log: bool = True) -> torch.Tensor:
        """S(t) = exp(−Σ_d Λ_d(t_last → t)).  ``(B,[Q])``.  ``log=False`` == ``1 − CDF``."""
        t, at_inf = self._finite_queries(t, ts)
        t = self._prep_query_time(t, ts, atol=_SNAP_ATOL_CDF)  # CDF snap tolerance
        total_ci = self._windowed_total_ci(t, ts)  # (B,[Q])
        if at_inf is not None:
            # S(∞) = 1 for a row with no possible next event, else 0.
            done = self._no_next_event(ts)
            s_inf = (done.unsqueeze(1) if total_ci.dim() == 2 else done).expand_as(total_ci)
            total_ci = torch.where(at_inf, torch.where(s_inf, 0.0, float("inf")).to(total_ci.dtype), total_ci)
        return -total_ci if log else torch.exp(-total_ci)


    def cdf_time(self, t: torch.Tensor, ts: BatchedMVEventData, *, log: bool = True) -> torch.Tensor:
        """P(t_next < t | H) = 1 − S(t).  ``(B,[Q])``.  ``log=False`` equals ``CDF``."""
        cdf = 1.0 - self.survival_time(t, ts, log=False)
        return torch.log(cdf.clamp(min=torch.finfo(cdf.dtype).tiny)) if log else cdf

    def prob_mark(self, ts: BatchedMVEventData, *, log: bool = True) -> torch.Tensor:
        """p(m | H) — the time-integrated mark marginal (grid).  ``(B, D)``."""
        # Native shell: the grid runs in scaled units; the mark distribution is
        # dimensionless (a normalised probability), so there is no output conversion.
        p = self._marginal_class_distribution(self._scaled_ts(ts))  # (B, D), linear
        return torch.log(p.clamp(min=torch.finfo(p.dtype).tiny)) if log else p

    @torch.no_grad()  # point estimate from grid inversion / bisection: no gradient
    def time_mean(self, ts: BatchedMVEventData) -> torch.Tensor:
        """E[t_next | H] — the grid survival-integral mean.  ``(B,)``."""
        # Native shell: scaled-grid mean × s (a time).
        return self._mean_next_event_time(self._scaled_ts(ts)) * self.time_scale

    @torch.no_grad()  # point estimate from grid inversion / bisection: no gradient
    def time_quantile(self, u, ts: BatchedMVEventData) -> torch.Tensor:
        """Inverse time CDF (median = ``u=0.5``).  ``(B,)``.  ``u`` is a scalar or ``(B,)``."""
        if not torch.is_tensor(u):
            u = torch.full((ts.time_points.shape[0],), float(u), device=ts.time_points.device, dtype=torch.float64)
        # Native shell: invert on the scaled batch (u is dimensionless), then × s.
        return self._inverse_CDF(u, self._scaled_ts(ts)) * self.time_scale

    def _build_integration_grid(
        self,
        ts: BatchedMVEventData,
        T_max: Optional[torch.Tensor] = None,
        num_points: Optional[int] = None,
        points_per_unit_time: Optional[float] = None,
        integrate_marginal_via_grid: bool = True,
        ignore_indices: Optional[Sequence[int]] = None,
    ) -> IntensityGrid:
        """
        Build an IntensityGrid from the forecast origin to T_max for marginal computations.

        This is the shared grid construction used by _marginal_class_distribution
        and _mean_next_event_time. It grows the grid incrementally until
        CDF >= 0.999 or T_max is reached.

        When T_max is provided, the grid is built with a fixed number of points
        (num_points) over [t_last, T_max]. When T_max is None, the grid grows
        incrementally using _build_intensity_grid_until with the specified
        points_per_unit_time density.

        Args:
            ts: Historical events (batched, padded). Shape: (batch_size, seq_len)
            T_max: Upper integration limit. Shape: (batch_size,) or None.
                   If None, automatically determined by growing until CDF >= 0.999.
            num_points: Number of grid points (only used when T_max is provided).
            points_per_unit_time: Grid density (only used when T_max is None).
            integrate_marginal_via_grid: When True, always derive cumulative
                intensity via trapezoidal prefix sums over the intensity grid,
                even for analytical-CI models.  This avoids calling the
                analytical CI function at every grid point (≈ 2× speedup for
                Hawkes).  Set False for exact analytical CI on the grid.
            ignore_indices: Types removed from the process (in addition to the context
                types); the grid grows until the retained process reaches the target CDF.

        Returns:
            IntensityGrid over [origin, T_max] with intensities and CI prefix sums;
            removed types have zero intensity and CI.
        """
        origin = self._forecast_origin(ts)  # (B,)
        type_mask = self._retained_types(ignore_indices)

        if T_max is not None:
            return self._build_intensity_grid(
                origin,
                T_max,
                ts,
                num_points=num_points,
                integrate_marginal_via_grid=integrate_marginal_via_grid,
                type_mask=type_mask,
            )

        return self._build_intensity_grid_until(
            t_start=origin,
            ts=ts,
            points_per_unit_time=points_per_unit_time,
            T_max=None,
            integrate_marginal_via_grid=integrate_marginal_via_grid,
            type_mask=type_mask,
        )

    def _zero_context_cols(self, x: torch.Tensor) -> torch.Tensor:
        """Zero the trailing type-D context (non-predictable) columns of ``x``.

        The single role-masking primitive: context/``<bos>`` types contribute neither
        to the survival Σ_d Λ_d nor to the grid mark marginal.  Uses ``masked_fill``
        (not a multiply) so a spurious inf/nan in a context column is zeroed outright
        rather than turned into a nan; the ``(D,)`` role mask broadcasts over any
        leading dims.  A no-op under the default all-True mask.
        """
        return x.masked_fill(~self.predictable_mask, 0.0)

    def _predictable_sum(self, x: torch.Tensor) -> torch.Tensor:
        """Sum over the trailing type dimension with context columns zeroed.

        The shared survival/inversion total behind ``_likelihood_scaled``,
        ``_windowed_total_ci``, the Newton ``ci_func`` / ``intensity_func`` closures,
        and the ``_inverse_CDF`` offset.  Equals
        ``x.masked_fill(~predictable_mask, 0).sum(dim=1)``.
        """
        return self._zero_context_cols(x).sum(dim=-1)

    @staticmethod
    def _grid_trapezoid(values: torch.Tensor, grid: IntensityGrid) -> torch.Tensor:
        """Trapezoid over the grid axis (dim 1), each row only up to ``grid.row_end``.

        Points past a row's ``row_end`` exist only because other rows needed a longer grid;
        repeating the row's last valid point gives them zero width.
        """
        n = grid.t_grid.shape[1]
        t = grid.t_grid if values.dim() == 2 else grid.t_grid.unsqueeze(-1).expand_as(values)
        if grid.row_end is None or bool((grid.row_end >= n).all()):
            return torch.trapezoid(values, t, dim=1)
        shape = (-1, n) + (1,) * (values.dim() - 2)
        valid = (torch.arange(n, device=values.device).unsqueeze(0) < grid.row_end.unsqueeze(1)).view(shape)
        last = (grid.row_end - 1).view((-1, 1) + (1,) * (values.dim() - 2)).expand((-1, 1) + values.shape[2:])
        t_cut = torch.where(valid, t, t.gather(1, last))
        v_cut = torch.where(valid, values, values.gather(1, last))
        return torch.trapezoid(v_cut, t_cut, dim=1)

    @staticmethod
    def _marginal_class_distribution_from_grid(grid: IntensityGrid) -> torch.Tensor:
        """
        Compute marginal class distribution from a pre-built IntensityGrid.

            p(m | H_t) = ∫ λ_m(s) S(s) ds / ∑_d ∫ λ_d(s) S(s) ds

        Args:
            grid: Pre-built IntensityGrid over [t_last, T_max].

        Returns:
            Marginal class probabilities. Shape: (B, D)
        """
        total_ci = grid.ci_prefix.sum(dim=2, keepdim=True)  # (B, num_points, 1)
        survival = torch.exp(-total_ci)  # (B, num_points, 1)

        integrands = grid.intensities * survival  # (B, num_points, D)
        marginal_probs = IntensityBasedTPP._grid_trapezoid(integrands, grid)  # (B, D)

        total_prob = marginal_probs.sum(dim=1, keepdim=True).clamp(min=_DIV_FLOOR)
        return marginal_probs / total_prob

    @staticmethod
    def _mean_next_event_time_from_grid(grid: IntensityGrid, target_cdf: Optional[float] = None) -> torch.Tensor:
        """
        Compute the mean next event time from a pre-built IntensityGrid.

        Uses the survival function identity:
            E[t_next] = t_N + ∫_{t_N}^{T_max} S(t) dt

        where S(t) = exp(-∑_d ΔΛ_d(t)) is the survival function.

        Args:
            grid: Pre-built IntensityGrid over [t_last, T_max].
            target_cdf: The CDF the grid was grown to.  Rows whose CDF never reached it
                (probability mass at infinity, or a horizon beyond the grid budget) get
                ``inf``: they have no finite mean.

        Returns:
            Mean next event time for each batch element. Shape: (B,)
        """
        total_ci = grid.ci_prefix.sum(dim=2)  # (B, num_points)
        survival = torch.exp(-total_ci)  # (B, num_points)

        # E[t] = t_last + ∫_{t_last}^{T_max} S(t) dt, each row up to its own stopping point
        # (points beyond it were added only for other rows of the batch).
        mean = grid.t_start + IntensityBasedTPP._grid_trapezoid(survival, grid)  # (B,)
        if target_cdf is None:
            return mean
        unresolved = total_ci[:, -1] < -math.log1p(-target_cdf) * (1.0 - 1e-9)
        return torch.where(unresolved, torch.full_like(mean, float("inf")), mean)

    def _marginal_class_distribution(
        self,
        ts: BatchedMVEventData,
        T_max: Optional[torch.Tensor] = None,
        num_points: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute the marginal class distribution p(m | H_t) for the next event.

        This marginalizes over time to get the probability of each event type
        for the next event, regardless of when it occurs:

            p(m | H_t) = ∫_{t_N}^{∞} p(m, s | H_t) ds
                       = ∫_{t_N}^{∞} λ_m(s) exp(-∑_d (Λ_d(s) - Λ_d(t_N))) ds

        Since we cannot integrate to infinity, we integrate up to T_max where
        the survival probability is negligible (CDF ≈ 1).

        Args:
            ts: Historical events (batched, padded). Shape: (batch_size, seq_len)
            T_max: Upper integration limit. Shape: (batch_size,) or None.
                   If None, automatically determined based on when CDF ≈ 0.999.
            num_points: Number of quadrature points for numerical integration.

        Returns:
            Marginal class probabilities for each batch. Shape: (batch_size, D)
            Each row sums to approximately 1 (may be slightly less if T_max is too small).

        Mathematical derivation:
            The joint PDF of the next event at time s with type m is:
                p(s, m | H_t) = λ_m(s) × exp(-∑_d (Λ_d(s) - Λ_d(t_N)))

            Integrating over time gives the marginal over event types:
                p(m | H_t) = ∫_{t_N}^{∞} λ_m(s) × S(s) ds

            where S(s) = exp(-∑_d (Λ_d(s) - Λ_d(t_N))) is the survival function.
        """
        grid = self._build_integration_grid(ts, T_max=T_max, num_points=num_points)
        return self._marginal_class_distribution_from_grid(grid)

    def _mean_next_event_time(
        self,
        ts: BatchedMVEventData,
        T_max: Optional[torch.Tensor] = None,
        num_points: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute the expected next event time E[t_next | H_t].

        Uses the survival function identity:
            E[t_next] = t_N + ∫_{t_N}^{T_max} S(t) dt

        where S(t) = exp(-∑_d (Λ_d(t) - Λ_d(t_N))) is the survival function
        and t_N is the time of the last observed event.

        Args:
            ts: Historical events (batched, padded). Shape: (batch_size, seq_len)
            T_max: Upper integration limit. Shape: (batch_size,) or None.
                   If None, automatically determined based on when CDF ≈ 0.999.
            num_points: Number of quadrature points for numerical integration.

        Returns:
            Expected next event time for each batch element. Shape: (batch_size,)
        """
        grid = self._build_integration_grid(ts, T_max=T_max, num_points=num_points)
        target_cdf = None if T_max is not None else self.ci_config.grid_target_cdf
        return self._mean_next_event_time_from_grid(grid, target_cdf=target_cdf)

    @staticmethod
    def _inverse_cdf_from_grid(grid: IntensityGrid, u: torch.Tensor) -> torch.Tensor:
        """
        Invert the CDF using a pre-built IntensityGrid to find t such that CDF(t) = u.

        The CDF is:
            CDF(t) = 1 - exp(-∑_d ΔΛ_d(t))

        so we need ∑_d ΔΛ_d(t) = -log(1 - u). We find this by searching the
        grid's total CI prefix sums and linearly interpolating.

        Args:
            grid: Pre-built IntensityGrid over [t_last, T_max].
            u: Target CDF values ∈ (0, 1). Shape: (B,)

        Returns:
            Time values t such that CDF(t) ≈ u. Shape: (B,).  ``inf`` where the grid's
            CDF never reaches ``u`` (probability mass at infinity).
        """
        # Target total CI: CDF(t) = 1 - exp(-total_ci(t)) = u
        # => total_ci(t) = -log(1 - u)
        target_ci = -torch.log1p(-u)  # (B,)

        # Total CI at each grid point: sum over dimensions
        total_ci_grid = grid.ci_prefix.sum(dim=2)  # (B, num_points)

        # For each batch element, find the grid segment where total CI crosses target
        # searchsorted finds the index k such that total_ci_grid[:, k-1] < target <= total_ci_grid[:, k]
        k = torch.searchsorted(total_ci_grid, target_ci.unsqueeze(1), right=False).squeeze(1)  # (B,)
        k = k.clamp(min=1, max=grid.num_points - 1)  # (B,)

        # Gather CI and time values at the bracketing grid points
        ci_lo = torch.gather(total_ci_grid, 1, (k - 1).unsqueeze(1)).squeeze(1)  # (B,)
        ci_hi = torch.gather(total_ci_grid, 1, k.unsqueeze(1)).squeeze(1)  # (B,)
        t_lo = torch.gather(grid.t_grid, 1, (k - 1).unsqueeze(1)).squeeze(1)  # (B,)
        t_hi = torch.gather(grid.t_grid, 1, k.unsqueeze(1)).squeeze(1)  # (B,)

        # Linear interpolation within the segment
        dci = (ci_hi - ci_lo).clamp(min=_DIV_FLOOR)
        frac = (target_ci - ci_lo) / dci  # (B,)
        frac = frac.clamp(0.0, 1.0)

        t_star = t_lo + frac * (t_hi - t_lo)  # (B,)
        unreached = target_ci > total_ci_grid[:, -1]
        return torch.where(unreached, torch.full_like(t_star, float("inf")), t_star)

    # =========================================================================
    # Sampling
    # =========================================================================

    @torch.no_grad()
    def sample(
        self,
        ts: Optional[MVEventData] = None,
        num_steps: int = 1,
        rng: Optional[torch.Generator | int] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ) -> Sample:
        """
        Generate new events by sampling from the temporal point process.

        The draw scheme is set by :class:`SamplingConfig` (resolution order call-arg
        > ``use_sampling`` context > instance default).  The default ``inversion``
        mode uses inverse-CDF sampling; ``thinning`` uses Ogata's thinning.

        For terminating processes, sampling stops when all event types have occurred.

        Args:
            ts: Initial event history (single sequence, not batched)
            num_steps: Number of events to generate (for terminating: min(num_steps, remaining types))
            rng: Random number generator or seed for reproducibility
            sampling_config: Optional per-call SamplingConfig override.

        Returns:
            :class:`Sample` ``(sequence, times, mark_probs)``: the history with the sampled
            events appended, their absolute times (native units) and the mark distribution
            at each sampled time. Fewer than ``num_steps`` events are returned when a
            terminating process has used up its types or the next event has no finite time.
        """
        # Resolve the config (call-arg > active context > instance default) and set it
        # active for the duration so the inversion helpers (which read
        # self.sampling_config) honour a per-call override.  Dispatch straight to the
        # mode implementation — never re-call self.sample, which would recurse through
        # subclass overrides (e.g. PoissonProcess.sample delegating thinning to super()).
        # Native shell: draw in scaled time (the inner loop uses the scaled cores), then
        # de-scale the returned event times back to native units (× s).
        ts = self._resolve_sample_history(ts)
        cfg = sampling_config if sampling_config is not None else self.sampling_config
        ts_internal = self._to_internal_time(ts)
        with self.use_sampling(cfg):
            if cfg.mode == "thinning":
                out_ts, times, dists = self._sample_thinning(ts_internal, num_steps=num_steps, rng=rng, cfg=cfg)
            else:
                out_ts, times, dists = self._sample_inversion(ts_internal, num_steps=num_steps, rng=rng, cfg=cfg)
        out_ts, times = self._to_native_sample(out_ts, times)
        return Sample(out_ts, times, dists)

    def _sample_inversion(
        self,
        ts: MVEventData,
        num_steps: int,
        rng: Optional[torch.Generator | int],
        cfg: SamplingConfig,
    ) -> Tuple[MVEventData, List[float], List[torch.Tensor]]:
        """Inverse-CDF sampling loop (the ``inversion`` mode of :meth:`sample`)."""
        # rng=None falls through to the device's global RNG (generator=None in
        # torch.rand/randint/multinomial) so repeated calls draw fresh samples.
        # Never substitute a fresh torch.Generator() here: it is default-seeded,
        # which would make every call replay the identical trajectory.
        if isinstance(rng, int):
            rng = torch.Generator(device=ts.time_points.device).manual_seed(rng)

        device = ts.time_points.device
        T_obs = getattr(ts, "T_obs", None)
        batched = BatchedMVEventData([ts.time_points], [ts.event_types], T_obs=[T_obs])

        time_samples = []
        dist_samples = []

        for step in range(num_steps):
            # For terminating processes, check if all types have occurred
            if self.terminating:
                mask = self._get_termination_mask(batched)  # (1, D)
                remaining_types = mask[0].sum().item()
                if remaining_types == 0:
                    # All types have occurred, stop sampling
                    break

            u = torch.rand(size=(), generator=rng, device=device)

            # CI is tracked in absolute time to simplify the likelihood integral.
            # We search for delta such that:
            # CDF(delta) = 1 - exp(- (CI(t_last + delta) - CI(t_last)) ) = u
            # CI(t_last + delta) - CI(t_last) = -log(1-u)
            # Invert CDF to get new event time. The offset with t_last is handled in the inversion implementation.
            t_star = self._inverse_CDF(u.unsqueeze(0), batched)  # (1,)
            if not bool(torch.isfinite(t_star).all()):
                break  # the draw falls in the mass at infinity: no further event occurs

            # Ensure the sampled time is after the forecast origin (important for terminating
            # processes where _inverse_CDF may not properly account for reduced marginal intensity).
            # The jitter (``cfg.inversion_jitter``) is a fallback; thinning handles the
            # reduced terminating-mode marginal exactly.
            origin = self._forecast_origin(batched)
            if t_star.item() < origin.item():
                t_star = origin + torch.rand(size=(), generator=rng, device=device) * cfg.inversion_jitter

            # Only predictable types can be drawn: context types are read-only.
            lam_vec = self._zero_context_cols(self._intensity_scaled(t_star, batched)[0])  # (D,) scaled
            type_idx, dist = self._draw_thinning_type(lam_vec, batched, rng, device, cfg)
            if type_idx is None:
                break

            time_samples.append(t_star)
            dist_samples.append(dist)

            new_time_points = torch.cat([ts.time_points, t_star], dim=0)
            new_event_types = torch.cat([ts.event_types, type_idx], dim=0)
            ts = MVEventData(new_time_points, new_event_types, T_obs=extend_T_obs(T_obs, new_time_points))
            batched = BatchedMVEventData([ts.time_points], [ts.event_types], T_obs=[getattr(ts, "T_obs", None)])

        return ts, time_samples, dist_samples

    # =========================================================================
    # Ogata thinning sampler
    # =========================================================================
    #
    # Draws each next event by proposing candidate times from a dominating
    # homogeneous process of rate λ̄ ≥ Σ_d λ_d over a forward window and accepting
    # a candidate with probability λ_sum/λ̄; a rejected candidate advances the
    # search (standard Ogata).  Like ``sample`` this operates on a single sequence
    # (batch-of-1).  The sup-scan window / point count are ``SamplingConfig`` fields
    # (``thinning_sup_points`` / ``thinning_window_events`` / ``thinning_global_window_events``).

    def _sample_thinning(
        self,
        ts: MVEventData,
        num_steps: int,
        rng: Optional[torch.Generator | int],
        cfg: SamplingConfig,
    ) -> Tuple[MVEventData, List[torch.Tensor], List[torch.Tensor]]:
        """Draw ``num_steps`` events by Ogata thinning.

        rng handling mirrors the inversion path: an ``int`` seeds a device
        ``Generator`` (deterministic), ``None`` falls through to the global RNG.
        """
        device = ts.time_points.device
        if isinstance(rng, int):
            rng = torch.Generator(device=device).manual_seed(rng)

        T_obs = getattr(ts, "T_obs", None)
        batched = BatchedMVEventData([ts.time_points], [ts.event_types], T_obs=[T_obs])
        time_samples: List[torch.Tensor] = []
        dist_samples: List[torch.Tensor] = []

        for _ in range(num_steps):
            if self.terminating:
                mask = self._get_termination_mask(batched)  # (1, D)
                if mask[0].sum().item() == 0:
                    break  # all types have occurred

            drawn = self._thin_next_event(batched, rng, cfg, device)
            if drawn is None:
                break  # no acceptance within max_attempts (or no remaining type)
            t_star, type_idx, dist = drawn

            time_samples.append(t_star)
            dist_samples.append(dist)
            new_time_points = torch.cat([ts.time_points, t_star], dim=0)
            ts = MVEventData(
                new_time_points,
                torch.cat([ts.event_types, type_idx], dim=0),
                T_obs=extend_T_obs(T_obs, new_time_points),
            )
            batched = BatchedMVEventData([ts.time_points], [ts.event_types], T_obs=[getattr(ts, "T_obs", None)])

        return ts, time_samples, dist_samples

    def _intensity_breakpoints(self, batched: BatchedMVEventData, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """Times in ``(lo, hi]`` where a piecewise-linear baseline or kernel changes slope.  ``(K,)``.

        Collected from the ``baseline`` / ``kernel`` modules' :meth:`breakpoints` (absolute knots
        for a baseline; lag knots placed after every real event for a kernel), each with its
        right limit.  Between these times the intensity of such a model is linear (or constant)
        in ``t``, so a scan over them and the window edges finds its supremum.
        """
        dtype = lo.dtype
        found = []
        baseline = getattr(self, "baseline", None)
        if baseline is not None and baseline.breakpoints() is not None:
            found.append(baseline.breakpoints().to(dtype).flatten())
        kernel = getattr(self, "kernel", None)
        if kernel is not None and kernel.breakpoints() is not None:
            events = batched.time_points[0][batched.event_mask[0]].to(dtype)  # (N,) real events
            found.append((events.unsqueeze(1) + kernel.breakpoints().to(dtype).unsqueeze(0)).flatten())
        if not found:
            return torch.empty(0, dtype=dtype, device=lo.device)
        pts = torch.cat(found)
        pts = pts[(pts > lo) & (pts <= hi)]
        pts = torch.cat([pts, right_limit(pts, self._time_compare_dtype(batched))])
        return torch.unique(pts[pts <= hi])

    def _thinning_upper_bound(
        self, batched: BatchedMVEventData, t: torch.Tensor, cfg: SamplingConfig
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dominating rate λ̄ and the window ``[t, t+W]`` it bounds.

        Scans the predictable Σ_d λ_d over the window via the multi-query ``intensity``
        primitive (which also applies termination masking) and returns ``max ×
        upper_bound_slack``.  The scan points are the right limit ``t⁺`` (an event at ``t``
        jumps there), a grid clustered near ``t``, the window end, and the knots of
        piecewise-linear baselines and kernels (:meth:`_intensity_breakpoints`): for the
        spline and piecewise models the scan therefore contains the exact supremum.  For
        other models it is a grid estimate; a violation detected at a candidate raises the
        bound (see :meth:`_thin_next_event`).  ``W`` scales with the local rate so a handful
        of expected events span it (``global_sup`` uses a wider window).
        """
        t_rl = right_limit(t, self._time_compare_dtype(batched))
        lam_here = self._predictable_sum(self._intensity_scaled(t_rl, batched)).clamp(min=cfg.degenerate_floor)  # (1,)
        events = (
            cfg.thinning_global_window_events if cfg.upper_bound == "global_sup" else cfg.thinning_window_events
        )
        W = events / lam_here  # (1,)
        # Square the spacing so points concentrate near t (the sup of a decaying
        # post-event intensity sits just after t) while still covering the window.
        frac = torch.linspace(0.0, 1.0, cfg.thinning_sup_points, device=t.device, dtype=t.dtype) ** 2  # (P,)
        grid = t_rl + W * frac  # (P,)
        query = torch.cat([grid, self._intensity_breakpoints(batched, t_rl, t + W)])
        sup = torch.zeros((), dtype=lam_here.dtype, device=t.device)
        for chunk in query.split(_THINNING_SCAN_CHUNK):
            sup = torch.maximum(sup, self._predictable_sum(self._intensity_scaled(chunk.unsqueeze(0), batched)).max())
        return sup * cfg.upper_bound_slack, t + W

    def _thin_next_event(
        self,
        batched: BatchedMVEventData,
        rng: Optional[torch.Generator],
        cfg: SamplingConfig,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Propose/accept one event after the forecast origin.

        Returns ``(t_star (1,), type_idx (1,), dist (D,))``, or ``None`` if no type remains or
        no candidate is accepted within ``max_attempts`` proposals (with a warning).  Moving to
        the next window does not count as an attempt.

        A candidate whose intensity exceeds the bound ``λ̄`` raises it to ``λ̄'``.  The stretch
        ``[t, t_cand)`` already carries a rate-``λ̄`` proposal stream with no point in it; adding
        an independent rate-``(λ̄' − λ̄)`` stream there makes it rate ``λ̄'``, and its points plus
        ``t_cand`` are then thinned in time order with probability ``λ/λ̄'``, which keeps the
        draw exact.
        """
        t = self._forecast_origin(batched)  # (1,) current search position
        lam_bar, window_end = self._thinning_upper_bound(batched, t, cfg)
        attempts, windows = 0, 0

        def next_window():
            nonlocal t, lam_bar, window_end, windows
            t = window_end
            lam_bar, window_end = self._thinning_upper_bound(batched, t, cfg)
            windows += 1

        while attempts < cfg.max_attempts and windows < _THINNING_MAX_WINDOWS:
            if lam_bar.item() <= cfg.degenerate_floor:
                next_window()  # intensity ~0 across the window
                continue
            u1 = torch.rand(size=(), generator=rng, device=device)
            t_cand = t - torch.log(u1) / lam_bar  # Δ ~ Exp(λ̄); (1,)
            if (t_cand > window_end).item():
                next_window()  # left the window where λ̄ dominates
                continue

            pending = [t_cand]  # candidates on [t, t_cand], thinned in time order
            while pending:
                attempts += 1
                s_cand = pending.pop(0)
                lam_vec = self._zero_context_cols(self._intensity_scaled(s_cand, batched)[0])  # (D,) predictable
                lam_sum = lam_vec.sum()
                if (lam_sum > lam_bar).item():
                    new_bar = torch.maximum(2.0 * lam_bar, lam_sum * cfg.upper_bound_slack)
                    extra = self._poisson_points(t, s_cand, new_bar - lam_bar, rng, device)
                    pending = sorted(extra + [s_cand], key=float) + pending
                    lam_bar = new_bar
                    continue
                u2 = torch.rand(size=(), generator=rng, device=device)
                if (u2 * lam_bar <= lam_sum).item():  # accept w.p. λ_sum/λ̄
                    type_idx, dist = self._draw_thinning_type(lam_vec, batched, rng, device, cfg)
                    if type_idx is None:
                        return None
                    return s_cand, type_idx, dist
                t = s_cand  # reject: the stream continues after this point
        warnings.warn(
            f"Thinning stopped without an event after {attempts} proposals and {windows} windows "
            "(SamplingConfig.max_attempts); the sampled sequence ends early.",
            stacklevel=4,
        )
        return None

    @staticmethod
    def _poisson_points(
        lo: torch.Tensor, hi: torch.Tensor, rate: torch.Tensor, rng: Optional[torch.Generator], device: torch.device
    ) -> List[torch.Tensor]:
        """Points of a homogeneous Poisson process of ``rate`` on ``[lo, hi)``, as ``(1,)`` tensors."""
        mean = (rate * (hi - lo)).clamp(min=0.0).reshape(())
        n = int(torch.poisson(mean.to(torch.float64), generator=rng).item())
        if n == 0:
            return []
        u = torch.rand(n, generator=rng, device=device, dtype=lo.dtype)
        return [lo + (hi - lo) * x for x in u.sort().values]

    def _draw_thinning_type(
        self,
        lam_vec: torch.Tensor,
        batched: BatchedMVEventData,
        rng: Optional[torch.Generator],
        device: torch.device,
        cfg: SamplingConfig,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Draw the accepted event's type from λ/λ_sum.

        Shared by both samplers.  ``lam_vec`` has the context types zeroed.  When
        λ_sum ≤ ``degenerate_floor`` (or NaN) the type is drawn uniformly over the
        predictable types — the remaining ones for a terminating process.  Returns
        ``(None, None)`` when no type can be drawn.
        """
        lam_sum = float(lam_vec.sum().item())
        if lam_sum <= cfg.degenerate_floor or lam_sum != lam_sum:  # degenerate or NaN
            allowed = self.predictable_mask.clone()
            if self.terminating:
                allowed = allowed & self._get_termination_mask(batched)[0]  # (D,)
            remaining = torch.where(allowed)[0]
            if remaining.numel() == 0:
                return None, None
            pick = torch.randint(high=remaining.numel(), size=(1,), generator=rng, device=device)
            type_idx = remaining[pick]
            dist = allowed.to(lam_vec.dtype) / allowed.sum()
        else:
            dist = lam_vec / lam_sum
            type_idx = torch.multinomial(dist, num_samples=1, generator=rng)
        return type_idx, dist


class DensityBasedTPP(AutoregressiveTPP):
    """Root for inter-event-time density models — the native density realization.

    Realizes the contract directly from a **mark-conditional** inter-event-time density
    ``p(t | m, H)`` and a history-conditioned mark head ``p(m | H)``, with **no derived
    hazard on the contract path**.  This is the density family's natural factorisation —
    the dual of the intensity family's ``p(t)·p(m|t)`` — so its marginals fall out by a
    finite **sum over marks**, no time integration::

        p(t, m | H) = p(m | H) · p(t | m, H)                    (joint)
        p(t | H)    = Σ_m p(m | H) · p(t | m, H)                (time marginal — a sum, not an integral)
        p(m | t, H) = p(m | H) · p(t | m, H) / p(t | H)         (mark | time)
        S(t | H)    = Σ_m p(m | H) · S(t | m, H)

    A concrete model roots the family by implementing the per-mark density hooks
    (``_log_prob_mark`` / ``_log_density_time_given_mark`` / ``_log_survival_time_given_mark`` /
    ``_event_log_density_sum`` / ``_draw_next`` / ``time_mean``); everything else is derived
    here.  The independent-mark model (mark ⟂ time given history, IFTPP Appendix F.2) is the
    special case ``p(t | m, H) = f(τ)`` for every ``m``: :class:`IntensityFreeProcess` roots
    the per-mark hooks with that broadcast form (plus its terminating reshaping), and then
    ``p(m | t) = p(m)``, ``p(t | m) = p(t)`` fall out of the identities above.  Unlike
    :class:`IntensityBasedTPP` there is no intensity / cumulative intensity — a subclass may
    expose a diagnostic hazard off the contract path, but nothing here depends on it.
    """

    def __init__(
        self,
        D: int,
        *,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        # The SamplingConfig override + resolution plumbing live on the root
        # ``TemporalPointProcess`` (sampling is universal); forward the override up.
        super().__init__(
            D=D,
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            sampling_config=sampling_config,
        )

    # =========================================================================
    # Density hooks (concrete model roots the family)
    # =========================================================================

    @abstractmethod
    def _log_density_time_given_mark(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """log p(t | m, H), τ = t − t_last per query, one density per mark.  ``t`` is ``(B,)``
        or ``(B, Q)``; returns ``(B,[Q,]D)``.  (Independent-mark models return the same density
        broadcast over the ``D`` axis.)"""
        raise NotImplementedError

    @abstractmethod
    def _log_survival_time_given_mark(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """log S(t | m, H) = log P(t_next ≥ t | m, H), one survival per mark.  ``(B,[Q,]D)``."""
        raise NotImplementedError

    @abstractmethod
    def _log_prob_mark(self, ts: BatchedMVEventData) -> torch.Tensor:
        """log p(m | H) for the next event.  ``(B, D)``."""
        raise NotImplementedError

    @abstractmethod
    def _event_log_density_sum(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Σ_i [log p(m_i | H_i) + log p(t_i | m_i, H_i)] over valid events.  ``(B,)``.

        Padding / empty-sequence handling and the terminating repeated-type
        impossibility (log p = −inf) are the concrete model's responsibility, mirroring
        :meth:`IntensityBasedTPP._event_log_intensity_sum`.
        """
        raise NotImplementedError

    @abstractmethod
    def _draw_next(self, ts_current: MVEventData, rng):
        """Draw the next event from the current history.

        Returns ``(t_next, type_idx, dist)`` — ``t_next`` a ``(1,)`` absolute-time
        tensor, ``type_idx`` a ``(1,)`` long tensor, ``dist`` the ``(D,)`` mark
        distribution — or ``None`` when no draw is possible (terminating, no type left).
        """
        raise NotImplementedError

    @abstractmethod
    def time_mean(self, ts: BatchedMVEventData) -> torch.Tensor:
        """E[t_next | H].  ``(B,)``."""
        raise NotImplementedError

    # =========================================================================
    # Realized contract
    # =========================================================================

    def _prep_query_time(self, t: torch.Tensor, ts: BatchedMVEventData, atol: float) -> torch.Tensor:
        """Cast to double, snap near-origin queries onto the forecast origin, check ``t ≥ origin``.

        Mirrors :meth:`IntensityBasedTPP._prep_query_time`; the origin is ``T_obs`` (at
        least the padding-aware last event).  Handles ``(B,)`` and ``(B, Q)``.
        """
        t = t.double()
        mx = self._forecast_origin(ts).double()  # (B,)
        mx_b = mx.unsqueeze(1) if t.dim() == 2 else mx
        close = torch.abs(t - mx_b) < atol
        t = torch.where(close, mx_b.expand_as(t), t)
        if not bool((t >= mx_b).all()):
            raise ValueError("Predictive densities are only defined for t >= T_obs (the last event by default).")
        return t

    def _origin_log_survival(self, ts: BatchedMVEventData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Log survival to the forecast origin: per mark ``(B, D)`` and marginal ``(B,)``.

        Zero where the origin is the last event; the predictive contract divides by these
        to condition on no event in ``(t_last, T_obs]``.
        """
        origin = self._forecast_origin(ts).double()
        B = origin.shape[0]
        zeros_m = torch.zeros(B, self.D, dtype=torch.float64, device=origin.device)
        zeros = torch.zeros(B, dtype=torch.float64, device=origin.device)
        censored = origin > self._last_event_time(ts).double()  # (B,)
        if not bool(censored.any()):
            return zeros_m, zeros
        lsgm = self._log_survival_time_given_mark(origin, ts).double()  # (B, D)
        ls = self._log_survival_marginal(origin, ts).double()  # (B,)
        return torch.where(censored.unsqueeze(1), lsgm, zeros_m), torch.where(censored, ls, zeros)

    def _conditional_log_survival(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """log S(t | H, no event before the origin); ``t`` already prepped.  ``(B,[Q])``."""
        ls = self._log_survival_marginal(t, ts)
        _, ls0 = self._origin_log_survival(ts)
        return ls - (ls0.unsqueeze(1) if ls.dim() == 2 else ls0).to(ls.dtype)

    def _log_density_joint(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """log p(t, m | H) = log p(m | H) + log p(t | m, H); ``t`` already prepped.  ``(B,[Q,]D)``."""
        lptm = self._log_density_time_given_mark(t, ts)  # (B,[Q,]D)
        lm = self._log_prob_mark(ts)  # (B, D)
        return lptm + (lm.unsqueeze(1) if lptm.dim() == 3 else lm)

    def _log_survival_marginal(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """log S(t | H) = log Σ_m p(m | H)·S(t | m, H).  ``(B,[Q])``.  (``t`` need not be prepped.)

        A degenerate mark law (``Σ_m p(m | H) = 0`` — e.g. a fully terminated history) means no
        next event can occur, so the survival is 1 for all t (``log S = 0``) rather than the
        empty-sum ``−inf``.
        """
        lsgm = self._log_survival_time_given_mark(t, ts)  # (B,[Q,]D)
        lm = self._log_prob_mark(ts)  # (B, D)
        ls = torch.logsumexp(lsgm + (lm.unsqueeze(1) if lsgm.dim() == 3 else lm), dim=-1)  # (B,[Q])
        no_next = torch.isneginf(torch.logsumexp(lm, dim=-1))  # (B,) — no mark can still occur
        return torch.where(no_next.unsqueeze(1) if ls.dim() == 2 else no_next, torch.zeros_like(ls), ls)

    # Time-unit shell: the density core works in scaled units.  Query
    # and history are normalised to ``t/s`` on entry; a per-unit-time density is
    # de-normalised by ``− log s`` (log space), while marks / survival / CDF are
    # dimensionless (no conversion).  With ``s = 1`` these are no-ops.

    def _log_survival_with_limit(self, ls: torch.Tensor, at_inf, ts: BatchedMVEventData) -> torch.Tensor:
        """Set ``log S(∞)``: 0 for a row with no possible next event, ``-inf`` otherwise."""
        if at_inf is None:
            return ls
        done = self._no_next_event(ts)
        done = (done.unsqueeze(1) if ls.dim() == 2 else done).expand_as(ls)
        return torch.where(at_inf, torch.where(done, 0.0, float("-inf")).to(ls.dtype), ls)

    def density_joint(self, t: torch.Tensor, ts: BatchedMVEventData, *, log: bool = True) -> torch.Tensor:
        """p(t, m | H) = p(m | H) · p(t | m, H).  ``(B,[Q,]D)``."""
        s = self.time_scale
        t, ts = self._to_internal(t, ts), self._scaled_ts(ts)
        t = self._prep_query_time(t, ts, atol=_SNAP_ATOL_DENSITY)
        lj = self._log_density_joint(t, ts)
        _, ls0 = self._origin_log_survival(ts)
        lj = lj - (ls0.view(-1, *([1] * (lj.dim() - 1)))).to(lj.dtype) - torch.log(s)  # condition; density → ÷ s
        return lj if log else lj.exp()

    def density_time(self, t: torch.Tensor, ts: BatchedMVEventData, *, log: bool = True) -> torch.Tensor:
        """p(t | H) = Σ_m p(m | H)·p(t | m, H) — mark-marginal, a finite sum (no integration).  ``(B,[Q])``."""
        s = self.time_scale
        t, ts = self._to_internal(t, ts), self._scaled_ts(ts)
        t = self._prep_query_time(t, ts, atol=_SNAP_ATOL_DENSITY)
        lt = torch.logsumexp(self._log_density_joint(t, ts), dim=-1)
        _, ls0 = self._origin_log_survival(ts)
        lt = lt - (ls0.unsqueeze(1) if lt.dim() == 2 else ls0).to(lt.dtype) - torch.log(s)  # condition; ÷ s
        return lt if log else lt.exp()

    def survival_time(self, t: torch.Tensor, ts: BatchedMVEventData, *, log: bool = True) -> torch.Tensor:
        """S(t) = Σ_m p(m | H)·S(t | m, H).  ``(B,[Q])``."""
        t, ts = self._to_internal(t, ts), self._scaled_ts(ts)
        t, at_inf = self._finite_queries(t, ts)
        t = self._prep_query_time(t, ts, atol=_SNAP_ATOL_CDF)
        ls = self._log_survival_with_limit(self._conditional_log_survival(t, ts), at_inf, ts)  # dimensionless
        return ls if log else ls.exp()

    def cdf_time(self, t: torch.Tensor, ts: BatchedMVEventData, *, log: bool = True) -> torch.Tensor:
        """P(t_next < t | H) = 1 − S(t).  ``(B,[Q])``."""
        t, ts = self._to_internal(t, ts), self._scaled_ts(ts)
        t, at_inf = self._finite_queries(t, ts)
        t = self._prep_query_time(t, ts, atol=_SNAP_ATOL_CDF)
        ls = self._log_survival_with_limit(self._conditional_log_survival(t, ts), at_inf, ts)  # dimensionless
        if not log:
            return -torch.expm1(ls)  # 1 − exp(log S), stable
        return _log1mexp(ls)

    def prob_mark(self, ts: BatchedMVEventData, *, log: bool = True) -> torch.Tensor:
        """p(m | H) (given no event before the forecast origin).  ``(B, D)``."""
        ts = self._scaled_ts(ts)
        lp = self._log_prob_mark(ts)  # dimensionless
        lsgm0, ls0 = self._origin_log_survival(ts)
        lp = lp + (lsgm0 - ls0.unsqueeze(1)).to(lp.dtype)  # p(m)·S_m(origin) / S(origin)
        return lp if log else lp.exp()

    def prob_mark_given_time(self, t: torch.Tensor, ts: BatchedMVEventData, *, log: bool = True) -> torch.Tensor:
        """p(m | t, H) = p(t, m | H) / p(t | H).  ``(B,[Q,]D)``.

        For an independent-mark model this reduces to ``p(m | H)`` (the per-mark densities
        are equal, so they cancel); the general form is kept so a time-dependent mark falls
        out automatically.
        """
        t, ts = self._to_internal(t, ts), self._scaled_ts(ts)
        t = self._prep_query_time(t, ts, atol=_SNAP_ATOL_DENSITY)
        lj = self._log_density_joint(t, ts)  # (B,[Q,]D) — the ÷s cancels in the ratio
        lt = torch.logsumexp(lj, dim=-1, keepdim=True)  # (B,[Q,]1) = log p(t | H)
        # A degenerate joint (no next event, all −inf) gives −inf − −inf = nan; define zero mass.
        lr = torch.where(torch.isneginf(lt).expand_as(lj), lj, lj - lt)
        return lr if log else lr.exp()

    def density_time_given_mark(self, t: torch.Tensor, ts: BatchedMVEventData, *, log: bool = True) -> torch.Tensor:
        """p(t | m, H) — the rooted per-mark inter-event density.  ``(B,[Q,]D)``."""
        s = self.time_scale
        t, ts = self._to_internal(t, ts), self._scaled_ts(ts)
        t = self._prep_query_time(t, ts, atol=_SNAP_ATOL_DENSITY)
        lr = self._log_density_time_given_mark(t, ts)
        lsgm0, _ = self._origin_log_survival(ts)
        lr = lr - (lsgm0.unsqueeze(1) if lr.dim() == 3 else lsgm0).to(lr.dtype) - torch.log(s)  # condition; ÷ s
        return lr if log else lr.exp()

    def _log_prob_events(self, ts_scaled: BatchedMVEventData, T_scaled: torch.Tensor) -> torch.Tensor:
        """Autoregressive per-event conditional: observed-event density sum + censoring survival ``(B,)``."""
        return self._event_log_density_sum(ts_scaled) + self._log_survival_marginal(T_scaled, ts_scaled)

    @torch.no_grad()  # point estimate from grid inversion / bisection: no gradient
    def time_quantile(self, u, ts: BatchedMVEventData) -> torch.Tensor:
        """Inverse time CDF via bisection over ``[origin, hi]`` (median = ``u=0.5``).  ``(B,)``.

        The density-family quantile: grow ``hi`` until the CDF at ``hi`` reaches ``max(u)``
        (capped), then bisect.  ``u`` is a scalar or ``(B,)`` and is capped at
        ``SamplingConfig.quantile_cap`` so ``u → 1`` cannot grow the bracket without bound.
        """
        s = self.time_scale
        return self._quantile_scaled(u, self._scaled_ts(ts)) * s

    def _quantile_scaled(self, u, ts: BatchedMVEventData) -> torch.Tensor:
        """:meth:`time_quantile` on a scaled history, in scaled units."""
        cfg = self.sampling_config
        device = ts.time_points.device
        origin = self._forecast_origin(ts).double()  # (B,) T_obs, at least the last real event
        B = origin.shape[0]
        if not torch.is_tensor(u):
            u = torch.full((B,), float(u), device=device, dtype=torch.float64)
        u = u.to(device=device, dtype=torch.float64).clamp(max=cfg.quantile_cap)

        def cdf(t: torch.Tensor) -> torch.Tensor:
            t = self._prep_query_time(t, ts, atol=_SNAP_ATOL_CDF)
            return -torch.expm1(self._conditional_log_survival(t, ts))

        lo = origin.clone()
        hi = origin + cfg.quantile_init_bracket_width
        for _ in range(cfg.quantile_bracket_grow_iters):  # grow the bracket until it covers the largest requested quantile
            need = cdf(hi) < u
            if not bool(need.any()):
                break
            hi = torch.where(need, origin + (hi - origin) * 2.0, hi)
        # A defective / saturated survival (CDF never reaches u — e.g. a fully terminated
        # history, or mass at infinity) has no finite quantile: report +inf rather than the
        # meaningless grown-bracket endpoint (~1e19).  Callers (eval) drop non-finite predictions.
        unreached = cdf(hi) < u  # (B,)
        for _ in range(cfg.quantile_bisection_iters):  # bisection
            mid = 0.5 * (lo + hi)
            go_right = cdf(mid) < u
            lo = torch.where(go_right, mid, lo)
            hi = torch.where(go_right, hi, mid)
        result = 0.5 * (lo + hi)
        return torch.where(unreached, torch.full_like(result, float("inf")), result)

    # =========================================================================
    # Sampling — direct draw from the density + mark head
    # =========================================================================

    @torch.no_grad()
    def sample(
        self,
        ts: Optional[MVEventData] = None,
        num_steps: int = 1,
        rng: Optional[torch.Generator | int] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ) -> Sample:
        """Generate events by drawing each next event directly from the density + mark head.

        Returns a :class:`Sample` like :meth:`IntensityBasedTPP.sample`. For terminating
        processes, sampling stops once all types have occurred and never repeats a type.
        """
        # Native shell, mirroring :meth:`IntensityBasedTPP.sample`: the per-event density is
        # parameterised in SCALED time (the log-normal mixture is fit on tau / time_scale), so the
        # draw must happen on a scaled history and the returned absolute times must be de-scaled.
        # Omitting this is inert at time_scale == 1 and silently inflates every sampled gap by
        # 1 / time_scale otherwise.
        ts = self._resolve_sample_history(ts)
        cfg = sampling_config if sampling_config is not None else self.sampling_config
        ts_internal = self._to_internal_time(ts)
        with self.use_sampling(cfg):
            out_ts, times, dists = self._sample_density(ts_internal, num_steps=num_steps, rng=rng)
        out_ts, times = self._to_native_sample(out_ts, times)
        return Sample(out_ts, times, dists)

    def _sample_density(
        self,
        ts: MVEventData,
        num_steps: int,
        rng: Optional[torch.Generator | int],
    ) -> Tuple[MVEventData, List[torch.Tensor], List[torch.Tensor]]:
        # rng=None -> global RNG (fresh each call); an int seeds a device Generator.
        if isinstance(rng, int):
            rng = torch.Generator(device=ts.time_points.device).manual_seed(rng)

        T_obs = getattr(ts, "T_obs", None)
        time_samples: List[torch.Tensor] = []
        dist_samples: List[torch.Tensor] = []
        for _ in range(num_steps):
            if self.terminating:
                batched = BatchedMVEventData([ts.time_points], [ts.event_types])
                if self._get_termination_mask(batched)[0].sum().item() == 0:
                    break  # all types have occurred
            batched = BatchedMVEventData([ts.time_points], [ts.event_types], T_obs=[getattr(ts, "T_obs", None)])
            if bool((self._forecast_origin(batched) > self._last_event_time(batched)).any()):
                drawn = self._draw_after_origin(batched, rng)  # observed beyond the last event
            else:
                drawn = self._draw_next(ts, rng)
            if drawn is None:
                break
            t_next, type_idx, dist = drawn
            time_samples.append(t_next)
            dist_samples.append(dist)
            new_time_points = torch.cat([ts.time_points, t_next.reshape(1).to(ts.time_points.dtype)], dim=0)
            ts = MVEventData(
                new_time_points,
                torch.cat([ts.event_types, type_idx.reshape(1)], dim=0),
                T_obs=extend_T_obs(T_obs, new_time_points),
            )
        return ts, time_samples, dist_samples

    def _draw_after_origin(self, batched: BatchedMVEventData, rng):
        """Draw the next event given no event before the forecast origin (scaled units).

        Inverse-CDF on the conditional time law, then the mark from ``p(m | t, H)``.
        Returns ``None`` when the draw falls in the mass at infinity or no mark can occur.
        """
        device = batched.time_points.device
        u = torch.rand(1, generator=rng, device=device, dtype=torch.float64)
        t_next = self._quantile_scaled(u, batched)  # (1,)
        if not bool(torch.isfinite(t_next).all()):
            return None
        t = self._prep_query_time(t_next, batched, atol=_SNAP_ATOL_DENSITY)
        lj = self._log_density_joint(t, batched)[0]  # (D,)
        if bool(torch.isneginf(lj).all()):
            return None
        probs = torch.softmax(lj, dim=-1)
        type_idx = torch.multinomial(probs, num_samples=1, generator=rng)
        return t_next, type_idx, probs


_wrap_entry_points(TemporalPointProcess)
TemporalPointProcess.__init__ = _seeded_constructor(TemporalPointProcess.__init__)
