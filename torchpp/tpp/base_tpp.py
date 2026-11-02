"""
Abstract Base Classes for Temporal Point Processes (TPPs)

This module defines the foundational abstract base classes for implementing
multivariate temporal point processes, extracting common interfaces from
implementations like Hawkes processes, Poisson processes, and spline-based processes.

Key features:
- Flexible intensity computation (standard or log-space)
- Optional numerical integration for cumulative intensity (trapezoidal + MC hybrid)
- CDF inversion via rootfinding (uses existing Hawkes utilities) or direct implementation
- Time-rescaling for goodness-of-fit testing
- Helper methods for common TPP operations
"""

import inspect
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal, Optional, Tuple, List
import torch
import torch.nn as nn

from ..utils.event_data import (
    MVEventData,
    BatchedMVEventData,
    termination_times as _termination_times,
    termination_mask as _termination_mask,
    terminated_query_mask as _terminated_query_mask,
    has_repeated_event_types as _has_repeated_event_types_fn,
)
from ..utils.math import invert_CI, numerical_integration, IntensityGrid


# Map the public quadrature-rule names (CIConfig.rule) onto the numerical_integration
# method names. "mc" is the pure Monte-Carlo estimator (utils.math "monte_carlo").
_RULE_TO_METHOD = {"trapezoidal": "trapezoidal", "mc_trapezoidal": "mc_trapezoidal", "mc": "monte_carlo"}

# ---------------------------------------------------------------------------
# Numerical constants (tolerances / grid + inversion heuristics).  All are
# value-preserving names for previously-inlined literals — see each comment.
# ---------------------------------------------------------------------------

# Query-time snap tolerance: a predictive query within this of t_last is snapped
# onto t_last.  Split by quantity to preserve the legacy PDF vs CDF numbers.
_SNAP_ATOL_DENSITY = 1e-8  # density / PDF / joint queries
_SNAP_ATOL_CDF = 1e-6  # CDF / survival queries

# log(1 - exp(x)) stable-branch switch (Mächler, 2012): log(0.5).
_LOG1MEXP_SWITCH = -0.6931471805599453

# Marginal-/mean-time integration grid defaults.
_GRID_NUM_POINTS = 200  # fixed grid points when T_max is given
_GRID_POINTS_PER_UNIT_TIME = 200.0  # grid density when growing to a target CDF
_GRID_TARGET_CDF = 0.999  # stop grid growth once the CDF reaches this
_GRID_CHUNK_DURATION = 0.1  # scaled-time span of each growth chunk
_GRID_MAX_CHUNKS = 100  # safety cap on the number of growth chunks

# Divisor floor guarding grid interpolation / normalisation against divide-by-zero.
_DIV_FLOOR = 1e-12

# Build the inversion grid slightly past the requested quantile (safety margin).
_GRID_TARGET_CDF_MARGIN = 0.001

# Density-family inverse-CDF bisection (DensityBasedTPP.time_quantile).
_QUANTILE_BRACKET_GROW_ITERS = 64  # bracket-doubling iterations
_QUANTILE_BISECTION_ITERS = 60  # bisection iterations
_QUANTILE_INIT_BRACKET_WIDTH = 1.0  # initial bracket width above t_last

# Legacy post-t_last jitter in inversion sampling (kept for number preservation).
_INVERSION_JITTER = 0.1


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
        mode: ``"closed_form"`` (analytic ``_interval_ci`` / ``_raw_ci``) or
            ``"numerical"`` (quadrature on the Layer-1 intensity hooks).
        placement: numerical only — ``"interval"`` (event-knotted, P points per
            inter-event interval; unbiased at event discontinuities; default) or
            ``"global"`` (M uniform points over the whole window; fixed budget,
            cheap for long sequences, the natural choice for an event-free forecast
            horizon).
        rule: numerical only — ``"trapezoidal"`` (deterministic linspace),
            ``"mc"`` (uniform-random mean×width) or ``"mc_trapezoidal"`` (sorted
            random + trapezoid weights).  **Default ``mc_trapezoidal``.**  WARNING:
            fixed ``trapezoidal`` nodes let a flexible-intensity neural head learn to
            hide intensity spikes between the nodes and evade the survival penalty
            during training (the B1 gaming finding) — use ``mc``/``mc_trapezoidal``
            for those.
        n_points: per interval (``interval``) or total (``global``).
        chunk_size: eval/forecast memory knob, ``no_grad`` only (relevant to AttNHP,
            whose nodes fold through an attention encoder).  ``None`` = off.
    """

    mode: Literal["closed_form", "numerical"] = "numerical"
    placement: Literal["interval", "global"] = "interval"
    rule: Literal["trapezoidal", "mc", "mc_trapezoidal"] = "mc_trapezoidal"
    n_points: int = 100
    chunk_size: Optional[int] = None


@dataclass(frozen=True)
class SamplingConfig:
    """How to draw events from the process.

    Value object mirroring :class:`CIConfig`: the draw scheme and its numeric knobs
    live here rather than as magic constants inside ``sample``/inversion code.
    Resolution order is call-arg > active ``use_sampling`` context > instance default.
    The default (``inversion`` / ``inversion_method="auto"``) reproduces the legacy
    sampler's exact same-seed trajectory.

    Attributes:
        mode: ``"inversion"`` (inverse-CDF sampling; number-preserving) or
            ``"thinning"`` (Ogata; implemented in a later step).
        inversion_method: ``"auto"`` picks per model (``closed_form`` where a
            closed-form inverter is overridden, ``newton`` for closed-form CI,
            ``grid`` for numerical CI — matching the legacy ``inverse_CDF``
            branching); the other values force a path.
        newton_tol / newton_max_iters: bracketed-Newton convergence controls
            (``invert_CI`` defaults; number-preserving).
        bracket_width: initial Newton bracket width above ``t_last``.  A wider
            bracket shifts the Newton iterate within ``newton_tol``.
        grid_density: points/unit-time for the grid inverter.  ``None`` derives the
            legacy ``ci_config.n_points / 0.1`` (number-preserving across all
            ``n_points``).
        quantile_cap: upper cap on the grid-build target CDF (legacy ``0.9999``).
        upper_bound / upper_bound_slack / max_attempts: Ogata-thinning knobs,
            present for the later thinning step; unused by ``inversion``.
        degenerate_floor: ``λ_sum ≤ floor`` ⇒ uniform type fallback (legacy ``1e-12``).
    """

    mode: Literal["inversion", "thinning"] = "inversion"
    # inversion-only
    inversion_method: Literal["auto", "closed_form", "newton", "grid"] = "auto"
    newton_tol: float = 1e-6
    newton_max_iters: int = 50
    bracket_width: float = 0.1
    grid_density: Optional[float] = None
    quantile_cap: float = 0.9999
    # thinning-only (Ogata) — later step
    upper_bound: Literal["interval_sup", "global_sup"] = "interval_sup"
    upper_bound_slack: float = 1.0
    max_attempts: int = 100
    # shared
    degenerate_floor: float = 1e-12


class TemporalPointProcess(nn.Module, ABC):
    """Abstract root for all temporal point processes — the family-agnostic contract.

    Subclassed by the model families: ``IntensityBasedTPP`` (the intensity harness,
    below) and ``DensityBasedTPP`` (inter-event density), plus future generative
    families.  It holds what every family shares — the event-type count ``D``, the
    sampling ``generator``/``seed``/``SamplingConfig``, the ``terminating`` flag — and
    defines the universal predictive/likelihood/sampling contract right here:
    the abstract ``likelihood`` / ``sample`` plus the ``density_*`` / ``prob_*`` /
    ``time_*`` methods, with the mark-axis identities as fallbacks each family base
    overrides for the subset it roots.  Each family *realizes* the contract (intensity
    via λ/Λ, density via the inter-event density + mark head).
    """

    def __init__(
        self,
        D: int,
        seed: Optional[int] = 42,
        terminating: bool = False,
        sampling_config: Optional["SamplingConfig"] = None,
    ):
        super().__init__()
        self.D = D
        self.seed = seed
        self.terminating = terminating
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)
        else:
            self.generator = torch.Generator()
        # Active SamplingConfig override (set here, by ``use_sampling``, or by direct
        # assignment); ``None`` lets the ``sampling_config`` property derive the family
        # default via :meth:`_default_sampling_config`.  Sampling is universal, so the
        # override + its resolution plumbing live on the root (CIConfig, being an
        # intensity-realization knob, stays on ``IntensityBasedTPP``).
        self._sampling_config_override: Optional[SamplingConfig] = sampling_config

    # =========================================================================
    # SamplingConfig plumbing (resolution order: call-arg > use_sampling > default)
    # =========================================================================

    def _default_sampling_config(self) -> "SamplingConfig":
        """Per-family default SamplingConfig.

        The default is the number-preserving ``inversion`` sampler; a family whose
        natural sampler is thinning may override this (not done in this step)."""
        return SamplingConfig()

    @property
    def sampling_config(self) -> "SamplingConfig":
        """Active SamplingConfig: explicit override if set, else the per-family default."""
        return self._sampling_config_override if self._sampling_config_override is not None else self._default_sampling_config()

    @sampling_config.setter
    def sampling_config(self, cfg: Optional["SamplingConfig"]) -> None:
        self._sampling_config_override = cfg

    @contextmanager
    def use_sampling(self, cfg: "SamplingConfig") -> Iterator[None]:
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

    @abstractmethod
    def transform_params(self) -> Tuple[torch.Tensor, ...]:
        """Return the model parameters in their constrained/transformed form.

        Many TPP parameters have natural constraints (e.g. rates must be positive);
        this returns them after the appropriate transform (softplus, exp, …).  The
        structure is model-specific (``(mu,)`` for Poisson, ``(mu, alpha, beta)`` for
        Hawkes, …).
        """
        raise NotImplementedError("Subclasses must implement parameter transformation")

    # =========================================================================
    # Universal predictive / likelihood / sampling contract
    # =========================================================================
    #
    # The paper's next-event decomposition, named uniformly: ``density_*`` for the
    # continuous time axis, ``prob_*`` for the discrete mark axis.  Every method
    # takes ``log: bool = True`` (log-space is the implemented primitive; ``log=False``
    # returns the ``exp``) and accepts time queries of shape ``(B,)`` or ``(B, Q)``.
    #
    # Only ``likelihood`` and ``sample`` are hard-universal (abstract).  The rest are
    # fallbacks expressing the decomposition identities; each *family base* overrides
    # the subset it roots and inherits the others.  ``density_joint`` is the
    # hub: rooted by the intensity family (λ_m·S) or derived exactly (p(t)·p(m|t)).

    def __init_subclass__(cls, **kwargs):
        """Fail-fast root-sufficiency guard.

        ``density_joint`` ⇄ ``{density_time, prob_mark_given_time}`` are mutually
        derivable, so a concrete family that roots *neither* side would send the
        fallbacks into infinite mutual recursion.  Reject it at class-definition
        time (mirrors the intensity-hook guard on ``IntensityBasedTPP``).  Abstract
        intermediate bases are skipped.
        """
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        root = TemporalPointProcess
        roots_joint = cls.density_joint is not root.density_joint
        roots_pair = (
            cls.density_time is not root.density_time
            and cls.prob_mark_given_time is not root.prob_mark_given_time
        )
        if not (roots_joint or roots_pair):
            raise TypeError(
                f"{cls.__name__} must root density_joint, or both density_time and "
                "prob_mark_given_time (they derive each other; at least one side must exist)."
            )

    @abstractmethod
    def likelihood(
        self, ts: BatchedMVEventData, T: Optional[torch.Tensor] = None, log: bool = True
    ) -> torch.Tensor:
        """Log-likelihood of the observed events (the canonical training objective)."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, ts, num_steps: int = 1, rng=None, sampling_config=None):
        """Generate events by sampling from the process."""
        raise NotImplementedError

    # -- density_joint hub + exact mark-axis fallbacks ----------------------

    def density_joint(self, t: torch.Tensor, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """p(t, m | H) — the joint next-event density over (time, mark).  ``(B,[Q,]D)``.

        Fallback (for families rooting the pair): ``p(t)·p(m|t)``.  The intensity
        family roots this directly as ``λ_m(t)·S(t)``.
        """
        lt = self.density_time(t, ts, log=True)  # (B,[Q])
        lc = self.prob_mark_given_time(t, ts, log=True)  # (B,[Q,]D)
        lj = lt.unsqueeze(-1) + lc
        return lj if log else lj.exp()

    def density_time(self, t: torch.Tensor, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """p(t | H) = Σ_m p(t, m | H) — the time-marginal density.  ``(B,[Q])``. Exact."""
        if log:
            return torch.logsumexp(self.density_joint(t, ts, log=True), dim=-1)
        return self.density_joint(t, ts, log=False).sum(dim=-1)

    def prob_mark_given_time(self, t: torch.Tensor, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """p(m | t, H) = p(t, m) / p(t) — the conditional mark distribution.  ``(B,[Q,]D)``. Exact."""
        lr = self.density_joint(t, ts, log=True) - self.density_time(t, ts, log=True).unsqueeze(-1)
        return lr if log else lr.exp()

    def density_time_given_mark(self, t: torch.Tensor, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """p(t | m, H) = p(t, m) / p(m) — the mark-conditional time density.  ``(B,[Q,]D)``."""
        lj = self.density_joint(t, ts, log=True)  # (B,[Q,]D)
        lpm = self.prob_mark(ts, log=True)  # (B, D)
        lr = lj - (lpm.unsqueeze(1) if lj.dim() == 3 else lpm)
        return lr if log else lr.exp()

    # -- integrated quantities (family-specific; overridden by every family) --

    def prob_mark(self, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """p(m | H) — the mark marginal, integrated over the next-event time.  ``(B, D)``."""
        raise NotImplementedError(f"{type(self).__name__} must implement prob_mark.")

    def survival_time(self, t: torch.Tensor, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """S(t) = P(t_next ≥ t | H) — the survival function.  ``(B,[Q])``."""
        raise NotImplementedError(f"{type(self).__name__} must implement survival_time.")

    def cdf_time(self, t: torch.Tensor, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
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
        """Boolean ``(B,)``: True where a sequence repeats an event type."""
        return _has_repeated_event_types_fn(ts, self.D)


class IntensityBasedTPP(TemporalPointProcess):
    """
    Abstract base class for multivariate temporal point processes.

    A temporal point process models the occurrence of events over time, where each
    event has a timestamp and an associated type/mark. The key component is the
    intensity function λ_d(t) which characterizes the instantaneous rate of events
    of type d at time t.

    Core Methods (Required):
    - _intensity_at_t(t, ts): The raw intensity function (before termination masking)
    - transform_params(): Return parameters in their constrained form

    Likelihood Methods (Must implement one of):
    - cumulative_intensity(T, ts): Analytical integral (preferred)
    - OR provide neither and use numerical_integration (slower but flexible)

    Helper Methods (Optional overrides):
    - _event_intensities(ts): raw λ at observed events (the positive likelihood term)
    - inverse_cumulative_intensity(u, ts): Invert CDF analytically
    - OR use root-finding solver (automatic fallback)

    Default Behavior:
    - intensity(t, ts) automatically applies termination masking to _intensity_at_t(t, ts)
    - If cumulative_intensity is not analytical, numerical integration is used
    - If CDF inversion is not provided, root-finding solver is used (Brent's method)
    - PDF and CDF use intensity and cumulative_intensity
    """

    def __init_subclass__(cls, **kwargs):
        """Require every concrete model to implement one Layer-1 intensity hook.

        ``_intensity_at_t`` and ``_intensities_at_queries`` delegate to each other
        by default, so a model overrides *either* and both work.  This guard
        breaks the otherwise-infinite mutual recursion by rejecting a concrete
        subclass that overrides neither.  Abstract intermediate bases (e.g.
        ``NeuralParametrizedTPP``) are skipped — ``inspect.isabstract`` scans
        manually during class creation, before ``__abstractmethods__`` is set.
        """
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        base = IntensityBasedTPP
        if (
            cls._intensity_at_t is base._intensity_at_t
            and cls._intensities_at_queries is base._intensities_at_queries
        ):
            raise TypeError(
                f"{cls.__name__} must implement _intensity_at_t or _intensities_at_queries "
                "(they delegate to each other; at least one must be overridden)."
            )

    def __init__(
        self,
        D: int,
        seed: Optional[int] = 42,
        terminating: bool = False,
        ci_config: Optional["CIConfig"] = None,
        sampling_config: Optional["SamplingConfig"] = None,
    ):
        """
        Initialize the temporal point process.

        Args:
            D: Number of event types (dimensions)
            seed: Random seed for reproducibility
            terminating: If True, each event type can only occur once (first occurrence data).
                After an event type occurs, its intensity becomes zero and it cannot occur again.
            ci_config: Cumulative-intensity scheme.  ``None`` (default) falls back to the
                family default via :meth:`_default_ci_config`; an explicit value overrides it.
            sampling_config: Draw scheme.  ``None`` (default) falls back to the family
                default via :meth:`_default_sampling_config`; an explicit value overrides it.
        """
        super().__init__(D=D, seed=seed, terminating=terminating, sampling_config=sampling_config)
        # Active CIConfig override (set here, by ``use_ci``, or by direct assignment).
        # ``None`` lets the ``ci_config`` property derive the per-family default from
        # :meth:`_default_ci_config`.  (The SamplingConfig override + plumbing live on the
        # root ``TemporalPointProcess``; only ``ci_config`` is intensity-specific.)
        self._ci_config_override: Optional[CIConfig] = ci_config

    # =========================================================================
    # Intensity Functions (Must be implemented by subclasses)
    # =========================================================================

    # -- Layer 1: intensity access (the multi-query primitive) --------------
    #
    # Three hooks of increasing structure; a model overrides whichever fits:
    #   _intensity_at_t        (B,)     -> (B,D)         leaf — one time/sequence
    #   _intensities_at_queries (B,Q)   -> (B,Q,D)       flat — arbitrary times
    #   _interval_intensities  (B,R,P)  -> (B,R,P,D)     structured — per interval
    #
    # The leaf and flat hooks mutually delegate, so overriding either makes
    # the other work; the __init_subclass__ guard requires one of the pair.  The
    # structured hook always terminates by flattening into the flat hook, so it is
    # a pure fast-path seam, never required.  All three return RAW intensities
    # (no termination masking) — masking is applied by the public ``intensity``.
    #
    # Conditioning contract: the tie-break at an exact event time is
    # strictly-before / left-limit — an event never contributes to its own
    # intensity (the standard TPP convention).  Every model's leaf/flat override
    # must honour this; the classical and neural causal lookups already do.

    def _intensity_at_t(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Raw (unmasked) intensity λ_d(t) for each type d at a single time/sequence.

        Default: delegate to the flat hook with a single query (Q=1).  Override
        this (or ``_intensities_at_queries``) to define the intensity function.

        Args:
            t: Time points where to evaluate intensity. Shape: (B,)
            ts: Historical events (batched, padded). Shape: (B, L)

        Returns:
            Raw intensity values for each type and batch. Shape: (B, D)
        """
        return self._intensities_at_queries(t.unsqueeze(1), ts).squeeze(1)  # (B, D)

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Raw intensity at Q arbitrary query times per sequence (the flat primitive).

        Default: loop the leaf ``_intensity_at_t`` over the query axis — exactly the
        per-point evaluation the rest of the framework used before this hook
        existed, so models are bit-identical until they opt into a fast override
        (classical broadcast, neural state-gather, AttNHP attention re-run).

        Args:
            query_times: Absolute query times. Shape: (B, Q)
            ts: Historical events. Shape: (B, L)

        Returns:
            Raw intensities at each query. Shape: (B, Q, D)
        """
        Q = query_times.shape[1]
        return torch.stack([self._intensity_at_t(query_times[:, k], ts) for k in range(Q)], dim=1)  # (B, Q, D)

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
        is reused here; it is bit-identical to looping ``_intensity_at_t`` at the
        event times.  Override directly for a cheaper all-positions pass.

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

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute the intensity function λ_d(t) for each event type d.

        The intensity function represents the instantaneous rate of events of each
        type conditional on the history of events up to time t.

        For terminating processes, the intensity is zero for event types that have
        already occurred (strictly before time t).

        Subclasses should override _intensity_at_t() instead of this method to ensure
        termination is properly handled.

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

        assert (t >= 0.0).all(), "Intensity is only defined for t >= 0."

        if t.dim() == 1:
            # (B,) single query — the exact prior path (leaf hook + (B, D) mask).
            intensities = self._intensity_at_t(t, ts)  # (B, D)
            if self.terminating:
                mask = self._get_termination_mask(ts, t=t)  # (B, D)
                intensities = intensities * mask.float()
            return intensities

        # (B, Q) multi-query — route through the flat primitive and mask per query.
        intensities = self._intensities_at_queries(t, ts)  # (B, Q, D)
        if self.terminating:
            mask = self._terminated_mask(t, ts)  # (B, Q, D)
            intensities = intensities * mask.to(intensities.dtype)
        return intensities

    # =========================================================================
    # Cumulative Intensity Functions. If analytical solution is not provided, numerical integration will be used
    # =========================================================================

    def _raw_ci(
        self,
        effective_T: torch.Tensor,
        ts: BatchedMVEventData,
    ) -> torch.Tensor:
        """
        Compute the cumulative intensity Λ_d(effective_T[:, d]) for each type d from 0 to effective_T.

        DEPRECATED on the live path — superseded by ``_closed_form_ci``.  Retained
        only as the equivalence-test oracle (``analytical_cumulative_intensity`` →
        ``_raw_ci``; gate ``Σ _interval_ci ≡ _raw_ci``).  Closed-form families still override it so the
        oracle can validate the per-interval path; no production code calls it.

        The base class has already folded termination into ``effective_T``: for
        terminating processes ``effective_T[:, d] = min(T, t_term_d)``, so
        subclasses never need to inspect termination times themselves.

        Args:
            effective_T: Per-type upper integration limits. Shape: (B, D)
            ts: Historical events. Shape: (B, L)

        Returns:
            Cumulative intensity Λ_d(effective_T[:, d]). Shape: (B, D)
        """
        raise NotImplementedError("Subclasses with an analytical CI must implement _raw_ci(effective_T, ts).")

    def _effective_T(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Per-type upper integration limit ``min(T, t_term_d)``. Shape ``(B, D)``.

        For terminating processes a type stops accruing intensity at its termination
        time; non-terminating reduces to ``T`` broadcast across all types.
        """
        if self.terminating:
            termination_times = self._get_termination_times(ts)  # (B, D)
            return torch.minimum(T.unsqueeze(1).expand_as(termination_times), termination_times)  # (B, D)
        return T.unsqueeze(1).expand(T.shape[0], self.D)  # (B, D)

    def analytical_cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData):
        """Deprecated global closed-form CI oracle (``_raw_ci`` over ``[0, min(T, t_term)]``).

        Kept only as the trust-chain test oracle for ``Σ _interval_ci ≡ _raw_ci``; the
        live closed-form path is ``_closed_form_ci``.  Termination is folded into
        the per-type effective upper limit before ``_raw_ci``.

        Args:
            T: Observation window end times. Shape: (B,)
            ts: Historical events. Shape: (B, L)

        Returns:
            Cumulative intensity. Shape: (B, D)
        """
        return self._raw_ci(self._effective_T(T, ts), ts)

    def _closed_form_ci(
        self, ts: BatchedMVEventData, T_low: Optional[torch.Tensor], T: torch.Tensor
    ) -> torch.Tensor:
        """Closed-form Λ over ``[T_low or 0, T]`` — the ``mode="closed_form"`` path.

        The full ``[0, T]`` integral is ``Σ _interval_ci`` (intervals start at events, so
        the per-interval heads see the correct relative origin; ``_active_interval_mask``
        zeroes post-termination intervals per type; ``Σ`` telescopes to ``_raw_ci(T)``).
        Windowing recurses for the lower limit — ``Λ(T) − Λ(T_low)`` — rather than clipping
        the interval grid: a mid-interval ``T_low`` would shift a relative-time head's
        origin.  The subtraction is exact for an analytic closed form (no quadrature error),
        so there is no cancellation concern.  Efficient for neural closed-form heads
        (``_interval_ci`` is ``O(L)``).  Hawkes/Baseline override this with the ``O(N·D)``
        global module CI (they provide no per-interval closed form).
        """
        lo, hi = self.interval_bounds(ts, None, T)  # (B, L+1), intervals start at events
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

        return numerical_integration(
            self.intensity,
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
    # Layer 2: cumulative-intensity construction
    # =========================================================================

    def _interval_ci(self, ts: BatchedMVEventData, t_lo: torch.Tensor, t_hi: torch.Tensor) -> torch.Tensor:
        """Closed-form ∫λ_d over ``[t_lo_i, t_hi_i]`` within interval ``i``.

        Parallel in shape to ``_interval_intensities``: the per-interval analytic
        integral, history held constant at ``H_i`` within interval ``i``.  The default
        derives it from the model's global analytic CI as ``Λ(hi_i) − Λ(lo_i)``: no
        event lies strictly inside an inter-event interval, so the history seen on
        ``[lo_i, hi_i]`` is exactly ``H_i`` and the difference is the genuine
        per-interval integral.  Summing over ``i`` telescopes to ``_raw_ci(T)`` — the
        closed-form trust-chain anchor (gate test ``Σ _interval_ci ≡ _raw_ci``).  A
        model without an analytic ``_raw_ci`` routes through numerical quadrature
        instead (``CIConfig`` ``mode="numerical"``).

        Closed-form families override this (neural: causal param-gather; classical:
        baseline+kernel module per-interval CI) — none route through ``_raw_ci``.  A
        model without a closed form leaves it unimplemented and uses a numerical
        ``CIConfig`` instead.

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

    def _default_ci_config(self) -> "CIConfig":
        """Per-family default CIConfig.

        The safe majority is closed-form; families that integrate numerically override this
        to declare their placement/rule/n_points explicitly (fail-loud: a numerical model
        that forgets hits ``closed_form`` and its missing ``_interval_ci`` raises).  The
        factory's ``num_ci_integration_steps`` still wins by assigning ``self.ci_config``
        after construction."""
        return CIConfig(mode="closed_form")

    @property
    def ci_config(self) -> "CIConfig":
        """Active CIConfig: explicit override if set, else the per-family default."""
        return self._ci_config_override if self._ci_config_override is not None else self._default_ci_config()

    @ci_config.setter
    def ci_config(self, cfg: Optional["CIConfig"]) -> None:
        self._ci_config_override = cfg

    @contextmanager
    def use_ci(self, cfg: "CIConfig") -> Iterator[None]:
        """Temporarily force a CIConfig for a block (eval / forecast / diagnostic sweep).

        Resolution is call-arg > this context > instance default, so wrapping an
        evaluation loop in ``with model.use_ci(cfg):`` redirects every CI call inside
        without touching model internals (e.g. to sweep n_points/rule).
        """
        prev = self._ci_config_override
        self._ci_config_override = cfg
        try:
            yield
        finally:
            self._ci_config_override = prev

    def _resolve_inversion_method(self, cfg: "SamplingConfig") -> str:
        """Resolve ``SamplingConfig.inversion_method`` to a concrete path.

        ``"auto"`` reproduces the legacy ``inverse_CDF`` dispatch: ``grid`` for a
        numerical CI, else ``closed_form`` when the model overrides
        :meth:`inverse_marginal_cumulative_intensity`, else ``newton``.
        """
        if cfg.inversion_method != "auto":
            return cfg.inversion_method
        if self.ci_config.mode != "closed_form":
            return "grid"
        overridden = type(self).inverse_marginal_cumulative_intensity is not IntensityBasedTPP.inverse_marginal_cumulative_intensity
        return "closed_form" if overridden else "newton"

    def _grid_inversion_density(self, cfg: "SamplingConfig") -> float:
        """Points/unit-time for the grid inverter (legacy ``ci_config.n_points / 0.1``)."""
        if cfg.grid_density is not None:
            return float(cfg.grid_density)
        return float(self.ci_config.n_points) / 0.1

    def interval_bounds(
        self, ts: BatchedMVEventData, T_low: Optional[torch.Tensor], T: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The L+1 inter-event interval bounds, clipped to ``[T_low or 0, T]``.

        Returns ``(lo, hi)`` of shape ``(B, L+1)``.  Interval ``i`` is ``[t_{i-1}, t_i]``
        (with ``t_{-1}=0``, ``t_L=T``); padding positions push to ``+inf`` so they clip
        to zero width.  Direct windowed integration — clipping replaces the old
        ``CI(T) − CI(T_low)`` double pass, so intervals outside the window contribute 0.
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
        self, lo: torch.Tensor, hi: torch.Tensor, cfg: "CIConfig"
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
        self, T: torch.Tensor, ts: BatchedMVEventData, T_low: Optional[torch.Tensor], cfg: "CIConfig"
    ) -> torch.Tensor:
        """Event-knotted per-interval numerical CI (the ``placement="interval"`` path).

        One self-contained path: build the L+1 inter-event interval grid (P nodes per
        interval, ``cfg.rule``), evaluate the structured intensity hook with its
        per-interval conditioning, integrate ``λ·w`` within each interval, and
        sum.  Termination is whole-interval masking (see ``_active_interval_mask``) —
        no cross-branch to the global path.  ``chunk_size`` bounds peak memory by
        accumulating over the node axis (eval/forecast, ``no_grad``).
        """
        lo, hi = self.interval_bounds(ts, T_low, T)  # (B, L+1) each
        nodes, weights = self._interval_quadrature_grid(lo, hi, cfg)  # (B, L+1, P) each
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
        if self.terminating:
            contrib = contrib * self._active_interval_mask(ts, lo)  # zero post-termination intervals
        return contrib.sum(dim=1)  # (B, D)

    def cumulative_intensity(
        self,
        T: torch.Tensor,
        ts: BatchedMVEventData,
        T_low: Optional[torch.Tensor] = None,
        ci_config: Optional["CIConfig"] = None,
    ):
        """Cumulative intensity Λ_d(T) (or over ``[T_low, T]``), dispatched by CIConfig.

        Single entry for both the survival term and point-CI queries — this is what
        kills the old hybrid (the ``per_interval_survival_quadrature`` boolean).  The
        scheme is resolved call-arg > active ``use_ci`` context > instance default; the
        instance default reproduces the model's prior scheme, so default outputs are
        preserved.

        Args:
            T: End times. Shape: (B,)
            ts: Historical events. Shape: (B, L)
            T_low: Optional start times. Shape: (B,)
            ci_config: Force a specific CIConfig for this call (else the instance/context one).
        """
        # Check legality of T:
        assert (T >= 0.0).all(), "Cumulative intensity is only defined for T >= 0."
        assert T.shape[0] == ts.time_points.shape[0], "T must have the same batch size as ts"

        if T_low is not None:
            assert (T_low >= 0.0).all(), "Cumulative intensity is only defined for T_low >= 0."
            assert T_low.shape[0] == ts.time_points.shape[0], "T_low must have the same batch size as ts"
            assert (T >= T_low).all(), "Cumulative intensity requires T >= T_low."

        cfg = ci_config if ci_config is not None else self.ci_config

        if cfg.mode == "closed_form":
            return self._closed_form_ci(ts, T_low, T)

        if cfg.placement == "global":
            return self._numerical_cumulative_intensity(
                T, ts, T_low=T_low, method=_RULE_TO_METHOD[cfg.rule], num_points=cfg.n_points
            )
        return self._numerical_interval_ci(T, ts, T_low, cfg)

    # ========================================================================
    # Intensity Grid (Pre-computed intensity with fast CI lookup)
    # ========================================================================

    def build_intensity_grid(
        self,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        ts: BatchedMVEventData,
        num_points: int = _GRID_NUM_POINTS,
        integrate_marginal_via_grid: bool = True,
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

        Returns:
            IntensityGrid with pre-computed values.
        """
        ci_func = None
        if self.ci_config.mode == "closed_form" and not integrate_marginal_via_grid:

            def ci_func(t, ts):
                return self.cumulative_intensity(t, ts)

        return IntensityGrid.build(
            intensity_func=self.intensity,
            t_start=t_start,
            t_end=t_end,
            ts=ts,
            num_points=num_points,
            ci_func=ci_func,
        )

    def build_intensity_grid_until(
        self,
        t_start: torch.Tensor,
        ts: BatchedMVEventData,
        points_per_unit_time: float = _GRID_POINTS_PER_UNIT_TIME,
        target_cdf: Optional[float] = _GRID_TARGET_CDF,
        T_max: Optional[torch.Tensor] = None,
        chunk_duration: float = _GRID_CHUNK_DURATION,
        max_chunks: int = _GRID_MAX_CHUNKS,
        integrate_marginal_via_grid: bool = True,
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

        Returns:
            IntensityGrid covering [t_start, t_stop].
        """
        ci_func = None
        if self.ci_config.mode == "closed_form" and not integrate_marginal_via_grid:

            def ci_func(t, ts):
                return self.cumulative_intensity(t, ts)

        return IntensityGrid.build_until(
            intensity_func=self.intensity,
            t_start=t_start,
            ts=ts,
            points_per_unit_time=points_per_unit_time,
            target_cdf=target_cdf,
            T_max=T_max,
            ci_func=ci_func,
            chunk_duration=chunk_duration,
            max_chunks=max_chunks,
        )

    # ========================================================================
    # CDF/CI Inversion (Override for analytical solution)
    # ========================================================================

    def inverse_marginal_cumulative_intensity(
        self,
        u: torch.Tensor,
        ts: BatchedMVEventData,
    ) -> torch.Tensor:
        """
        Invert the marginal cumulative intensity to find t such that Λ(t) = u.
        t is here in absolute time.

        Default implementation: Uses Bracketed-Newton's method from utils.py.
        Override for analytical solutions or more efficient implementations.

        Args:
            u: Target cumulative intensity values. Shape: (batch_size,)
            ts: Historical events (batched, padded sequences). Shape: (batch_size, seq_len)

        Returns:
            Time values t such that cumulative_intensity(t) ≈ u. Shape: same as u

        Useful for sampling and other applications requiring CI inversion.
        """
        assert u.shape[0] == ts.time_points.shape[0], "u must have the same batch size as ts"

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

            grid = IntensityGrid.build_until(
                intensity_func=self.intensity,
                t_start=t_start,
                ts=ts,
                points_per_unit_time=self._grid_inversion_density(cfg),
                target_cdf=target_cdf,
                T_max=None,
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

        # Analytical CI or grid fallback: use bracketed Newton's method
        def ci_func(t: torch.Tensor) -> torch.Tensor:
            ci = self.cumulative_intensity(t, ts)  # (B,D)
            return ci.sum(dim=1)  # (B,)

        def intensity_func(t: torch.Tensor) -> torch.Tensor:
            lam = self.intensity(t, ts)  # (B,D)
            return lam.sum(dim=1)  # (B,)

        t_star = invert_CI(
            cumulative_intensity_func=ci_func,
            intensity_func=intensity_func,
            target=u,
            bracket_init=(ts.max_time, ts.max_time + cfg.bracket_width),
            tol=cfg.newton_tol,
            max_newton_iters=cfg.newton_max_iters,
        )

        return t_star

    def inverse_CDF(self, u: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Invert the CDF to find T such that P( t_next < T) = u.
        Also known as the quantile function. It is only defined for T >= last event time.
        If you want to sample inbetween events, use the inverse_marginal_cumulative_intensity function,
        or cut of the history ts.

        For analytical CI models, transforms u → target CI and delegates to
        inverse_marginal_cumulative_intensity (Newton's method).

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
        if self._resolve_inversion_method(cfg) == "grid":
            t_last = ts.max_time  # (B,)

            # Build grid until CDF reaches the maximum requested quantile
            target_cdf = float(u.max().item())
            target_cdf = min(target_cdf + _GRID_TARGET_CDF_MARGIN, cfg.quantile_cap)

            grid = self.build_intensity_grid_until(
                t_start=t_last,
                ts=ts,
                points_per_unit_time=self._grid_inversion_density(cfg),
                target_cdf=target_cdf,
                T_max=None,
            )

            return self._inverse_cdf_from_grid(grid, u)

        # Analytical CI: transform u to target cumulative intensity
        device = ts.time_points.device
        target_ci = -torch.log1p(-u).to(device)  # (B,)

        # CDF is only defined for t >= last event time
        target_ci += self.cumulative_intensity(ts.max_time, ts).sum(dim=1)  # (B,)

        t_star = self.inverse_marginal_cumulative_intensity(target_ci, ts)  # (B,)

        return t_star

    # =========================================================================
    # Likelihood
    # =========================================================================

    def _event_log_intensity_sum(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Σ_i log λ_{m_i}(t_i) over valid events — the positive log-likelihood term.  ``(B,)``.

        Delegates the raw per-event intensity to ``_event_intensities`` (override *that*
        to speed up the positive term).  Padding, empty sequences, and the
        terminating-repeat impossibility (log p = −inf) are handled here so subclasses
        supply only raw λ.  Folded from the former ``positive_likelihood``.
        """
        B = ts.time_points.shape[0]
        L = ts.time_points.shape[1]
        device = ts.time_points.device

        if L == 0:
            return torch.zeros(B, device=device)

        valid_events = ts.event_mask  # (B, L)

        # --- core: get λ_{m_i}(t_i) for every i, from the subclass hook ---
        lam = self._event_intensities(ts)  # (B, L) — raw, before masking
        # Catch a broken parameterization (NaN / negative λ) at the source rather
        # than silently clamping it; an actual structurally-zero λ is fine and is
        # floored to finfo.tiny inside log() below. Guard is debug-only: a
        # per-element .all() check on a GPU tensor forces a host sync every step,
        # which is too costly for the imminent training/tuning runs.
        if __debug__ and not lam.is_cuda:
            valid_lam = lam[valid_events]
            assert torch.isfinite(valid_lam).all() and (valid_lam >= 0).all(), (
                "intensity at observed events must be finite and non-negative"
            )
        # Floor only inside log(): finfo.tiny keeps log() finite for a literally
        # zero intensity (log(tiny) ≈ −87 f32 / −708 f64) without the −27.6 nat
        # compression the old 1e-12 floor caused. NOTE: clamp the log() argument
        # only — `lam` itself stays unclamped so the survival integral Λ=∫λ is not
        # fed a spurious positive rate.  Mirrors utils/evaluation.py (commit 1cdee1d).
        log_lam = torch.log(lam.clamp(min=torch.finfo(lam.dtype).tiny))  # (B, L)

        # Zero out padding positions
        log_lam = log_lam.masked_fill(~valid_events, 0.0)  # (B, L)
        log_positive = log_lam.sum(dim=1)  # (B,)

        # Sequences with no valid events contribute 0 to the positive term.
        no_events = valid_events.sum(dim=1) == 0  # (B,)
        log_positive = log_positive.masked_fill(no_events, 0.0)

        # Terminating: a repeated type makes the event impossible (log p = -inf).
        if self.terminating:
            has_repeated = self._has_repeated_event_types(ts)  # (B,)
            log_positive = log_positive.masked_fill(has_repeated, float("-inf"))

        return log_positive

    def likelihood(
        self,
        ts: BatchedMVEventData,
        T: Optional[torch.Tensor] = None,
        log: bool = True,
    ) -> torch.Tensor:
        """
        Full log-likelihood of the observed events (the canonical training objective).

        Combines the positive term (log-intensities at events) and the negative
        (survival) term (cumulative intensity over the window) inline:

            log L = ∑ᵢ log λ_{m_i}(tᵢ) − ∑_d Λ_d(T)


        Args:
            ts: Observed event sequences (batched, padded). Shape: (batch_size, seq_len)
            T: End time of observation window. Shape: (batch_size,). If None, defaults
               to ``ts.T_obs`` (the per-sequence observation window stored on the batch).
            log: If True, return the log-likelihood; otherwise its ``exp``.

        Returns:
            Likelihood for each batch. Shape: (batch_size,)
        """
        if T is None:
            T = ts.T_obs

        assert (T >= ts.max_time).all(), "Likelihood is only defined for T >= last event time."

        log_positive = self._event_log_intensity_sum(ts)  # (B,)  Σ_i log λ_{m_i}(t_i)
        log_negative = -self.cumulative_intensity(T=T, ts=ts).sum(dim=1)  # (B,)  −Σ_d Λ_d(T)
        log_lik = log_positive + log_negative

        return log_lik if log else torch.exp(log_lik)

    # =========================================================================
    # Predictive contract realization — via λ/Λ
    # =========================================================================

    def _prep_query_time(self, t: torch.Tensor, ts: BatchedMVEventData, atol: float) -> torch.Tensor:
        """Cast to double, snap near-``t_last`` queries onto ``t_last``, assert ``t ≥ t_last``.

        Shared by the density/survival methods; ``atol`` is the near-boundary snap
        tolerance (kept per-method to preserve the legacy PDF/CDF numbers).  Handles
        ``(B,)`` and ``(B, Q)`` queries.
        """
        t = t.double()
        mx = ts.max_time  # (B,)
        mx_b = mx.unsqueeze(1) if t.dim() == 2 else mx
        close = torch.abs(t - mx_b) < atol
        t = torch.where(close, mx_b.expand_as(t), t)
        assert (t >= mx_b).all(), "predictive densities are only defined for t >= last event time."
        return t

    def _windowed_total_ci(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Σ_d Λ_d over ``[t_last, t]``.  ``(B,)`` → ``(B,)``; ``(B, Q)`` → ``(B, Q)``.

        The ``(B,)`` path is one validated ``cumulative_intensity`` call (bit-identical
        to the legacy PDF/CDF survival).  The ``(B, Q)`` path loops that call over the
        query axis — all queries share ``T_low = t_last`` and history ``ts`` — pending a
        fused windowed multi-query CI.
        """
        t_last = ts.max_time  # (B,)
        if t.dim() == 1:
            return self.cumulative_intensity(t, ts, T_low=t_last).sum(dim=1)  # (B,)
        Q = t.shape[1]
        cols = [self.cumulative_intensity(t[:, q], ts, T_low=t_last).sum(dim=1) for q in range(Q)]
        return torch.stack(cols, dim=1)  # (B, Q)

    def density_joint(self, t: torch.Tensor, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """p(t, m | H) = λ_m(t)·S(t) — the intensity family's rooted joint density.

        ``(B,[Q,]D)``.  ``log=False`` reproduces the legacy ``PDF`` exactly (raw λ·exp(−ΣΛ));
        ``log=True`` floors λ at ``finfo.tiny`` inside the log (matching the likelihood /
        eval log-accumulation), so ``exp(log)`` and the linear form agree except at a
        structurally-zero intensity.
        """
        t = self._prep_query_time(t, ts, atol=_SNAP_ATOL_DENSITY)  # legacy PDF snap tolerance
        lam = self.intensity(t, ts)  # (B,[Q,]D)
        total_ci = self._windowed_total_ci(t, ts)  # (B,[Q])
        if log:
            tiny = torch.finfo(lam.dtype).tiny
            return torch.log(lam.clamp(min=tiny)) - total_ci.unsqueeze(-1)
        return lam * torch.exp(-total_ci).unsqueeze(-1)

    def survival_time(self, t: torch.Tensor, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """S(t) = exp(−Σ_d Λ_d(t_last → t)).  ``(B,[Q])``.  ``log=False`` == ``1 − CDF``."""
        t = self._prep_query_time(t, ts, atol=_SNAP_ATOL_CDF)  # legacy CDF snap tolerance
        total_ci = self._windowed_total_ci(t, ts)  # (B,[Q])
        return -total_ci if log else torch.exp(-total_ci)

    def cdf_time(self, t: torch.Tensor, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """P(t_next < t | H) = 1 − S(t).  ``(B,[Q])``.  ``log=False`` reproduces the legacy ``CDF``."""
        cdf = 1.0 - self.survival_time(t, ts, log=False)
        return torch.log(cdf.clamp(min=torch.finfo(cdf.dtype).tiny)) if log else cdf

    def prob_mark(self, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """p(m | H) — the time-integrated mark marginal (grid).  ``(B, D)``."""
        p = self.marginal_class_distribution(ts)  # (B, D), linear
        return torch.log(p.clamp(min=torch.finfo(p.dtype).tiny)) if log else p

    def time_mean(self, ts: BatchedMVEventData) -> torch.Tensor:
        """E[t_next | H] — the grid survival-integral mean.  ``(B,)``."""
        return self.mean_next_event_time(ts)

    def time_quantile(self, u, ts: BatchedMVEventData) -> torch.Tensor:
        """Inverse time CDF (median = ``u=0.5``).  ``(B,)``.  ``u`` is a scalar or ``(B,)``."""
        if not torch.is_tensor(u):
            u = torch.full((ts.time_points.shape[0],), float(u), device=ts.time_points.device, dtype=torch.float64)
        return self.inverse_CDF(u, ts)

    # -- legacy-name delegators (kept for call-sites; the contract names are primary) --

    def PDF(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Deprecated alias for :meth:`density_joint` (``log=False``): p(t, m | H)."""
        return self.density_joint(T, ts, log=False)

    def CDF(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Deprecated alias for :meth:`cdf_time` (``log=False``): P(t_next < T | H)."""
        return self.cdf_time(T, ts, log=False)

    def build_integration_grid(
        self,
        ts: BatchedMVEventData,
        T_max: Optional[torch.Tensor] = None,
        num_points: int = _GRID_NUM_POINTS,
        points_per_unit_time: float = _GRID_POINTS_PER_UNIT_TIME,
        integrate_marginal_via_grid: bool = True,
    ) -> IntensityGrid:
        """
        Build an IntensityGrid from t_last to T_max for marginal computations.

        This is the shared grid construction used by marginal_class_distribution
        and mean_next_event_time. It grows the grid incrementally until
        CDF >= 0.999 or T_max is reached.

        When T_max is provided, the grid is built with a fixed number of points
        (num_points) over [t_last, T_max]. When T_max is None, the grid grows
        incrementally using build_intensity_grid_until with the specified
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

        Returns:
            IntensityGrid over [t_last, T_max] with intensities and CI prefix sums.
        """
        t_last = ts.max_time  # (B,)

        if T_max is not None:
            return self.build_intensity_grid(
                t_last,
                T_max,
                ts,
                num_points=num_points,
                integrate_marginal_via_grid=integrate_marginal_via_grid,
            )

        return self.build_intensity_grid_until(
            t_start=t_last,
            ts=ts,
            points_per_unit_time=points_per_unit_time,
            target_cdf=_GRID_TARGET_CDF,
            T_max=None,
            integrate_marginal_via_grid=integrate_marginal_via_grid,
        )

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
        marginal_probs = torch.trapezoid(integrands, grid.t_grid.unsqueeze(-1).expand_as(integrands), dim=1)  # (B, D)

        total_prob = marginal_probs.sum(dim=1, keepdim=True).clamp(min=_DIV_FLOOR)
        return marginal_probs / total_prob

    @staticmethod
    def _mean_next_event_time_from_grid(grid: IntensityGrid) -> torch.Tensor:
        """
        Compute the mean next event time from a pre-built IntensityGrid.

        Uses the survival function identity:
            E[t_next] = t_N + ∫_{t_N}^{T_max} S(t) dt

        where S(t) = exp(-∑_d ΔΛ_d(t)) is the survival function.

        Args:
            grid: Pre-built IntensityGrid over [t_last, T_max].

        Returns:
            Mean next event time for each batch element. Shape: (B,)
        """
        total_ci = grid.ci_prefix.sum(dim=2)  # (B, num_points)
        survival = torch.exp(-total_ci)  # (B, num_points)

        # E[t] = t_last + ∫_{t_last}^{T_max} S(t) dt
        survival_integral = torch.trapezoid(survival, grid.t_grid, dim=1)  # (B,)
        return grid.t_start + survival_integral  # (B,)

    def marginal_class_distribution(
        self,
        ts: BatchedMVEventData,
        T_max: Optional[torch.Tensor] = None,
        num_points: int = _GRID_NUM_POINTS,
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
        grid = self.build_integration_grid(ts, T_max=T_max, num_points=num_points)
        return self._marginal_class_distribution_from_grid(grid)

    def mean_next_event_time(
        self,
        ts: BatchedMVEventData,
        T_max: Optional[torch.Tensor] = None,
        num_points: int = _GRID_NUM_POINTS,
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
        grid = self.build_integration_grid(ts, T_max=T_max, num_points=num_points)
        return self._mean_next_event_time_from_grid(grid)

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
            Time values t such that CDF(t) ≈ u. Shape: (B,)
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

        return t_lo + frac * (t_hi - t_lo)  # (B,)

    # =========================================================================
    # Sampling
    # =========================================================================

    def sample(
        self,
        ts: MVEventData,
        num_steps: int = 1,
        rng: Optional[torch.Generator | int] = None,
        sampling_config: Optional["SamplingConfig"] = None,
    ) -> Tuple[MVEventData, List[float], List[torch.Tensor]]:
        """
        Generate new events by sampling from the temporal point process.

        The draw scheme is set by :class:`SamplingConfig` (resolution order call-arg
        > ``use_sampling`` context > instance default).  The default ``inversion``
        mode uses inverse-CDF sampling; ``thinning`` (Ogata) is implemented in a
        later step.

        For terminating processes, sampling stops when all event types have occurred.

        Args:
            ts: Initial event history (single sequence, not batched)
            num_steps: Number of events to generate (for terminating: min(num_steps, remaining types))
            rng: Random number generator or seed for reproducibility
            sampling_config: Optional per-call SamplingConfig override.

        Returns:
            Tuple containing:
            - Updated MVEventData with new events appended
            - List of sampled inter-arrival times
            - List of type probability distributions at each sampled time

        Note:
            Sampling strategies vary by model:
            - Poisson: Direct inverse CDF sampling
            - Hawkes: Thinning algorithm or inverse CDF (if tractable)
            - Spline: Inverse CDF via numerical inversion

            Not all models may have efficient sampling implementations.

            For terminating processes, sampling automatically stops when all event
            types have occurred, even if num_steps is larger.
        """
        # Resolve the config (call-arg > active context > instance default) and set it
        # active for the duration so the inversion helpers (which read
        # self.sampling_config) honour a per-call override.  Dispatch straight to the
        # mode implementation — never re-call self.sample, which would recurse through
        # subclass overrides (e.g. PoissonProcess.sample delegating thinning to super()).
        cfg = sampling_config if sampling_config is not None else self.sampling_config
        with self.use_sampling(cfg):
            if cfg.mode == "thinning":
                return self._sample_thinning(ts, num_steps=num_steps, rng=rng, cfg=cfg)
            return self._sample_inversion(ts, num_steps=num_steps, rng=rng, cfg=cfg)

    def _sample_inversion(
        self,
        ts: MVEventData,
        num_steps: int,
        rng: Optional[torch.Generator | int],
        cfg: "SamplingConfig",
    ) -> Tuple[MVEventData, List[float], List[torch.Tensor]]:
        """Inverse-CDF sampling loop (the ``inversion`` mode of :meth:`sample`)."""
        # rng=None falls through to the device's global RNG (generator=None in
        # torch.rand/randint/multinomial) so repeated calls draw fresh samples.
        # Never substitute a fresh torch.Generator() here: it is default-seeded,
        # which would make every call replay the identical trajectory.
        if isinstance(rng, int):
            rng = torch.Generator(device=ts.time_points.device).manual_seed(rng)

        device = ts.time_points.device
        batched = BatchedMVEventData([ts.time_points], [ts.event_types])

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
            t_star = self.inverse_CDF(u.unsqueeze(0), batched)  # (1,)

            # Ensure the sampled time is after the last event (important for terminating processes
            # where inverse_CDF may not properly account for reduced marginal intensity).
            # The jitter (_INVERSION_JITTER) is a legacy band-aid retained for number
            # preservation; the root cause (reduced terminating-mode marginal) is
            # addressed with thinning.
            t_last = batched.max_time if hasattr(batched, "max_time") else batched.time_points.max()
            if t_star.item() < t_last.item():
                t_star = t_last + torch.rand(size=(), generator=rng, device=device) * _INVERSION_JITTER

            lam_vec = self.intensity(t_star, batched)[0]  # Shape: (D,)
            lam_sum = float(lam_vec.sum().item())

            if lam_sum <= cfg.degenerate_floor or not (lam_sum == lam_sum):
                # Degenerate case
                if self.terminating:
                    # For terminating: choose uniformly from remaining types
                    mask = self._get_termination_mask(batched)[0]  # (D,) - squeeze batch dim
                    remaining_indices = torch.where(mask)[0]
                    if len(remaining_indices) > 0:
                        random_idx = torch.randint(high=len(remaining_indices), size=(1,), generator=rng, device=device)
                        type_idx = remaining_indices[random_idx]
                    else:
                        # Should not happen as we check above, but handle gracefully
                        break
                else:
                    # Non-terminating: choose uniformly from all types
                    type_idx = torch.randint(high=self.D, size=(1,), generator=rng, device=device)
            else:
                probs = lam_vec / lam_sum
                type_idx = torch.multinomial(probs, num_samples=1, generator=rng)

            time_samples.append(t_star)
            dist_samples.append(lam_vec / lam_sum if lam_sum > 0 else torch.ones(self.D, device=device) / self.D)

            new_time_points = torch.cat([ts.time_points, t_star], dim=0)
            new_event_types = torch.cat([ts.event_types, type_idx], dim=0)
            ts = MVEventData(new_time_points, new_event_types)
            batched = BatchedMVEventData([ts.time_points], [ts.event_types])

        return ts, time_samples, dist_samples

    # =========================================================================
    # Ogata thinning sampler
    # =========================================================================
    #
    # Draws each next event by proposing candidate times from a dominating
    # homogeneous process of rate λ̄ ≥ Σ_d λ_d over a forward window and accepting
    # a candidate with probability λ_sum/λ̄; a rejected candidate advances the
    # search (standard Ogata).  Like ``sample`` this operates on a single sequence
    # (batch-of-1).

    #: Number of grid points used to scan for the upper-bound sup.
    _THINNING_SUP_POINTS = 64
    #: Forward-window size, in expected events at the local rate, for the sup scan.
    _THINNING_WINDOW_EVENTS = 10.0
    _THINNING_GLOBAL_WINDOW_EVENTS = 50.0

    def _sample_thinning(
        self,
        ts: MVEventData,
        num_steps: int,
        rng: Optional[torch.Generator | int],
        cfg: "SamplingConfig",
    ) -> Tuple[MVEventData, List[torch.Tensor], List[torch.Tensor]]:
        """Draw ``num_steps`` events by Ogata thinning.

        rng handling mirrors the inversion path: an ``int`` seeds a device
        ``Generator`` (deterministic), ``None`` falls through to the global RNG.
        """
        device = ts.time_points.device
        if isinstance(rng, int):
            rng = torch.Generator(device=device).manual_seed(rng)

        batched = BatchedMVEventData([ts.time_points], [ts.event_types])
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
            ts = MVEventData(
                torch.cat([ts.time_points, t_star], dim=0),
                torch.cat([ts.event_types, type_idx], dim=0),
            )
            batched = BatchedMVEventData([ts.time_points], [ts.event_types])

        return ts, time_samples, dist_samples

    def _thinning_upper_bound(
        self, batched: BatchedMVEventData, t: torch.Tensor, cfg: "SamplingConfig"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dominating rate λ̄ and the window ``[t, t+W]`` it bounds.

        Scans Σ_d λ_d on a grid over the window via the multi-query ``intensity``
        primitive (which also applies termination masking) and returns
        ``max × upper_bound_slack``.  ``W`` auto-scales with the local rate so a
        handful of expected events span it (``global_sup`` uses a wider window);
        grid points cluster near ``t`` so the post-event peak of an excitatory
        kernel is resolved.  The window edge is returned so the caller can
        re-bound when a proposal overshoots.  Caveat: for an intensity that spikes
        between grid points this sup is a heuristic — raise ``upper_bound_slack``.
        """
        lam_here = self.intensity(t, batched).sum(dim=1).clamp(min=cfg.degenerate_floor)  # (1,)
        events = (
            self._THINNING_GLOBAL_WINDOW_EVENTS if cfg.upper_bound == "global_sup" else self._THINNING_WINDOW_EVENTS
        )
        W = events / lam_here  # (1,)
        # Square the spacing so points concentrate near t (the sup of a decaying
        # post-event intensity sits just after t) while still covering the window.
        frac = torch.linspace(0.0, 1.0, self._THINNING_SUP_POINTS, device=t.device, dtype=t.dtype) ** 2  # (P,)
        query = t.unsqueeze(1) + W.unsqueeze(1) * frac.unsqueeze(0)  # (1, P)
        lam_bar = self.intensity(query, batched).sum(dim=-1).max() * cfg.upper_bound_slack  # scalar
        return lam_bar, t + W

    def _thin_next_event(
        self,
        batched: BatchedMVEventData,
        rng: Optional[torch.Generator],
        cfg: "SamplingConfig",
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Propose/accept one event after the last history time.

        Returns ``(t_star (1,), type_idx (1,), dist (D,))`` or ``None`` if no
        candidate is accepted within ``max_attempts`` (or no type remains).
        """
        t = batched.max_time  # (1,) current search position
        lam_bar, window_end = self._thinning_upper_bound(batched, t, cfg)

        for _ in range(cfg.max_attempts):
            if lam_bar.item() <= cfg.degenerate_floor:
                # Intensity is ~0 across the window; jump to its edge and re-bound.
                t = window_end
                lam_bar, window_end = self._thinning_upper_bound(batched, t, cfg)
                continue

            u1 = torch.rand(size=(), generator=rng, device=device)
            t_cand = t - torch.log(u1) / lam_bar  # Δ ~ Exp(λ̄); (1,)

            if (t_cand > window_end).item():
                # Left the window where λ̄ dominates; re-bound at its edge.
                t = window_end
                lam_bar, window_end = self._thinning_upper_bound(batched, t, cfg)
                continue

            lam_vec = self.intensity(t_cand, batched)[0]  # (D,), termination-masked
            lam_sum = lam_vec.sum()
            u2 = torch.rand(size=(), generator=rng, device=device)
            if (u2 * lam_bar <= lam_sum).item():  # accept w.p. λ_sum/λ̄
                type_idx, dist = self._draw_thinning_type(lam_vec, batched, rng, device, cfg)
                if type_idx is None:
                    return None
                return t_cand, type_idx, dist

            t = t_cand  # reject: advance the search and retry
        return None

    def _draw_thinning_type(
        self,
        lam_vec: torch.Tensor,
        batched: BatchedMVEventData,
        rng: Optional[torch.Generator],
        device: torch.device,
        cfg: "SamplingConfig",
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Draw the accepted event's type from λ/λ_sum.

        Mirrors the inversion path's degenerate handling: when λ_sum ≤
        ``degenerate_floor`` (or NaN) the type is drawn uniformly — among the
        remaining types for a terminating process, else over all types.  Returns
        ``(None, None)`` when a terminating process has no remaining type.
        """
        lam_sum = float(lam_vec.sum().item())
        if lam_sum <= cfg.degenerate_floor or lam_sum != lam_sum:  # degenerate or NaN
            if self.terminating:
                mask = self._get_termination_mask(batched)[0]  # (D,)
                remaining = torch.where(mask)[0]
                if remaining.numel() == 0:
                    return None, None
                pick = torch.randint(high=remaining.numel(), size=(1,), generator=rng, device=device)
                type_idx = remaining[pick]
            else:
                type_idx = torch.randint(high=self.D, size=(1,), generator=rng, device=device)
            dist = torch.ones(self.D, device=device) / self.D
        else:
            dist = lam_vec / lam_sum
            type_idx = torch.multinomial(dist, num_samples=1, generator=rng)
        return type_idx, dist


class DensityBasedTPP(TemporalPointProcess):
    """Root for inter-event-time density models — the native density realization.

    Realizes the predictive contract directly from a **mark-conditional** inter-event-time density
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
        seed: Optional[int] = 42,
        terminating: bool = False,
        sampling_config: Optional["SamplingConfig"] = None,
    ):
        # The SamplingConfig override + resolution plumbing live on the root
        # ``TemporalPointProcess`` (sampling is universal); forward the override up.
        super().__init__(D=D, seed=seed, terminating=terminating, sampling_config=sampling_config)

    def transform_params(self) -> Tuple[torch.Tensor, ...]:
        """Not meaningful for density TPPs (parameters depend on history)."""
        raise NotImplementedError(
            "Density-based TPP parameters are history-dependent; there is no static transform."
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

    def _last_event_time(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Padding-aware time of the last observed (real) event per sequence.  ``(B,)`` (0 for empty).

        Unlike ``ts.max_time`` this uses ``real_counts`` (the event mask), so a batch padded
        with dummy timestamps still measures the inter-event time from the true last event —
        keeping the density contract padding-invariant.
        """
        tp = ts.time_points  # (B, L)
        B = tp.shape[0]
        if tp.shape[1] == 0:
            return torch.zeros(B, dtype=tp.dtype, device=tp.device)
        tp0 = torch.cat([torch.zeros(B, 1, dtype=tp.dtype, device=tp.device), tp], dim=1)  # (B, L+1)
        return tp0.gather(1, ts.real_counts.unsqueeze(1)).squeeze(1)  # (B,) = t at index real_counts-1

    def _prep_query_time(self, t: torch.Tensor, ts: BatchedMVEventData, atol: float) -> torch.Tensor:
        """Cast to double, snap near-``t_last`` queries onto ``t_last``, assert ``t ≥ t_last``.

        Mirrors :meth:`IntensityBasedTPP._prep_query_time` but keys on the padding-aware
        last-event time; handles ``(B,)`` and ``(B, Q)``.
        """
        t = t.double()
        mx = self._last_event_time(ts)  # (B,)
        mx_b = mx.unsqueeze(1) if t.dim() == 2 else mx
        close = torch.abs(t - mx_b) < atol
        t = torch.where(close, mx_b.expand_as(t), t)
        assert (t >= mx_b).all(), "predictive densities are only defined for t >= last event time."
        return t

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

    def density_joint(self, t: torch.Tensor, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """p(t, m | H) = p(m | H) · p(t | m, H).  ``(B,[Q,]D)``."""
        t = self._prep_query_time(t, ts, atol=_SNAP_ATOL_DENSITY)
        lj = self._log_density_joint(t, ts)
        return lj if log else lj.exp()

    def density_time(self, t: torch.Tensor, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """p(t | H) = Σ_m p(m | H)·p(t | m, H) — mark-marginal, a finite sum (no integration).  ``(B,[Q])``."""
        t = self._prep_query_time(t, ts, atol=_SNAP_ATOL_DENSITY)
        lt = torch.logsumexp(self._log_density_joint(t, ts), dim=-1)  # (B,[Q])
        return lt if log else lt.exp()

    def survival_time(self, t: torch.Tensor, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """S(t) = Σ_m p(m | H)·S(t | m, H).  ``(B,[Q])``."""
        t = self._prep_query_time(t, ts, atol=_SNAP_ATOL_CDF)
        ls = self._log_survival_marginal(t, ts)
        return ls if log else ls.exp()

    def cdf_time(self, t: torch.Tensor, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """P(t_next < t | H) = 1 − S(t).  ``(B,[Q])``."""
        t = self._prep_query_time(t, ts, atol=_SNAP_ATOL_CDF)
        ls = self._log_survival_marginal(t, ts)
        if not log:
            return -torch.expm1(ls)  # 1 − exp(log S), stable
        return _log1mexp(ls)

    def prob_mark(self, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """p(m | H).  ``(B, D)``."""
        lp = self._log_prob_mark(ts)
        return lp if log else lp.exp()

    def prob_mark_given_time(self, t: torch.Tensor, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """p(m | t, H) = p(t, m | H) / p(t | H).  ``(B,[Q,]D)``.

        For an independent-mark model this reduces to ``p(m | H)`` (the per-mark densities
        are equal, so they cancel); the general form is kept so a time-dependent mark falls
        out automatically.
        """
        t = self._prep_query_time(t, ts, atol=_SNAP_ATOL_DENSITY)
        lj = self._log_density_joint(t, ts)  # (B,[Q,]D)
        lt = torch.logsumexp(lj, dim=-1, keepdim=True)  # (B,[Q,]1) = log p(t | H)
        # A degenerate joint (no next event, all −inf) gives −inf − −inf = nan; define zero mass.
        lr = torch.where(torch.isneginf(lt).expand_as(lj), lj, lj - lt)
        return lr if log else lr.exp()

    def density_time_given_mark(self, t: torch.Tensor, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """p(t | m, H) — the rooted per-mark inter-event density.  ``(B,[Q,]D)``."""
        t = self._prep_query_time(t, ts, atol=_SNAP_ATOL_DENSITY)
        lr = self._log_density_time_given_mark(t, ts)
        return lr if log else lr.exp()

    def likelihood(
        self, ts: BatchedMVEventData, T: Optional[torch.Tensor] = None, log: bool = True
    ) -> torch.Tensor:
        """Log-likelihood: ``Σ_i [log p(m_i | H_i) + log p(t_i | m_i, H_i)] + log S(τ_censor)``.  ``(B,)``.

        The observed-event term and the censoring survival term are both native (no
        derived hazard); for IFTPP this equals the harness likelihood up to fp — the
        per-interval ``+log S`` / ``−log S`` cancel.
        """
        if T is None:
            T = ts.T_obs
        assert (T >= self._last_event_time(ts)).all(), "Likelihood is only defined for T >= last event time."
        log_lik = self._event_log_density_sum(ts) + self._log_survival_marginal(T, ts)  # (B,)
        return log_lik if log else torch.exp(log_lik)

    def time_quantile(self, u, ts: BatchedMVEventData) -> torch.Tensor:
        """Inverse time CDF via bisection on ``cdf_time`` over ``[t_last, hi]`` (median = ``u=0.5``).  ``(B,)``.

        The density-family quantile: grow ``hi`` until ``cdf_time(hi) ≥ max(u)`` (capped),
        then bisect.  ``u`` is a scalar or ``(B,)`` and is capped at
        ``SamplingConfig.quantile_cap`` so ``u → 1`` cannot grow the bracket without bound.
        """
        device = ts.time_points.device
        t_last = self._last_event_time(ts).double()  # (B,) padding-aware last real event
        B = t_last.shape[0]
        if not torch.is_tensor(u):
            u = torch.full((B,), float(u), device=device, dtype=torch.float64)
        u = u.to(device=device, dtype=torch.float64).clamp(max=self.sampling_config.quantile_cap)

        lo = t_last.clone()
        hi = t_last + _QUANTILE_INIT_BRACKET_WIDTH
        for _ in range(_QUANTILE_BRACKET_GROW_ITERS):  # grow the bracket until it covers the largest requested quantile
            need = self.cdf_time(hi, ts, log=False) < u
            if not bool(need.any()):
                break
            hi = torch.where(need, t_last + (hi - t_last) * 2.0, hi)
        # A defective / saturated survival (CDF never reaches u — e.g. a fully terminated
        # history, or mass at infinity) has no finite quantile: report +inf rather than the
        # meaningless grown-bracket endpoint (~1e19).  Callers (eval) drop non-finite predictions.
        unreached = self.cdf_time(hi, ts, log=False) < u  # (B,)
        for _ in range(_QUANTILE_BISECTION_ITERS):  # bisection
            mid = 0.5 * (lo + hi)
            go_right = self.cdf_time(mid, ts, log=False) < u
            lo = torch.where(go_right, mid, lo)
            hi = torch.where(go_right, hi, mid)
        result = 0.5 * (lo + hi)
        return torch.where(unreached, torch.full_like(result, float("inf")), result)

    # =========================================================================
    # Sampling — direct draw from the density + mark head
    # =========================================================================

    def sample(
        self,
        ts: MVEventData,
        num_steps: int = 1,
        rng: Optional[torch.Generator | int] = None,
        sampling_config: Optional["SamplingConfig"] = None,
    ) -> Tuple[MVEventData, List[torch.Tensor], List[torch.Tensor]]:
        """Generate events by drawing each next event directly from the density + mark head.

        The tuple shape mirrors :meth:`IntensityBasedTPP.sample`: ``(events, times,
        dists)`` where ``times`` are the sampled absolute-time ``(1,)`` tensors and
        ``dists`` the per-step ``(D,)`` mark distributions.  For terminating processes,
        sampling stops once all types have occurred and never repeats a type.
        """
        cfg = sampling_config if sampling_config is not None else self.sampling_config
        with self.use_sampling(cfg):
            return self._sample_density(ts, num_steps=num_steps, rng=rng)

    def _sample_density(
        self,
        ts: MVEventData,
        num_steps: int,
        rng: Optional[torch.Generator | int],
    ) -> Tuple[MVEventData, List[torch.Tensor], List[torch.Tensor]]:
        # rng=None -> global RNG (fresh each call); an int seeds a device Generator.
        if isinstance(rng, int):
            rng = torch.Generator(device=ts.time_points.device).manual_seed(rng)

        time_samples: List[torch.Tensor] = []
        dist_samples: List[torch.Tensor] = []
        for _ in range(num_steps):
            if self.terminating:
                batched = BatchedMVEventData([ts.time_points], [ts.event_types])
                if self._get_termination_mask(batched)[0].sum().item() == 0:
                    break  # all types have occurred
            drawn = self._draw_next(ts, rng)
            if drawn is None:
                break
            t_next, type_idx, dist = drawn
            time_samples.append(t_next)
            dist_samples.append(dist)
            ts = MVEventData(
                torch.cat([ts.time_points, t_next.reshape(1)], dim=0),
                torch.cat([ts.event_types, type_idx.reshape(1)], dim=0),
            )
        return ts, time_samples, dist_samples

    # =========================================================================
    # Legacy-name delegators (keep existing call-sites/tests working)
    # =========================================================================

    def PDF(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Deprecated alias for :meth:`density_joint` (``log=False``): p(t, m | H)."""
        return self.density_joint(T, ts, log=False)

    def CDF(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Deprecated alias for :meth:`cdf_time` (``log=False``): P(t_next < T | H)."""
        return self.cdf_time(T, ts, log=False)

    def marginal_class_distribution(
        self, ts: BatchedMVEventData, T_max: Optional[torch.Tensor] = None, num_points: int = _GRID_NUM_POINTS
    ) -> torch.Tensor:
        """Deprecated alias for :meth:`prob_mark` (``log=False``): p(m | H)."""
        return self.prob_mark(ts, log=False)

    def mean_next_event_time(
        self, ts: BatchedMVEventData, T_max: Optional[torch.Tensor] = None, num_points: int = _GRID_NUM_POINTS
    ) -> torch.Tensor:
        """Deprecated alias for :meth:`time_mean`: E[t_next | H]."""
        return self.time_mean(ts)

    def inverse_CDF(self, u: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Deprecated alias for :meth:`time_quantile`: t such that P(t_next < t | H) = u."""
        return self.time_quantile(u, ts)
