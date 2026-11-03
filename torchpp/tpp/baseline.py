"""Baseline-only (history-free) point processes.

The intensity depends on time only, never on past events: homogeneous Poisson
(``PoissonProcess``), log-linear (``LogLinearPoissonProcess``), linear-spline
(``SplinePoissonProcess``) and piecewise-constant (``PiecewisePoissonProcess``) rates.
All have a closed-form cumulative intensity, quantile and sampler. They are lower bounds for
history-dependent models and the baseline part of the Hawkes models.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import torch

from ..utils.event_data import MVEventData, BatchedMVEventData, DatasetStats, TokenRoles, extend_T_obs
from ..utils.math import LinearSpline, PiecewiseConstant
from . import _inspection
from .base_tpp import CIConfig, IntensityBasedTPP, Sample, SamplingConfig
from ..modules.intensity import (
    BaselineIntensityModule,
    ConstantBaselineModule,
    ConstantBaselineParams,
    LogLinearBaselineModule,
    LogLinearBaselineParams,
    SplineBaselineModule,
    SplineBaselineParams,
    PiecewiseConstantBaselineModule,
    PiecewiseConstantBaselineParams,
)


class BaselineTPP(IntensityBasedTPP, ABC):
    """Abstract base for TPPs whose intensity is provided by a single
    :class:`~torchpp.modules.intensity.BaselineIntensityModule`.

    Subclasses must:
    - Assign ``self.baseline`` (a :class:`BaselineIntensityModule`) in ``__init__``
      before calling any of the methods below.
    - Call ``super().__init__(D, seed, terminating=...)`` (closed-form CI by default).

    The core intensity hooks are forwarded to the module:

    * ``transform_params``       → ``self.baseline.transform_params``
    * ``_intensities_at_queries``→ ``self.baseline.intensities_at_queries``
    * ``_closed_form_ci``        → ``self.baseline._cumulative_intensity``
    * ``_event_intensities``→ ``self.baseline.event_intensities``

    Any of these can be overridden in a subclass when the process requires
    additional logic (e.g. the terminating spline/piecewise fallback).
    """

    baseline: BaselineIntensityModule


    def baseline_values(self, t) -> torch.Tensor:
        """Intensity ``μ_d(t)`` at native times ``t`` ``(Q,)``, in native units, before termination.  ``(Q, D)``."""
        return _inspection.baseline_values(self, t)

    def transform_params(self) -> Tuple[torch.Tensor, ...]:
        """Return constrained parameters — delegates to the baseline module."""
        return self.baseline.transform_params()

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Flat multi-query raw baseline intensity, vectorised over queries. ``(B, Q)`` -> ``(B, Q, D)``.

        Delegates to :meth:`BaselineIntensityModule.intensities_at_queries`.
        """
        return self.baseline.intensities_at_queries(query_times, ts)

    def _closed_form_ci(self, ts: BatchedMVEventData, T_low, T: torch.Tensor) -> torch.Tensor:
        """Efficient closed-form CI via the global baseline module CI.

        The baseline is history-free, so the global ``_cumulative_intensity`` is the
        cheapest form; windowing is exact subtraction.  Termination folds into the
        per-type ``_effective_T``.
        """
        ci = self.baseline._cumulative_intensity(self._effective_T(T, ts), ts)
        if T_low is not None:
            ci = ci - self.baseline._cumulative_intensity(self._effective_T(T_low, ts), ts)
        return ci

    def _event_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Intensity of the observed event type at each sequence position. Shape: (B, L).

        Delegates to :meth:`BaselineIntensityModule.event_intensities`.
        Subclasses that need termination-aware gathering (e.g. spline/piecewise
        variants) should override this method.
        """
        return self.baseline.event_intensities(ts)


@dataclass
class PoissonProcessParams:
    mu: torch.Tensor  # Shape: (D,)


class PoissonProcess(BaselineTPP):
    """Homogeneous Poisson process: a constant rate per event type, independent of the history.

    .. math:: \\lambda_d(t) = \\mu_d, \\qquad \\Lambda_d(a, b) = \\mu_d\\,(b - a)

    The next event time is exponential with rate :math:`\\sum_d \\mu_d` and its type is
    :math:`d` with probability :math:`\\mu_d / \\sum_{d'} \\mu_{d'}`; quantiles and sampling
    are closed-form. A natural lower bound for history-dependent models.

    Args:
        D: Number of event types.
        params: Initial rates; random (or data-driven) if ``None``.
        seed: Seed for the parameter initialisation.
        terminating: Each event type occurs at most once per sequence.
        dataset_stats: Statistics of the training data; initialises :math:`\\mu` from the
            observed rates.
    """

    def __init__(
        self,
        D: Optional[int] = None,
        *,
        params: Optional[PoissonProcessParams] = None,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        if isinstance(D, PoissonProcessParams):
            raise TypeError("Pass parameters as params=...; the first argument is D.")
        if params is None and D is None:
            raise ValueError("Either params or D (number of dimensions) must be provided.")

        if D is None:
            D = params.mu.shape[0]  # type: ignore[union-attr]

        super().__init__(
            D,
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )

        baseline_params = ConstantBaselineParams(mu=params.mu) if params is not None else None

        self.baseline = ConstantBaselineModule(
            D=D,
            params=baseline_params,
            seed=seed,
            dataset_stats=dataset_stats,
        )

    # ------------------------------------------------------------------
    # CI inversion — kept on the TPP because it depends on termination state
    # ------------------------------------------------------------------

    def _inverse_marginal_cumulative_intensity(
        self,
        u: torch.Tensor,
        ts: BatchedMVEventData,
    ) -> torch.Tensor:
        """
        Invert the marginal cumulative intensity to find t such that Λ_marginal(t) = u.

        For terminating processes, the marginal intensity only includes types that
        haven't occurred yet, and the cumulative intensity accounts for termination:
            Λ_marginal(t) = Σ_d μ_d * min(t, t_term_d)

        For non-terminating: Λ_marginal(t) = (Σ_d μ_d) * t, so t = u / Σ_d μ_d

        Args:
            u: Target marginal cumulative intensity. Shape: (B,)
            ts: Historical events. Shape: (B, N)

        Returns:
            Time t such that Λ_marginal(t) = u. Shape: (B,)
        """
        if u.shape[0] != ts.time_points.shape[0]:
            raise ValueError("u must have one entry per sequence.")

        device = ts.time_points.device
        (mu,) = self.transform_params()  # (D,)
        mu = self._zero_context_cols(mu)  # context types do not compete for the next event
        lamb = mu.sum()
        B = u.shape[0]

        if not self.terminating:
            # Simple case: Λ(t) = (Σ μ_d) * t => t = u / Σ μ_d
            return u / lamb

        # For terminating processes, we need to handle the piecewise nature.
        # Λ_marginal(t) = Σ_d μ_d * min(t, t_term_d)
        # This is piecewise linear with breakpoints at each t_term_d.

        termination_times = self._get_termination_times(ts)  # (B, D)

        # Batch elements with no terminations yet (all inf) use the simple formula.
        no_terminations = torch.isinf(termination_times).all(dim=1)  # (B,)
        t_simple = u / lamb  # (B,)

        if no_terminations.all():
            return t_simple

        # For batch elements with some terminations, we need the piecewise approach.
        # To avoid inf - inf = nan, replace inf with a large finite value that's
        # larger than any realistic t we'll need to compute.
        # The max CI we might need to invert is u.max(), so max t ≈ u.max() / min(μ)
        # Use a conservative upper bound.
        max_t_needed = (u.max() / mu[self.predictable_mask].min() + 1.0).detach()
        termination_times_finite = torch.where(
            torch.isinf(termination_times), torch.full_like(termination_times, max_t_needed.item()), termination_times
        )

        # Strategy: Sort termination times, then solve piecewise.
        # Before any termination: slope = Σ μ_d
        # After t_term_d: slope decreases by μ_d

        sorted_terms, sort_indices = torch.sort(termination_times_finite, dim=1)  # (B, D)
        mu_sorted = mu[sort_indices]  # (B, D)

        # Cumulative sum of mu in reverse order gives remaining intensity:
        # mu_cumsum_rev[:, k] = Σ_{j>=k} μ_sorted[:, j] = slope in segment k
        mu_cumsum_rev = torch.flip(torch.cumsum(torch.flip(mu_sorted, [1]), dim=1), [1])  # (B, D)

        # Prepend 0 to sorted_terms for the segment from 0 to first breakpoint
        zeros = torch.zeros(B, 1, device=device, dtype=sorted_terms.dtype)
        breakpoints = torch.cat([zeros, sorted_terms], dim=1)  # (B, D+1)

        segment_lengths = breakpoints[:, 1:] - breakpoints[:, :-1]  # (B, D)
        slopes = mu_cumsum_rev  # (B, D)
        ci_increments = slopes * segment_lengths  # (B, D)

        # Cumulative CI at end of each segment; ci_at_start prepends 0 for the first segment.
        ci_cumsum = torch.cumsum(ci_increments, dim=1)  # (B, D)
        ci_at_start = torch.cat([zeros, ci_cumsum[:, :-1]], dim=1)  # (B, D)

        # u falls in segment k if ci_at_start[:, k] <= u < ci_at_start[:, k] + ci_increments[:, k]
        segment_idx = torch.searchsorted(ci_cumsum, u.unsqueeze(1)).squeeze(1)  # (B,)
        segment_idx = segment_idx.clamp(max=self.D - 1)

        batch_indices = torch.arange(B, device=device)

        ci_start = ci_at_start[batch_indices, segment_idx]  # (B,)
        slope = slopes[batch_indices, segment_idx]  # (B,)
        t_start = breakpoints[batch_indices, segment_idx]  # (B,)

        # Solve: ci_start + slope * (t - t_start) = u
        # => t = t_start + (u - ci_start) / slope
        # Handle zero slope (all types terminated) - return inf
        safe_slope = torch.where(slope > 1e-12, slope, torch.ones_like(slope))
        delta_t = (u - ci_start) / safe_slope
        delta_t = torch.where(slope > 1e-12, delta_t, torch.full_like(delta_t, float("inf")))

        t_star = t_start + delta_t

        # If all types terminated and u exceeds max achievable CI, return inf.
        all_terminated = ~torch.isinf(termination_times).any(dim=1)
        max_ci_when_all_terminated = (mu.unsqueeze(0) * termination_times).sum(dim=1)

        exceeds_max = all_terminated & (u > max_ci_when_all_terminated)
        t_star = torch.where(exceeds_max, torch.full_like(t_star, float("inf")), t_star)

        t_star = torch.where(no_terminations, t_simple, t_star)

        return t_star

    @torch.no_grad()
    def sample(
        self,
        ts: Optional[MVEventData] = None,
        num_steps: int = 1,
        rng: Optional[torch.Generator | int] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ) -> Sample:
        """Closed-form sampling for homogeneous Poisson processes; returns a :class:`Sample`.

        The inter-arrival times follow an exponential distribution, so exact
        inversion sampling is used instead of the general base-class method.  This
        is the ``inversion_method="closed_form"`` path; ``thinning`` delegates to the
        base Ogata sampler (no closed-form shortcut).

        For terminating processes, sampling stops when all event types have occurred.

        Args:
            ts (MVEventData): Conditioned past events.
            num_steps (int, optional): Number of steps for generation (for terminating: min(num_steps, remaining types)). Defaults to 1.
            rng (Optional[torch.Generator  |  int], optional): Random state. Defaults to None.
            sampling_config: Optional per-call SamplingConfig override (only the draw
                ``mode`` is consulted; this closed-form sampler ignores the inversion knobs).
        """
        ts = self._resolve_sample_history(ts)
        cfg = sampling_config if sampling_config is not None else self.sampling_config
        if cfg.mode == "thinning":
            # Ogata thinning has no closed-form shortcut — use the base implementation.
            return super().sample(ts, num_steps=num_steps, rng=rng, sampling_config=cfg)

        # Native shell, as in :meth:`IntensityBasedTPP.sample`: ``transform_params`` returns the
        # rate of the SCALED core, so the inversion below must run on a scaled history and its
        # absolute times be de-scaled on the way out.  Inert at time_scale == 1; without it every
        # sampled time is off by 1 / time_scale.
        ts = self._to_internal_time(ts)
        T_obs = getattr(ts, "T_obs", None)

        step = 0
        time_samples = []
        dist_samples = []

        (mu,) = self.transform_params()
        mu = self._zero_context_cols(mu)  # only predictable types are drawn
        device = ts.time_points.device

        # rng=None falls through to the global RNG (fresh samples per call);
        # a fresh torch.Generator() is default-seeded and would replay the
        # identical trajectory on every call. The generator must live on the
        # data device — a CPU generator fails on CUDA multinomial/rand calls.
        if isinstance(rng, int):
            rng = torch.Generator(device=device).manual_seed(rng)

        for step in range(0, num_steps):
            if self.terminating:
                batched = BatchedMVEventData([ts.time_points], [ts.event_types])
                mask = self._get_termination_mask(batched)[0]  # (D,)
                remaining_types = mask.sum().item()
                if remaining_types == 0:
                    break
                active_mu = mu * mask.float()
            else:
                active_mu = mu

            t_last = ts.time_points[-1].item() if ts.time_points.numel() > 0 else 0.0
            if T_obs is not None:
                t_last = max(t_last, float(T_obs))  # forecast from the end of observation

            u = torch.rand(size=(), generator=rng, device=device).item()

            # Rate of the superimposed process is the sum of per-type rates.
            lamb = active_mu.sum()

            if lamb <= 1e-12:
                break

            CI_at_last = (active_mu * t_last).sum().item()

            # Invert the exponential CDF: u = 1 - exp(-lambda·Delta_t)  => Delta_t = -log(1-u)/lambda.
            # Since CI is tracked in absolute time, shift by the CI accumulated up to t_last.
            t_star = (-torch.log1p(-torch.tensor(u)).item() + CI_at_last) / lamb

            probs = active_mu / lamb
            type_idx = torch.multinomial(probs, num_samples=1, generator=rng)

            time_samples.append(t_star)
            dist_samples.append(probs)

            T_tensor = t_star.detach().reshape(1).to(dtype=ts.time_points.dtype, device=device)

            new_time_points = torch.cat([ts.time_points, T_tensor], dim=0)
            new_event_types = torch.cat([ts.event_types, type_idx], dim=0)

            ts = MVEventData(new_time_points, new_event_types, T_obs=extend_T_obs(T_obs, new_time_points))

        # De-scale back to native units.  Done inline rather than via ``_to_native_sample`` so the
        # float entries of ``time_samples`` stay floats (this sampler's documented shape).
        scale = float(self.time_scale)
        if scale != 1.0:
            ts = MVEventData(
                ts.time_points * scale, ts.event_types, T_obs=None if T_obs is None else float(ts.T_obs) * scale
            )
            time_samples = [t * scale for t in time_samples]
        return Sample(ts, time_samples, dist_samples)


@dataclass
class LogLinearPoissonParams:
    mu: torch.Tensor  # (D,) log-rate intercepts
    slope: torch.Tensor  # (D,) log-rate slopes


class LogLinearPoissonProcess(BaselineTPP):
    """Inhomogeneous Poisson process with log-linear intensity λ_d(t) = exp(μ_d + slope_d·t).

    Delegates all intensity math to a :class:`LogLinearBaselineModule`.
    """

    def __init__(
        self,
        D: Optional[int] = None,
        *,
        params: Optional[LogLinearPoissonParams] = None,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """Initialize the log-linear inhomogeneous Poisson process.

        Args:
            params: Process parameters or None to initialize randomly.
            D: Event dimension (required if params is None).
            seed: Random seed for initialization.
            terminating: If True, each event type can only occur once.
            dataset_stats: Optional empirical statistics.  When provided, the
                log-intercept mu is initialised at log(mean_rate) so the initial
                intensity matches the empirically observed per-type rate.
        """
        if isinstance(D, LogLinearPoissonParams):
            raise TypeError("Pass parameters as params=...; the first argument is D.")
        if params is None and D is None:
            raise ValueError("Either params or D (number of dimensions) must be provided.")

        if D is None:
            D = params.mu.shape[0]  # type: ignore[union-attr]

        super().__init__(
            D,
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )

        baseline_params = LogLinearBaselineParams(mu=params.mu, slope=params.slope) if params is not None else None

        self.baseline = LogLinearBaselineModule(
            D=D,
            params=baseline_params,
            seed=seed,
            dataset_stats=dataset_stats,
        )



@dataclass
class SplineProcessParams:
    h_knots: torch.Tensor  # Shape: (D, num_knots)
    # Shape: (num_knots,). Knot locations are cumsum(delta_ts), so the first
    # entry must be 0 (knots start at t=0); entry i>0 is the gap to knot i-1.
    delta_ts: torch.Tensor


class SplinePoissonProcess(BaselineTPP):
    """Inhomogeneous Poisson process with piecewise-linear (linear spline) intensity.

    Delegates all intensity math to a :class:`SplineBaselineModule`. The TPP
    layer retains the analytical CI inversion, which depends on termination state.
    """

    def __init__(
        self,
        D: int,
        *,
        baseline_num_knots: int,
        baseline_delta_t: torch.Tensor | float,
        params: Optional[SplineProcessParams] = None,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """Build the model.

        Args:
            D: Number of event types.
            baseline_num_knots: Number of spline knots.
            baseline_delta_t: Spacing between knots — scalar for uniform spacing, or a
                tensor of per-knot gaps whose first entry must be 0
                (knot locations = cumsum(baseline_delta_t), starting at t=0).
            params: Pre-computed parameters.  If given, dataset_stats is ignored.
            seed: Random seed.
            terminating: If True, each event type can occur at most once.
            dataset_stats: Optional empirical statistics.  When provided, knot
                heights are initialised near the per-type observed rate.
        """
        super().__init__(
            D,
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )

        baseline_params: Optional[SplineBaselineParams] = (
            SplineBaselineParams(h_knots=params.h_knots, delta_ts=params.delta_ts) if params is not None else None
        )

        self.baseline = SplineBaselineModule(
            D=D,
            num_knots=baseline_num_knots,
            delta_t=baseline_delta_t,
            params=baseline_params,
            seed=seed,
            dataset_stats=dataset_stats,
        )
        self.num_knots = self.baseline.num_knots

    # ------------------------------------------------------------------
    # Accessors and overrides
    # ------------------------------------------------------------------

    def _get_heights(self) -> Tuple[torch.Tensor, ...]:
        """Non-negative knot heights in internal time units, ``((D, num_knots),)``."""
        return self.baseline.get_heights()

    def _event_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Intensity of the observed event type at each position.  Shape: (B, L).

        For the non-terminating case the module uses a batched flatten+view path
        (efficient since intensity is history-independent).  The terminating case
        uses the base-class default (``_sequence_intensities`` → gather); a repeated
        type is impossible and is handled by the likelihood's terminating gate.
        """
        if self.terminating:
            # Fall back to base-class default: _sequence_intensities → gather.
            return super()._event_intensities(ts)
        return self.baseline.event_intensities(ts)

    # ------------------------------------------------------------------
    # CI inversion — kept on the TPP because it depends on termination state
    # ------------------------------------------------------------------

    def _inverse_marginal_cumulative_intensity(
        self,
        u: torch.Tensor,
        ts: BatchedMVEventData,
    ) -> torch.Tensor:
        """Invert the marginal CI to find t such that Λ(t) = u.

        For terminating processes delegates to the base-class numerical root
        finder (termination breakpoints make this non-trivial analytically).
        For non-terminating processes collapses the spline across dimensions
        and uses the closed-form spline inverse.

        Args:
            u: Target marginal CI values. Shape: (B,)
            ts: Historical events. Shape: (B, L)

        Returns:
            Absolute time t such that Λ_marginal(t) = u. Shape: (B,)
        """
        if u.shape[0] != ts.time_points.shape[0]:
            raise ValueError("u must have one entry per sequence.")

        if self.terminating:
            return super()._inverse_marginal_cumulative_intensity(u, ts)

        (h,) = self._get_heights()  # (D, K)
        h_marginal = self._zero_context_cols(h.T).T.sum(dim=0, keepdim=True)  # (1, K) — over predictable types
        t_next = LinearSpline.inverse_integral(
            self.baseline.knot_locs,
            h_marginal,
            u.unsqueeze(1),  # (B, 1)
        )  # (B, 1)
        return t_next.squeeze(1)  # (B,)


class PiecewisePoissonProcess(BaselineTPP):
    """Inhomogeneous Poisson process with piecewise-constant intensity.

    Between consecutive knots the intensity is constant (equal to the left
    knot height).  More flexible than a homogeneous Poisson process but
    simpler than a spline process — no slope parameters.

    Intensity:          λ_d(t) = h_k   for x_k ≤ t < x_{k+1}
    Cumulative intens.: Λ_d(T) = Σ_{j<k} h_j Δx_j  +  h_k (T - x_k)

    Delegates all intensity math to a :class:`PiecewiseConstantBaselineModule`.
    """

    def __init__(
        self,
        D: int,
        *,
        baseline_num_knots: int,
        baseline_delta_t: torch.Tensor | float,
        params: Optional[SplineProcessParams] = None,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """Build the model.

        Args:
            D: Number of event types.
            baseline_num_knots: Number of piecewise-constant bins.
            baseline_delta_t: Bin width — scalar for uniform bins, or a tensor of
                per-edge gaps whose first entry must be 0
                (left bin edges = cumsum(baseline_delta_t), starting at t=0).
            params: Pre-computed parameters.  If given, dataset_stats is ignored.
            seed: Random seed.
            terminating: If True, each event type can occur at most once.
            dataset_stats: Optional empirical statistics.  When provided, bin
                heights are initialised near the per-type observed rate.
        """
        super().__init__(
            D,
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )

        baseline_params: Optional[PiecewiseConstantBaselineParams] = (
            PiecewiseConstantBaselineParams(h_knots=params.h_knots, delta_ts=params.delta_ts)
            if params is not None
            else None
        )

        self.baseline = PiecewiseConstantBaselineModule(
            D=D,
            num_knots=baseline_num_knots,
            delta_t=baseline_delta_t,
            params=baseline_params,
            seed=seed,
            dataset_stats=dataset_stats,
        )
        self.num_knots = self.baseline.num_knots

    # ------------------------------------------------------------------
    # Accessors and overrides
    # ------------------------------------------------------------------

    def _get_heights(self) -> Tuple[torch.Tensor, ...]:
        """Non-negative bin heights in internal time units, ``((D, num_knots),)``."""
        return self.baseline.get_heights()

    def _event_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Intensity of the observed event type at each position.  Shape: (B, L).

        For the non-terminating case the module uses a batched flatten+view path.
        The terminating case uses the base-class default (``_sequence_intensities``
        → gather); a repeated type is impossible and is handled by the likelihood's
        terminating gate.
        """
        if self.terminating:
            return super()._event_intensities(ts)
        return self.baseline.event_intensities(ts)

    # ------------------------------------------------------------------
    # CI inversion — kept on the TPP because it depends on termination state
    # ------------------------------------------------------------------

    def _inverse_marginal_cumulative_intensity(
        self,
        u: torch.Tensor,
        ts: BatchedMVEventData,
    ) -> torch.Tensor:
        """Invert the marginal CI: find t such that Λ_marginal(t) = u.

        For terminating processes delegates to the base-class numerical root
        finder.  For non-terminating processes collapses bins across dimensions
        and uses the closed-form piecewise-constant inverse.

        Args:
            u: Target marginal CI values. Shape: (B,)
            ts: Historical events. Shape: (B, L)

        Returns:
            Absolute time t such that Λ_marginal(t) = u. Shape: (B,)
        """
        if u.shape[0] != ts.time_points.shape[0]:
            raise ValueError("u must have one entry per sequence.")

        if self.terminating:
            return super()._inverse_marginal_cumulative_intensity(u, ts)

        (h,) = self._get_heights()  # (D, K)
        h_marginal = self._zero_context_cols(h.T).T.sum(dim=0, keepdim=True)  # (1, K) — over predictable types
        t_next = PiecewiseConstant.inverse_integral(
            self.baseline.knot_locs,
            h_marginal,
            u.unsqueeze(1),  # (B, 1)
        )  # (B, 1)
        return t_next.squeeze(1)  # (B,)
