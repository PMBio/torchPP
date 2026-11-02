from abc import ABC
from dataclasses import dataclass
from typing import Optional, Tuple
import torch

from ..utils.event_data import MVEventData, BatchedMVEventData, DatasetStats
from ..utils.math import LinearSpline, PiecewiseConstant
from .base_tpp import IntensityBasedTPP, SamplingConfig
from ..modules.intensity import (
    BaselineIntensityModule,
    ConstantBaselineModule,
    ConstantBaselineParams,
    LinearBaselineModule,
    LinearBaselineParams,
    SplineBaselineModule,
    SplineBaselineParams,
    PiecewiseConstantBaselineModule,
    PiecewiseConstantBaselineParams,
)


class BaselineTPP(IntensityBasedTPP, ABC):
    """Abstract base for TPPs whose intensity is provided by a single
    :class:`~torchpp.intensity_modules.BaselineIntensityModule`.

    Subclasses must:
    - Assign ``self.baseline`` (a :class:`BaselineIntensityModule`) in ``__init__``
      before calling any of the methods below.
    - Call ``super().__init__(D, seed, terminating=...)`` (closed-form CI by default).

    The three core TPP abstractions are forwarded to the module:

    * ``transform_params``  → ``self.baseline.transform_params``
    * ``_intensity_at_t``   → ``self.baseline.intensity``
    * ``_raw_ci``           → ``self.baseline._cumulative_intensity``
    * ``_event_intensities``→ ``self.baseline.event_intensities``

    Any of these can be overridden in a subclass when the process requires
    additional logic (e.g. the terminating spline/piecewise fallback).
    """

    baseline: BaselineIntensityModule

    def transform_params(self) -> Tuple[torch.Tensor, ...]:
        """Return constrained parameters — delegates to the baseline module."""
        return self.baseline.transform_params()

    def _intensity_at_t(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Raw intensity at time *t* (before termination masking). Shape: (B, D).

        Delegates to :meth:`BaselineIntensityModule.intensity`.
        """
        return self.baseline.intensity(t, ts)

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Flat multi-query raw baseline intensity, vectorised over queries. ``(B, Q)`` -> ``(B, Q, D)``.

        Delegates to :meth:`BaselineIntensityModule.intensities_at_queries`.
        """
        return self.baseline.intensities_at_queries(query_times, ts)

    def _raw_ci(self, effective_T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Analytical cumulative intensity up to *effective_T*. Shape: (B, D).

        ``effective_T`` is already clamped to ``min(T, t_term_d)`` for terminating
        processes by the base class before this method is called.
        Delegates to :meth:`BaselineIntensityModule._cumulative_intensity`.
        """
        return self.baseline._cumulative_intensity(effective_T, ts)

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
    """Homogeneous Poisson process with constant intensity per event type.

    Delegates all intensity math to a :class:`ConstantBaselineModule`. The TPP
    layer retains the closed-form sampler and the analytical CI inversion, both
    of which depend on termination state and therefore cannot live in the module.
    """

    def __init__(
        self,
        params: Optional[PoissonProcessParams] = None,
        D: Optional[int] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """Initialize the Poisson process.

        Args:
            params: Poisson process parameters or None to initialize randomly.
            D: Event dimension (required if params is None).
            seed: Random seed for initialization.
            terminating: If True, each event type can only occur once.
            dataset_stats: Optional empirical statistics.  When provided, mu is
                initialised near the per-type observed rate rather than a fixed
                random range.
        """
        if params is None and D is None:
            raise ValueError("Either params or D (number of dimensions) must be provided.")

        if D is None:
            D = params.mu.shape[0]  # type: ignore[union-attr]

        super().__init__(D, seed, terminating=terminating)

        baseline_params = ConstantBaselineParams(mu=params.mu) if params is not None else None

        self.baseline = ConstantBaselineModule(
            D=D,
            params=baseline_params,
            seed=seed,
            dataset_stats=dataset_stats,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_baserate(self) -> dict:
        """Return the constrained base rates as a dict."""
        (mu,) = self.transform_params()
        return {"mu": mu}

    # ------------------------------------------------------------------
    # CI inversion — kept on the TPP because it depends on termination state
    # ------------------------------------------------------------------

    def inverse_marginal_cumulative_intensity(
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
        assert u.shape[0] == ts.time_points.shape[0], "u must have the same batch size as ts"

        device = ts.time_points.device
        (mu,) = self.transform_params()  # (D,)
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
        max_t_needed = (u.max() / mu.min() + 1.0).detach()
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
        last_finite_term = termination_times.masked_fill(torch.isinf(termination_times), 0.0).max(dim=1).values
        all_terminated = ~torch.isinf(termination_times).any(dim=1)
        max_ci_when_all_terminated = (mu.unsqueeze(0) * termination_times).sum(dim=1)

        exceeds_max = all_terminated & (u > max_ci_when_all_terminated)
        t_star = torch.where(exceeds_max, torch.full_like(t_star, float("inf")), t_star)

        t_star = torch.where(no_terminations, t_simple, t_star)

        return t_star

    def sample(
        self,
        ts: MVEventData,
        num_steps: int = 1,
        rng: Optional[torch.Generator | int] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        """Closed-form sampling for homogeneous Poisson processes.

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
        cfg = sampling_config if sampling_config is not None else self.sampling_config
        if cfg.mode == "thinning":
            # Ogata thinning has no closed-form shortcut — use the base implementation.
            return super().sample(ts, num_steps=num_steps, rng=rng, sampling_config=cfg)

        step = 0
        time_samples = []
        dist_samples = []

        (mu,) = self.transform_params()
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

            T_tensor = torch.tensor([t_star], dtype=ts.time_points.dtype, device=device)

            new_time_points = torch.cat([ts.time_points, T_tensor], dim=0)
            new_event_types = torch.cat([ts.event_types, type_idx], dim=0)

            ts = MVEventData(new_time_points, new_event_types)

        return ts, time_samples, dist_samples


@dataclass
class ConditionalIPPParams:
    mu: torch.Tensor  # D intercepts per disease
    slope: torch.Tensor  # D slopes per disease


class ConditionalInhomogeniousPoissonProcess(BaselineTPP):
    """Inhomogeneous Poisson process with log-linear intensity λ_d(t) = exp(μ_d + slope_d·t).

    Delegates all intensity math to a :class:`LinearBaselineModule`.
    """

    def __init__(
        self,
        params: Optional[ConditionalIPPParams] = None,
        D: Optional[int] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
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
        if params is None and D is None:
            raise ValueError("Either params or D (number of dimensions) must be provided.")

        if D is None:
            D = params.mu.shape[0]  # type: ignore[union-attr]

        super().__init__(D, seed, terminating=terminating)

        baseline_params = LinearBaselineParams(mu=params.mu, slope=params.slope) if params is not None else None

        self.baseline = LinearBaselineModule(
            D=D,
            params=baseline_params,
            seed=seed,
            dataset_stats=dataset_stats,
        )

    # ------------------------------------------------------------------
    # Convenience accessor
    # ------------------------------------------------------------------

    def get_rate_slope(self) -> dict:
        """Return the current (mu, slope) parameters as a dict."""
        mu, slope = self.transform_params()
        return {"mu": mu, "slope": slope}


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
        num_knots: int,
        delta_t: torch.Tensor | float,
        params: Optional[SplineProcessParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """Args:
        D: Number of event types.
        num_knots: Number of spline knots.
        delta_t: Spacing between knots — scalar for uniform spacing, or a
            tensor of per-knot gaps whose first entry must be 0
            (knot locations = cumsum(delta_t), starting at t=0).
        params: Pre-computed parameters.  If given, dataset_stats is ignored.
        seed: Random seed.
        terminating: If True, each event type can occur at most once.
        dataset_stats: Optional empirical statistics.  When provided, knot
            heights are initialised near the per-type observed rate.
        """
        super().__init__(D, seed, terminating=terminating)

        baseline_params: Optional[SplineBaselineParams] = (
            SplineBaselineParams(h_knots=params.h_knots, delta_ts=params.delta_ts) if params is not None else None
        )

        self.baseline = SplineBaselineModule(
            D=D,
            num_knots=num_knots,
            delta_t=delta_t,
            params=baseline_params,
            seed=seed,
            dataset_stats=dataset_stats,
        )
        self.num_knots = self.baseline.num_knots

    # ------------------------------------------------------------------
    # Accessors and overrides
    # ------------------------------------------------------------------

    def get_heights(self) -> Tuple[torch.Tensor, ...]:
        """Return non-negative knot heights: Shape (D, num_knots).

        Returns a tuple for consistency with transform_params().
        Use ``(h,) = self.get_heights()`` to unpack.
        """
        return self.baseline.get_heights()

    def _event_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Intensity of the observed event type at each position.  Shape: (B, L).

        For the non-terminating case the module uses a batched flatten+view path
        (efficient since intensity is history-independent).  For the terminating
        case the base-class default is used so that termination masking in
        ``intensity()`` is applied before gathering.
        """
        if self.terminating:
            # Fall back to base-class default: _sequence_intensities → gather.
            return super()._event_intensities(ts)
        return self.baseline.event_intensities(ts)

    # ------------------------------------------------------------------
    # CI inversion — kept on the TPP because it depends on termination state
    # ------------------------------------------------------------------

    def inverse_marginal_cumulative_intensity(
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
        assert u.shape[0] == ts.time_points.shape[0], "u must have the same batch size as ts"

        if self.terminating:
            return super().inverse_marginal_cumulative_intensity(u, ts)

        (h,) = self.get_heights()  # (D, K)
        h_marginal = h.sum(dim=0, keepdim=True)  # (1, K) — marginal over all types
        t_next = LinearSpline.inverse_integral(
            self.baseline.knot_locs,
            h_marginal,
            u.unsqueeze(1),  # (B, 1)
        )  # (B, 1)
        return t_next.squeeze(1)  # (B,)


class PiecewiseConstantPoissonProcess(BaselineTPP):
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
        num_knots: int,
        delta_t: torch.Tensor | float,
        params: Optional[SplineProcessParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """Args:
        D: Number of event types.
        num_knots: Number of piecewise-constant bins.
        delta_t: Bin width — scalar for uniform bins, or a tensor of
            per-edge gaps whose first entry must be 0
            (left bin edges = cumsum(delta_t), starting at t=0).
        params: Pre-computed parameters.  If given, dataset_stats is ignored.
        seed: Random seed.
        terminating: If True, each event type can occur at most once.
        dataset_stats: Optional empirical statistics.  When provided, bin
            heights are initialised near the per-type observed rate.
        """
        super().__init__(D, seed, terminating=terminating)

        baseline_params: Optional[PiecewiseConstantBaselineParams] = (
            PiecewiseConstantBaselineParams(h_knots=params.h_knots, delta_ts=params.delta_ts)
            if params is not None
            else None
        )

        self.baseline = PiecewiseConstantBaselineModule(
            D=D,
            num_knots=num_knots,
            delta_t=delta_t,
            params=baseline_params,
            seed=seed,
            dataset_stats=dataset_stats,
        )
        self.num_knots = self.baseline.num_knots

    # ------------------------------------------------------------------
    # Accessors and overrides
    # ------------------------------------------------------------------

    def get_heights(self) -> Tuple[torch.Tensor, ...]:
        """Return non-negative bin heights: Shape (D, num_knots)."""
        return self.baseline.get_heights()

    def _event_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Intensity of the observed event type at each position.  Shape: (B, L).

        For the non-terminating case the module uses a batched flatten+view path.
        For the terminating case the base-class default is used so that
        termination masking in ``intensity()`` is applied before gathering.
        """
        if self.terminating:
            return super()._event_intensities(ts)
        return self.baseline.event_intensities(ts)

    # ------------------------------------------------------------------
    # CI inversion — kept on the TPP because it depends on termination state
    # ------------------------------------------------------------------

    def inverse_marginal_cumulative_intensity(
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
        assert u.shape[0] == ts.time_points.shape[0]

        if self.terminating:
            return super().inverse_marginal_cumulative_intensity(u, ts)

        (h,) = self.get_heights()  # (D, K)
        h_marginal = h.sum(dim=0, keepdim=True)  # (1, K) — marginal over all types
        t_next = PiecewiseConstant.inverse_integral(
            self.baseline.knot_locs,
            h_marginal,
            u.unsqueeze(1),  # (B, 1)
        )  # (B, 1)
        return t_next.squeeze(1)  # (B,)
