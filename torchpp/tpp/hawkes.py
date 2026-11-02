# Hawkes process implementations.
#
# Contains HawkesProcess, InhibitiveHawkesProcess (abstract bases) and all
# concrete composite Hawkes model classes.
from abc import ABC
from typing import Callable, Optional, Tuple, cast
import torch
import torch.nn.functional as F

from .base_tpp import IntensityBasedTPP, CIConfig
from ..utils.event_data import BatchedMVEventData, DatasetStats
from ..modules.intensity import (
    BaselineIntensityModule,
    KernelIntensityModule,
    ConstantBaselineParams,
    ConstantBaselineModule,
    LinearBaselineParams,
    LinearBaselineModule,
    SplineBaselineParams,
    SplineBaselineModule,
    UnconstrainedSplineBaselineParams,
    PiecewiseConstantBaselineParams,
    PiecewiseConstantBaselineModule,
    ExpKernelParams,
    ExpKernelModule,
    UnconstrainedExpKernelParams,
    SplineKernelParams,
    SplineKernelModule,
    PiecewiseConstantKernelParams,
    PiecewiseConstantKernelModule,
    NonStationaryExpKernelParams,
    NonStationarySplineExpKernelModule,
    NonStationarySplineKernelParams,
    NonStationarySplineKernelModule,
    NeuralKernelModule,
    NeuralSplineKernelModule,
)

# ---------------------------------------------------------------------------
# Clamping-function registry
# ---------------------------------------------------------------------------

#: Valid ``clamping_fn`` string names for inhibitive Hawkes processes.
VALID_CLAMPING_FNS: list[str] = ["softplus", "relu", "exp"]


def resolve_clamping_fn(name: str):
    """Return the callable corresponding to a clamping-function name.

    Args:
        name: One of ``"softplus"``, ``"relu"``, ``"exp"``.

    Returns:
        A callable ``f: Tensor → Tensor`` that maps any real input to ℝ≥0.

    Raises:
        ValueError: For unknown names.
    """
    name = name.lower()
    if name == "softplus":
        return F.softplus
    elif name == "relu":
        return F.relu
    elif name == "exp":
        return torch.exp
    else:
        raise ValueError(f"Unknown clamping_fn '{name}'. Valid choices: {VALID_CLAMPING_FNS}")


class HawkesProcess(IntensityBasedTPP, ABC):
    """
    Abstract base class for Hawkes processes with modular baseline and kernel.

    A Hawkes process combines:
    - A baseline module (exogenous intensity)
    - A kernel module (endogenous self-exciting component)

    The total intensity is: λ(t) = baseline(t) + kernel(t)
    """

    def __init__(
        self,
        baseline: BaselineIntensityModule,
        kernel: KernelIntensityModule,
        D: int,
        seed: Optional[int] = 42,
        terminating: bool = False,
    ):
        """
        Args:
            baseline: BaselineIntensityModule instance
            kernel: KernelIntensityModule instance
            D: Number of event types
            seed: Random seed
            terminating: If True, each event type can only occur once
        """
        super().__init__(D, seed, terminating=terminating)
        self.baseline = baseline
        self.kernel = kernel

        assert self.baseline.D == D, "Baseline dimension mismatch"
        assert self.kernel.D == D, "Kernel dimension mismatch"

    def transform_params(self) -> Tuple[torch.Tensor, ...]:
        """
        Return all parameters in constrained form.

        Returns:
            (baseline_params, *kernel_params)
        """
        baseline_params = self.baseline.transform_params()
        kernel_params = self.kernel.transform_params()
        return (*baseline_params, *kernel_params)

    def _intensity_at_t(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute total raw intensity = baseline + kernel (before termination masking).

        Args:
            t: Time points. Shape: (B,)
            ts: Historical events. Shape: (B,N)

        Returns:
            Total raw intensity. Shape: (B,D)
        """
        baseline_intensity = self.baseline.intensity(t, ts)  # (B,D)
        kernel_intensity = self.kernel.intensity(t, ts)  # (B,D)
        return baseline_intensity + kernel_intensity

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Flat multi-query raw intensity = baseline + kernel, vectorised over queries.

        ``(B, Q)`` -> ``(B, Q, D)``.  Composes the modules' ``intensities_at_queries`` (fast
        for constant-baseline + exp-kernel; loop-default otherwise), so the eval grid /
        numerical CI evaluate the whole grid in one pass instead of per point.
        """
        return self.baseline.intensities_at_queries(query_times, ts) + self.kernel.intensities_at_queries(
            query_times, ts
        )

    def _raw_ci(self, effective_T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute total cumulative intensity = baseline CI + kernel CI.

        Args:
            effective_T: Per-type upper integration limits. Shape: (B, D)
            ts: Historical events. Shape: (B, N)

        Returns:
            Total cumulative intensity. Shape: (B, D)
        """
        baseline_ci = self.baseline.cumulative_intensity(effective_T, ts)  # (B, D)
        kernel_ci = self.kernel.cumulative_intensity(effective_T, ts)  # (B, D)

        return baseline_ci + kernel_ci

    def _closed_form_ci(self, ts: BatchedMVEventData, T_low, T: torch.Tensor) -> torch.Tensor:
        """Efficient ``O(N·D)`` closed-form CI via the global module CI.

        Overrides the per-interval default: the per-interval kernel CI is ``O(L²)``,
        whereas the global ``baseline+kernel.cumulative_intensity`` is one ``O(N·D)``
        vectorised pass.  Windowing is exact subtraction (closed-form, no quadrature
        error); termination folds into the per-type ``_effective_T``.
        """

        def total(t):
            eff = self._effective_T(t, ts)  # (B, D)
            return self.baseline.cumulative_intensity(eff, ts) + self.kernel.cumulative_intensity(eff, ts)

        ci = total(T)
        if T_low is not None:
            ci = ci - total(T_low)
        return ci

    def _event_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Intensity of the observed event type at each position.  Shape: (B, L).

        Uses ``event_intensities`` from the baseline and kernel modules
        to compute the total intensity at each observed event in a single efficient pass.
        All masking is handled by the base ``_event_log_intensity_sum``.
        """
        baseline_intensities = self.baseline.event_intensities(ts)  # (B, L)
        interaction_terms = self.kernel.event_intensities(ts)  # (B, L)
        total = baseline_intensities + interaction_terms  # (B, L)
        return total


class InhibitiveHawkesProcess(IntensityBasedTPP, ABC):
    """
    Abstract base class for Inhibitive Hawkes processes with modular baseline and kernel.

    A Hawkes process combines:
    - A baseline module (exogenous intensity)
    - A kernel module (endogenous self-exciting component)

    The total intensity is: λ(t) = positivity_constraint(baseline(t) + kernel(t))

    Uses numerical integration for cumulative intensity since the positivity
    constraint (e.g., softplus) makes analytical solutions intractable.
    Grid-accelerated CI inversion (inverse_CDF, inverse_marginal_cumulative_intensity)
    is provided by the base class for the numerical CIConfig.
    """

    def __init__(
        self,
        positive_constraint: Callable,
        baseline: BaselineIntensityModule,
        kernel: KernelIntensityModule,
        D: int,
        seed: Optional[int] = 42,
        terminating: bool = False,
    ):
        """
        Args:
            positive_constraint: Positivity function (e.g., softplus)
            baseline: BaselineIntensityModule instance
            kernel: KernelIntensityModule instance
            D: Number of event types
            seed: Random seed
            terminating: If True, each event type can only occur once
        """
        super().__init__(D=D, seed=seed, terminating=terminating)
        self.baseline = baseline
        self.kernel = kernel
        self.positivity_constraint = positive_constraint

        assert self.baseline.D == D, "Baseline dimension mismatch"
        assert self.kernel.D == D, "Kernel dimension mismatch"

    def _default_ci_config(self) -> CIConfig:
        """Whole-window numerical CI: the clamped ``clamping_fn(baseline + kernel)`` may
        kink within an interval, so the global placement (not event-knotted) is used."""
        return CIConfig("numerical", "global", "mc_trapezoidal", n_points=100)

    def transform_params(self) -> Tuple[torch.Tensor, ...]:
        """
        Return all parameters in constrained form.

        Returns:
            (baseline_params, *kernel_params)
        """
        baseline_params = self.baseline.transform_params()
        kernel_params = self.kernel.transform_params()
        return (*baseline_params, *kernel_params)

    def _intensity_at_t(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute total raw intensity = positivity_constraint(baseline + kernel) (before termination masking).

        Args:
            t: Time points. Shape: (B,)
            ts: Historical events. Shape: (B,N)

        Returns:
            Total raw intensity. Shape: (B,D)
        """
        baseline_intensity = self.baseline.intensity(t, ts)  # (B,D)
        kernel_intensity = self.kernel.intensity(t, ts)  # (B,D)
        return self.positivity_constraint(baseline_intensity + kernel_intensity)

    def _event_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Intensity of the observed event type at each position.  Shape: (B, L).

        Applies the positivity constraint, matching the behaviour of
        ``_intensity_at_t`` for inhibitive Hawkes processes.
        All masking is handled by the base ``_event_log_intensity_sum``.
        """
        baseline_intensities = self.baseline.event_intensities(ts)  # (B, L)
        interaction_terms = self.kernel.event_intensities(ts)  # (B, L)
        total = self.positivity_constraint(baseline_intensities + interaction_terms)  # (B, L)
        return total


class ConstBaselineExpKernelHawkesProcess(HawkesProcess):
    """
    Reimplementation of Hawkes process with exponential kernel using modular architecture.

    This version uses the HawkesProcess base class with ConstantBaselineModule and ExpKernelModule.
    """

    def __init__(
        self,
        D: int,
        baseline_params: Optional[ConstantBaselineParams] = None,
        kernel_params: Optional[ExpKernelParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Initialize exponential kernel Hawkes process.

        Args:
            D: Number of event types
            baseline_mu: Baseline intensity (D,). If None, initialized randomly.
            alpha: Branching ratio matrix (D,D). If None, initialized randomly.
            beta: Decay rate matrix (D,D). If None, initialized randomly.
            seed: Random seed
            terminating: If True, each event type can only occur once
            dataset_stats: Optional empirical statistics for data-driven init.
        """
        kernel_seed = seed + 1 if seed is not None else None
        baseline = ConstantBaselineModule(D, baseline_params, seed=seed, dataset_stats=dataset_stats)
        kernel = ExpKernelModule(D, kernel_params, kernel_seed, dataset_stats=dataset_stats)

        super().__init__(baseline, kernel, D, seed, terminating=terminating)


class LinearBaselineExpKernelHawkesProcess(HawkesProcess):
    """
    Hawkes process with log-linear baseline and exponential kernel.

    This version uses the HawkesProcess base class with LinearBaselineModule and ExpKernelModule.
    """

    def __init__(
        self,
        D: int,
        baseline_params: Optional[LinearBaselineParams] = None,
        kernel_params: Optional[ExpKernelParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Initialize exponential kernel Hawkes process with a linear function baseline intensity

        Args:
            D: Number of event types
            baseline_params: If None, initialized randomly.
            kernel_params: If None, initialized randomly.
            seed: Random seed
            terminating: If True, each event type can only occur once
            dataset_stats: Optional empirical statistics for data-driven init.
        """
        kernel_seed = seed + 1 if seed is not None else None
        baseline = LinearBaselineModule(baseline_params, D, seed, dataset_stats=dataset_stats)
        kernel = ExpKernelModule(D, kernel_params, kernel_seed, dataset_stats=dataset_stats)

        super().__init__(baseline, kernel, D, seed, terminating=terminating)


class PiecewiseBaselineExpKernelHawkesProcess(HawkesProcess):
    """Hawkes process with piecewise-constant baseline and exponential kernel.

    More flexible than constant baseline; simpler than spline (no slopes).
    """

    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        baseline_params: Optional[PiecewiseConstantBaselineParams] = None,
        kernel_params: Optional[ExpKernelParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        kernel_seed = seed + 1 if seed is not None else None
        baseline = PiecewiseConstantBaselineModule(
            D, num_knots=num_knots, delta_t=delta_t, params=baseline_params, seed=seed, dataset_stats=dataset_stats
        )
        kernel = ExpKernelModule(D, kernel_params, kernel_seed, dataset_stats=dataset_stats)
        super().__init__(baseline, kernel, D, seed, terminating=terminating)


class SplineBaselineExpKernelHawkesProcess(HawkesProcess):
    """
    Hawkes process with spline-based baseline intensity and exponential kernel.

    This version uses the HawkesProcess base class with SplineBaselineModule and ExpKernelModule.
    """

    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        baseline_params: Optional[SplineBaselineParams] = None,
        kernel_params: Optional[ExpKernelParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Initialize spline-based baseline exponential kernel Hawkes process.

        Args:
            D: Number of event types
            baseline_knots: Knots for spline baseline (num_knots,). If None, initialized randomly.
            baseline_coefs: Coefficients for spline baseline (D, num_coefs). If None, initialized randomly.
            alpha: Branching ratio matrix (D,D). If None, initialized randomly.
            beta: Decay rate matrix (D,D). If None, initialized randomly.
            seed: Random seed
            terminating: If True, each event type can only occur once
            dataset_stats: Optional empirical statistics for data-driven init.
        """
        kernel_seed = seed + 1 if seed is not None else None
        baseline = SplineBaselineModule(
            D, num_knots=num_knots, delta_t=delta_t, params=baseline_params, seed=seed, dataset_stats=dataset_stats
        )
        kernel = ExpKernelModule(D, kernel_params, kernel_seed, dataset_stats=dataset_stats)

        super().__init__(baseline, kernel, D, seed, terminating=terminating)


class ConstBaselinePiecewiseKernelHawkesProcess(HawkesProcess):
    """Hawkes process with constant baseline and piecewise-constant kernel.

    The kernel function is a step function over elapsed time Δt, with one
    step function per (receiver, trigger) pair.  Constant baseline provides
    a spontaneous event rate.

    Cumulative intensity is computed analytically.
    """

    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        baseline_params: Optional[ConstantBaselineParams] = None,
        kernel_params: Optional[PiecewiseConstantKernelParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        kernel_seed = seed + 1 if seed is not None else None
        baseline = ConstantBaselineModule(D, baseline_params, seed=seed, dataset_stats=dataset_stats)
        kernel = PiecewiseConstantKernelModule(
            D,
            num_knots=num_knots,
            delta_t=delta_t,
            params=kernel_params,
            seed=kernel_seed,
            dataset_stats=dataset_stats,
        )
        super().__init__(baseline, kernel, D, seed, terminating=terminating)


class ConstBaselineSplineKernelHawkesProcess(HawkesProcess):
    """
    Hawkes process with constant baseline and spline kernel.

    The kernel function is a linear spline over elapsed time Δt, with one
    spline per (receiver, trigger) pair. Knot heights are positive (softplus).
    The constant baseline provides a spontaneous event rate.

    Cumulative intensity is computed analytically via LinearSpline.integrate.
    """

    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        baseline_params: Optional[ConstantBaselineParams] = None,
        kernel_params: Optional[SplineKernelParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Initialize constant baseline spline kernel Hawkes process.

        Args:
            D: Number of event types
            num_knots: Number of spline knots for the kernel (on Δt axis)
            delta_t: Knot specification for the kernel (float/scalar for uniform spacing,
                or 1-D tensor of absolute knot positions).
            baseline_params: Constant baseline parameters. If None, initialized randomly.
            kernel_params: Spline kernel parameters. If None, initialized randomly.
            seed: Random seed
            terminating: If True, each event type can only occur once
            dataset_stats: Optional empirical statistics for data-driven init.
        """
        kernel_seed = seed + 1 if seed is not None else None
        baseline = ConstantBaselineModule(D, baseline_params, seed=seed, dataset_stats=dataset_stats)
        kernel = SplineKernelModule(
            D,
            num_knots=num_knots,
            delta_t=delta_t,
            params=kernel_params,
            seed=kernel_seed,
            dataset_stats=dataset_stats,
        )

        super().__init__(baseline, kernel, D, seed, terminating=terminating)


class SplineBaselineSplineKernelHawkesProcess(HawkesProcess):
    """
    Hawkes process with spline-based baseline and spline kernel.

    Both the baseline intensity and the kernel are linear splines. The baseline
    spline is over absolute time t, while the kernel spline is over elapsed
    time Δt since the trigger event.
    """

    def __init__(
        self,
        D: int,
        num_knots_baseline: int,
        num_knots_kernel: int,
        delta_t_baseline: torch.Tensor | float,
        delta_t_kernel: torch.Tensor | float,
        baseline_params: Optional[SplineBaselineParams] = None,
        kernel_params: Optional[SplineKernelParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Initialize spline baseline + spline kernel Hawkes process.

        Args:
            D: Number of event types
            num_knots_baseline: Number of spline knots for the baseline
            num_knots_kernel: Number of spline knots for the kernel
            delta_t_baseline: Spacing between baseline knots (float for uniform, tensor for variable)
            delta_t_kernel: Knot specification for the kernel (float/scalar for uniform spacing,
                or 1-D tensor of absolute knot positions).
            baseline_params: Spline baseline parameters. If None, initialized randomly.
            kernel_params: Spline kernel parameters. If None, initialized randomly.
            seed: Random seed
            terminating: If True, each event type can only occur once
            dataset_stats: Optional empirical statistics for data-driven init.
        """
        kernel_seed = seed + 1 if seed is not None else None
        baseline = SplineBaselineModule(
            D,
            num_knots=num_knots_baseline,
            delta_t=delta_t_baseline,
            params=baseline_params,
            seed=seed,
            dataset_stats=dataset_stats,
        )
        kernel = SplineKernelModule(
            D,
            num_knots=num_knots_kernel,
            delta_t=delta_t_kernel,
            params=kernel_params,
            seed=kernel_seed,
            dataset_stats=dataset_stats,
        )
        super().__init__(baseline, kernel, D, seed, terminating=terminating)


class NumericalSplineBaselineExpKernelHawkesProcess(HawkesProcess):
    """
    Hawkes process with spline-based baseline intensity and exponential kernel.

    This version uses numerical integration for cumulative intensity computation,
    which is necessary when analytical solutions are intractable or for testing purposes.
    """

    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        baseline_params: Optional[SplineBaselineParams] = None,
        kernel_params: Optional[ExpKernelParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Initialize spline-based baseline exponential kernel Hawkes process with numerical integration.

        Args:
            D: Number of event types
            num_knots: Number of knots for spline baseline
            delta_t: Spacing between knots (float for uniform, tensor for variable)
            baseline_params: Spline baseline parameters. If None, initialized randomly.
            kernel_params: Kernel parameters. If None, initialized randomly.
            seed: Random seed
            terminating: If True, each event type can only occur once
            dataset_stats: Optional empirical statistics for data-driven init.
        """
        kernel_seed = seed + 1 if seed is not None else None
        baseline = SplineBaselineModule(
            D, num_knots=num_knots, delta_t=delta_t, params=baseline_params, seed=seed, dataset_stats=dataset_stats
        )
        kernel = ExpKernelModule(D, kernel_params, kernel_seed, dataset_stats=dataset_stats)

        super().__init__(baseline, kernel, D, seed, terminating=terminating)

    def _default_ci_config(self) -> CIConfig:
        """Closed-form-capable but intentionally numerical (whole-window global grid) — it must
        declare this explicitly, since it inherits ``HawkesProcess``'s closed-form CI hooks."""
        return CIConfig("numerical", "global", "mc_trapezoidal", n_points=100)


class InhibitiveConstExpHawkesProcess(InhibitiveHawkesProcess):
    """
    Inhibitive Hawkes process with constant baseline and unconstrained exponential kernel.

    Total intensity: λ(t) = clamping_fn(μ + kernel(t))

    Both baseline μ and kernel α are unconstrained; the clamping function is applied
    only to the final combined intensity, avoiding gradient attenuation from nested
    positivity constraints.

    Kernel α can be negative (inhibition) or positive (excitation).
    Numerical integration is always used for cumulative intensity.
    """

    def __init__(
        self,
        D: int,
        baseline_params: Optional[ConstantBaselineParams] = None,
        kernel_params: Optional[ExpKernelParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        clamping_fn: str = "softplus",
        constrain_kernel: bool = False,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Args:
            D: Number of event types
            baseline_params: Constant baseline parameters. If None, initialized randomly.
            kernel_params: Kernel parameters. If None, initialized randomly.
            seed: Random seed
            terminating: If True, each event type can only occur once
            clamping_fn: Positivity function applied to the raw intensity sum.
                One of "softplus" (default), "relu", "exp".
            constrain_kernel: If True, kernel alpha is constrained positive (excitation only).
                Useful as a debug/sanity-check mode to compare against analytical Hawkes.
            dataset_stats: Optional empirical statistics for data-driven init.
        """
        kernel_seed = seed + 1 if seed is not None else None
        baseline = ConstantBaselineModule(
            D,
            baseline_params,
            seed=seed,
            constrained=True if clamping_fn == "relu" else False,
            dataset_stats=dataset_stats,
        )
        kernel = ExpKernelModule(
            D, kernel_params, kernel_seed, constrained=constrain_kernel, dataset_stats=dataset_stats
        )
        super().__init__(
            resolve_clamping_fn(clamping_fn),
            baseline,
            kernel,
            D,
            seed,
            terminating=terminating,
        )


# Backwards-compat alias
SoftplusConstExpIHawkesProcess = InhibitiveConstExpHawkesProcess


class InhibitiveSplineExpHawkesProcess(InhibitiveHawkesProcess):
    """
    Inhibitive Hawkes process with spline baseline and unconstrained exponential kernel.

    Total intensity: λ(t) = clamping_fn(spline(t) + kernel(t))

    Both the spline heights and kernel α are unconstrained; the clamping function
    is applied only to the combined intensity.

    Kernel α can be negative (inhibition) or positive (excitation).
    Numerical integration is always used for cumulative intensity.
    """

    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        baseline_params: Optional[UnconstrainedSplineBaselineParams] = None,
        kernel_params: Optional[UnconstrainedExpKernelParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        clamping_fn: str = "softplus",
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Args:
            D: Number of event types
            num_knots: Number of knots for the spline baseline
            delta_t: Spacing between knots (float for uniform, tensor for variable)
            baseline_params: Spline baseline parameters. If None, initialized randomly.
            kernel_params: Kernel parameters. If None, initialized randomly.
            seed: Random seed
            terminating: If True, each event type can only occur once
            clamping_fn: Positivity function applied to the raw intensity sum.
                One of "softplus" (default), "relu", "exp".
            dataset_stats: Optional empirical statistics for data-driven init.
        """
        kernel_seed = seed + 1 if seed is not None else None
        baseline = SplineBaselineModule(
            D,
            num_knots=num_knots,
            delta_t=delta_t,
            params=baseline_params,
            seed=seed,
            constrained=True if clamping_fn == "relu" else False,
            dataset_stats=dataset_stats,
        )
        kernel = ExpKernelModule(D, kernel_params, kernel_seed, constrained=False, dataset_stats=dataset_stats)
        super().__init__(
            resolve_clamping_fn(clamping_fn),
            baseline,
            kernel,
            D,
            seed,
            terminating=terminating,
        )


class InhibitiveSplineBaselineSplineKernelHawkesProcess(InhibitiveHawkesProcess):
    """
    Inhibitive Hawkes process with spline baseline and unconstrained spline kernel.

    Total intensity: λ(t) = clamping_fn(spline(t) + kernel(t))

    Both the spline heights and kernel α are unconstrained; the clamping function
    is applied only to the combined intensity.

    Kernel α can be negative (inhibition) or positive (excitation).
    Numerical integration is always used for cumulative intensity.
    """

    def __init__(
        self,
        D: int,
        num_knots_baseline: int,
        num_knots_kernel: int,
        delta_t_baseline: torch.Tensor | float,
        delta_t_kernel: torch.Tensor | float,
        baseline_params: Optional[SplineBaselineParams] = None,
        kernel_params: Optional[SplineKernelParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        clamping_fn: str = "softplus",
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Args:
            D: Number of event types
            num_knots: Number of knots for the spline baseline
            delta_t: Spacing between knots (float for uniform, tensor for variable)
            baseline_params: Spline baseline parameters. If None, initialized randomly.
            kernel_params: Kernel parameters. If None, initialized randomly.
            seed: Random seed
            terminating: If True, each event type can only occur once
            clamping_fn: Positivity function applied to the raw intensity sum.
                One of "softplus" (default), "relu", "exp".
            dataset_stats: Optional empirical statistics for data-driven init.
        """
        kernel_seed = seed + 1 if seed is not None else None
        baseline = SplineBaselineModule(
            D,
            num_knots=num_knots_baseline,
            delta_t=delta_t_baseline,
            params=baseline_params,
            seed=seed,
            constrained=True if clamping_fn == "relu" else False,
            dataset_stats=dataset_stats,
        )
        kernel = SplineKernelModule(
            D,
            num_knots=num_knots_kernel,
            delta_t=delta_t_kernel,
            params=kernel_params,
            seed=kernel_seed,
            constrained=False,
            dataset_stats=dataset_stats,
        )
        super().__init__(
            resolve_clamping_fn(clamping_fn),
            baseline,
            kernel,
            D,
            seed,
            terminating=terminating,
        )


# Backwards-compat alias
SoftplusSplineExpIHawkesProcess = InhibitiveSplineExpHawkesProcess


class ConstBaselineNSExpKernelHawkesProcess(HawkesProcess):
    """
    Hawkes process with constant baseline and non-stationary exponential kernel.

    The kernel parameters α and β are time-varying, defined by linear spline
    interpolation over knot values. The baseline intensity is constant per type.

    This uses the HawkesProcess base class with ConstantBaselineModule and
    NonStationarySplineExpKernelModule.
    """

    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        baseline_params: Optional[ConstantBaselineParams] = None,
        kernel_params: Optional[NonStationaryExpKernelParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Initialize constant baseline non-stationary exponential kernel Hawkes process.

        Args:
            D: Number of event types
            num_knots: Number of spline knots for the kernel parameters
            delta_t: Spacing between knots (float for uniform, tensor for variable)
            baseline_params: Constant baseline parameters. If None, initialized randomly.
            kernel_params: Non-stationary kernel parameters. If None, initialized randomly.
            seed: Random seed
            terminating: If True, each event type can only occur once
            dataset_stats: Optional empirical statistics for data-driven baseline init.
        """
        kernel_seed = seed + 1 if seed is not None else None
        baseline = ConstantBaselineModule(D, baseline_params, seed=seed, dataset_stats=dataset_stats)
        kernel = NonStationarySplineExpKernelModule(
            D,
            num_knots=num_knots,
            delta_t=delta_t,
            params=kernel_params,
            seed=kernel_seed,
            dataset_stats=dataset_stats,
        )

        super().__init__(baseline, kernel, D, seed, terminating=terminating)


class SplineBaselineNSExpKernelHawkesProcess(HawkesProcess):
    """
    Hawkes process with spline baseline and non-stationary exponential kernel.

    Both the baseline (spline over absolute time) and the kernel parameters α, β
    (splines over absolute time, interpolated at the trigger time) are time-varying;
    the Δt-shape of the kernel remains exponential.
    """

    def __init__(
        self,
        D: int,
        num_knots_baseline: int,
        num_knots_kernel: int,
        delta_t_baseline: torch.Tensor | float,
        delta_t_kernel: torch.Tensor | float,
        baseline_params: Optional[SplineBaselineParams] = None,
        kernel_params: Optional[NonStationaryExpKernelParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Args:
            D: Number of event types
            num_knots_baseline: Number of spline knots for the baseline
            num_knots_kernel: Number of knots for the non-stationary kernel parameters
                (on the absolute-time axis)
            delta_t_baseline: Spacing between baseline knots (float for uniform, tensor for variable)
            delta_t_kernel: Spacing between the kernel's absolute-time knots
            baseline_params: Spline baseline parameters. If None, initialized randomly.
            kernel_params: Non-stationary kernel parameters. If None, initialized randomly.
            seed: Random seed
            terminating: If True, each event type can only occur once
            dataset_stats: Optional empirical statistics for data-driven init.
        """
        kernel_seed = seed + 1 if seed is not None else None
        baseline = SplineBaselineModule(
            D,
            num_knots=num_knots_baseline,
            delta_t=delta_t_baseline,
            params=baseline_params,
            seed=seed,
            dataset_stats=dataset_stats,
        )
        kernel = NonStationarySplineExpKernelModule(
            D,
            num_knots=num_knots_kernel,
            delta_t=delta_t_kernel,
            params=kernel_params,
            seed=kernel_seed,
            dataset_stats=dataset_stats,
        )
        super().__init__(baseline, kernel, D, seed, terminating=terminating)


class SplineBaselineNSSplineKernelHawkesProcess(HawkesProcess):
    """
    Hawkes process with spline baseline and non-stationary spline kernel.

    The baseline is a linear spline over absolute time.  The kernel is a linear
    spline over elapsed time Δt whose knot heights vary over absolute time
    (interpolated at the trigger event's time), giving a control that is both
    non-stationary and flexible in shape while keeping closed-form cumulative
    intensity.
    """

    def __init__(
        self,
        D: int,
        num_knots_baseline: int,
        num_knots_ns: int,
        num_knots_kernel: int,
        delta_t_baseline: torch.Tensor | float,
        delta_t_ns: torch.Tensor | float,
        delta_t_kernel: torch.Tensor | float,
        baseline_params: Optional[SplineBaselineParams] = None,
        kernel_params: Optional[NonStationarySplineKernelParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Args:
            D: Number of event types
            num_knots_baseline: Number of spline knots for the baseline
            num_knots_ns: Number of kernel knots on the absolute-time axis (K_ns)
            num_knots_kernel: Number of kernel knots on the Δt axis (K_Δt)
            delta_t_baseline: Spacing between baseline knots (float for uniform, tensor for variable)
            delta_t_ns: Spacing between the kernel's absolute-time knots
            delta_t_kernel: Δt knot spec for the kernel (float for uniform, or K_Δt-1 gaps)
            baseline_params: Spline baseline parameters. If None, initialized randomly.
            kernel_params: Non-stationary spline kernel parameters. If None, initialized randomly.
            seed: Random seed
            terminating: If True, each event type can only occur once
            dataset_stats: Optional empirical statistics for data-driven init.
        """
        kernel_seed = seed + 1 if seed is not None else None
        baseline = SplineBaselineModule(
            D,
            num_knots=num_knots_baseline,
            delta_t=delta_t_baseline,
            params=baseline_params,
            seed=seed,
            dataset_stats=dataset_stats,
        )
        kernel = NonStationarySplineKernelModule(
            D,
            num_knots_ns=num_knots_ns,
            num_knots_kernel=num_knots_kernel,
            delta_t_ns=delta_t_ns,
            delta_t_kernel=delta_t_kernel,
            params=kernel_params,
            seed=kernel_seed,
            dataset_stats=dataset_stats,
        )
        super().__init__(baseline, kernel, D, seed, terminating=terminating)


class SplineBaselineNeuralKernelHawkesProcess(InhibitiveHawkesProcess):
    """Hawkes process with spline baseline and neural MLP kernel.

    Intensity: ``λ(t) = softplus(spline_baseline(t) + Σ_j MLP(emb(k_j), Δt_j))``

    The spline baseline is unconstrained (can go negative before softplus).
    The neural kernel uses a learned event-type embedding and sinusoidal
    time encoding fed through an MLP, allowing arbitrary excitation and
    inhibition patterns.

    Cumulative intensity is always numerical.

    Args:
        D: Number of event types.
        num_knots: Number of spline baseline knots.
        delta_t: Knot spacing (float for uniform, tensor for variable).
        baseline_params: Optional spline baseline init params.
        seed: Random seed.
        clamping_fn: Positivity function (``"softplus"``, ``"relu"``, ``"exp"``).
        terminating: If True, each event type can occur at most once.
        dataset_stats: Empirical statistics for data-driven initialisation.
        kernel_emb_dim: Event-type embedding dimension in the neural kernel.
        kernel_time_embd_dim: Sinusoidal time encoding dimension.
        kernel_n_hidden: Width of hidden layers in the kernel MLP.
        kernel_n_mlp_layers: Number of hidden layers in the kernel MLP.
    """

    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        baseline_params: Optional[UnconstrainedSplineBaselineParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        clamping_fn: str = "softplus",
        dataset_stats: Optional[DatasetStats] = None,
        kernel_emb_dim: int = 32,
        kernel_time_embd_dim: int = 32,
        kernel_n_hidden: int = 64,
        kernel_n_mlp_layers: int = 2,
    ):
        kernel_seed = seed + 1 if seed is not None else None
        baseline = SplineBaselineModule(
            D,
            num_knots=num_knots,
            delta_t=delta_t,
            params=baseline_params,
            seed=seed,
            constrained=True if clamping_fn == "relu" else False,
            dataset_stats=dataset_stats,
        )
        kernel = NeuralKernelModule(
            D,
            emb_dim=kernel_emb_dim,
            time_embd_dim=kernel_time_embd_dim,
            n_hidden=kernel_n_hidden,
            n_mlp_layers=kernel_n_mlp_layers,
            seed=kernel_seed,
        )
        super().__init__(
            resolve_clamping_fn(clamping_fn),
            baseline,
            kernel,
            D,
            seed,
            terminating=terminating,
        )

    def _default_ci_config(self) -> CIConfig:
        """Softplus(baseline + neural kernel) is smooth within an interval and jumps only at
        events → event-knotted per-interval numerical compensator (not the family global default)."""
        return CIConfig("numerical", "interval", "mc_trapezoidal", n_points=100)


class SplineBaselineNeuralSplineKernelHawkesProcess(InhibitiveHawkesProcess):
    """Hawkes process with spline baseline and neural-spline kernel.

    Intensity::

        λ(t) = softplus(spline_baseline(t) + Σ_j spline(Δt_j; MLP(emb(k_j))))

    For each past event of type ``k``, an MLP maps a learned event-type
    embedding to per-receiver spline knot heights.  This gives the temporal
    flexibility of a spline kernel without the ``O(D² × K)`` parameter cost:
    parameters scale as ``O(D × emb + MLP)``.

    The kernel has **analytical cumulative intensity** via
    ``LinearSpline.integrate``, so each past event's CI contribution is
    computed in closed form.  Because the kernel sum + baseline can be
    negative, the whole thing is wrapped in a softplus (hence
    InhibitiveHawkesProcess), and the top-level CI is numerical.

    Args:
        D: Number of event types.
        num_baseline_knots: Number of spline baseline knots.
        baseline_delta_t: Baseline knot spacing.
        num_kernel_knots: Number of spline kernel knots.
        kernel_delta_t: Kernel knot spacing.
        baseline_params: Optional baseline init.
        seed: Random seed.
        clamping_fn: Positivity function (``"softplus"``, ``"relu"``, ``"exp"``).
        terminating: If True, each event type can occur at most once.
        dataset_stats: Empirical statistics for data-driven initialisation.
        kernel_emb_dim: Event-type embedding dimension in the kernel MLP.
        kernel_n_hidden: Width of hidden layers in the kernel MLP.
        kernel_n_mlp_layers: Number of hidden layers in the kernel MLP.
    """

    def __init__(
        self,
        D: int,
        num_baseline_knots: int,
        baseline_delta_t: torch.Tensor | float,
        num_kernel_knots: int,
        kernel_delta_t: torch.Tensor | float,
        baseline_params: Optional[UnconstrainedSplineBaselineParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
        clamping_fn: str = "softplus",
        dataset_stats: Optional[DatasetStats] = None,
        kernel_emb_dim: int = 32,
        kernel_n_hidden: int = 64,
        kernel_n_mlp_layers: int = 2,
    ):
        kernel_seed = seed + 1 if seed is not None else None
        baseline = SplineBaselineModule(
            D,
            num_knots=num_baseline_knots,
            delta_t=baseline_delta_t,
            params=baseline_params,
            seed=seed,
            constrained=True if clamping_fn == "relu" else False,
            dataset_stats=dataset_stats,
        )
        kernel = NeuralSplineKernelModule(
            D,
            num_knots=num_kernel_knots,
            delta_t=kernel_delta_t,
            emb_dim=kernel_emb_dim,
            n_hidden=kernel_n_hidden,
            n_mlp_layers=kernel_n_mlp_layers,
            seed=kernel_seed,
        )
        super().__init__(
            resolve_clamping_fn(clamping_fn),
            baseline,
            kernel,
            D,
            seed,
            terminating=terminating,
        )

    def _default_ci_config(self) -> CIConfig:
        """Softplus(baseline + neural-spline kernel) is smooth within an interval, jumps only at
        events → event-knotted per-interval numerical compensator (not the family global default)."""
        return CIConfig("numerical", "interval", "mc_trapezoidal", n_points=100)

    # ------------------------------------------------------------------
    # Kernel-cache plumbing
    # ------------------------------------------------------------------
    # ``NeuralSplineKernelModule`` exposes a ``fix_context(ts)`` that runs the
    # MLP once per unique trigger type and caches outputs for the duration of
    # the block.  Wrapping ``likelihood``, ``cumulative_intensity`` and
    # ``intensity`` ensures the MLP runs once per call instead of once per
    # numerical-CI grid point (~100×).

    def likelihood(self, ts, T=None, log: bool = True):  # type: ignore[override]
        with cast(NeuralSplineKernelModule, self.kernel).fix_context(ts):
            return super().likelihood(ts, T, log=log)

    def cumulative_intensity(self, T, ts, T_low=None, ci_config=None):  # type: ignore[override]
        with cast(NeuralSplineKernelModule, self.kernel).fix_context(ts):
            return super().cumulative_intensity(T, ts, T_low=T_low, ci_config=ci_config)

    def intensity(self, t, ts):  # type: ignore[override]
        with cast(NeuralSplineKernelModule, self.kernel).fix_context(ts):
            return super().intensity(t, ts)
