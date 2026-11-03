"""Hawkes processes: a baseline plus kernels triggered by past events.

``HawkesProcess`` sums a baseline module and a kernel module,
:math:`\\lambda_d(t) = \\mu_d(t) + \\sum_{t_i < t} \\phi_{d,m_i}(t - t_i)`, so the cumulative
intensity is closed-form when both modules have one. ``InhibitiveHawkesProcess`` lets the
kernels be negative and applies a positivity map to the sum (numerical cumulative intensity).
The concrete classes pair one baseline (constant, log-linear, spline, piecewise) with one kernel
(exponential, spline, piecewise, MLP); ``baseline_values``, ``kernel_values`` and
``branching_matrix`` read a fitted model in native units.
"""
from abc import ABC
from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Sequence, Tuple, cast
import torch
import torch.nn.functional as F

from . import _inspection
from .base_tpp import CIConfig, IntensityBasedTPP, SamplingConfig
from ..utils.event_data import BatchedMVEventData, DatasetStats, TokenRoles
from ..modules.intensity import (
    BaselineIntensityModule,
    KernelIntensityModule,
    ConstantBaselineParams,
    ConstantBaselineModule,
    LogLinearBaselineParams,
    LogLinearBaselineModule,
    SplineBaselineParams,
    SplineBaselineModule,
    PiecewiseConstantBaselineParams,
    PiecewiseConstantBaselineModule,
    ExpKernelParams,
    ExpKernelModule,
    SplineKernelParams,
    SplineKernelModule,
    PiecewiseConstantKernelParams,
    PiecewiseConstantKernelModule,
    MLPKernelModule,
    MLPSplineKernelModule,
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
        *,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        """
        Args:
            baseline: BaselineIntensityModule instance
            kernel: KernelIntensityModule instance
            D: Number of event types
            seed: Random seed
            terminating: If True, each event type can only occur once
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
        self.baseline = baseline
        self.kernel = kernel

        if self.baseline.D != D or self.kernel.D != D:
            raise ValueError(f"baseline (D={self.baseline.D}) and kernel (D={self.kernel.D}) must both have D={D}.")

    def baseline_values(self, t) -> torch.Tensor:
        """Baseline intensity ``μ_d(t)`` at native times ``t`` ``(Q,)``, in native units.  ``(Q, D)``."""
        return _inspection.baseline_values(self, t)

    def kernel_values(self, dt) -> torch.Tensor:
        """Kernel ``φ_{d,j}(Δt)`` at native lags ``dt`` ``(Q,)``, in native units.  ``(Q, D, D)``.

        Indexed ``[q, target d, source j]``: the rate added to type ``d`` at lag ``Δt`` after an
        event of type ``j`` (the layout of the exponential kernel's ``alpha``).
        """
        return _inspection.kernel_values(self, dt)

    def branching_matrix(self, t_max=None) -> torch.Tensor:
        """Expected number of type-``d`` events triggered by one type-``j`` event, ``∫_0^∞ φ_{d,j}``.

        ``(D, D)`` indexed ``[target, source]``. Exact for exponential, spline and piecewise
        kernels; kernels without a closed-form integral need ``t_max`` (native units).
        """
        return _inspection.branching_matrix(self, rate_scale=True, t_max=t_max)

    def transform_params(self) -> Tuple[torch.Tensor, ...]:
        """
        Return all parameters in constrained form.

        Returns:
            (baseline_params, *kernel_params)
        """
        baseline_params = self.baseline.transform_params()
        kernel_params = self.kernel.transform_params()
        return (*baseline_params, *kernel_params)

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Flat multi-query raw intensity = baseline + kernel, vectorised over queries.

        ``(B, Q)`` -> ``(B, Q, D)``.  Composes the modules' ``intensities_at_queries`` (fast
        for constant-baseline + exp-kernel; loop-default otherwise), so the eval grid /
        numerical CI evaluate the whole grid in one pass instead of per point.
        """
        return self.baseline.intensities_at_queries(query_times, ts) + self.kernel.intensities_at_queries(
            query_times, ts
        )

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

    The kernels may be negative (inhibition), so the sum is passed through a positivity
    map: λ(t) = clamping_fn(baseline(t) + kernel(t)). The map makes the cumulative
    intensity non-decomposable, so it is integrated numerically.
    Grid-accelerated CI inversion (_inverse_CDF, _inverse_marginal_cumulative_intensity)
    is provided by the base class for the numerical CIConfig.
    """

    def __init__(
        self,
        clamping_fn: Callable,
        baseline: BaselineIntensityModule,
        kernel: KernelIntensityModule,
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
        Args:
            clamping_fn: Positivity map applied to the summed intensity (e.g. softplus).
            baseline: BaselineIntensityModule instance
            kernel: KernelIntensityModule instance
            D: Number of event types
            seed: Random seed
            terminating: If True, each event type can only occur once
        """
        super().__init__(
            D=D,
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )
        self.baseline = baseline
        self.kernel = kernel
        self.clamping_fn = clamping_fn

        if self.baseline.D != D or self.kernel.D != D:
            raise ValueError(f"baseline (D={self.baseline.D}) and kernel (D={self.kernel.D}) must both have D={D}.")

    def _default_ci_config(self) -> CIConfig:
        """Whole-window numerical CI: the clamped ``clamping_fn(baseline + kernel)`` may
        kink within an interval, so the global placement (not event-knotted) is used."""
        return CIConfig("numerical", "global", "mc_trapezoidal", n_points=100)

    def baseline_values(self, t) -> torch.Tensor:
        """Baseline term ``μ_d(t)`` at native times ``t`` ``(Q,)``.  ``(Q, D)``.

        A pre-link term in the model's internal parametrisation, not a rate:
        ``λ_d(t) = clamping_fn(μ_d(t) + Σ φ_{d,m_i}(t − t_i)) / time_scale``.
        """
        return _inspection.baseline_values(self, t, rate_scale=False)

    def kernel_values(self, dt) -> torch.Tensor:
        """Kernel term ``φ_{d,j}(Δt)`` at native lags ``dt`` ``(Q,)``.  ``(Q, D, D)``, ``[q, target, source]``.

        Signed pre-link terms (negative = inhibition), in the same parametrisation as
        :meth:`baseline_values`.
        """
        return _inspection.kernel_values(self, dt, rate_scale=False)

    def branching_matrix(self, t_max=None) -> torch.Tensor:
        """Signed integral ``∫_0^∞ φ_{d,j}(Δt) dΔt`` over native lags.  ``(D, D)``, ``[target, source]``.

        Not an expected offspring count: the terms act before the link function.
        """
        return _inspection.branching_matrix(self, rate_scale=False, t_max=t_max)

    def transform_params(self) -> Tuple[torch.Tensor, ...]:
        """
        Return all parameters in constrained form.

        Returns:
            (baseline_params, *kernel_params)
        """
        baseline_params = self.baseline.transform_params()
        kernel_params = self.kernel.transform_params()
        return (*baseline_params, *kernel_params)

    def _intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Flat multi-query raw intensity, vectorised over queries.  ``(B, Q)`` -> ``(B, Q, D)``.

        The inhibitive analogue of :meth:`HawkesProcess._intensities_at_queries`: composes the
        modules' vectorised ``intensities_at_queries`` and applies the element-wise
        ``clamping_fn`` to the whole grid at once, evaluating the eval /
        numerical-CI grid in a single pass.
        """
        return self.clamping_fn(
            self.baseline.intensities_at_queries(query_times, ts)
            + self.kernel.intensities_at_queries(query_times, ts)
        )

    def _event_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Intensity of the observed event type at each position.  Shape: (B, L).

        Applies the positivity constraint, matching the behaviour of
        ``_intensities_at_queries`` for inhibitive Hawkes processes.
        All masking is handled by the base ``_event_log_intensity_sum``.
        """
        baseline_intensities = self.baseline.event_intensities(ts)  # (B, L)
        interaction_terms = self.kernel.event_intensities(ts)  # (B, L)
        total = self.clamping_fn(baseline_intensities + interaction_terms)  # (B, L)
        return total


class ConstBaselineExpKernelHawkesProcess(HawkesProcess):
    """Hawkes process with a constant baseline and an exponential kernel.

    .. math::
        \\lambda_d(t) = \\mu_d + \\sum_{i:\\, t_i < t} \\alpha_{d,m_i}\\,\\beta_{d,m_i}\\,
        e^{-\\beta_{d,m_i}(t - t_i)}

    :math:`\\mu_d > 0` is the background rate of type :math:`d`; :math:`\\alpha_{d,j} \\ge 0`
    is the expected number of type-:math:`d` events triggered by one type-:math:`j` event, and
    :math:`1/\\beta_{d,j}` how long that effect lasts. The cumulative intensity is closed-form.

    Parameters (internal time unit): ``baseline.mu`` ``(D,)``, ``kernel.log_alpha`` and
    ``kernel.log_beta`` ``(D, D)``, indexed ``[target, source]`` and stored unconstrained. Read
    them in native units with :meth:`baseline_values`, :meth:`kernel_values` and
    :meth:`branching_matrix` (which equals :math:`\\alpha`).

    Args:
        D: Number of event types.
        baseline_params: Initial baseline parameters; random (or data-driven) if ``None``.
        kernel_params: Initial kernel parameters; random (or data-driven) if ``None``.
        seed: Seed for the parameter initialisation.
        terminating: Each event type occurs at most once per sequence.
        dataset_stats: Statistics of the training data for a data-driven initialisation.
    """

    def __init__(
        self,
        D: int,
        *,
        baseline_params: Optional[ConstantBaselineParams] = None,
        kernel_params: Optional[ExpKernelParams] = None,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        kernel_seed = seed + 1 if seed is not None else None
        baseline = ConstantBaselineModule(D, baseline_params, seed=seed, dataset_stats=dataset_stats)
        kernel = ExpKernelModule(D, kernel_params, kernel_seed, dataset_stats=dataset_stats)

        super().__init__(
            baseline,
            kernel,
            D,
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )


class LogLinearBaselineExpKernelHawkesProcess(HawkesProcess):
    """
    Hawkes process with log-linear baseline and exponential kernel.

    This version uses the HawkesProcess base class with LogLinearBaselineModule and ExpKernelModule.
    """

    def __init__(
        self,
        D: int,
        *,
        baseline_params: Optional[LogLinearBaselineParams] = None,
        kernel_params: Optional[ExpKernelParams] = None,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
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
        baseline = LogLinearBaselineModule(baseline_params, D, seed, dataset_stats=dataset_stats)
        kernel = ExpKernelModule(D, kernel_params, kernel_seed, dataset_stats=dataset_stats)

        super().__init__(
            baseline,
            kernel,
            D,
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )


class PiecewiseBaselineExpKernelHawkesProcess(HawkesProcess):
    """Hawkes process with piecewise-constant baseline and exponential kernel.

    More flexible than constant baseline; simpler than spline (no slopes).
    """

    def __init__(
        self,
        D: int,
        *,
        baseline_num_knots: int,
        baseline_delta_t: torch.Tensor | float,
        baseline_params: Optional[PiecewiseConstantBaselineParams] = None,
        kernel_params: Optional[ExpKernelParams] = None,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        kernel_seed = seed + 1 if seed is not None else None
        baseline = PiecewiseConstantBaselineModule(
            D, num_knots=baseline_num_knots, delta_t=baseline_delta_t, params=baseline_params, seed=seed, dataset_stats=dataset_stats
        )
        kernel = ExpKernelModule(D, kernel_params, kernel_seed, dataset_stats=dataset_stats)
        super().__init__(
            baseline,
            kernel,
            D,
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )


class SplineBaselineExpKernelHawkesProcess(HawkesProcess):
    """
    Hawkes process with spline-based baseline intensity and exponential kernel.

    This version uses the HawkesProcess base class with SplineBaselineModule and ExpKernelModule.
    """

    def __init__(
        self,
        D: int,
        *,
        baseline_num_knots: int,
        baseline_delta_t: torch.Tensor | float,
        baseline_params: Optional[SplineBaselineParams] = None,
        kernel_params: Optional[ExpKernelParams] = None,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Initialize spline-based baseline exponential kernel Hawkes process.

        Args:
            D: Number of event types
            baseline_num_knots: Number of spline-baseline knots.
            baseline_delta_t: Spacing between baseline knots (scalar or tensor).
            baseline_params: Spline-baseline parameters. If None, initialized randomly.
            kernel_params: Exponential-kernel parameters. If None, initialized randomly.
            seed: Random seed
            terminating: If True, each event type can only occur once
            dataset_stats: Optional empirical statistics for data-driven init.
        """
        kernel_seed = seed + 1 if seed is not None else None
        baseline = SplineBaselineModule(
            D, num_knots=baseline_num_knots, delta_t=baseline_delta_t, params=baseline_params, seed=seed, dataset_stats=dataset_stats
        )
        kernel = ExpKernelModule(D, kernel_params, kernel_seed, dataset_stats=dataset_stats)

        super().__init__(
            baseline,
            kernel,
            D,
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )


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
        *,
        kernel_num_knots: int,
        kernel_delta_t: torch.Tensor | float,
        baseline_params: Optional[ConstantBaselineParams] = None,
        kernel_params: Optional[PiecewiseConstantKernelParams] = None,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        kernel_seed = seed + 1 if seed is not None else None
        baseline = ConstantBaselineModule(D, baseline_params, seed=seed, dataset_stats=dataset_stats)
        kernel = PiecewiseConstantKernelModule(
            D,
            num_knots=kernel_num_knots,
            delta_t=kernel_delta_t,
            params=kernel_params,
            seed=kernel_seed,
            dataset_stats=dataset_stats,
        )
        super().__init__(
            baseline,
            kernel,
            D,
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )


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
        *,
        kernel_num_knots: int,
        kernel_delta_t: torch.Tensor | float,
        baseline_params: Optional[ConstantBaselineParams] = None,
        kernel_params: Optional[SplineKernelParams] = None,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Initialize constant baseline spline kernel Hawkes process.

        Args:
            D: Number of event types
            kernel_num_knots: Number of spline knots for the kernel (on Δt axis)
            kernel_delta_t: Knot specification for the kernel (float/scalar for uniform spacing,
                or a 1-D tensor of the K-1 gaps between knots; the first knot is at 0).
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
            num_knots=kernel_num_knots,
            delta_t=kernel_delta_t,
            params=kernel_params,
            seed=kernel_seed,
            dataset_stats=dataset_stats,
        )

        super().__init__(
            baseline,
            kernel,
            D,
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )


class SplineBaselineSplineKernelHawkesProcess(HawkesProcess):
    """Hawkes process with a linear-spline baseline and linear-spline kernels.

    .. math:: \\lambda_d(t) = \\mu_d(t) + \\sum_{i:\\, t_i < t} \\phi_{d,m_i}(t - t_i)

    :math:`\\mu_d(t)` is a non-negative linear spline over absolute time with
    ``baseline_num_knots`` equally spaced knots; each kernel :math:`\\phi_{d,j}(\\Delta t)` is a
    non-negative linear spline over the elapsed time with ``kernel_num_knots`` knots starting
    at 0, and zero after its last knot. Both are integrated in closed form. The kernels can take
    any shape (delayed, multi-modal), unlike the exponential kernel.

    Args:
        D: Number of event types.
        baseline_num_knots: Number of baseline knots.
        kernel_num_knots: Number of kernel knots.
        baseline_delta_t: Baseline knot spacing in the internal time unit.
        kernel_delta_t: The ``kernel_num_knots - 1`` kernel knot gaps (or one spacing) in the
            internal time unit.
        baseline_params: Initial baseline heights; random (or data-driven) if ``None``.
        kernel_params: Initial kernel heights; random (or data-driven) if ``None``.
        seed: Seed for the parameter initialisation.
        terminating: Each event type occurs at most once per sequence.
        dataset_stats: Statistics of the training data for a data-driven initialisation.
    """

    def __init__(
        self,
        D: int,
        *,
        baseline_num_knots: int,
        kernel_num_knots: int,
        baseline_delta_t: torch.Tensor | float,
        kernel_delta_t: torch.Tensor | float,
        baseline_params: Optional[SplineBaselineParams] = None,
        kernel_params: Optional[SplineKernelParams] = None,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        kernel_seed = seed + 1 if seed is not None else None
        baseline = SplineBaselineModule(
            D,
            num_knots=baseline_num_knots,
            delta_t=baseline_delta_t,
            params=baseline_params,
            seed=seed,
            dataset_stats=dataset_stats,
        )
        kernel = SplineKernelModule(
            D,
            num_knots=kernel_num_knots,
            delta_t=kernel_delta_t,
            params=kernel_params,
            seed=kernel_seed,
            dataset_stats=dataset_stats,
        )
        super().__init__(
            baseline,
            kernel,
            D,
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )


class InhibitiveConstBaselineExpKernelHawkesProcess(InhibitiveHawkesProcess):
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
        *,
        baseline_params: Optional[ConstantBaselineParams] = None,
        kernel_params: Optional[ExpKernelParams] = None,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
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
            constrained=clamping_fn.lower() == "relu",
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
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )


class InhibitiveSplineBaselineExpKernelHawkesProcess(InhibitiveHawkesProcess):
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
        *,
        baseline_num_knots: int,
        baseline_delta_t: torch.Tensor | float,
        baseline_params: Optional[SplineBaselineParams] = None,
        kernel_params: Optional[ExpKernelParams] = None,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        clamping_fn: str = "softplus",
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Args:
            D: Number of event types
            baseline_num_knots: Number of knots for the spline baseline
            baseline_delta_t: Spacing between knots (float for uniform, tensor for variable)
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
            num_knots=baseline_num_knots,
            delta_t=baseline_delta_t,
            params=baseline_params,
            seed=seed,
            constrained=clamping_fn.lower() == "relu",
            dataset_stats=dataset_stats,
        )
        kernel = ExpKernelModule(D, kernel_params, kernel_seed, constrained=False, dataset_stats=dataset_stats)
        super().__init__(
            resolve_clamping_fn(clamping_fn),
            baseline,
            kernel,
            D,
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
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
        *,
        baseline_num_knots: int,
        kernel_num_knots: int,
        baseline_delta_t: torch.Tensor | float,
        kernel_delta_t: torch.Tensor | float,
        baseline_params: Optional[SplineBaselineParams] = None,
        kernel_params: Optional[SplineKernelParams] = None,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        clamping_fn: str = "softplus",
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Args:
            D: Number of event types
            baseline_num_knots: Number of spline baseline knots.
            kernel_num_knots: Number of spline kernel knots.
            baseline_delta_t: Baseline knot spacing in the internal time unit (float for uniform,
                tensor of gaps for variable spacing).
            kernel_delta_t: Kernel knot gaps in the internal time unit.
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
            num_knots=baseline_num_knots,
            delta_t=baseline_delta_t,
            params=baseline_params,
            seed=seed,
            constrained=clamping_fn.lower() == "relu",
            dataset_stats=dataset_stats,
        )
        kernel = SplineKernelModule(
            D,
            num_knots=kernel_num_knots,
            delta_t=kernel_delta_t,
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
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )


class InhibitiveSplineBaselineMLPKernelHawkesProcess(InhibitiveHawkesProcess):
    """Inhibitive Hawkes process with spline baseline and MLP kernel (experimental).

    Intensity: ``λ(t) = softplus(spline_baseline(t) + Σ_j MLP(emb(k_j), Δt_j))``

    The spline baseline is unconstrained (can go negative before softplus).
    The MLP kernel uses a learned event-type embedding and sinusoidal
    time encoding fed through an MLP, allowing arbitrary excitation and
    inhibition patterns.

    Cumulative intensity is always numerical.

    Args:
        D: Number of event types.
        baseline_num_knots: Number of spline baseline knots.
        baseline_delta_t: Baseline knot spacing in the internal time unit (float for uniform,
            tensor of gaps for variable spacing).
        baseline_params: Optional spline baseline init params.
        seed: Random seed.
        clamping_fn: Positivity function (``"softplus"``, ``"relu"``, ``"exp"``).
        terminating: If True, each event type can occur at most once.
        dataset_stats: Empirical statistics for data-driven initialisation.
        kernel_emb_dim: Event-type embedding dimension in the neural kernel.
        kernel_time_emb_dim: Sinusoidal time encoding dimension.
        kernel_n_hidden: Width of hidden layers in the kernel MLP.
        kernel_n_layers: Number of hidden layers in the kernel MLP.
    """

    def __init__(
        self,
        D: int,
        *,
        baseline_num_knots: int,
        baseline_delta_t: torch.Tensor | float,
        baseline_params: Optional[SplineBaselineParams] = None,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        clamping_fn: str = "softplus",
        dataset_stats: Optional[DatasetStats] = None,
        kernel_emb_dim: int = 32,
        kernel_time_emb_dim: int = 32,
        kernel_n_hidden: int = 64,
        kernel_n_layers: int = 2,
    ):
        kernel_seed = seed + 1 if seed is not None else None
        baseline = SplineBaselineModule(
            D,
            num_knots=baseline_num_knots,
            delta_t=baseline_delta_t,
            params=baseline_params,
            seed=seed,
            constrained=clamping_fn.lower() == "relu",
            dataset_stats=dataset_stats,
        )
        kernel = MLPKernelModule(
            D,
            emb_dim=kernel_emb_dim,
            time_embd_dim=kernel_time_emb_dim,
            n_hidden=kernel_n_hidden,
            n_mlp_layers=kernel_n_layers,
            seed=kernel_seed,
        )
        super().__init__(
            resolve_clamping_fn(clamping_fn),
            baseline,
            kernel,
            D,
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )

    def _default_ci_config(self) -> CIConfig:
        """Softplus(baseline + neural kernel) is smooth within an interval and jumps only at
        events → event-knotted per-interval numerical compensator (not the family global default)."""
        return CIConfig("numerical", "interval", "mc_trapezoidal", n_points=100)


class InhibitiveSplineBaselineMLPSplineKernelHawkesProcess(InhibitiveHawkesProcess):
    """Inhibitive Hawkes process with spline baseline and MLP-predicted spline kernel (experimental).

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
        baseline_num_knots: Number of spline baseline knots.
        baseline_delta_t: Baseline knot spacing in the internal time unit.
        kernel_num_knots: Number of spline kernel knots.
        kernel_delta_t: Kernel knot gaps in the internal time unit.
        baseline_params: Optional baseline init.
        seed: Random seed.
        clamping_fn: Positivity function (``"softplus"``, ``"relu"``, ``"exp"``).
        terminating: If True, each event type can occur at most once.
        dataset_stats: Empirical statistics for data-driven initialisation.
        kernel_emb_dim: Event-type embedding dimension in the kernel MLP.
        kernel_n_hidden: Width of hidden layers in the kernel MLP.
        kernel_n_layers: Number of hidden layers in the kernel MLP.
    """

    def __init__(
        self,
        D: int,
        *,
        baseline_num_knots: int,
        baseline_delta_t: torch.Tensor | float,
        kernel_num_knots: int,
        kernel_delta_t: torch.Tensor | float,
        baseline_params: Optional[SplineBaselineParams] = None,
        seed: Optional[int] = None,
        terminating: bool = False,
        time_scale: float = 1.0,
        token_roles: Optional[TokenRoles] = None,
        ci_config: Optional[CIConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        clamping_fn: str = "softplus",
        dataset_stats: Optional[DatasetStats] = None,
        kernel_emb_dim: int = 32,
        kernel_n_hidden: int = 64,
        kernel_n_layers: int = 2,
    ):
        kernel_seed = seed + 1 if seed is not None else None
        baseline = SplineBaselineModule(
            D,
            num_knots=baseline_num_knots,
            delta_t=baseline_delta_t,
            params=baseline_params,
            seed=seed,
            constrained=clamping_fn.lower() == "relu",
            dataset_stats=dataset_stats,
        )
        kernel = MLPSplineKernelModule(
            D,
            num_knots=kernel_num_knots,
            delta_t=kernel_delta_t,
            emb_dim=kernel_emb_dim,
            n_hidden=kernel_n_hidden,
            n_mlp_layers=kernel_n_layers,
            seed=kernel_seed,
        )
        super().__init__(
            resolve_clamping_fn(clamping_fn),
            baseline,
            kernel,
            D,
            seed=seed,
            terminating=terminating,
            time_scale=time_scale,
            token_roles=token_roles,
            ci_config=ci_config,
            sampling_config=sampling_config,
        )

    def _default_ci_config(self) -> CIConfig:
        """Softplus(baseline + neural-spline kernel) is smooth within an interval, jumps only at
        events → event-knotted per-interval numerical compensator (not the family global default)."""
        return CIConfig("numerical", "interval", "mc_trapezoidal", n_points=100)

    # ------------------------------------------------------------------
    # Kernel-cache plumbing
    # ------------------------------------------------------------------
    # ``MLPSplineKernelModule`` exposes a ``fix_context(ts)`` that runs the
    # MLP once per unique trigger type and caches outputs for the duration of
    # the block.  Wrapping ``likelihood``, ``cumulative_intensity`` and
    # ``intensity`` ensures the MLP runs once per call instead of once per
    # numerical-CI grid point (~100×).
    #
    # The cache must be primed on the SCALED batch: the public shells here take
    # native times, but the base methods run the core on ``_scaled_ts(ts)``, so the
    # kernel is queried in scaled units.  Priming on native left the (content-keyed)
    # cache unmatched at ``time_scale != 1`` — the guard then raised on every real
    # dataset.  ``_prime_kernel_scaled`` scales once, so no contract override repeats
    # ``_scaled_ts`` at its call site.

    @contextmanager
    def _prime_kernel_scaled(self, ts: BatchedMVEventData) -> Iterator[None]:
        """Prime the neural-spline kernel cache on the SCALED batch for one contract call.

        ``_scaled_ts`` is deterministic, so the primed content key matches the scaled
        core's kernel queries (MLP runs once per call instead of once per CI grid point).
        """
        with cast(MLPSplineKernelModule, self.kernel).fix_context(self._scaled_ts(ts)):
            yield

    def likelihood(self, ts, T=None, *, log: bool = True):  # type: ignore[override]
        with self._prime_kernel_scaled(ts):
            return super().likelihood(ts, T, log=log)

    def cumulative_intensity(self, T, ts, T_low=None, ci_config=None):  # type: ignore[override]
        with self._prime_kernel_scaled(ts):
            return super().cumulative_intensity(T, ts, T_low=T_low, ci_config=ci_config)

    def intensity(self, t, ts):  # type: ignore[override]
        with self._prime_kernel_scaled(ts):
            return super().intensity(t, ts)
