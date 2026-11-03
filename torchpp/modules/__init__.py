"""Reusable intensity building blocks for Hawkes-family models.

Provides the abstract baseline and kernel module interfaces together with
all concrete implementations (constant, log-linear, spline, piecewise-constant
baselines; exponential, spline, piecewise-constant and MLP kernels).
"""

from torchpp.modules.intensity import (
    BaselineIntensityModule,
    KernelIntensityModule,
    ConstantBaselineModule,
    LogLinearBaselineModule,
    SplineBaselineModule,
    PiecewiseConstantBaselineModule,
    ExpKernelModule,
    SplineKernelModule,
    PiecewiseConstantKernelModule,
    MLPKernelModule,
    MLPSplineKernelModule,
)

__all__ = [
    # Abstract interfaces
    "BaselineIntensityModule",
    "KernelIntensityModule",
    # Baseline implementations
    "ConstantBaselineModule",
    "LogLinearBaselineModule",
    "SplineBaselineModule",
    "PiecewiseConstantBaselineModule",
    # Kernel implementations
    "ExpKernelModule",
    "SplineKernelModule",
    "PiecewiseConstantKernelModule",
    "MLPKernelModule",
    "MLPSplineKernelModule",
]
