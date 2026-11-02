"""Reusable intensity building blocks for Hawkes-family models.

Provides the abstract baseline and kernel module interfaces together with
all concrete implementations (constant, linear, spline, piecewise-constant
baselines; exponential, spline, piecewise-constant kernels).
"""

from torchpp.modules.intensity import (
    BaselineIntensityModule,
    KernelIntensityModule,
    ConstantBaselineModule,
    LinearBaselineModule,
    SplineBaselineModule,
    PiecewiseConstantBaselineModule,
    ExpKernelModule,
    SplineKernelModule,
    PiecewiseConstantKernelModule,
)

__all__ = [
    # Abstract interfaces
    "BaselineIntensityModule",
    "KernelIntensityModule",
    # Baseline implementations
    "ConstantBaselineModule",
    "LinearBaselineModule",
    "SplineBaselineModule",
    "PiecewiseConstantBaselineModule",
    # Kernel implementations
    "ExpKernelModule",
    "SplineKernelModule",
    "PiecewiseConstantKernelModule",
]
