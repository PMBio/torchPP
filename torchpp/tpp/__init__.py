"""Temporal point process model classes.

The abstract bases, the integration/sampling configs and every model of the
registry (Poisson, Hawkes, inhibitive Hawkes and neural TPPs).
"""

from torchpp.tpp.base_tpp import (
    AutoregressiveTPP,
    CIConfig,
    DensityBasedTPP,
    IntensityBasedTPP,
    Sample,
    SamplingConfig,
    TemporalPointProcess,
)
from torchpp.tpp.baseline import (
    BaselineTPP,
    LogLinearPoissonProcess,
    PiecewisePoissonProcess,
    PoissonProcess,
    SplinePoissonProcess,
)
from torchpp.tpp.hawkes import (
    ConstBaselineExpKernelHawkesProcess,
    ConstBaselinePiecewiseKernelHawkesProcess,
    ConstBaselineSplineKernelHawkesProcess,
    HawkesProcess,
    InhibitiveConstBaselineExpKernelHawkesProcess,
    InhibitiveHawkesProcess,
    InhibitiveSplineBaselineExpKernelHawkesProcess,
    InhibitiveSplineBaselineMLPKernelHawkesProcess,
    InhibitiveSplineBaselineMLPSplineKernelHawkesProcess,
    InhibitiveSplineBaselineSplineKernelHawkesProcess,
    LogLinearBaselineExpKernelHawkesProcess,
    PiecewiseBaselineExpKernelHawkesProcess,
    SplineBaselineExpKernelHawkesProcess,
    SplineBaselineSplineKernelHawkesProcess,
)
from torchpp.tpp.neural import (
    AttNHPProcess,
    IntensityFreeProcess,
    NeuralLogLinearProcess,
    NeuralMLPProcess,
    NeuralParametrizedTPP,
    NeuralPiecewiseConstantProcess,
    NeuralPoissonProcess,
    NeuralSplineProcess,
    NHPProcess,
    SAHPProcess,
    THPProcess,
    TransformerParametrizedTPP,
)

__all__ = [
    # Bases and configs
    "TemporalPointProcess",
    "AutoregressiveTPP",
    "IntensityBasedTPP",
    "DensityBasedTPP",
    "CIConfig",
    "SamplingConfig",
    "Sample",
    "BaselineTPP",
    "HawkesProcess",
    "InhibitiveHawkesProcess",
    "NeuralParametrizedTPP",
    "TransformerParametrizedTPP",
    # Poisson processes
    "PoissonProcess",
    "LogLinearPoissonProcess",
    "SplinePoissonProcess",
    "PiecewisePoissonProcess",
    # Hawkes processes
    "ConstBaselineExpKernelHawkesProcess",
    "LogLinearBaselineExpKernelHawkesProcess",
    "SplineBaselineExpKernelHawkesProcess",
    "PiecewiseBaselineExpKernelHawkesProcess",
    "ConstBaselineSplineKernelHawkesProcess",
    "ConstBaselinePiecewiseKernelHawkesProcess",
    "SplineBaselineSplineKernelHawkesProcess",
    # Inhibitive Hawkes processes
    "InhibitiveConstBaselineExpKernelHawkesProcess",
    "InhibitiveSplineBaselineExpKernelHawkesProcess",
    "InhibitiveSplineBaselineSplineKernelHawkesProcess",
    "InhibitiveSplineBaselineMLPKernelHawkesProcess",
    "InhibitiveSplineBaselineMLPSplineKernelHawkesProcess",
    # Neural TPPs
    "NeuralPoissonProcess",
    "NeuralLogLinearProcess",
    "NeuralSplineProcess",
    "NeuralPiecewiseConstantProcess",
    "NeuralMLPProcess",
    "THPProcess",
    "SAHPProcess",
    "IntensityFreeProcess",
    "NHPProcess",
    "AttNHPProcess",
]
