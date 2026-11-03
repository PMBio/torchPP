"""Shared test helpers: constants, the model factory, and event-data builders.

Importable from any test module (``pythonpath = tests`` in ``pytest.ini``), so
files in ``tests/unit/…``, ``tests/contract/…`` etc. all share one source of
truth for the model matrix and the synthetic sequences. Pytest *fixtures* live
in ``conftest.py``; plain values and constructors live here.
"""

import os
from dataclasses import replace

import torch

from torchpp.training.config import VALID_MODEL_TYPES
from torchpp.utils.event_data import MVEventData, BatchedMVEventData
from torchpp.tpp.baseline import (
    PoissonProcess,
    LogLinearPoissonProcess,
    SplinePoissonProcess,
    PiecewisePoissonProcess,
)
from torchpp.tpp.hawkes import (
    ConstBaselineExpKernelHawkesProcess,
    LogLinearBaselineExpKernelHawkesProcess,
    SplineBaselineExpKernelHawkesProcess,
    PiecewiseBaselineExpKernelHawkesProcess,
    InhibitiveConstBaselineExpKernelHawkesProcess,
    InhibitiveSplineBaselineExpKernelHawkesProcess,
    InhibitiveSplineBaselineSplineKernelHawkesProcess,
    ConstBaselineSplineKernelHawkesProcess,
    ConstBaselinePiecewiseKernelHawkesProcess,
    SplineBaselineSplineKernelHawkesProcess,
    InhibitiveSplineBaselineMLPKernelHawkesProcess,
    InhibitiveSplineBaselineMLPSplineKernelHawkesProcess,
)
from torchpp.tpp.neural import (
    NeuralPoissonProcess,
    NeuralLogLinearProcess,
    NeuralSplineProcess,
    NeuralPiecewiseConstantProcess,
    THPProcess,
    SAHPProcess,
    IntensityFreeProcess,
    NHPProcess,
    AttNHPProcess,
    NeuralMLPProcess,
)
from torchpp.modules.nn.transformer import TransformerConfig
from torchpp.modules.nn.encoder import GRUEncoder, ContTimeLSTMEncoder
from torchpp.modules.nn.attnhp import AttNHPEncoder


# =============================================================================
# Test Configuration Constants
# =============================================================================

D = 3  # Number of event types
SEED = 42
SPLINE_K = 15
SPLINE_DELTA_T = 4.0 / SPLINE_K


def _resolve_device() -> torch.device:
    """Test device: ``TPP_TEST_DEVICE`` env override, else cuda-if-available.

    Set ``TPP_TEST_DEVICE=cpu`` to run the device-stable unit/contract gate
    anywhere (the default picks CUDA when present).
    """
    requested = os.environ.get("TPP_TEST_DEVICE")
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = _resolve_device()

# Tolerances
TIGHT_TOLERANCE = 1e-4
STANDARD_TOLERANCE = 1e-2
SAMPLING_TOLERANCE = 0.1

# Numerical settings
NUM_INTEGRATION_STEPS = 500
NUM_SAMPLING_ITERATIONS = 500
NUMERICAL_CI_POINTS = 200
NUMERICAL_CI_METHOD = "trapezoidal"
EPSILON = 1e-4

# Numerical-CI models whose tests pin *both* the integration method and the point
# count (create_model forces rule + n_points on their CIConfig).  All
# other numerical models keep their natural rule and only override n_points.
_CI_METHOD_FORCED_MODEL_TYPES = {
    "inhibitive_const_exp_hawkes",
    "inhibitive_spline_exp_hawkes",
    "inhibitive_spline_spline_hawkes",
    "inhibitive_spline_mlp_hawkes",
    "inhibitive_spline_mlpspline_hawkes",
}


# =============================================================================
# Model matrix
# =============================================================================

# Single source of truth: the full matrix is exactly VALID_MODEL_TYPES, so any
# model added to the registry is automatically exercised (``_BUILDERS`` below must
# grow a matching entry — importing this module fails on a missing type).
MODEL_TYPES = list(VALID_MODEL_TYPES)

# Contract archetypes: a representative model per distinct base-class branch.
# Relation/shape contracts run over THIS set instead of the full model fan;
# the fan is reserved for genuine per-backend numerical (CI-value) checks.
#
# The set is chosen to span the CI-taxonomy dispatch (CIConfig =
# mode {closed_form|numerical} x placement {interval|global}) plus the
# density-based branch, so every ``cumulative_intensity`` code path is exercised:
#   const_exp_hawkes             -> closed_form CI (analytic _interval_ci); d/dt Λ ≈ λ valid
#   piecewise_poisson            -> closed_form CI, step intensity (piecewise-linear CI / kink)
#   neural_log_linear            -> closed_form CI on a neural (log-linear/exp) head
#   inhibitive_const_exp_hawkes  -> numerical CI, placement="global" (uniform-window quadrature)
#   nhp                          -> numerical CI, placement="interval" (encoder-backed, event-knotted quadrature)
#   intensity_free               -> density-based (DensityBasedTPP); native log-normal density x mark head
# (attnhp — numerical interval + multi-query/chunk backend — is NOT an archetype;
#  its hook overrides are covered per-backend in tests/models/, which fan the full matrix.)
ARCHETYPES = [
    "const_exp_hawkes",
    "inhibitive_const_exp_hawkes",
    "piecewise_poisson",
    "nhp",
    "intensity_free",
    "neural_log_linear",
]

# ---------------------------------------------------------------------------
# Numerical-property families (per-model NUMERICAL tests in tests/models/ use
# these to skip ill-posed cases). Relation/shape contracts are universal and do
# NOT consult these sets.
# ---------------------------------------------------------------------------

# Step-function intensity (piecewise-constant baseline or kernel): cumulative
# intensity is piecewise-linear with kinks, so a finite-difference derivative of
# the CI does not approximate the intensity near the knots.
STEP_INTENSITY_MODEL_TYPES = {
    "piecewise_poisson",
    "piecewise_exp_hawkes",
    "const_piecewise_hawkes",
    "neural_piecewise_constant",
}

# Heavy-tailed log-normal density: coarse trapezoid grids undershoot the
# marginal mass / CDF boundary (closed-form quantities are correct).
HEAVY_TAIL_DENSITY_MODEL_TYPES = {
    "intensity_free",
}

# No genuine per-type intensities (IntensityFree = shared hazard * p(d|H)), so
# the partial-termination "CDF -> 1" invariant is ill-posed.
NO_PER_TYPE_INTENSITY_MODEL_TYPES = {
    "intensity_free",
}

# Defective inter-event density at random init: the active-type intensity decays
# toward zero, so the survival function never reaches 0 and the partial-termination
# CDF plateaus below 1. The "CDF -> 1 as t -> inf" invariant is ill-posed for these.
DEFECTIVE_CDF_MODEL_TYPES = {
    "thp",
}

# Log-linear (exponential) tail extrapolation: with a positive tail slope
# lambda(t) ~ exp(b*t) genuinely diverges, so lambda and Lambda overflow float32
# far before T=1e6 — finiteness at extreme T is mathematically unsatisfiable.
DIVERGENT_INTENSITY_MODEL_TYPES = {
    "neural_log_linear",
}

# Heads whose parameters ARE the rate (no exp remapping of column 0): the zeros
# no-history column 0 is a literal-zero no-history intensity.
# Column 0 is structurally never scored (no-origin) nor integrated ([0, t₁] excluded), so
# a zero there is the correct contract — not a floored rate. (Contrast neural_log_linear,
# whose exp head maps the zeros column to exp(0)=1, so its no-history intensity is positive.)
ZERO_NO_HISTORY_INTENSITY_MODEL_TYPES = {
    "neural_poisson",
    "neural_spline",
    "neural_piecewise_constant",
}


# =============================================================================
# Contract-fan helper
# =============================================================================


def skip_if_density(model):
    """Skip an intensity/CI-specific assertion for density-based models.

    The universal predictive contract (likelihood / density_* / prob_* / sample / time_*) runs for
    every archetype; assertions that reach into ``intensity`` / ``cumulative_intensity`` or
    the intensity hooks are intensity-family-specific and do not apply to a
    :class:`~torchpp.tpp.base_tpp.DensityBasedTPP` (e.g. ``intensity_free``).
    """
    import pytest
    from torchpp.tpp.base_tpp import IntensityBasedTPP

    if not isinstance(model, IntensityBasedTPP):
        pytest.skip("intensity/CI-specific assertion; not applicable to density-based models")


# =============================================================================
# Model factory
# =============================================================================


def _tiny_transformer(d):
    return TransformerConfig(num_event_types=d, n_layer=1, n_head=2, n_embd=16, dropout=0.0)


# Small test configurations per model_type: (d, seed, terminating) -> model.
_BUILDERS = {
    "poisson": lambda d, s, t: PoissonProcess(D=d, seed=s, terminating=t),
    "loglinear_poisson": lambda d, s, t: LogLinearPoissonProcess(D=d, seed=s, terminating=t),
    "spline_poisson": lambda d, s, t: SplinePoissonProcess(
        d, baseline_num_knots=SPLINE_K, baseline_delta_t=SPLINE_DELTA_T, seed=s, terminating=t
    ),
    "piecewise_poisson": lambda d, s, t: PiecewisePoissonProcess(
        d, baseline_num_knots=SPLINE_K, baseline_delta_t=SPLINE_DELTA_T, seed=s, terminating=t
    ),
    "const_exp_hawkes": lambda d, s, t: ConstBaselineExpKernelHawkesProcess(D=d, seed=s, terminating=t),
    "loglinear_exp_hawkes": lambda d, s, t: LogLinearBaselineExpKernelHawkesProcess(D=d, seed=s, terminating=t),
    "spline_exp_hawkes": lambda d, s, t: SplineBaselineExpKernelHawkesProcess(
        D=d, baseline_num_knots=SPLINE_K, baseline_delta_t=SPLINE_DELTA_T, seed=s, terminating=t
    ),
    "piecewise_exp_hawkes": lambda d, s, t: PiecewiseBaselineExpKernelHawkesProcess(
        D=d, baseline_num_knots=SPLINE_K, baseline_delta_t=SPLINE_DELTA_T, seed=s, terminating=t
    ),
    "const_spline_hawkes": lambda d, s, t: ConstBaselineSplineKernelHawkesProcess(
        D=d, kernel_num_knots=SPLINE_K, kernel_delta_t=SPLINE_DELTA_T, seed=s, terminating=t
    ),
    "const_piecewise_hawkes": lambda d, s, t: ConstBaselinePiecewiseKernelHawkesProcess(
        D=d, kernel_num_knots=SPLINE_K, kernel_delta_t=SPLINE_DELTA_T, seed=s, terminating=t
    ),
    "spline_spline_hawkes": lambda d, s, t: SplineBaselineSplineKernelHawkesProcess(
        D=d,
        baseline_num_knots=SPLINE_K,
        kernel_num_knots=SPLINE_K,
        baseline_delta_t=SPLINE_DELTA_T,
        kernel_delta_t=SPLINE_DELTA_T,
        seed=s,
        terminating=t,
    ),
    "inhibitive_const_exp_hawkes": lambda d, s, t: InhibitiveConstBaselineExpKernelHawkesProcess(
        D=d, seed=s, terminating=t
    ),
    "inhibitive_spline_exp_hawkes": lambda d, s, t: InhibitiveSplineBaselineExpKernelHawkesProcess(
        D=d, baseline_num_knots=SPLINE_K, baseline_delta_t=SPLINE_DELTA_T, seed=s, terminating=t
    ),
    "inhibitive_spline_spline_hawkes": lambda d, s, t: InhibitiveSplineBaselineSplineKernelHawkesProcess(
        D=d,
        baseline_num_knots=SPLINE_K,
        kernel_num_knots=SPLINE_K,
        baseline_delta_t=SPLINE_DELTA_T,
        kernel_delta_t=SPLINE_DELTA_T,
        seed=s,
        terminating=t,
    ),
    "inhibitive_spline_mlp_hawkes": lambda d, s, t: InhibitiveSplineBaselineMLPKernelHawkesProcess(
        D=d,
        baseline_num_knots=4,
        baseline_delta_t=0.2,
        seed=s,
        terminating=t,
        kernel_emb_dim=8,
        kernel_time_emb_dim=8,
        kernel_n_hidden=16,
        kernel_n_layers=1,
    ),
    "inhibitive_spline_mlpspline_hawkes": lambda d, s, t: InhibitiveSplineBaselineMLPSplineKernelHawkesProcess(
        D=d,
        baseline_num_knots=4,
        baseline_delta_t=0.2,
        kernel_num_knots=4,
        kernel_delta_t=0.2,
        seed=s,
        terminating=t,
        kernel_emb_dim=8,
        kernel_n_hidden=16,
        kernel_n_layers=1,
    ),
    "neural_poisson": lambda d, s, t: NeuralPoissonProcess(config=_tiny_transformer(d), terminating=t, seed=s),
    "neural_log_linear": lambda d, s, t: NeuralLogLinearProcess(config=_tiny_transformer(d), terminating=t, seed=s),
    "neural_spline": lambda d, s, t: NeuralSplineProcess(
        config=_tiny_transformer(d), num_bins=5, t_max=4.0, terminating=t, seed=s
    ),
    "neural_mlp": lambda d, s, t: NeuralMLPProcess(config=_tiny_transformer(d), terminating=t, seed=s),
    "neural_piecewise_constant": lambda d, s, t: NeuralPiecewiseConstantProcess(
        config=_tiny_transformer(d), num_bins=5, t_max=4.0, terminating=t, seed=s
    ),
    "thp": lambda d, s, t: THPProcess(config=_tiny_transformer(d), terminating=t, seed=s),
    "sahp": lambda d, s, t: SAHPProcess(config=_tiny_transformer(d), terminating=t, seed=s),
    "intensity_free": lambda d, s, t: IntensityFreeProcess(
        GRUEncoder(num_event_types=d, n_embd=16, num_layers=1, dropout=0.0), num_mix_components=4, terminating=t, seed=s
    ),
    "nhp": lambda d, s, t: NHPProcess(ContTimeLSTMEncoder(num_event_types=d, hidden_dim=16), terminating=t, seed=s),
    "attnhp": lambda d, s, t: AttNHPProcess(
        AttNHPEncoder(num_event_types=d, d_model=16, d_time=8, n_layers=1, n_head=2), terminating=t, seed=s
    ),
}
assert set(_BUILDERS) == set(VALID_MODEL_TYPES), sorted(set(VALID_MODEL_TYPES) ^ set(_BUILDERS))


def create_model(model_type: str, d: int = D, seed: int = SEED, terminating: bool = False):
    """Construct a TPP model by registry name (small test configuration), moved onto :data:`DEVICE`.

    Args:
        model_type: A name from :data:`MODEL_TYPES`.
        d: Number of event types.
        seed: Initialisation seed.
        terminating: If True, each event type can occur at most once per sequence.
    """
    if model_type not in _BUILDERS:
        raise ValueError(f"Unknown model type: {model_type}")
    # Neural encoders/heads init from the GLOBAL torch RNG at construction; pin it here so a given
    # ``seed`` yields reproducible weights regardless of ambient RNG / collection order.
    torch.manual_seed(seed)
    model = _BUILDERS[model_type](d, seed, terminating)

    # Pin the numerical-CI settings the tests rely on: the methods-forced group pins rule +
    # n_points, the rest only n_points (natural rule kept). Closed-form models keep their
    # default (no-op here). Density models (DensityBasedTPP, e.g. intensity_free) have no
    # CIConfig — skip.
    if hasattr(model, "ci_config") and model.ci_config.mode == "numerical":
        if model_type in _CI_METHOD_FORCED_MODEL_TYPES:
            model.ci_config = replace(model.ci_config, n_points=NUMERICAL_CI_POINTS, rule=NUMERICAL_CI_METHOD)
        else:
            model.ci_config = replace(model.ci_config, n_points=NUMERICAL_CI_POINTS)

    return model.to(DEVICE)


def create_model_pair(model_type: str, d: int = D, seed: int = SEED):
    """A (terminating, non-terminating) pair sharing identical parameters.

    Parameters are copied from the terminating model so the two differ only in
    the termination semantics — the basis for the trust-chain comparisons.
    """
    model_term = create_model(model_type, d, seed, terminating=True)
    model_non_term = create_model(model_type, d, seed, terminating=False)
    model_non_term.load_state_dict(model_term.state_dict())
    return model_term, model_non_term


# =============================================================================
# Event-data builders
# =============================================================================


def build_test_data(device: torch.device = DEVICE) -> dict:
    """Standard non-terminating sequences (single, batched, padded, simultaneous)."""
    empty_seq = MVEventData(torch.tensor([]), torch.tensor([], dtype=int)).to(device)

    seq1 = MVEventData(
        torch.tensor([0.2, 0.8, 2.0, 2.3], dtype=float), torch.tensor([0, 0, 1, 0], dtype=int)
    ).to(device)

    seq2 = MVEventData(
        torch.tensor([0.1, 0.5, 1.2, 1.8, 2.5], dtype=float), torch.tensor([1, 0, 1, 0, 1], dtype=int)
    ).to(device)

    # Trailing -1 entries are padding; padding exists only inside a batch.
    padded_times = torch.tensor([0.2, 0.8, 2.0, 2.3, 2.5, 2.8], dtype=float)
    padded_types = torch.tensor([0, 0, 1, 0, -1, -1], dtype=int)

    # Two simultaneous events: types 1 and 2 at t=1.0
    simul_seq = MVEventData(
        torch.tensor([0.2, 1.0, 1.0, 2.3], dtype=float),
        torch.tensor([0, 1, 2, 0], dtype=int),
    ).to(device)

    # A run of three simultaneous events right at the start
    simul_leading_seq = MVEventData(
        torch.tensor([0.5, 0.5, 0.5, 1.5], dtype=float),
        torch.tensor([0, 1, 2, 0], dtype=int),
    ).to(device)

    return {
        "empty_seq": empty_seq,
        "seq1": seq1,
        "seq2": seq2,
        "simul_seq": simul_seq,
        "simul_leading_seq": simul_leading_seq,
        "batch_single": BatchedMVEventData([seq1.time_points], [seq1.event_types]).to(device),
        "batch_multi": BatchedMVEventData(
            [seq1.time_points, seq2.time_points], [seq1.event_types, seq2.event_types]
        ).to(device),
        "empty_batch": BatchedMVEventData([empty_seq.time_points], [empty_seq.event_types]).to(device),
        "padded_batch": BatchedMVEventData([padded_times], [padded_types]).to(device),
        "simul_batch": BatchedMVEventData([simul_seq.time_points], [simul_seq.event_types]).to(device),
        "simul_leading_batch": BatchedMVEventData(
            [simul_leading_seq.time_points], [simul_leading_seq.event_types]
        ).to(device),
        "simul_multi_batch": BatchedMVEventData(
            [simul_seq.time_points, seq1.time_points],
            [simul_seq.event_types, seq1.event_types],
        ).to(device),
    }


def build_terminating_test_data(device: torch.device = DEVICE, d: int = D) -> dict:
    """Sequences for terminating tests (single, partial/full termination, repeated, padded)."""
    empty_seq = MVEventData(torch.tensor([]), torch.tensor([], dtype=torch.long)).to(device)

    single_event = MVEventData(
        torch.tensor([0.5], dtype=torch.float64), torch.tensor([0], dtype=torch.long)
    ).to(device)

    # Partial termination: types 0 and 2 occurred, type 1 still active
    partial_seq = MVEventData(
        torch.tensor([0.5, 1.5], dtype=torch.float64), torch.tensor([0, 2], dtype=torch.long)
    ).to(device)

    # Full termination: all d types occurred
    full_seq = MVEventData(
        torch.tensor([0.3, 0.8, 1.5][:d], dtype=torch.float64),
        torch.tensor(list(range(d)), dtype=torch.long),
    ).to(device)

    # Repeated types: type 0 appears twice
    repeated_seq = MVEventData(
        torch.tensor([0.3, 0.8, 1.2], dtype=torch.float64), torch.tensor([0, 1, 0], dtype=torch.long)
    ).to(device)

    out_of_order_seq = MVEventData(
        torch.tensor([0.4, 0.9, 1.3][:d], dtype=torch.float64),
        torch.tensor([d - 1, 0, d // 2][:d], dtype=torch.long),
    ).to(device)

    batched_padded_seq = BatchedMVEventData(
        [torch.tensor([0.5, 1.5, 2.0, 2.5], dtype=torch.float64)], [torch.tensor([0, 2, -1, -1], dtype=torch.long)]
    ).to(device)
    # event_types [0, 2, -1, -1] already yield real_counts == 2 (derived seq_lengths).

    seq2 = MVEventData(
        torch.tensor([0.2, 0.7, 1.1], dtype=torch.float64),
        torch.tensor([1, 0, 2][:d] if d >= 3 else [1, 0][:d], dtype=torch.long),
    ).to(device)

    return {
        "empty_seq": empty_seq,
        "single_event": single_event,
        "partial_seq": partial_seq,
        "full_seq": full_seq,
        "repeated_seq": repeated_seq,
        "out_of_order_seq": out_of_order_seq,
        "seq2": seq2,
        "empty_batch": BatchedMVEventData([empty_seq.time_points], [empty_seq.event_types]).to(device),
        "single_batch": BatchedMVEventData([single_event.time_points], [single_event.event_types]).to(device),
        "partial_batch": BatchedMVEventData([partial_seq.time_points], [partial_seq.event_types]).to(device),
        "full_batch": BatchedMVEventData([full_seq.time_points], [full_seq.event_types]).to(device),
        "repeated_batch": BatchedMVEventData([repeated_seq.time_points], [repeated_seq.event_types]).to(device),
        "padded_batch": batched_padded_seq,
        "multi_batch": BatchedMVEventData(
            [partial_seq.time_points, seq2.time_points], [partial_seq.event_types, seq2.event_types]
        ).to(device),
    }
