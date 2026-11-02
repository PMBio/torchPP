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
    ConditionalInhomogeniousPoissonProcess,
    SplinePoissonProcess,
    PiecewiseConstantPoissonProcess,
)
from torchpp.tpp.hawkes import (
    ConstBaselineExpKernelHawkesProcess,
    LinearBaselineExpKernelHawkesProcess,
    SplineBaselineExpKernelHawkesProcess,
    PiecewiseBaselineExpKernelHawkesProcess,
    NumericalSplineBaselineExpKernelHawkesProcess,
    InhibitiveConstExpHawkesProcess,
    InhibitiveSplineExpHawkesProcess,
    InhibitiveSplineBaselineSplineKernelHawkesProcess,
    ConstBaselineSplineKernelHawkesProcess,
    ConstBaselinePiecewiseKernelHawkesProcess,
    SplineBaselineSplineKernelHawkesProcess,
    SplineBaselineNSSplineKernelHawkesProcess,
    ConstBaselineNSExpKernelHawkesProcess,
    SplineBaselineNSExpKernelHawkesProcess,
    SplineBaselineNeuralKernelHawkesProcess,
    SplineBaselineNeuralSplineKernelHawkesProcess,
)
from torchpp.tpp.neural import (
    NeuralPoissonProcess,
    NeuralLogLinearProcess,
    NeuralSplineProcess,
    NeuralPiecewiseConstantProcess,
    NeuralTHPProcess,
    NeuralSAHPProcess,
    EasyTPPTHPProcess,
    EasyTPPSAHPProcess,
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

# Numerical-CI models whose legacy ctor pinned *both* the integration method and the point
# count (create_model reproduces that by forcing rule + n_points on their CIConfig).  All
# other numerical models keep their natural rule and only override n_points.
_CI_METHOD_FORCED_MODEL_TYPES = {
    "numerical_spline_exp_hawkes",
    "inhibitive_const_exp_hawkes",
    "inhibitive_spline_exp_hawkes",
    "inhibitive_spline_spline_kernel",
    "neural_hawkes",
    "neural_spline_hawkes",
}


# =============================================================================
# Model matrix
# =============================================================================

# Single source of truth: the full matrix is exactly VALID_MODEL_TYPES, so any
# model added to the registry is automatically exercised (create_model below
# must grow a matching branch — the registry smoke test fails loudly on a
# missing type rather than skipping it silently).
MODEL_TYPES = list(VALID_MODEL_TYPES)

# Contract archetypes: a representative model per distinct base-class branch.
# Relation/shape contracts run over THIS set instead of the full 29-model fan;
# the fan is reserved for genuine per-backend numerical (CI-value) checks.
#
# The set is chosen to span the CI-taxonomy dispatch (CIConfig =
# mode {closed_form|numerical} x placement {interval|global}) plus the
# density-based branch, so every ``cumulative_intensity`` code path is exercised:
#   hawkes                       -> closed_form CI (analytic _interval_ci); d/dt Λ ≈ λ valid
#   piecewise_constant           -> closed_form CI, step intensity (piecewise-linear CI / kink)
#   neural_log_linear            -> closed_form CI on a neural (log-linear/exp) head
#   inhibitive_const_exp_hawkes  -> numerical CI, placement="global" (uniform-window quadrature)
#   nhp                          -> numerical CI, placement="interval" (encoder-backed, event-knotted quadrature)
#   intensity_free               -> density-based (DensityBasedTPP); native log-normal density x mark head
# (attnhp — numerical interval + multi-query/chunk backend — is NOT an archetype;
#  its hook overrides are covered per-backend in tests/models/, which fan the full matrix.)
ARCHETYPES = [
    "hawkes",
    "inhibitive_const_exp_hawkes",
    "piecewise_constant",
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
    "piecewise_constant",
    "piecewise_exp_hawkes",
    "const_piecewise_kernel",
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
    "easytpp_thp",
}

# Log-linear (exponential) tail extrapolation: with a positive tail slope
# lambda(t) ~ exp(b*t) genuinely diverges, so lambda and Lambda overflow float32
# far before T=1e6 — finiteness at extreme T is mathematically unsatisfiable.
DIVERGENT_INTENSITY_MODEL_TYPES = {
    "neural_log_linear",
    "neural_thp",
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


def create_model(model_type: str, d: int = D, seed: int = SEED, terminating: bool = False):
    """Construct a TPP model by registry name, moved onto :data:`DEVICE`.

    Args:
        model_type: A name from :data:`MODEL_TYPES`.
        d: Number of event types.
        seed: Initialisation seed.
        terminating: If True, each event type can occur at most once per sequence.
    """
    # Neural encoders/heads init from the GLOBAL torch RNG at construction; pin it here so a given
    # ``seed`` yields reproducible weights regardless of ambient RNG / collection order (the ``seed``
    # ctor arg only builds the sampling generator).
    torch.manual_seed(seed)

    if model_type == "poisson":
        model = PoissonProcess(D=d, seed=seed, terminating=terminating)

    elif model_type == "inhomogeneous_poisson":
        model = ConditionalInhomogeniousPoissonProcess(D=d, seed=seed, terminating=terminating)

    elif model_type == "spline_poisson":
        model = SplinePoissonProcess(d, SPLINE_K, SPLINE_DELTA_T, seed=seed, terminating=terminating)

    elif model_type == "piecewise_constant":
        model = PiecewiseConstantPoissonProcess(d, SPLINE_K, SPLINE_DELTA_T, seed=seed, terminating=terminating)

    elif model_type == "hawkes":
        model = ConstBaselineExpKernelHawkesProcess(D=d, seed=seed, terminating=terminating)

    elif model_type == "linear_exp_hawkes":
        model = LinearBaselineExpKernelHawkesProcess(D=d, seed=seed, terminating=terminating)

    elif model_type == "spline_exp_hawkes":
        model = SplineBaselineExpKernelHawkesProcess(
            D=d, num_knots=SPLINE_K, delta_t=SPLINE_DELTA_T, seed=seed, terminating=terminating
        )

    elif model_type == "piecewise_exp_hawkes":
        model = PiecewiseBaselineExpKernelHawkesProcess(
            D=d, num_knots=SPLINE_K, delta_t=SPLINE_DELTA_T, seed=seed, terminating=terminating
        )

    elif model_type == "numerical_spline_exp_hawkes":
        model = NumericalSplineBaselineExpKernelHawkesProcess(
            D=d,
            num_knots=SPLINE_K,
            delta_t=SPLINE_DELTA_T,
            seed=seed,
            terminating=terminating,
        )

    elif model_type == "inhibitive_const_exp_hawkes":
        model = InhibitiveConstExpHawkesProcess(
            D=d,
            seed=seed,
            terminating=terminating,
        )

    elif model_type == "inhibitive_spline_exp_hawkes":
        model = InhibitiveSplineExpHawkesProcess(
            D=d,
            num_knots=SPLINE_K,
            delta_t=SPLINE_DELTA_T,
            seed=seed,
            terminating=terminating,
        )

    elif model_type == "const_spline_kernel":
        model = ConstBaselineSplineKernelHawkesProcess(
            D=d,
            num_knots=SPLINE_K,
            delta_t=SPLINE_DELTA_T,
            seed=seed,
            terminating=terminating,
        )

    elif model_type == "const_piecewise_kernel":
        model = ConstBaselinePiecewiseKernelHawkesProcess(
            D=d,
            num_knots=SPLINE_K,
            delta_t=SPLINE_DELTA_T,
            seed=seed,
            terminating=terminating,
        )

    elif model_type == "spline_spline_kernel":
        model = SplineBaselineSplineKernelHawkesProcess(
            D=d,
            num_knots_baseline=SPLINE_K,
            num_knots_kernel=SPLINE_K,
            delta_t_baseline=SPLINE_DELTA_T,
            delta_t_kernel=SPLINE_DELTA_T,
            seed=seed,
            terminating=terminating,
        )

    elif model_type == "spline_ns_spline_kernel":
        model = SplineBaselineNSSplineKernelHawkesProcess(
            D=d,
            num_knots_baseline=SPLINE_K,
            num_knots_ns=3,
            num_knots_kernel=SPLINE_K,
            delta_t_baseline=SPLINE_DELTA_T,
            delta_t_ns=2.0,
            delta_t_kernel=SPLINE_DELTA_T,
            seed=seed,
            terminating=terminating,
        )

    elif model_type == "ns_exp_hawkes":
        model = ConstBaselineNSExpKernelHawkesProcess(
            D=d,
            num_knots=SPLINE_K,
            delta_t=SPLINE_DELTA_T,
            seed=seed,
            terminating=terminating,
        )

    elif model_type == "spline_ns_exp_hawkes":
        model = SplineBaselineNSExpKernelHawkesProcess(
            D=d,
            num_knots_baseline=SPLINE_K,
            num_knots_kernel=SPLINE_K,
            delta_t_baseline=SPLINE_DELTA_T,
            delta_t_kernel=SPLINE_DELTA_T,
            seed=seed,
            terminating=terminating,
        )

    elif model_type == "inhibitive_spline_spline_kernel":
        model = InhibitiveSplineBaselineSplineKernelHawkesProcess(
            D=d,
            num_knots_baseline=SPLINE_K,
            num_knots_kernel=SPLINE_K,
            delta_t_baseline=SPLINE_DELTA_T,
            delta_t_kernel=SPLINE_DELTA_T,
            seed=seed,
            terminating=terminating,
        )

    elif model_type == "neural_poisson":
        cfg = TransformerConfig(num_event_types=d, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
        model = NeuralPoissonProcess(cfg, terminating=terminating, seed=seed)

    elif model_type == "neural_log_linear":
        cfg = TransformerConfig(num_event_types=d, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
        model = NeuralLogLinearProcess(cfg, terminating=terminating, seed=seed)

    elif model_type == "neural_spline":
        cfg = TransformerConfig(num_event_types=d, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
        model = NeuralSplineProcess(cfg, num_spline_bins=5, spline_t_max=4.0, terminating=terminating, seed=seed)

    elif model_type == "neural_piecewise_constant":
        cfg = TransformerConfig(num_event_types=d, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
        model = NeuralPiecewiseConstantProcess(cfg, num_bins=5, bin_t_max=4.0, terminating=terminating, seed=seed)

    elif model_type == "neural_thp":
        cfg = TransformerConfig(num_event_types=d, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
        model = NeuralTHPProcess(cfg, terminating=terminating, seed=seed)

    elif model_type == "neural_sahp":
        cfg = TransformerConfig(num_event_types=d, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
        model = NeuralSAHPProcess(cfg, terminating=terminating, seed=seed)

    elif model_type == "easytpp_thp":
        cfg = TransformerConfig(num_event_types=d, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
        model = EasyTPPTHPProcess(cfg, terminating=terminating, seed=seed)

    elif model_type == "easytpp_sahp":
        cfg = TransformerConfig(num_event_types=d, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
        model = EasyTPPSAHPProcess(cfg, terminating=terminating, seed=seed)

    elif model_type == "intensity_free":
        encoder = GRUEncoder(num_event_types=d, n_embd=16, num_layers=1, dropout=0.0)
        model = IntensityFreeProcess(
            encoder, num_mix_components=4, terminating=terminating, seed=seed
        )

    elif model_type == "nhp":
        encoder = ContTimeLSTMEncoder(num_event_types=d, hidden_dim=16)
        model = NHPProcess(encoder, terminating=terminating, seed=seed)

    elif model_type == "attnhp":
        encoder = AttNHPEncoder(num_event_types=d, d_model=16, d_time=8, n_layers=1, n_head=2)
        model = AttNHPProcess(encoder, terminating=terminating, seed=seed)

    elif model_type == "neural_mlp":
        cfg = TransformerConfig(num_event_types=d, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
        model = NeuralMLPProcess(cfg, terminating=terminating, seed=seed)

    elif model_type == "neural_hawkes":
        model = SplineBaselineNeuralKernelHawkesProcess(
            D=d,
            num_knots=4,
            delta_t=0.2,
            seed=seed,
            terminating=terminating,
            kernel_emb_dim=8,
            kernel_time_embd_dim=8,
            kernel_n_hidden=16,
            kernel_n_mlp_layers=1,
        )

    elif model_type == "neural_spline_hawkes":
        model = SplineBaselineNeuralSplineKernelHawkesProcess(
            D=d,
            num_baseline_knots=4,
            baseline_delta_t=0.2,
            num_kernel_knots=4,
            kernel_delta_t=0.2,
            seed=seed,
            terminating=terminating,
            kernel_emb_dim=8,
            kernel_n_hidden=16,
            kernel_n_mlp_layers=1,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # CIConfig-D: reproduce the legacy numerical-CI knobs the old ctor args set.  The
    # methods-forced group pinned rule + n_points; the rest pinned only n_points (natural
    # rule kept).  Closed-form models keep their default (no-op here).  Density models
    # (DensityBasedTPP, e.g. intensity_free) have no CIConfig — skip.
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

    padded_seq1 = MVEventData(
        torch.tensor([0.2, 0.8, 2.0, 2.3, 2.5, 2.8], dtype=float), torch.tensor([0, 0, 1, 0, -1, -1], dtype=int)
    ).to(device)

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
        "padded_seq1": padded_seq1,
        "simul_seq": simul_seq,
        "simul_leading_seq": simul_leading_seq,
        "batch_single": BatchedMVEventData([seq1.time_points], [seq1.event_types]).to(device),
        "batch_multi": BatchedMVEventData(
            [seq1.time_points, seq2.time_points], [seq1.event_types, seq2.event_types]
        ).to(device),
        "empty_batch": BatchedMVEventData([empty_seq.time_points], [empty_seq.event_types]).to(device),
        "padded_batch": BatchedMVEventData([padded_seq1.time_points], [padded_seq1.event_types]).to(device),
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

    padded_seq = MVEventData(
        torch.tensor([0.5, 1.5, 2.0, 2.5], dtype=torch.float64), torch.tensor([0, 2, -1, -1], dtype=torch.long)
    ).to(device)

    batched_padded_seq = BatchedMVEventData(
        [torch.tensor([0.5, 1.5, 2.0, 2.5], dtype=torch.float64)], [torch.tensor([0, 2, -1, -1], dtype=torch.long)]
    ).to(device)
    batched_padded_seq.seq_lengths = torch.tensor([2], dtype=torch.long)  # Only first 2 events are valid

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
        "padded_seq": padded_seq,
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
