"""Shared pytest fixtures for the TPP test suite.

Layout (see ``tests/README.md``):
    unit/        modules in isolation (CPU-stable, no model factory)
    contract/    base-class API contracts over the ARCHETYPE set
    models/      per-family CI-VALUE correctness (kept fanned over the matrix)
    integration/ pointer to scripts/equivalence (full-train; not pytest)

Plain values and the model factory live in ``helpers.py`` (importable from any
subdir via ``pythonpath = tests``); this module wraps them as fixtures.
"""

import random

import numpy as np
import pytest
import torch

# Re-exported so existing ``from conftest import …`` keeps working; new modules
# should ``from helpers import …`` directly.
from helpers import (  # noqa: F401
    D,
    SEED,
    DEVICE,
    SPLINE_K,
    SPLINE_DELTA_T,
    TIGHT_TOLERANCE,
    STANDARD_TOLERANCE,
    SAMPLING_TOLERANCE,
    NUM_INTEGRATION_STEPS,
    NUM_SAMPLING_ITERATIONS,
    NUMERICAL_CI_POINTS,
    NUMERICAL_CI_METHOD,
    EPSILON,
    MODEL_TYPES,
    ARCHETYPES,
    STEP_INTENSITY_MODEL_TYPES,
    HEAVY_TAIL_DENSITY_MODEL_TYPES,
    NO_PER_TYPE_INTENSITY_MODEL_TYPES,
    DEFECTIVE_CDF_MODEL_TYPES,
    DIVERGENT_INTENSITY_MODEL_TYPES,
    create_model,
    create_model_pair,
    build_test_data,
    build_terminating_test_data,
)


_LAYER_MARKERS = ("unit", "contract", "models", "integration")


def pytest_collection_modifyitems(config, items):
    """Auto-apply the layer marker matching each test's directory."""
    for item in items:
        path = item.fspath.strpath.replace("\\", "/")
        for layer in _LAYER_MARKERS:
            if f"/{layer}/" in path:
                item.add_marker(layer)
                break


# =============================================================================
# Global fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed every RNG before each test to pin the stochastic test bodies (sampling, MC
    integration), removing cross-test ordering flakiness.

    Note: this does NOT by itself make neural model init reproducible — encoders/heads draw
    from the global RNG at construction, and module-scoped fixtures build before this
    function-scoped fixture runs.  Reproducible init is ensured by ``create_model`` (test helper
    and production factory), which reseeds the global RNG from its ``seed`` argument.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


@pytest.fixture(scope="session")
def device():
    """The resolved test device (see ``TPP_TEST_DEVICE``)."""
    return DEVICE


@pytest.fixture
def d():
    """Number of event types used across the suite."""
    return D


@pytest.fixture
def tolerances():
    return {"tight": TIGHT_TOLERANCE, "standard": STANDARD_TOLERANCE, "sampling": SAMPLING_TOLERANCE}


@pytest.fixture
def numerical_settings():
    return {
        "integration_steps": NUM_INTEGRATION_STEPS,
        "sampling_iterations": NUM_SAMPLING_ITERATIONS,
        "ci_points": NUMERICAL_CI_POINTS,
        "ci_method": NUMERICAL_CI_METHOD,
        "epsilon": EPSILON,
    }


# =============================================================================
# Per-model fan (tests/models/) — full matrix; function-scoped so gradient
# tests get a fresh model each time.
# =============================================================================


@pytest.fixture(params=MODEL_TYPES)
def model_type(request):
    """Every model type in the registry."""
    return request.param


@pytest.fixture
def model(model_type):
    return create_model(model_type, d=D, seed=SEED, terminating=False)


@pytest.fixture
def terminating_model(model_type):
    return create_model(model_type, d=D, seed=SEED, terminating=True)


@pytest.fixture
def model_pair(model_type):
    return create_model_pair(model_type, d=D, seed=SEED)


# =============================================================================
# Archetype fan (tests/contract/) — representative set; module-scoped, built
# once per archetype per module (contract tests are read-only forward passes).
# =============================================================================


@pytest.fixture(params=ARCHETYPES, scope="module")
def archetype(request):
    """A representative model type per distinct base-class branch."""
    return request.param


@pytest.fixture(scope="module")
def representative_model(archetype):
    return create_model(archetype, d=D, seed=SEED, terminating=False)


@pytest.fixture(scope="module")
def representative_terminating_model(archetype):
    return create_model(archetype, d=D, seed=SEED, terminating=True)


@pytest.fixture(scope="module")
def representative_model_pair(archetype):
    return create_model_pair(archetype, d=D, seed=SEED)


# =============================================================================
# Event data — function-scoped (cheap to build; fresh tensors avoid any
# cross-test in-place mutation hazard).
# =============================================================================


@pytest.fixture
def test_data():
    """Standard non-terminating sequences (single, batched, padded, simultaneous)."""
    return build_test_data(DEVICE)


@pytest.fixture
def terminating_test_data():
    """Sequences for terminating tests (single, partial/full termination, repeated, padded)."""
    return build_terminating_test_data(DEVICE, d=D)
