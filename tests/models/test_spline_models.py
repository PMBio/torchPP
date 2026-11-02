"""
Full-model tests for the spline-kernel Hawkes processes.

Covers ConstBaselineSplineKernelHawkesProcess and
SplineBaselineSplineKernelHawkesProcess end-to-end (intensity, cumulative
intensity, likelihood, gradients, and terminating mode).
"""

import pytest
import torch
from helpers import DEVICE

from torchpp.tpp.hawkes import (
    ConstBaselineSplineKernelHawkesProcess,
    SplineBaselineSplineKernelHawkesProcess,
)
from torchpp.utils.event_data import MVEventData, BatchedMVEventData


# =============================================================================
# Constants & Fixtures
# =============================================================================

D = 3
SEED = 42
NUM_KNOTS = 15
DELTA_T = 0.3  # knots at 0.0, 0.3, 0.6, ..., 4.2


@pytest.fixture
def test_data():
    """Create test event data."""
    seq1 = MVEventData(
        torch.tensor([0.2, 0.8, 2.0, 2.3], dtype=torch.float64).to(DEVICE),
        torch.tensor([0, 1, 2, 0], dtype=torch.long).to(DEVICE),
    )
    seq2 = MVEventData(
        torch.tensor([0.1, 0.5, 1.2, 1.8, 2.5], dtype=torch.float64).to(DEVICE),
        torch.tensor([1, 0, 2, 0, 1], dtype=torch.long).to(DEVICE),
    )
    return {
        "seq1": seq1,
        "seq2": seq2,
        "batch_single": BatchedMVEventData([seq1.time_points], [seq1.event_types]).to(DEVICE),
        "batch_multi": BatchedMVEventData(
            [seq1.time_points, seq2.time_points],
            [seq1.event_types, seq2.event_types],
        ).to(DEVICE),
    }


@pytest.fixture
def model():
    """Create a ConstBaselineSplineKernelHawkesProcess."""
    torch.manual_seed(SEED)
    return ConstBaselineSplineKernelHawkesProcess(D=D, num_knots=NUM_KNOTS, delta_t=DELTA_T, seed=SEED).to(DEVICE)


@pytest.fixture
def spline_spline_model():
    """Create a SplineBaselineSplineKernelHawkesProcess."""
    torch.manual_seed(SEED)
    return SplineBaselineSplineKernelHawkesProcess(
        D=D,
        num_knots_baseline=NUM_KNOTS,
        num_knots_kernel=NUM_KNOTS,
        delta_t_baseline=DELTA_T,
        delta_t_kernel=DELTA_T,
        seed=SEED,
    ).to(DEVICE)


# =============================================================================
# Full Model Tests
# =============================================================================


class TestSplineKernelModel:
    """Test the full ConstBaselineSplineKernelHawkesProcess."""

    def test_model_intensity_shape(self, model, test_data):
        """Full model intensity should return (B, D)."""
        ts = test_data["batch_multi"]
        t = torch.tensor([3.0, 3.5], dtype=torch.float64).to(DEVICE)
        intensity = model.intensity(t, ts)
        assert intensity.shape == (2, D)

    def test_model_intensity_positive(self, model, test_data):
        """Full model intensity should be positive."""
        ts = test_data["batch_multi"]
        t = torch.tensor([3.0, 3.5], dtype=torch.float64).to(DEVICE)
        intensity = model.intensity(t, ts)
        assert (intensity > 0).all()

    def test_model_ci_shape(self, model, test_data):
        """Full model CI should return (B, D)."""
        ts = test_data["batch_multi"]
        T = torch.tensor([4.0, 3.5], dtype=torch.float64).to(DEVICE)
        ci = model.cumulative_intensity(T, ts)
        assert ci.shape == (2, D)

    def test_model_likelihood_shape(self, model, test_data):
        """Full model log-likelihood should return (B,)."""
        ts = test_data["batch_multi"]
        T = torch.tensor([4.0, 3.5], dtype=torch.float64).to(DEVICE)
        ll = model.likelihood(ts, T, log=True)
        assert ll.shape == (2,)

    def test_model_likelihood_finite(self, model, test_data):
        """Log-likelihood should be finite."""
        ts = test_data["batch_multi"]
        T = torch.tensor([4.0, 3.5], dtype=torch.float64).to(DEVICE)
        ll = model.likelihood(ts, T, log=True)
        assert torch.isfinite(ll).all()

    def test_model_likelihood_gradient(self, model, test_data):
        """Log-likelihood should have gradients."""
        ts = test_data["batch_multi"]
        T = torch.tensor([4.0, 3.5], dtype=torch.float64).to(DEVICE)
        ll = model.likelihood(ts, T, log=True)
        loss = -ll.sum()
        loss.backward()

        # Check kernel params have gradients
        assert model.kernel.log_h_knots.grad is not None
        assert model.kernel.log_h_knots.grad.abs().sum() > 0

    def test_spline_spline_model(self, spline_spline_model, test_data):
        """SplineBaselineSplineKernelHawkesProcess should work end-to-end."""
        ts = test_data["batch_multi"]
        T = torch.tensor([4.0, 3.5], dtype=torch.float64).to(DEVICE)
        ll = spline_spline_model.likelihood(ts, T, log=True)
        assert ll.shape == (2,)
        assert torch.isfinite(ll).all()

    def test_terminating_model(self, test_data):
        """Terminating model should work correctly."""
        model = ConstBaselineSplineKernelHawkesProcess(
            D=D, num_knots=NUM_KNOTS, delta_t=DELTA_T, seed=SEED, terminating=True
        ).to(DEVICE)
        # Use data where each type occurs at most once (valid for terminating mode)
        seq1 = MVEventData(
            torch.tensor([0.2, 0.8, 2.0], dtype=torch.float64).to(DEVICE),
            torch.tensor([0, 1, 2], dtype=torch.long).to(DEVICE),
        )
        seq2 = MVEventData(
            torch.tensor([0.5, 1.5], dtype=torch.float64).to(DEVICE),
            torch.tensor([1, 0], dtype=torch.long).to(DEVICE),
        )
        ts = BatchedMVEventData(
            [seq1.time_points, seq2.time_points],
            [seq1.event_types, seq2.event_types],
        )
        T = torch.tensor([4.0, 3.5], dtype=torch.float64).to(DEVICE)
        ll = model.likelihood(ts, T, log=True)
        assert ll.shape == (2,)
        assert torch.isfinite(ll).all()
