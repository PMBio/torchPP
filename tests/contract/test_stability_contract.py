"""Contract: parameter-transformation, numerical-stability and empty-batch API.

These are relation/shape/API contracts that take the same code path for every
model, so they fan over the ARCHETYPE set.

Note: ``evaluate_at_time`` is no longer a model method — it lives in
:mod:`torchpp.utils.evaluation` as a free function, so it is exercised through
the evaluation/pipeline tests rather than here.
"""

import pytest
import torch
from helpers import DEVICE, DIVERGENT_INTENSITY_MODEL_TYPES, create_model, skip_if_density
from torchpp.utils.event_data import BatchedMVEventData

# transform_params is ill-posed for neural models (history-dependent params);
# among the archetypes these are the neural backends.
NEURAL_ARCHETYPES = {"nhp", "intensity_free", "neural_log_linear"}


class TestParameterTransformation:
    """Tests for transform_params() method."""

    def test_transform_params_idempotent(self, representative_model, archetype):
        """Test transform_params is idempotent."""
        model = representative_model
        if archetype in NEURAL_ARCHETYPES:
            pytest.skip("Neural TPP parameters are history-dependent")
        params1 = model.transform_params()
        params2 = model.transform_params()

        assert len(params1) == len(params2), "transform_params should return same number of parameters"
        for p1, p2 in zip(params1, params2):
            assert torch.equal(p1, p2), "transform_params should be idempotent (same result each call)"

    def test_transform_params_gradient_flow(self, representative_model, test_data):
        """Test gradients flow through parameter transformations."""
        model = representative_model
        batch_single = test_data["batch_single"]

        # Zero out gradients
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        # Compute likelihood (requires transformed params)
        log_lik = model.likelihood(batch_single, torch.tensor([4.0]).to(DEVICE), log=True)
        log_lik.sum().backward()

        # Check that at least one parameter has non-zero gradient
        has_gradient = any(param.grad is not None and torch.any(param.grad != 0) for param in model.parameters())
        assert has_gradient, "At least one parameter should have non-zero gradient after backward pass"

    def test_transform_params_type_consistency(self, representative_model, archetype):
        """Test all transformed parameters have consistent types."""
        model = representative_model
        if archetype in NEURAL_ARCHETYPES:
            pytest.skip("Neural TPP parameters are history-dependent")
        params = model.transform_params()

        if len(params) > 0:
            first_dtype = params[0].dtype
            first_device = params[0].device

            for i, param in enumerate(params):
                assert param.dtype == first_dtype, (
                    f"Parameter {i} has inconsistent dtype: {param.dtype} vs {first_dtype}"
                )
                assert param.device == first_device, (
                    f"Parameter {i} has inconsistent device: {param.device} vs {first_device}"
                )


class TestNumericalStability:
    """Tests for numerical edge cases."""

    def test_likelihood_very_small_intensities(self, representative_model, test_data):
        """Test likelihood handles near-zero intensities gracefully."""
        model = representative_model
        batch_single = test_data["batch_single"]

        # Force very late time where intensities may decay to near-zero
        T_large = batch_single.last_event_time + 1000.0

        log_lik = model.likelihood(batch_single, T_large, log=True)

        # Should be finite (not -inf from log(0))
        assert torch.isfinite(log_lik).all(), (
            f"Likelihood should remain finite even with small intensities, got {log_lik}"
        )

    def test_cdf_near_one(self, representative_model, test_data, tolerances):
        """Test CDF computation near 1.0 (cancellation risk)."""
        model = representative_model
        batch_single = test_data["batch_single"]

        # Find time where CDF ≈ 0.99999
        T_high = model.time_quantile(torch.tensor([0.99999]).to(DEVICE), batch_single)

        cdf = model.cdf_time(T_high, batch_single, log=False)

        # Should be in [0, 1] despite numerical issues
        assert torch.all(cdf >= 0) and torch.all(cdf <= 1), f"CDF should remain in [0,1], got {cdf}"
        assert torch.all(cdf > 0.999), f"CDF should be near 1.0, got {cdf}"

    def test_ci_very_large_T(self, representative_model, archetype, test_data):
        """Test CI with very large T values."""
        model = representative_model
        skip_if_density(model)  # cumulative_intensity is intensity-family-specific
        if archetype in DIVERGENT_INTENSITY_MODEL_TYPES:
            pytest.skip("Log-linear intensity extrapolation diverges; CI at extreme T overflows by design")
        batch_single = test_data["batch_single"]
        T_large = torch.tensor([1e6]).to(DEVICE)

        ci = model.cumulative_intensity(T_large, batch_single)

        # Should be finite (no overflow)
        assert torch.isfinite(ci).all(), f"CI should remain finite even for large T, got {ci}"

    def test_intensity_very_early_time(self, representative_model, test_data, d):
        """Test intensity at very early time (near t=0)."""
        model = representative_model
        batch_single = test_data["batch_single"]
        t_early = torch.tensor([1e-10]).to(DEVICE)

        intensity = model.intensity(t_early, batch_single)

        # Should be finite and positive
        assert torch.isfinite(intensity).all(), "Intensity should be finite at early times"
        assert torch.all(intensity > 0), "Intensity should be positive at early times"
        assert intensity.shape == (1, d), f"Intensity shape should be (1, {d})"

    def test_pdf_very_close_to_last_event(self, representative_model, test_data, tolerances):
        """Test PDF just after last event.

        Uses an offset of 1e-6 to stay above the PDF method's internal
        snapping threshold (|T - last_event_time| < 1e-8 → T is snapped to last_event_time).
        With 1e-6, the CI over the tiny interval is negligible so survival ≈ 1
        and PDF ≈ intensity.  For Hawkes-type models the kernel uses a strict
        past-event mask (t_event < t), so evaluating at exactly last_event_time excludes
        the last event from the kernel sum while last_event_time + ε includes it; a
        sufficiently small but un-snapped ε avoids that discrepancy.
        """
        model = representative_model
        batch_single = test_data["batch_single"]
        t_just_after = batch_single.last_event_time + 1e-6

        pdf = model.density_joint(t_just_after, batch_single, log=False)

        # Should be finite
        assert torch.isfinite(pdf).all(), "PDF should be finite just after last event"
        # Should be close to intensity at that time
        intensity = model.intensity(t_just_after, batch_single)
        # survival ≈ 1 for very small Δt, so PDF ≈ intensity
        assert torch.allclose(pdf.float(), intensity.float(), atol=tolerances["standard"]), (
            "PDF should be close to intensity just after last event"
        )

    def test_log_likelihood_numerical_stability(self, representative_model, test_data):
        """Test log-likelihood vs likelihood numerical consistency."""
        model = representative_model
        batch_single = test_data["batch_single"]
        T = torch.tensor([4.0]).to(DEVICE)

        log_lik = model.likelihood(batch_single, T, log=True)
        lik = model.likelihood(batch_single, T, log=False)

        # Check that log(lik) ≈ log_lik
        log_lik_from_lik = torch.log(lik)

        # Should be close (within numerical precision)
        # Use larger tolerance since we're comparing log of potentially small numbers
        assert torch.allclose(log_lik.float(), log_lik_from_lik.float(), atol=1e-2), (
            f"log-likelihood {log_lik} should match log(likelihood) {log_lik_from_lik}"
        )


class TestMixedEmptySequence:
    """An all-padding (empty) sequence inside a non-empty batch (no-origin regime).

    Empty sequences are reachable after dataset filtering (e.g. length or event-type filters),
    so a mixed batch containing one must not crash the no-origin likelihood.  The
    regression: ``_likelihood_scaled`` set ``T_low = time_points[:, 0]`` batch-wide, which
    for an empty row is the padding sentinel (``> T_obs=0``) — tripping the CI's
    ``T >= T_low`` assertion.  ``T_low`` is now per-sequence (0 where the row has no first
    real event), so the empty row gets the degenerate window ``[0, 0]`` and contributes 0.
    """

    @pytest.mark.parametrize("model_type", ["poisson", "neural_log_linear", "nhp"])
    def test_empty_row_in_mixed_batch(self, model_type):
        model = create_model(model_type).to(DEVICE)
        model.eval()
        assert not model.has_origin  # the crashing branch is the no-origin one

        mixed = BatchedMVEventData(
            [torch.tensor([0.7, 1.3, 2.2], dtype=torch.float64), torch.tensor([], dtype=torch.float64)],
            [torch.tensor([0, 1, 2]), torch.tensor([], dtype=torch.long)],
        ).to(DEVICE)
        solo = BatchedMVEventData(
            [torch.tensor([0.7, 1.3, 2.2], dtype=torch.float64)], [torch.tensor([0, 1, 2])]
        ).to(DEVICE)

        with torch.no_grad():
            ll = model.likelihood(mixed, log=True)  # must not raise
            ll_solo = model.likelihood(solo, log=True)

        assert torch.isfinite(ll).all(), f"non-finite likelihood: {ll}"
        # Empty row: no positive term and a degenerate [0, 0] survival window → exactly 0.
        assert ll[1].abs().item() == 0.0, f"empty sequence should contribute 0, got {ll[1].item()}"
        # And it must not perturb the real row's likelihood.
        assert torch.allclose(ll[0], ll_solo[0], atol=1e-5), "empty row perturbed the real row"


class TestEmptyBatch:
    """Tests for B=0 batch handling."""

    def test_empty_batch_intensity(self, representative_model, d):
        """Test intensity with empty batch (B=0)."""
        model = representative_model
        from torchpp.utils.event_data import BatchedMVEventData

        empty_batch = BatchedMVEventData([], []).to(DEVICE)
        t = torch.tensor([], dtype=torch.float64).to(DEVICE)

        intensity = model.intensity(t, empty_batch)
        assert intensity.shape == (0, d), f"Empty batch should give shape (0, {d}), got {intensity.shape}"

    def test_empty_batch_cumulative_intensity(self, representative_model, d):
        """Test CI with empty batch (B=0)."""
        model = representative_model
        skip_if_density(model)  # cumulative_intensity is intensity-family-specific
        from torchpp.utils.event_data import BatchedMVEventData

        empty_batch = BatchedMVEventData([], []).to(DEVICE)
        T = torch.tensor([], dtype=torch.float64).to(DEVICE)

        ci = model.cumulative_intensity(T, empty_batch)
        assert ci.shape == (0, d), f"Empty batch should give shape (0, {d}), got {ci.shape}"

    def test_empty_batch_likelihood(self, representative_model):
        """Test likelihood with empty batch (B=0)."""
        model = representative_model
        from torchpp.utils.event_data import BatchedMVEventData

        empty_batch = BatchedMVEventData([], []).to(DEVICE)
        T = torch.tensor([], dtype=torch.float64).to(DEVICE)

        log_lik = model.likelihood(empty_batch, T, log=True)
        assert log_lik.shape == (0,), f"Empty batch should give shape (0,), got {log_lik.shape}"

    def test_empty_batch_pdf(self, representative_model, d):
        """Test PDF with empty batch (B=0)."""
        model = representative_model
        from torchpp.utils.event_data import BatchedMVEventData

        empty_batch = BatchedMVEventData([], []).to(DEVICE)
        T = torch.tensor([], dtype=torch.float64).to(DEVICE)

        pdf = model.density_joint(T, empty_batch, log=False)
        assert pdf.shape == (0, d), f"Empty batch should give shape (0, {d}), got {pdf.shape}"

    def test_empty_batch_cdf(self, representative_model):
        """Test CDF with empty batch (B=0)."""
        model = representative_model
        from torchpp.utils.event_data import BatchedMVEventData

        empty_batch = BatchedMVEventData([], []).to(DEVICE)
        T = torch.tensor([], dtype=torch.float64).to(DEVICE)

        cdf = model.cdf_time(T, empty_batch, log=False)
        assert cdf.shape == (0,), f"Empty batch should give shape (0,), got {cdf.shape}"
