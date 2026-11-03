"""Contract: universal model API/relation/identity checks.

Every property here holds for every model and takes an identical code path:
shapes, positivity, finiteness, bounds, batch==loop consistency, padding
invariance, empty-sequence handling, illegal-input raises, and the algebraic
identities tying intensity / cumulative intensity / PDF / CDF / likelihood /
sampling / inverse-CDF together. They therefore fan over the ARCHETYPE set
(``representative_model``) and carry NO skip-sets — they are universal.

Genuine per-family NUMERICAL-VALUE checks (finite-difference d/dt CI ≈ λ,
marginal quadrature, analytical-vs-empirical sampling, inverse-CDF boundary
magnitudes) live in ``tests/models/test_ci_value.py`` over the full matrix.
"""

import pytest
import torch
from helpers import DEVICE, SEED, skip_if_density, create_model
from torchpp.utils.event_data import MVEventData, BatchedMVEventData


def _drop_last_event(batch):
    """Return ``batch`` with each sequence's last real event removed.

    Replaces the old ``batch[:, :-1]`` 2-D slice, which no longer applies now that a
    ``BatchedMVEventData`` has ``batch_size=(B,)`` (no second batch dimension to slice);
    decollates to unpadded sequences, drops the last real event, and recollates.
    """
    return BatchedMVEventData.from_sequences(
        [MVEventData(s.time_points[:-1], s.event_types[:-1]) for s in batch.to_sequences()]
    )


class TestIntensity:
    """Tests for intensity computation."""

    def test_intensity_shape(self, representative_model, test_data, d):
        """Test that intensity has correct output shape."""
        model = representative_model
        batch_single = test_data["batch_single"]

        intensity = model.intensity(torch.tensor([4.0]).to(DEVICE), batch_single)
        assert intensity.shape == (1, d), f"Intensity shape mismatch: {intensity.shape} vs (1, {d})"

    def test_intensity_positive(self, representative_model, test_data):
        """Test that intensity is always positive."""
        model = representative_model
        batch_single = test_data["batch_single"]

        intensity = model.intensity(torch.tensor([4.0]).to(DEVICE), batch_single)
        assert torch.all(intensity > 0), "Intensity should be positive"

    def test_intensity_batch_consistency(self, representative_model, test_data, d, tolerances):
        """Test that batched intensity equals individual intensities combined."""
        model = representative_model
        batch_multi = test_data["batch_multi"]
        seq1 = test_data["seq1"]
        seq2 = test_data["seq2"]

        # Batched call
        intensity_batched = model.intensity(torch.tensor([4.0, 3.0]).to(DEVICE), batch_multi)

        # Individual calls
        from torchpp.utils.event_data import BatchedMVEventData

        batch1 = BatchedMVEventData([seq1.time_points], [seq1.event_types]).to(DEVICE)
        batch2 = BatchedMVEventData([seq2.time_points], [seq2.event_types]).to(DEVICE)
        intensity1 = model.intensity(torch.tensor([4.0]).to(DEVICE), batch1)
        intensity2 = model.intensity(torch.tensor([3.0]).to(DEVICE), batch2)
        intensity_combined = torch.cat([intensity1, intensity2], dim=0)

        assert torch.allclose(intensity_batched.float(), intensity_combined.float(), atol=tolerances["tight"]), (
            "Batched intensity should equal individual intensities combined"
        )

    def test_intensity_illegal_t(self, representative_model, test_data):
        """Test that intensity is always positive."""
        model = representative_model
        batch_single = test_data["batch_single"]

        with pytest.raises(ValueError):
            model.intensity(torch.tensor([-1.0]).to(DEVICE), batch_single)

    def test_intensity_padding_invariance(self, representative_model, test_data, tolerances):
        """Test that padding with -1 events does not affect intensity."""
        model = representative_model
        batch_single = test_data["batch_single"]
        padded_batch = test_data["padded_batch"]

        intensity_normal = model.intensity(torch.tensor([4.0]).to(DEVICE), batch_single)
        intensity_padded = model.intensity(torch.tensor([4.0]).to(DEVICE), padded_batch)

        assert torch.allclose(intensity_normal.float(), intensity_padded.float(), atol=tolerances["tight"]), (
            "Intensity should be same with padded events"
        )

    def test_intensity_empty_sequence(self, representative_model, test_data, d):
        """Test intensity on empty sequence equals baseline."""
        model = representative_model
        empty_batch = test_data["empty_batch"]

        intensity = model.intensity(torch.tensor([1.0]).to(DEVICE), empty_batch)
        assert intensity.shape == (1, d), "Intensity shape mismatch for empty sequence"
        assert torch.all(intensity > 0), "Intensity should be positive for empty sequence"

    def test_intensity_at_event_time(self, representative_model, test_data, tolerances):
        """Test intensity exactly at event time matches intensity just before."""
        model = representative_model
        batch_single = test_data["batch_single"]

        # Intensity exactly at last event
        intensity_at_event = model.intensity(torch.tensor([2.3]).to(DEVICE), batch_single)

        # Intensity computed without last event
        intensity_exactly_at_event_last_cut = model.intensity(torch.tensor([2.3]).to(DEVICE), _drop_last_event(batch_single))

        assert torch.allclose(
            intensity_at_event.float(), intensity_exactly_at_event_last_cut.float(), atol=tolerances["tight"]
        ), "Intensity at exactly event time should match intensity with last event cut"

    def test_intensity_inbetween_event_times(self, representative_model, test_data, tolerances):
        """Test intensity exactly at event time matches intensity just before."""
        model = representative_model
        batch_single = test_data["batch_single"]

        # Intensity just before last event
        intensity_before_event = model.intensity(torch.tensor([2.2]).to(DEVICE), batch_single)

        # Intensity computed without last event
        intensity_before_event_last_cut = model.intensity(torch.tensor([2.2]).to(DEVICE), _drop_last_event(batch_single))

        assert torch.allclose(
            intensity_before_event.float(), intensity_before_event_last_cut.float(), atol=tolerances["tight"]
        ), "Intensity just before event time should match intensity with last event cut"

    def test_intensity_baseline_at_t0(self, representative_model, test_data, tolerances):
        """Test intensity at t=0 matches baseline intensity."""
        model = representative_model
        batch_single = test_data["batch_single"]
        empty_batch = test_data["empty_batch"]

        intensity_at_0 = model.intensity(torch.tensor([0.0]).to(DEVICE), batch_single)
        intensity_baseline = model.intensity(torch.tensor([0.0]).to(DEVICE), empty_batch)

        assert torch.allclose(intensity_at_0.float(), intensity_baseline.float(), atol=tolerances["tight"]), (
            "Intensity at t=0 should match baseline"
        )


class TestCumulativeIntensity:
    """Tests for cumulative intensity computation (intensity family only)."""

    @pytest.fixture(autouse=True)
    def _intensity_only(self, representative_model):
        # cumulative_intensity is an intensity-family realization; density models have none.
        skip_if_density(representative_model)

    def test_cumulative_intensity_shape(self, representative_model, test_data, d):
        """Test cumulative intensity has correct output shape."""
        model = representative_model
        batch_single = test_data["batch_single"]

        ci = model.cumulative_intensity(torch.tensor([4.0]).to(DEVICE), batch_single)
        assert ci.shape == (1, d), f"CI shape mismatch: {ci.shape} vs (1, {d})"

    def test_cumulative_intensity_positive(self, representative_model, test_data):
        """Test cumulative intensity is positive."""
        model = representative_model
        batch_single = test_data["batch_single"]

        ci = model.cumulative_intensity(torch.tensor([4.0]).to(DEVICE), batch_single)
        assert torch.all(ci > 0), "Cumulative intensity should be positive"

    def test_cumulative_intensity_batch_consistency(self, representative_model, test_data, d, tolerances):
        """Test batched CI equals individual CIs combined."""
        model = representative_model
        batch_multi = test_data["batch_multi"]
        seq1 = test_data["seq1"]
        seq2 = test_data["seq2"]

        # Batched call
        ci_batched = model.cumulative_intensity(torch.tensor([4.0, 3.0]).to(DEVICE), batch_multi)
        assert ci_batched.shape == (2, d), "Cumulative Intensity shape mismatch for multi-batch."
        assert torch.all(ci_batched > 0), "Cumulative Intensity should be positive for multi-batch."

        # Individual calls
        from torchpp.utils.event_data import BatchedMVEventData

        batch1 = BatchedMVEventData([seq1.time_points], [seq1.event_types]).to(DEVICE)
        batch2 = BatchedMVEventData([seq2.time_points], [seq2.event_types]).to(DEVICE)
        ci1 = model.cumulative_intensity(torch.tensor([4.0]).to(DEVICE), batch1)
        ci2 = model.cumulative_intensity(torch.tensor([3.0]).to(DEVICE), batch2)
        ci_combined = torch.cat([ci1, ci2], dim=0)

        assert torch.allclose(ci_batched.float(), ci_combined.float(), atol=tolerances["tight"]), (
            "Batched CI should equal individual CIs combined"
        )

    def test_cumulative_intensity_padding_invariance(self, representative_model, test_data, tolerances):
        """Test padding does not affect cumulative intensity."""
        model = representative_model
        batch_single = test_data["batch_single"]
        padded_batch = test_data["padded_batch"]

        ci_normal = model.cumulative_intensity(torch.tensor([4.0]).to(DEVICE), batch_single)
        ci_padded = model.cumulative_intensity(torch.tensor([4.0]).to(DEVICE), padded_batch)

        assert torch.allclose(ci_normal.float(), ci_padded.float(), atol=tolerances["tight"]), (
            "CI should be same with padded events"
        )

    def test_cumulative_intensity_empty_sequence(self, representative_model, test_data, d):
        """Test padding does not affect cumulative intensity."""
        model = representative_model
        batch_empty = test_data["empty_batch"]

        ci_empty = model.cumulative_intensity(torch.tensor([4.0]).to(DEVICE), batch_empty)

        assert ci_empty.shape == (1, d), "Cumulative Intensity shape mismatch for empty sequence."
        assert torch.all(ci_empty > 0), "Cumulative Intensity should be positive for empty sequence."

    def test_cumulative_intensity_at_event_time(self, representative_model, test_data, tolerances):
        """Test cumulative intensity exactly at event time matches cumulative intensity just before."""
        model = representative_model
        batch_single = test_data["batch_single"]

        # Cumulative intensity exactly at last event
        ci_at_event = model.cumulative_intensity(torch.tensor([2.3]).to(DEVICE), batch_single)

        # Cumulative intensity computed without last event
        ci_exactly_at_event_last_cut = model.cumulative_intensity(torch.tensor([2.3]).to(DEVICE), _drop_last_event(batch_single))

        assert torch.allclose(ci_at_event.float(), ci_exactly_at_event_last_cut.float(), atol=tolerances["tight"]), (
            "Cumulative intensity at exactly event time should match cumulative intensity with last event cut"
        )

    def test_cumulative_intensity_inbetween_event_times(self, representative_model, test_data, tolerances):
        """Test cumulative intensity exactly at event time matches cumulative intensity just before."""
        model = representative_model
        batch_single = test_data["batch_single"]

        # Cumulative intensity just before last event
        ci_before_event = model.cumulative_intensity(torch.tensor([2.2]).to(DEVICE), batch_single)

        # Cumulative intensity computed without last event
        ci_before_event_last_cut = model.cumulative_intensity(torch.tensor([2.2]).to(DEVICE), _drop_last_event(batch_single))

        assert torch.allclose(ci_before_event.float(), ci_before_event_last_cut.float(), atol=tolerances["tight"]), (
            "Cumulative intensity just before event time should match cumulative intensity with last event cut"
        )

    def test_cumulative_intensity_baseline_at_t0(self, representative_model, test_data, tolerances):
        """Test cumulative intensity at t=0 matches baseline cumulative intensity."""
        model = representative_model
        batch_single = test_data["batch_single"]
        empty_batch = test_data["empty_batch"]

        ci_at_0 = model.cumulative_intensity(torch.tensor([0.0]).to(DEVICE), batch_single)
        ci_baseline = model.cumulative_intensity(torch.tensor([0.0]).to(DEVICE), empty_batch)
        assert torch.allclose(ci_at_0.float(), ci_baseline.float(), atol=tolerances["tight"]), (
            "Cumulative intensity at t=0 should match baseline"
        )

    def test_cumulative_illegal_t(self, representative_model, test_data, tolerances):
        """Test if illegal T are caught."""
        model = representative_model
        batch_single = test_data["batch_single"]

        # Except an exception raised for negative T
        with pytest.raises(ValueError):
            model.cumulative_intensity(torch.tensor([-1.0]).to(DEVICE), batch_single)

    def test_cumulative_intensity_t_low(self, representative_model, test_data, tolerances):
        """Test if the T_low parameter is handled correctly."""
        model = representative_model
        batch_single = test_data["batch_single"]

        ci_high = model.cumulative_intensity(torch.tensor([4.0]).to(DEVICE), batch_single)
        ci_low = model.cumulative_intensity(torch.tensor([2.0]).to(DEVICE), batch_single)
        ci_diff = model.cumulative_intensity(
            torch.tensor([4.0]).to(DEVICE), batch_single, T_low=torch.tensor([2.0]).to(DEVICE)
        )

        assert torch.allclose((ci_high - ci_low).float(), ci_diff.float(), atol=tolerances["standard"])

    def test_cumulative_intensity_monotonic(self, representative_model, test_data):
        """Test cumulative intensity is monotonically increasing."""
        model = representative_model
        batch_single = test_data["batch_single"]

        times = torch.linspace(0.0, 4.0, 20).to(DEVICE)
        cis = torch.stack([model.cumulative_intensity(t.unsqueeze(0), batch_single) for t in times])
        cis = cis.squeeze()

        # Check monotonicity for each dimension
        for dim in range(cis.shape[1]):
            diffs = cis[1:, dim] - cis[:-1, dim]
            assert torch.all(diffs >= 0.0), f"CI should be monotonic for dimension {dim}"


class TestPDFCDF:
    """Tests for PDF and CDF computation."""

    def test_pdf_shape(self, representative_model, test_data, d):
        """Test PDF has correct output shape."""
        model = representative_model
        batch_single = test_data["batch_single"]

        pdf = model.density_joint(torch.tensor([2.5]).to(DEVICE), batch_single, log=False)
        assert pdf.shape == (1, d), f"PDF shape mismatch: {pdf.shape}"

    def test_pdf_non_negative(self, representative_model, test_data):
        """Test PDF is non-negative."""
        model = representative_model
        batch_single = test_data["batch_single"]

        pdf = model.density_joint(torch.tensor([2.5]).to(DEVICE), batch_single, log=False)
        assert torch.all(pdf >= 0), "PDF should be non-negative"

    def test_pdf_illegal_before_last_event(self, representative_model, test_data):
        """Test PDF raises error if evaluated before last event."""
        model = representative_model
        batch_single = test_data["batch_single"]

        with pytest.raises(ValueError):
            model.density_joint(batch_single.last_event_time - 0.1, batch_single, log=False)

    def test_pdf_batch_consistency(self, representative_model, test_data, tolerances):
        """Test batched PDF equals individual PDFs combined."""
        model = representative_model
        batch_multi = test_data["batch_multi"]
        seq1 = test_data["seq1"]
        seq2 = test_data["seq2"]

        # Batched call
        pdf_batched = model.density_joint(torch.tensor([3.0, 3.0]).to(DEVICE), batch_multi, log=False)

        # Individual calls
        from torchpp.utils.event_data import BatchedMVEventData

        batch1 = BatchedMVEventData([seq1.time_points], [seq1.event_types]).to(DEVICE)
        batch2 = BatchedMVEventData([seq2.time_points], [seq2.event_types]).to(DEVICE)
        pdf1 = model.density_joint(torch.tensor([3.0]).to(DEVICE), batch1, log=False)
        pdf2 = model.density_joint(torch.tensor([3.0]).to(DEVICE), batch2, log=False)
        pdf_combined = torch.cat([pdf1, pdf2], dim=0)

        assert torch.allclose(pdf_batched.float(), pdf_combined.float(), atol=tolerances["tight"]), (
            "Batched PDF should equal individual PDFs combined"
        )

    def test_pdf_relation_to_intensity_and_ci(self, representative_model, test_data, tolerances):
        """Test PDF equals intensity times exp(-cumulative intensity)."""
        model = representative_model
        skip_if_density(model)  # relation is via intensity/CI; density realizes PDF natively
        batch_single = test_data["batch_single"]

        t_test = torch.tensor([2.5]).to(DEVICE)
        t_last = torch.tensor([2.3]).to(DEVICE)

        pdf = model.density_joint(t_test, batch_single, log=False)

        # Compute expected PDF from intensity and cumulative intensity
        intensity = model.intensity(t_test, batch_single)
        ci_diff = model.cumulative_intensity(t_test, batch_single) - model.cumulative_intensity(t_last, batch_single)
        pdf_expected = intensity * torch.exp(-ci_diff.sum(dim=1, keepdim=True))

        tolerance = tolerances["tight"] if (model.ci_config.mode == "closed_form") else tolerances["standard"]
        assert torch.allclose(pdf.float(), pdf_expected.float(), atol=tolerance), (
            "PDF should equal intensity * exp(-cumulative intensity)"
        )

    def test_cdf_shape(self, representative_model, test_data, d):
        """Test CDF has correct output shape."""
        model = representative_model
        batch_single = test_data["batch_single"]

        cdf = model.cdf_time(torch.tensor([2.5]).to(DEVICE), batch_single, log=False)
        assert cdf.shape == (1,), f"CDF shape mismatch: {cdf.shape}"

    def test_cdf_bounded(self, representative_model, test_data):
        """Test CDF is bounded between 0 and 1."""
        model = representative_model
        batch_single = test_data["batch_single"]

        cdf = model.cdf_time(torch.tensor([2.5]).to(DEVICE), batch_single, log=False)
        assert torch.all(cdf >= 0), "CDF should be >= 0"
        assert torch.all(cdf <= 1), "CDF should be <= 1"

    def test_cdf_batch_consistency(self, representative_model, test_data, tolerances):
        """Test batched CDF equals individual CDFs combined."""
        model = representative_model
        batch_multi = test_data["batch_multi"]
        seq1 = test_data["seq1"]
        seq2 = test_data["seq2"]

        # Batched call (CDF returns shape (batch_size,) not (batch_size, d))
        cdf_batched = model.cdf_time(torch.tensor([3.0, 3.0]).to(DEVICE), batch_multi, log=False)

        # Individual calls
        from torchpp.utils.event_data import BatchedMVEventData

        batch1 = BatchedMVEventData([seq1.time_points], [seq1.event_types]).to(DEVICE)
        batch2 = BatchedMVEventData([seq2.time_points], [seq2.event_types]).to(DEVICE)
        cdf1 = model.cdf_time(torch.tensor([3.0]).to(DEVICE), batch1, log=False)
        cdf2 = model.cdf_time(torch.tensor([3.0]).to(DEVICE), batch2, log=False)
        cdf_combined = torch.cat([cdf1, cdf2], dim=0)

        assert torch.allclose(cdf_batched.float(), cdf_combined.float(), atol=tolerances["tight"]), (
            "Batched CDF should equal individual CDFs combined"
        )

    def test_cdf_illegal_before_last_event(self, representative_model, test_data):
        """Test CDF raises error if evaluated before last event."""
        model = representative_model
        batch_single = test_data["batch_single"]

        with pytest.raises(ValueError):
            model.cdf_time(batch_single.last_event_time - 0.1, batch_single, log=False)

    def test_cdf_relation_to_ci(self, representative_model, test_data, tolerances):
        """Test CDF equals 1 - exp(-cumulative intensity)."""
        model = representative_model
        skip_if_density(model)  # relation is via CI; density realizes cdf_time natively
        batch_single = test_data["batch_single"]

        t_test = torch.tensor([2.5]).to(DEVICE)
        t_last = torch.tensor([2.3]).to(DEVICE)

        cdf = model.cdf_time(t_test, batch_single, log=False)

        # Compute expected CDF from cumulative intensity
        ci_diff = model.cumulative_intensity(t_test, batch_single, T_low=t_last)
        cdf_expected = 1 - torch.exp(-ci_diff.sum(dim=1))

        assert torch.allclose(cdf.float(), cdf_expected.float(), atol=tolerances["tight"]), (
            "CDF should equal 1 - exp(-cumulative intensity)"
        )

    def test_cdf_near_zero_after_last_event(self, representative_model, test_data, tolerances, numerical_settings):
        """Test CDF is near zero just after last event."""
        model = representative_model
        batch_single = test_data["batch_single"]
        seq1 = test_data["seq1"]

        last_time = seq1.time_points.max().item()
        t_just_after = torch.tensor([last_time + numerical_settings["epsilon"]]).to(DEVICE)

        cdf = model.cdf_time(t_just_after, batch_single, log=False)

        assert torch.allclose(cdf, torch.zeros_like(cdf), atol=tolerances["standard"]), (
            "CDF should be near zero just after last event"
        )

    def test_cdf_monotonic(self, representative_model, test_data):
        """Test CDF is monotonically increasing."""
        model = representative_model
        batch_single = test_data["batch_single"]

        times = torch.linspace(batch_single.last_event_time.item(), (batch_single.last_event_time + 2.0).item(), 20).to(DEVICE)
        cdfs = torch.stack([model.cdf_time(t.unsqueeze(0), batch_single, log=False) for t in times])

        # Check monotonicity for each dimension
        for dim in range(cdfs.shape[1]):
            diffs = cdfs[1:, dim] - cdfs[:-1, dim]
            assert torch.all(diffs >= -1e-6), f"CDF should be monotonic for dimension {dim}"

    def test_pdf_cdf_integration_relation(self, representative_model, test_data, tolerances, numerical_settings):
        """Test that integral of PDF approximates CDF."""
        model = representative_model
        batch_single = test_data["batch_single"]
        seq1 = test_data["seq1"]

        last_time = batch_single.last_event_time.item()
        start_time = last_time + numerical_settings["epsilon"]
        end_time = start_time + 2.0

        # Compute CDF difference (integral from start to end)
        cdf_high = model.cdf_time(torch.tensor([end_time]).to(DEVICE), batch_single, log=False)
        cdf_low = model.cdf_time(torch.tensor([start_time]).to(DEVICE), batch_single, log=False)
        cdf_diff = cdf_high - cdf_low

        # Compute via numerical integration of PDF
        times = torch.linspace(start_time, end_time, steps=numerical_settings["integration_steps"]).to(DEVICE)

        pdfs = torch.stack([model.density_joint(t.unsqueeze(0), batch_single, log=False) for t in times])
        pdfs = pdfs.squeeze()

        # Integrate over time dimension and sum over event types
        pdf_integrated = torch.trapz(pdfs, times, dim=0).sum()

        assert torch.allclose(cdf_diff.sum().float(), pdf_integrated.float(), atol=tolerances["standard"]), (
            "Integrated PDF should approximate CDF difference"
        )


class TestLikelihood:
    """Tests for likelihood computation."""

    def test_likelihood_finite(self, representative_model, test_data):
        """Test likelihood is finite."""
        model = representative_model
        batch_single = test_data["batch_single"]
        T = torch.tensor([4.0]).to(DEVICE)

        log_lik = model.likelihood(batch_single, T, log=True)
        assert torch.isfinite(log_lik), "Log-likelihood should be finite"

    def test_likelihood_shape(self, representative_model, test_data):
        """Test likelihood has correct shape."""
        model = representative_model
        batch_single = test_data["batch_single"]
        batch_multi = test_data["batch_multi"]

        # Single batch
        lik = model.likelihood(batch_single, torch.tensor([4.0]).to(DEVICE), log=False)
        assert lik.shape == (1,), "Likelihood shape mismatch"

        # Multi batch
        lik_multi = model.likelihood(batch_multi, torch.tensor([4.0, 3.0]).to(DEVICE), log=False)
        assert lik_multi.shape == (2,), "Multi-batch likelihood shape mismatch"

    def test_likelihood_positive(self, representative_model, test_data):
        """Test likelihood is positive."""
        model = representative_model
        batch_single = test_data["batch_single"]

        lik = model.likelihood(batch_single, torch.tensor([4.0]).to(DEVICE), log=False)
        assert torch.all(lik > 0), "Likelihood should be positive"

    def test_log_likelihood_non_positive(self, representative_model, test_data):
        """Test log-likelihood is non-positive."""
        model = representative_model
        batch_single = test_data["batch_single"]

        log_lik = model.likelihood(batch_single, torch.tensor([4.0]).to(DEVICE), log=True)
        assert torch.all(log_lik <= 0), "Log-likelihood should be non-positive"

    def test_likelihood_empty_sequence(self, representative_model, test_data):
        """Test likelihood on empty sequence."""
        model = representative_model
        empty_batch = test_data["empty_batch"]
        T = torch.tensor([2.0]).to(DEVICE)

        log_lik = model.likelihood(empty_batch, T, log=True)
        assert torch.isfinite(log_lik), "Empty sequence log-likelihood should be finite"
        # Empty sequence should have negative log-likelihood (only survival term)
        assert log_lik < 0, "Empty sequence log-likelihood should be negative"

    def test_likelihood_batch_consistency(self, representative_model, test_data, tolerances):
        """Test batched likelihood equals sum of individual likelihoods."""
        model = representative_model
        batch_multi = test_data["batch_multi"]
        seq1 = test_data["seq1"]
        seq2 = test_data["seq2"]

        # Batched call
        log_lik_batched = model.likelihood(batch_multi, torch.tensor([4.0, 3.0]).to(DEVICE), log=True)

        # Individual calls
        from torchpp.utils.event_data import BatchedMVEventData

        batch1 = BatchedMVEventData([seq1.time_points], [seq1.event_types]).to(DEVICE)
        batch2 = BatchedMVEventData([seq2.time_points], [seq2.event_types]).to(DEVICE)
        log_lik1 = model.likelihood(batch1, torch.tensor([4.0]).to(DEVICE), log=True)
        log_lik2 = model.likelihood(batch2, torch.tensor([3.0]).to(DEVICE), log=True)

        # For log-likelihood, batch result should be sum of individuals
        assert torch.allclose(log_lik_batched.sum(), log_lik1 + log_lik2, atol=tolerances["tight"]), (
            "Batched log-likelihood should equal sum of individual log-likelihoods"
        )

    def test_log_exp_consistency(self, representative_model, test_data, tolerances):
        """Test that exp(log_likelihood) equals likelihood."""
        model = representative_model
        batch_single = test_data["batch_single"]
        T = torch.tensor([4.0]).to(DEVICE)

        lik = model.likelihood(batch_single, T, log=False)
        log_lik = model.likelihood(batch_single, T, log=True)

        assert torch.allclose(torch.exp(log_lik).float(), lik.float(), atol=tolerances["tight"]), (
            "exp(log_likelihood) should equal likelihood"
        )

    def test_positive_term_shape(self, representative_model, test_data):
        """Test positive term has correct shape."""
        model = representative_model
        skip_if_density(model)  # _event_log_intensity_sum is intensity-family-specific
        batch_single = test_data["batch_single"]

        pos_lik = model._event_log_intensity_sum(batch_single).exp()
        assert pos_lik.shape == (1,), "Positive term shape mismatch"

    def test_positive_term_positive(self, representative_model, test_data):
        """Test positive term is positive."""
        model = representative_model
        skip_if_density(model)  # _event_log_intensity_sum is intensity-family-specific
        batch_single = test_data["batch_single"]

        pos_lik = model._event_log_intensity_sum(batch_single).exp()
        assert torch.all(pos_lik > 0), "Positive term should be positive"

    @staticmethod
    def _survival_ci(model, T, batch):
        """CI as used by the likelihood's survival term.

        with no origin the survival integrates from the first event t₁
        (condition-on-first); with an origin it integrates from 0.
        """
        # Match the data dtype so the window starts exactly at t₁, as in the likelihood.
        T = T.to(batch.time_points.dtype)
        T_low = None if (model.has_origin or batch.time_points.shape[1] == 0) else batch.time_points[:, 0]
        return model.cumulative_intensity(T, batch, T_low=T_low).sum(dim=1)

    def test_likelihood_decomposition(self, representative_model, test_data, tolerances):
        """Test likelihood = positive term * exp(-cumulative_intensity)."""
        model = representative_model
        skip_if_density(model)  # decomposition is via intensity/CI; density is native
        batch_single = test_data["batch_single"]
        T = torch.tensor([4.0]).to(DEVICE)

        lik = model.likelihood(batch_single, T, log=False)

        # Compute components
        pos_lik = model._event_log_intensity_sum(batch_single).exp()
        ci_end = self._survival_ci(model, T, batch_single)
        lik_expected = pos_lik * torch.exp(-ci_end)

        assert torch.allclose(lik.float(), lik_expected.float(), atol=tolerances["tight"]), (
            "Likelihood should equal positive term * exp(-cumulative_intensity)"
        )

    def test_log_likelihood_decomposition(self, representative_model, test_data, tolerances):
        """Test log_likelihood = log positive term - cumulative_intensity."""
        model = representative_model
        skip_if_density(model)  # decomposition is via intensity/CI; density is native
        batch_single = test_data["batch_single"]
        T = torch.tensor([4.0]).to(DEVICE)

        log_lik = model.likelihood(batch_single, T, log=True)

        # Compute components
        log_pos_lik = model._event_log_intensity_sum(batch_single)
        ci_end = self._survival_ci(model, T, batch_single)
        log_lik_expected = log_pos_lik - ci_end

        assert torch.allclose(log_lik.float(), log_lik_expected.float(), atol=tolerances["tight"]), (
            "Log-likelihood should equal log positive term - cumulative_intensity"
        )

    def test_empty_sequence_likelihood_relation(self, representative_model, test_data, tolerances):
        """Test empty sequence likelihood equals exp(-cumulative_intensity)."""
        model = representative_model
        empty_batch = test_data["empty_batch"]
        T = torch.tensor([2.0]).to(DEVICE)

        skip_if_density(model)  # asserted via cumulative_intensity (intensity family)
        # For empty sequence, the positive term should be 1 (no events)
        # So likelihood = exp(-CI)
        log_lik = model.likelihood(empty_batch, T, log=True)
        ci_end = model.cumulative_intensity(T, empty_batch).sum(dim=1)
        log_lik_expected = -ci_end

        assert torch.allclose(log_lik.float(), log_lik_expected.float(), atol=tolerances["tight"]), (
            "Empty sequence log-likelihood should equal -cumulative_intensity"
        )

    def test_empty_sequence_positive_term(self, representative_model, test_data, tolerances):
        """Test empty sequence positive term is 1, and log is 0."""
        model = representative_model
        skip_if_density(model)  # _event_log_intensity_sum is intensity-family-specific
        empty_batch = test_data["empty_batch"]

        log_lik = model._event_log_intensity_sum(empty_batch)
        assert torch.allclose(log_lik.float(), torch.zeros_like(log_lik).float(), atol=tolerances["tight"]), (
            "Empty sequence log positive term should be 0"
        )
        lik = model._event_log_intensity_sum(empty_batch).exp()
        assert torch.allclose(lik.float(), torch.ones_like(lik).float(), atol=tolerances["tight"]), (
            "Empty sequence positive term should be 1"
        )

    def test_positive_term_from_intensities(self, representative_model, test_data, tolerances):
        """Test positive term equals product of intensities at event times."""
        model = representative_model
        skip_if_density(model)  # positive term = Σ log λ at events (intensity family)
        batch_single = test_data["batch_single"]
        seq1 = test_data["seq1"]

        # Compute positive log-likelihood term
        log_pos_lik = model._event_log_intensity_sum(batch_single)

        # Compute log intensities at event times
        # Need to stack sequence on itself with increasing lengths
        from torchpp.utils.event_data import BatchedMVEventData

        stacked_batch = BatchedMVEventData(
            time_points=[seq1.time_points[:i] for i in range(len(seq1.time_points))],
            event_types=[seq1.event_types[:i] for i in range(len(seq1.event_types))],
        ).to(DEVICE)

        stacked_intensities = model.intensity(seq1.time_points, stacked_batch)
        intensity_at_events = stacked_intensities[torch.arange(len(seq1.time_points)), seq1.event_types]
        log_intensity_at_events = torch.log(intensity_at_events)
        # no-origin conditions on the first event — it is not in the positive term.
        if not model.has_origin:
            log_intensity_at_events = log_intensity_at_events[1:]
        log_pos_lik_expected = log_intensity_at_events.sum()

        assert torch.allclose(log_pos_lik.float(), log_pos_lik_expected.float(), atol=tolerances["tight"]), (
            "Log positive term should equal sum of log intensities at event times"
        )

    def test_likelihood_padding_invariance(self, representative_model, test_data, tolerances):
        """Test padding does not affect likelihood."""
        model = representative_model
        batch_single = test_data["batch_single"]
        padded_batch = test_data["padded_batch"]
        T = torch.tensor([4.0]).to(DEVICE)

        log_lik_normal = model.likelihood(batch_single, T, log=True)
        log_lik_padded = model.likelihood(padded_batch, T, log=True)

        assert torch.allclose(log_lik_normal.float(), log_lik_padded.float(), atol=tolerances["tight"]), (
            "Padding should not affect likelihood"
        )

    def test_likelihood_illegal_negative_T(self, representative_model, test_data):
        """Test that negative T raises an error."""
        model = representative_model
        batch_single = test_data["batch_single"]

        with pytest.raises(ValueError):
            model.likelihood(batch_single, torch.tensor([-1.0]).to(DEVICE), log=True)

    def test_likelihood_illegal_T_before_last_event(self, representative_model, test_data):
        """Test that T before last event raises an error."""
        model = representative_model
        batch_single = test_data["batch_single"]
        seq1 = test_data["seq1"]

        last_time = seq1.time_points.max().item()
        T_before_last = torch.tensor([last_time - 0.1]).to(DEVICE)

        with pytest.raises(ValueError):
            model.likelihood(batch_single, T_before_last, log=True)

    def test_likelihood_from_pdf_and_survival(self, representative_model, test_data, tolerances):
        """Test likelihood = product of PDFs at event times × survival function.

        The likelihood can be decomposed as:
        L = ∏ᵢ PDF(tᵢ | history) × S(T | history)
        where S(T) = 1 - CDF(T) is the survival function (probability of no event after last event).
        """
        model = representative_model
        skip_if_density(model)  # survival term recomputed via cumulative_intensity (intensity family)
        batch_single = test_data["batch_single"]
        seq1 = test_data["seq1"]
        T = torch.tensor([4.0]).to(DEVICE)

        log_lik = model.likelihood(batch_single, T, log=True)

        # Compute PDFs at each event time
        # Need to evaluate PDF at each event given history up to that point
        from torchpp.utils.event_data import BatchedMVEventData

        log_pdf_sum = 0.0
        for i in range(len(seq1.time_points)):
            # no-origin conditions on the first event, so its PDF factor is dropped;
            # the survival then telescopes over [t₁, T], matching ``likelihood``.
            if not model.has_origin and i == 0:
                continue
            # History up to (but not including) event i
            if i == 0:
                history_batch = test_data["empty_batch"]
            else:
                history_batch = BatchedMVEventData([seq1.time_points[:i]], [seq1.event_types[:i]]).to(DEVICE)

            # PDF at event i
            t_i = seq1.time_points[i].unsqueeze(0)
            pdf_i = model.density_joint(t_i, history_batch, log=False)

            # Select the PDF for the actual event type that occurred
            pdf_i_type = pdf_i[0, seq1.event_types[i]]
            log_pdf_sum += torch.log(pdf_i_type)

        # Compute log-survival: log S(T) = log(1 - CDF(T)) = -∑_d Λ_d(T)
        # We compute this directly from cumulative intensity to avoid the
        # catastrophic cancellation in log(1 - CDF) when CDF ≈ 1.
        ci_from_last = model.cumulative_intensity(T, batch_single, T_low=batch_single.last_event_time)
        log_survival = -ci_from_last.sum(dim=1)  # Shape: (B,)

        # Likelihood should equal product of PDFs times survival
        log_lik_expected = log_pdf_sum + log_survival

        assert torch.allclose(log_lik.float(), log_lik_expected.float(), atol=tolerances["standard"]), (
            f"Likelihood from PDF/survival {log_lik_expected.item():.6f} should match direct likelihood {log_lik.item():.6f}"
        )


class TestSampling:
    """Tests for sampling functionality."""

    def test_sampling_basic(self, representative_model, test_data, d):
        """Test basic sampling produces valid events."""
        model = representative_model
        # Seed from a real history: the representative models are has_origin=False, for which
        # unconditional (empty) sampling is guarded (Phase D) — see test_sampling_regime below.
        seq1 = test_data["seq1"]

        ts_sampled, time_samples, dist_samples = model.sample(seq1, num_steps=10, rng=SEED)

        # Check we got some samples
        assert len(time_samples) > 0, "Should produce at least one sample"
        assert len(time_samples) <= 10, "Should not produce more than requested samples"

        # Check event types are valid
        assert torch.all(ts_sampled.event_types >= 0), "Event types should be non-negative"
        assert torch.all(ts_sampled.event_types < d), f"Event types should be < {d}"

        # Check times are increasing
        # time_samples may be a list, convert to tensor if needed
        if len(time_samples) > 1:
            time_samples_tensor = (
                torch.tensor(time_samples).to(DEVICE) if not isinstance(time_samples, torch.Tensor) else time_samples
            )
            assert torch.all(time_samples_tensor[1:] > time_samples_tensor[:-1]), "Sample times should be increasing"

    def test_sampling_from_history(self, representative_model, test_data):
        """Test sampling can continue from existing history."""
        model = representative_model
        seq1 = test_data["seq1"]

        ts_sampled, time_samples, dist_samples = model.sample(seq1, num_steps=5, rng=SEED)

        # Check new samples come after last event
        last_event_time = seq1.time_points.max().item()
        if len(time_samples) > 0:
            # time_samples may be a list
            first_sample_time = time_samples[0] if isinstance(time_samples, list) else time_samples[0].item()
            assert first_sample_time > last_event_time, "New samples should come after last event"

    def test_sampling_reproducibility(self, representative_model, test_data):
        """Test sampling with same seed produces same results."""
        model = representative_model
        seq1 = test_data["seq1"]  # has_origin=False models require a seed (Phase D)

        # Sample twice with same seed
        ts1, times1, _ = model.sample(seq1, num_steps=5, rng=SEED)
        ts2, times2, _ = model.sample(seq1, num_steps=5, rng=SEED)

        # Convert to tensors if needed for comparison
        times1_tensor = torch.tensor(times1).to(DEVICE) if not isinstance(times1, torch.Tensor) else times1
        times2_tensor = torch.tensor(times2).to(DEVICE) if not isinstance(times2, torch.Tensor) else times2

        # Should produce same results
        assert torch.allclose(times1_tensor, times2_tensor), "Same seed should produce same sample times"
        assert torch.equal(ts1.event_types, ts2.event_types), "Same seed should produce same event types"


def _empty_history():
    return MVEventData(
        torch.zeros(0, dtype=torch.float64, device=DEVICE),
        torch.zeros(0, dtype=torch.long, device=DEVICE),
    )


def test_sampling_regime_guard():
    """Phase D: unconditional sampling (ts=None or empty) is allowed only for a model that
    learned the first-event distribution (``has_origin=True``); a condition-on-first model
    (``has_origin=False``) must be seeded with a non-empty history."""
    # has_origin=False (the default / condition-on-first regime): guarded.
    cond = create_model("poisson", seed=SEED)
    assert cond.has_origin is False
    for seed_ts in (None, _empty_history()):
        with pytest.raises(ValueError, match="condition-on-first"):
            cond.sample(seed_ts, num_steps=3, rng=SEED)
    # Seeding with a real history is fine.
    seed = MVEventData(
        torch.tensor([0.3, 0.7], dtype=torch.float64, device=DEVICE),
        torch.tensor([0, 1], dtype=torch.long, device=DEVICE),
    )
    out, times, _ = cond.sample(seed, num_steps=3, rng=SEED)
    assert len(times) > 0

    # has_origin=True: unconditional generation is defined (for homogeneous Poisson the
    # first event is drawn from the baseline over [0, t_1]).
    uncond = create_model("poisson", seed=SEED)
    uncond.has_origin = True
    for seed_ts in (None, _empty_history()):
        out, times, _ = uncond.sample(seed_ts, num_steps=3, rng=SEED)
        assert len(times) > 0 and torch.all(out.event_types >= 0)


class TestInverseCDF:
    """Tests for inverse CDF computation."""

    def test_inverse_cdf_shape(self, representative_model, test_data, d):
        """Test inverse CDF has correct output shape."""
        model = representative_model
        batch_single = test_data["batch_single"]
        probs = torch.tensor([0.3, 0.5, 0.7]).to(DEVICE)

        times = torch.stack([model.time_quantile(probs[i].unsqueeze(0), batch_single) for i in range(probs.shape[0])])
        assert times.shape == (3, 1), "Inverse CDF shape mismatch"

    def test_inverse_cdf_consistency(self, representative_model, test_data, tolerances):
        """Test that inverse CDF is consistent with CDF."""
        model = representative_model
        batch_single = test_data["batch_single"]

        # Pick a time point
        t_original = torch.tensor([2.5]).to(DEVICE)

        cdf_vals = model.cdf_time(t_original, batch_single, log=False)
        t_recovered = model.time_quantile(cdf_vals, batch_single)

        # Should recover original time (approximately)
        assert torch.allclose(
            t_original.expand_as(t_recovered).float(), t_recovered.float(), atol=tolerances["standard"]
        ), "Inverse CDF should recover original time"
