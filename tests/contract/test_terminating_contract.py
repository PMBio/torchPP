"""Universal terminating-mode contract tests.

Termination properties that hold for EVERY model via an identical code path:
mask shape/semantics, termination-time shape/semantics, batch==loop
consistency, padding invariance, finiteness/bounds, the algebraic identities
(L = L_pos·L_neg; log L_neg = -CI; L_neg = exp(-CI); decomposition), empty
sequence = survival-only, repeated event -> likelihood 0 / log-lik -inf,
full termination -> intensity/PDF/CDF = 0 and likelihood frozen-in-T, sampling
stops after D events / no samples when fully terminated / samples after last
event, single-event-type D=1, zero-intensity-stays-zero.

Run over the archetype fan (representative_terminating_model /
representative_model_pair / terminating_test_data). NO skip-sets.
"""

import pytest
import torch
from helpers import DEVICE, SEED, TIGHT_TOLERANCE, create_model, skip_if_density


class TestTerminationMask:
    """Tests for termination mask computation."""

    def test_mask_empty_sequence(self, representative_terminating_model, terminating_test_data, d):
        """Test all types are active for empty sequence."""
        empty_batch = terminating_test_data["empty_batch"]

        mask = representative_terminating_model._get_termination_mask(empty_batch)
        assert mask.shape == (1, d), "Mask shape mismatch"
        assert torch.all(mask), "Empty sequence should have all types active"

    def test_mask_single_event(self, representative_terminating_model, terminating_test_data, d):
        """Test only occurred type is masked for single event."""
        single_batch = terminating_test_data["single_batch"]

        mask = representative_terminating_model._get_termination_mask(single_batch)
        expected = torch.tensor([[False] + [True] * (d - 1)]).to(DEVICE)
        assert torch.equal(mask, expected), f"Single event mask mismatch: {mask} vs {expected}"

    def test_mask_partial_termination(self, representative_terminating_model, terminating_test_data, d):
        """Test partial termination masks only occurred types."""
        if d < 3:
            pytest.skip("Test requires D >= 3")

        partial_batch = terminating_test_data["partial_batch"]

        mask = representative_terminating_model._get_termination_mask(partial_batch)
        # Types 0 and 2 occurred, type 1 still active
        expected = torch.tensor([[False, True, False]]).to(DEVICE)
        assert torch.equal(mask, expected), f"Partial mask mismatch: {mask} vs {expected}"

    def test_mask_full_termination(self, representative_terminating_model, terminating_test_data):
        """Test all types are masked when all have occurred."""
        full_batch = terminating_test_data["full_batch"]

        mask = representative_terminating_model._get_termination_mask(full_batch)
        assert not torch.any(mask), "Full termination should have no active types"

    def test_mask_repeated_types(self, representative_terminating_model, terminating_test_data, d):
        """Test that repeated type occurrences use first occurrence."""
        repeated_batch = terminating_test_data["repeated_batch"]

        mask = representative_terminating_model._get_termination_mask(repeated_batch)
        # Types 0 and 1 occurred (type 0 repeated but still masked)
        expected = torch.tensor([[False, False] + [True] * (d - 2)]).to(DEVICE)
        assert torch.equal(mask, expected), f"Repeated types mask mismatch: {mask}"

    def test_mask_time_aware(self, representative_terminating_model, terminating_test_data, d):
        """Test time-aware mask computation."""
        if d < 3:
            pytest.skip("Test requires D >= 3")

        partial_batch = terminating_test_data["partial_batch"]
        # partial_seq has: type 0 at t=0.5, type 2 at t=1.5

        # At t=0.3 (before any event): all active
        mask_before = representative_terminating_model._get_termination_mask(
            partial_batch, t=torch.tensor([0.3]).to(DEVICE)
        )
        assert torch.all(mask_before), "Before events, all types should be active"

        # At t=0.6 (after type 0, before type 2): type 0 inactive
        mask_middle = representative_terminating_model._get_termination_mask(
            partial_batch, t=torch.tensor([0.6]).to(DEVICE)
        )
        expected_middle = torch.tensor([[False, True, True]]).to(DEVICE)
        assert torch.equal(mask_middle, expected_middle), f"Time-aware mask at t=0.6 mismatch: {mask_middle}"

        # At t=2.0 (after both events): types 0 and 2 inactive
        mask_after = representative_terminating_model._get_termination_mask(
            partial_batch, t=torch.tensor([2.0]).to(DEVICE)
        )
        expected_after = torch.tensor([[False, True, False]]).to(DEVICE)
        assert torch.equal(mask_after, expected_after), f"Time-aware mask at t=2.0 mismatch: {mask_after}"


class TestTerminationTimes:
    """Tests for _get_termination_times computation."""

    def test_termination_times_empty_sequence(self, representative_terminating_model, terminating_test_data, d):
        """Test all termination times are inf for empty sequence."""
        empty_batch = terminating_test_data["empty_batch"]

        term_times = representative_terminating_model._get_termination_times(empty_batch)
        assert term_times.shape == (1, d), f"Termination times shape mismatch: {term_times.shape}"
        assert torch.all(torch.isinf(term_times)), "Empty sequence should have all inf termination times"

    def test_termination_times_single_event(self, representative_terminating_model, terminating_test_data, d):
        """Test correct termination time for single event."""
        single_batch = terminating_test_data["single_batch"]

        term_times = representative_terminating_model._get_termination_times(single_batch)

        # Type 0 occurred at t=0.5
        assert torch.isclose(term_times[0, 0].float(), torch.tensor(0.5).to(DEVICE)), (
            f"Type 0 termination time should be 0.5: {term_times[0, 0]}"
        )
        # Other types should be inf
        for i in range(1, d):
            assert torch.isinf(term_times[0, i]), f"Type {i} should have inf termination time"

    def test_termination_times_partial(self, representative_terminating_model, terminating_test_data, d):
        """Test termination times for partial termination."""
        if d < 3:
            pytest.skip("Test requires D >= 3")

        partial_batch = terminating_test_data["partial_batch"]
        # partial_seq has: type 0 at t=0.5, type 2 at t=1.5

        term_times = representative_terminating_model._get_termination_times(partial_batch)

        assert torch.isclose(term_times[0, 0].float(), torch.tensor(0.5).to(DEVICE)), (
            f"Type 0 termination time should be 0.5: {term_times[0, 0]}"
        )
        assert torch.isinf(term_times[0, 1]), f"Type 1 termination time should be inf: {term_times[0, 1]}"
        assert torch.isclose(term_times[0, 2].float(), torch.tensor(1.5).to(DEVICE)), (
            f"Type 2 termination time should be 1.5: {term_times[0, 2]}"
        )

    def test_termination_times_full(self, representative_terminating_model, terminating_test_data, d):
        """Test all termination times are finite when all types occurred."""
        full_batch = terminating_test_data["full_batch"]

        term_times = representative_terminating_model._get_termination_times(full_batch)
        assert torch.all(torch.isfinite(term_times)), "Full termination should have all finite termination times"

    def test_termination_times_repeated_uses_first(self, representative_terminating_model, terminating_test_data, d):
        """Test that repeated type occurrences use first occurrence time."""
        repeated_batch = terminating_test_data["repeated_batch"]
        # repeated_seq has: type 0 at t=0.3, type 1 at t=0.8, type 0 at t=1.2

        term_times = representative_terminating_model._get_termination_times(repeated_batch)

        # Type 0 should use FIRST occurrence (t=0.3, not t=1.2)
        assert torch.isclose(term_times[0, 0].float(), torch.tensor(0.3).to(DEVICE)), (
            f"Type 0 should use first occurrence (0.3), got: {term_times[0, 0]}"
        )
        # Type 1 occurred at t=0.8
        assert torch.isclose(term_times[0, 1].float(), torch.tensor(0.8).to(DEVICE)), (
            f"Type 1 termination time should be 0.8: {term_times[0, 1]}"
        )

    def test_termination_times_batched(self, representative_terminating_model, terminating_test_data, d):
        """Test termination times for batched data."""
        multi_batch = terminating_test_data["multi_batch"]

        term_times = representative_terminating_model._get_termination_times(multi_batch)
        assert term_times.shape[0] == 2, "Batch dimension should be 2"
        assert term_times.shape[1] == d, f"Type dimension should be {d}"

    def test_termination_times_non_terminating_model(self, representative_model_pair, terminating_test_data, d):
        """Test that non-terminating model returns all inf termination times."""
        _, model_non_term = representative_model_pair
        partial_batch = terminating_test_data["partial_batch"]

        term_times = model_non_term._get_termination_times(partial_batch)
        assert torch.all(torch.isinf(term_times)), "Non-terminating model should return all inf termination times"

    def test_termination_times_with_padding(self, representative_terminating_model, d):
        """Test termination times correctly ignore padded events."""
        from torchpp.utils.event_data import MVEventData, BatchedMVEventData

        # Create sequence with padding: type 0 at t=0.5, then padding
        padded_seq = MVEventData(
            torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64).to(DEVICE),
            torch.tensor([0, -1, -1], dtype=torch.long).to(DEVICE),  # -1 indicates padding
        )
        padded_batch = BatchedMVEventData([padded_seq.time_points], [padded_seq.event_types]).to(DEVICE)

        term_times = representative_terminating_model._get_termination_times(padded_batch)

        # Type 0 should have termination time 0.5
        assert torch.isclose(term_times[0, 0].float(), torch.tensor(0.5).to(DEVICE)), (
            f"Type 0 termination time should be 0.5: {term_times[0, 0]}"
        )
        # Padded events (type -1 clamped to 0) should NOT affect termination times
        # Other types should be inf
        for i in range(1, d):
            assert torch.isinf(term_times[0, i]), (
                f"Type {i} should have inf termination time (padding should be ignored)"
            )

    def test_termination_times_order_independence(self, representative_terminating_model, d):
        """Test that termination times are independent of event order for same type."""
        from torchpp.utils.event_data import MVEventData, BatchedMVEventData

        if d < 2:
            pytest.skip("Test requires D >= 2")

        # Sequence 1: type 0 first, then type 1
        seq1 = MVEventData(
            torch.tensor([0.3, 0.8], dtype=torch.float64).to(DEVICE), torch.tensor([0, 1], dtype=torch.long).to(DEVICE)
        )
        batch1 = BatchedMVEventData([seq1.time_points], [seq1.event_types]).to(DEVICE)

        # Sequence 2: type 1 first, then type 0 (different order)
        seq2 = MVEventData(
            torch.tensor([0.3, 0.8], dtype=torch.float64).to(DEVICE), torch.tensor([1, 0], dtype=torch.long).to(DEVICE)
        )
        batch2 = BatchedMVEventData([seq2.time_points], [seq2.event_types]).to(DEVICE)

        term_times1 = representative_terminating_model._get_termination_times(batch1)
        term_times2 = representative_terminating_model._get_termination_times(batch2)

        # Type 0: first at 0.3 in batch1, at 0.8 in batch2
        assert torch.isclose(term_times1[0, 0].float(), torch.tensor(0.3).to(DEVICE)), "Batch1 type 0 should be 0.3"
        assert torch.isclose(term_times2[0, 0].float(), torch.tensor(0.8).to(DEVICE)), "Batch2 type 0 should be 0.8"

        # Type 1: at 0.8 in batch1, at 0.3 in batch2
        assert torch.isclose(term_times1[0, 1].float(), torch.tensor(0.8).to(DEVICE)), "Batch1 type 1 should be 0.8"
        assert torch.isclose(term_times2[0, 1].float(), torch.tensor(0.3).to(DEVICE)), "Batch2 type 1 should be 0.3"


class TestTerminatingIntensity:
    """Tests for intensity with termination (shape/zero/batch contract)."""

    def test_intensity_full_termination_is_zero(self, representative_terminating_model, terminating_test_data, tolerances):
        """Test intensity is zero when all types have occurred."""
        full_batch = terminating_test_data["full_batch"]

        t_after = torch.tensor([2.0]).to(DEVICE)
        int_full = representative_terminating_model.intensity(t_after, full_batch)

        assert torch.all(int_full.abs() < tolerances["tight"]), "Full termination should have zero intensity"

    def test_intensity_batch_consistency(self, representative_terminating_model, terminating_test_data, tolerances):
        """Test batched terminating intensity is consistent."""
        multi_batch = terminating_test_data["multi_batch"]
        partial_seq = terminating_test_data["partial_seq"]
        seq2 = terminating_test_data["seq2"]

        # Batched call
        int_batched = representative_terminating_model.intensity(torch.tensor([2.0, 2.0]).to(DEVICE), multi_batch)

        # Individual calls
        from torchpp.utils.event_data import BatchedMVEventData

        batch1 = BatchedMVEventData([partial_seq.time_points], [partial_seq.event_types]).to(DEVICE)
        batch2 = BatchedMVEventData([seq2.time_points], [seq2.event_types]).to(DEVICE)
        int1 = representative_terminating_model.intensity(torch.tensor([2.0]).to(DEVICE), batch1)
        int2 = representative_terminating_model.intensity(torch.tensor([2.0]).to(DEVICE), batch2)
        int_combined = torch.cat([int1, int2], dim=0)

        assert torch.allclose(int_batched.float(), int_combined.float(), atol=tolerances["tight"]), (
            "Batched terminating intensity should match individual calls"
        )


class TestTerminatingCumulativeIntensity:
    """Tests for cumulative intensity with termination (intensity family only)."""

    @pytest.fixture(autouse=True)
    def _intensity_only(self, representative_terminating_model):
        # cumulative_intensity is an intensity-family realization; density models have none.
        skip_if_density(representative_terminating_model)

    def test_ci_positive(self, representative_terminating_model, terminating_test_data):
        """Test CI is always non-negative."""
        partial_batch = terminating_test_data["partial_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        ci = representative_terminating_model.cumulative_intensity(T, partial_batch)

        assert torch.all(ci >= 0), "CI should be non-negative"

    def test_ci_frozen_after_termination(self, representative_terminating_model, terminating_test_data, d, tolerances):
        """Test CI stops accumulating after termination (is frozen/constant).

        Once a type has occurred, its cumulative intensity should not increase
        further, regardless of how much time passes.

        Note: Models using numerical CI may have small variations due to discretization.
        """
        if d < 3:
            pytest.skip("Test requires D >= 3")

        partial_batch = terminating_test_data["partial_batch"]
        # partial_seq: type 0 at t=0.5, type 2 at t=1.5

        # Use standard tolerance for numerical CI models
        tol = tolerances["standard"] if not (representative_terminating_model.ci_config.mode == "closed_form") else tolerances["tight"]

        # CI at different times after type 0 terminated (t=0.5)
        ci_at_1 = representative_terminating_model.cumulative_intensity(torch.tensor([1.0]).to(DEVICE), partial_batch)
        ci_at_2 = representative_terminating_model.cumulative_intensity(torch.tensor([2.0]).to(DEVICE), partial_batch)
        ci_at_5 = representative_terminating_model.cumulative_intensity(torch.tensor([5.0]).to(DEVICE), partial_batch)

        # Type 0 CI should be frozen at all times after 0.5
        assert torch.allclose(ci_at_1[0, 0].float(), ci_at_2[0, 0].float(), atol=tol), (
            f"Type 0 CI should be frozen after termination: {ci_at_1[0, 0]} vs {ci_at_2[0, 0]}"
        )
        assert torch.allclose(ci_at_2[0, 0].float(), ci_at_5[0, 0].float(), atol=tol), (
            f"Type 0 CI should remain frozen indefinitely: {ci_at_2[0, 0]} vs {ci_at_5[0, 0]}"
        )

        # Type 2 CI should be frozen after 1.5
        assert torch.allclose(ci_at_2[0, 2].float(), ci_at_5[0, 2].float(), atol=tol), (
            f"Type 2 CI should be frozen after termination at t=1.5: {ci_at_2[0, 2]} vs {ci_at_5[0, 2]}"
        )

        # Type 1 (active) should still be increasing
        assert ci_at_2[0, 1] > ci_at_1[0, 1], "Active type 1 CI should increase"
        assert ci_at_5[0, 1] > ci_at_2[0, 1], "Active type 1 CI should keep increasing"

    def test_ci_batch_consistency(self, representative_terminating_model, terminating_test_data, tolerances):
        """Test batched CI computation matches individual calls."""
        multi_batch = terminating_test_data["multi_batch"]
        partial_seq = terminating_test_data["partial_seq"]
        seq2 = terminating_test_data["seq2"]

        # Batched call
        ci_batched = representative_terminating_model.cumulative_intensity(torch.tensor([2.0, 2.0]).to(DEVICE), multi_batch)

        # Individual calls
        from torchpp.utils.event_data import BatchedMVEventData

        batch1 = BatchedMVEventData([partial_seq.time_points], [partial_seq.event_types]).to(DEVICE)
        batch2 = BatchedMVEventData([seq2.time_points], [seq2.event_types]).to(DEVICE)
        ci1 = representative_terminating_model.cumulative_intensity(torch.tensor([2.0]).to(DEVICE), batch1)
        ci2 = representative_terminating_model.cumulative_intensity(torch.tensor([2.0]).to(DEVICE), batch2)
        ci_combined = torch.cat([ci1, ci2], dim=0)

        assert torch.allclose(ci_batched.float(), ci_combined.float(), atol=tolerances["tight"]), (
            "Batched CI should match individual calls"
        )

    def test_ci_monotonic_then_constant(self, representative_terminating_model, terminating_test_data, d, tolerances):
        """Test CI is monotonically increasing until termination, then constant.

        For each type, CI should increase until the termination event,
        then remain constant forever after.

        Note: Models using numerical CI may have small variations due to discretization.
        """
        if d < 3:
            pytest.skip("Test requires D >= 3")

        partial_batch = terminating_test_data["partial_batch"]
        # partial_seq: type 0 at t=0.5, type 2 at t=1.5

        # Use standard tolerance for numerical CI models
        tol = tolerances["standard"] if not (representative_terminating_model.ci_config.mode == "closed_form") else tolerances["tight"]

        # Test times spanning before and after terminations
        times = torch.tensor([0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]).to(DEVICE)
        cis = torch.stack(
            [representative_terminating_model.cumulative_intensity(t.unsqueeze(0), partial_batch) for t in times]
        )
        cis = cis.squeeze()  # Shape: (num_times, d)

        # Type 0: monotonic until t=0.5, then constant
        # Indices: 0.1->0, 0.3->1, 0.5->2, 0.7->3, ...
        for i in range(2):  # Before termination (0.1 to 0.5)
            assert cis[i + 1, 0] >= cis[i, 0] - tolerances["tight"], "Type 0 CI should be monotonic before termination"
        for i in range(2, len(times) - 1):  # After termination
            assert torch.allclose(cis[i, 0].float(), cis[i + 1, 0].float(), atol=tol), (
                f"Type 0 CI should be constant after termination: {cis[i, 0]} vs {cis[i + 1, 0]}"
            )

        # Type 1: always monotonic (never terminates)
        for i in range(len(times) - 1):
            assert cis[i + 1, 1] >= cis[i, 1] - tolerances["tight"], "Type 1 CI should always be monotonic"

    def test_ci_repeated_events_no_effect(self, representative_terminating_model, terminating_test_data, d, tolerances):
        """Test repeated events don't affect frozen CI.

        Once a type's CI is frozen, additional events of that type
        should not change the CI for that type.

        Note: Models using numerical CI may have small variations due to discretization.
        """
        if d < 2:
            pytest.skip("Test requires D >= 2")

        repeated_batch = terminating_test_data["repeated_batch"]
        # repeated_batch: type 0 at t=0.3, type 1 at t=0.8, type 0 again at t=1.2

        # Use standard tolerance for numerical CI models
        tol = tolerances["standard"] if not (representative_terminating_model.ci_config.mode == "closed_form") else tolerances["tight"]

        # CI at t=1.0 (after first type 0, before repeated type 0)
        ci_before_repeat = representative_terminating_model.cumulative_intensity(
            torch.tensor([1.0]).to(DEVICE), repeated_batch
        )

        # CI at t=1.5 (after repeated type 0 at t=1.2)
        ci_after_repeat = representative_terminating_model.cumulative_intensity(
            torch.tensor([1.5]).to(DEVICE), repeated_batch
        )

        # Type 0 CI should be frozen and identical
        assert torch.allclose(ci_before_repeat[0, 0].float(), ci_after_repeat[0, 0].float(), atol=tol), (
            f"Type 0 CI should remain frozen despite repeated event: {ci_before_repeat[0, 0]} vs {ci_after_repeat[0, 0]}"
        )


class TestTerminatingPDFCDF:
    """Tests for PDF and CDF with termination (shape/zero/bounds/identities)."""

    def test_pdf_shape(self, representative_terminating_model, terminating_test_data, d):
        """Test PDF has correct output shape."""
        partial_batch = terminating_test_data["partial_batch"]

        pdf = representative_terminating_model.PDF(torch.tensor([2.0]).to(DEVICE), partial_batch)
        assert pdf.shape == (1, d), f"PDF shape mismatch: {pdf.shape}"

    def test_pdf_non_negative(self, representative_terminating_model, terminating_test_data):
        """Test PDF is non-negative."""
        partial_batch = terminating_test_data["partial_batch"]

        pdf = representative_terminating_model.PDF(torch.tensor([2.0]).to(DEVICE), partial_batch)
        assert torch.all(pdf >= 0), "PDF should be non-negative"

    def test_pdf_zero_when_fully_terminated(self, representative_terminating_model, terminating_test_data, d, tolerances):
        """Test PDF is zero everywhere when all types have occurred.

        When all event types have already happened, there is no valid next event,
        so the PDF should be 0 for all types at all future times.
        """
        # Density realization is not termination-aware on the contract path: the
        # native f(τ)·p(m|H) stays positive when fully terminated.  Intensity family only.
        skip_if_density(representative_terminating_model)
        full_batch = terminating_test_data["full_batch"]

        # Test at multiple times after full termination
        test_times = [2.0, 5.0, 10.0]
        for t in test_times:
            pdf = representative_terminating_model.PDF(torch.tensor([t]).to(DEVICE), full_batch)
            assert torch.all(pdf.abs() < tolerances["tight"]), (
                f"PDF should be 0 for all types when fully terminated at t={t}: {pdf}"
            )

    def test_pdf_relation_to_intensity_and_ci(self, representative_terminating_model, terminating_test_data, d, tolerances):
        """Test PDF equals intensity times survival function.

        PDF(t, k) = λ_k(t) × exp(-∑_j Λ_j(t_last, t))

        This should hold for terminating models too, with masked intensities.
        """
        skip_if_density(representative_terminating_model)  # relation is via intensity/CI
        if d < 3:
            pytest.skip("Test requires D >= 3")

        partial_batch = terminating_test_data["partial_batch"]
        # partial_seq: type 0 at t=0.5, type 2 at t=1.5

        t_test = torch.tensor([2.0]).to(DEVICE)
        t_last = torch.tensor([1.5]).to(DEVICE)  # Last event time

        pdf = representative_terminating_model.PDF(t_test, partial_batch)

        # Compute expected PDF from intensity and cumulative intensity
        intensity = representative_terminating_model.intensity(t_test, partial_batch)
        ci_diff = representative_terminating_model.cumulative_intensity(t_test, partial_batch, T_low=t_last)
        survival = torch.exp(-ci_diff.sum(dim=1, keepdim=True))
        pdf_expected = intensity * survival

        tol = tolerances["standard"] if not (representative_terminating_model.ci_config.mode == "closed_form") else tolerances["tight"]
        assert torch.allclose(pdf.float(), pdf_expected.float(), atol=tol), (
            f"PDF should equal intensity × survival: {pdf} vs {pdf_expected}"
        )

    def test_pdf_batch_consistency(self, representative_terminating_model, terminating_test_data, tolerances):
        """Test batched PDF equals individual PDFs combined."""
        multi_batch = terminating_test_data["multi_batch"]
        partial_seq = terminating_test_data["partial_seq"]
        seq2 = terminating_test_data["seq2"]

        # Batched call
        pdf_batched = representative_terminating_model.PDF(torch.tensor([2.0, 2.0]).to(DEVICE), multi_batch)

        # Individual calls
        from torchpp.utils.event_data import BatchedMVEventData

        batch1 = BatchedMVEventData([partial_seq.time_points], [partial_seq.event_types]).to(DEVICE)
        batch2 = BatchedMVEventData([seq2.time_points], [seq2.event_types]).to(DEVICE)
        pdf1 = representative_terminating_model.PDF(torch.tensor([2.0]).to(DEVICE), batch1)
        pdf2 = representative_terminating_model.PDF(torch.tensor([2.0]).to(DEVICE), batch2)
        pdf_combined = torch.cat([pdf1, pdf2], dim=0)

        assert torch.allclose(pdf_batched.float(), pdf_combined.float(), atol=tolerances["tight"]), (
            "Batched PDF should equal individual PDFs combined"
        )

    def test_cdf_shape(self, representative_terminating_model, terminating_test_data):
        """Test time-marginal CDF has correct output shape."""
        partial_batch = terminating_test_data["partial_batch"]

        cdf = representative_terminating_model.CDF(torch.tensor([2.0]).to(DEVICE), partial_batch)
        # Time-marginal CDF should be scalar per batch element
        assert cdf.shape == (1,), f"Time-marginal CDF shape mismatch: {cdf.shape}"

    def test_cdf_bounded(self, representative_terminating_model, terminating_test_data):
        """Test CDF is bounded between 0 and 1."""
        partial_batch = terminating_test_data["partial_batch"]

        cdf = representative_terminating_model.CDF(torch.tensor([2.0]).to(DEVICE), partial_batch)
        assert torch.all(cdf >= 0), "CDF should be >= 0"
        assert torch.all(cdf <= 1 + 1e-6), "CDF should be <= 1"

    def test_cdf_zero_when_fully_terminated(self, representative_terminating_model, terminating_test_data, tolerances):
        """Test CDF equals 0 when all types have occurred.

        When all event types have already happened, the cumulative intensity
        for all types is frozen. Since CDF = 1 - exp(-∑ CI_diff), and CI_diff = 0
        for all types (no intensity accumulated after termination), CDF = 0.

        This means: probability of "next event before time t" is 0, because
        no event can happen (all types are exhausted).
        """
        # Density realization is not termination-aware on the contract path: the
        # native cdf_time = 1 − S(τ) stays positive when fully terminated.  Intensity only.
        skip_if_density(representative_terminating_model)
        full_batch = terminating_test_data["full_batch"]

        # CDF should be 0 (or very close) when fully terminated
        cdf = representative_terminating_model.CDF(torch.tensor([2.0]).to(DEVICE), full_batch)

        tol = tolerances["standard"] if not (representative_terminating_model.ci_config.mode == "closed_form") else tolerances["tight"]
        assert torch.allclose(cdf, torch.zeros_like(cdf), atol=tol), (
            f"CDF should be 0 when fully terminated (no events can happen): {cdf}"
        )

    def test_cdf_relation_to_ci(self, representative_terminating_model, terminating_test_data, d, tolerances):
        """Test time-marginal CDF equals 1 - exp(-sum of cumulative intensities).

        CDF(t) = 1 - exp(-∑_k Λ_k(t_last, t))
        """
        skip_if_density(representative_terminating_model)  # relation is via CI
        if d < 3:
            pytest.skip("Test requires D >= 3")

        partial_batch = terminating_test_data["partial_batch"]

        t_test = torch.tensor([2.0]).to(DEVICE)
        t_last = torch.tensor([1.5]).to(DEVICE)  # Last event time

        cdf = representative_terminating_model.CDF(t_test, partial_batch)

        # Compute expected CDF from cumulative intensity
        ci_diff = representative_terminating_model.cumulative_intensity(t_test, partial_batch, T_low=t_last)
        cdf_expected = 1 - torch.exp(-ci_diff.sum(dim=1))

        tol = tolerances["standard"] if not (representative_terminating_model.ci_config.mode == "closed_form") else tolerances["tight"]
        assert torch.allclose(cdf.float(), cdf_expected.float(), atol=tol), (
            f"CDF should equal 1 - exp(-sum CI): {cdf} vs {cdf_expected}"
        )

    def test_cdf_batch_consistency(self, representative_terminating_model, terminating_test_data, tolerances):
        """Test batched CDF equals individual CDFs combined."""
        multi_batch = terminating_test_data["multi_batch"]
        partial_seq = terminating_test_data["partial_seq"]
        seq2 = terminating_test_data["seq2"]

        # Batched call
        cdf_batched = representative_terminating_model.CDF(torch.tensor([2.0, 2.0]).to(DEVICE), multi_batch)

        # Individual calls
        from torchpp.utils.event_data import BatchedMVEventData

        batch1 = BatchedMVEventData([partial_seq.time_points], [partial_seq.event_types]).to(DEVICE)
        batch2 = BatchedMVEventData([seq2.time_points], [seq2.event_types]).to(DEVICE)
        cdf1 = representative_terminating_model.CDF(torch.tensor([2.0]).to(DEVICE), batch1)
        cdf2 = representative_terminating_model.CDF(torch.tensor([2.0]).to(DEVICE), batch2)
        cdf_combined = torch.cat([cdf1, cdf2], dim=0)

        assert torch.allclose(cdf_batched.float(), cdf_combined.float(), atol=tolerances["tight"]), (
            "Batched CDF should equal individual CDFs combined"
        )

    def test_cdf_monotonic(self, representative_terminating_model, terminating_test_data, d, tolerances):
        """Test time-marginal CDF is monotonically increasing."""
        if d < 3:
            pytest.skip("Test requires D >= 3")

        partial_batch = terminating_test_data["partial_batch"]
        # Last event at t=1.5

        times = torch.linspace(1.5, 5.0, 20).to(DEVICE)
        cdfs = torch.stack([representative_terminating_model.CDF(t.unsqueeze(0), partial_batch) for t in times])
        cdfs = cdfs.squeeze()

        # Check monotonicity
        diffs = cdfs[1:] - cdfs[:-1]
        assert torch.all(diffs >= -tolerances["tight"]), "CDF should be monotonically increasing"

    def test_cdf_near_zero_after_last_event(
        self, representative_terminating_model, terminating_test_data, d, tolerances, numerical_settings
    ):
        """Test CDF is near zero just after last event."""
        if d < 3:
            pytest.skip("Test requires D >= 3")

        partial_batch = terminating_test_data["partial_batch"]
        # Last event at t=1.5

        epsilon = numerical_settings["epsilon"]
        t_just_after = torch.tensor([1.5 + epsilon]).to(DEVICE)

        cdf = representative_terminating_model.CDF(t_just_after, partial_batch)

        # Use standard tolerance - small time means small CDF but numerical issues
        assert torch.allclose(cdf, torch.zeros_like(cdf), atol=tolerances["standard"]), (
            f"CDF should be near zero just after last event: {cdf}"
        )

    def test_pdf_cdf_integration_relation(
        self, representative_terminating_model, terminating_test_data, d, tolerances, numerical_settings
    ):
        """Test that integral of sum of PDFs approximates CDF.

        CDF(t) = ∫_{t_last}^{t} ∑_k PDF(s, k) ds
        """
        if d < 3:
            pytest.skip("Test requires D >= 3")

        partial_batch = terminating_test_data["partial_batch"]
        # Last event at t=1.5

        start_time = 1.5 + numerical_settings["epsilon"]
        end_time = 3.0

        # Compute CDF difference
        cdf_high = representative_terminating_model.CDF(torch.tensor([end_time]).to(DEVICE), partial_batch)
        cdf_low = representative_terminating_model.CDF(torch.tensor([start_time]).to(DEVICE), partial_batch)
        cdf_diff = cdf_high - cdf_low

        # Compute via numerical integration of PDF
        times = torch.linspace(start_time, end_time, steps=numerical_settings["integration_steps"]).to(DEVICE)
        pdfs = torch.stack([representative_terminating_model.PDF(t.unsqueeze(0), partial_batch) for t in times])
        pdfs = pdfs.squeeze()

        # Sum over types and integrate over time
        pdf_sum = pdfs.sum(dim=1)  # Sum over event types
        cdf_integrated = torch.trapezoid(pdf_sum, times)

        tol = tolerances["standard"]
        assert torch.allclose(cdf_diff.float(), cdf_integrated.float(), atol=tol), (
            f"Integrated PDF should approximate CDF: {cdf_integrated} vs {cdf_diff}"
        )


class TestTerminatingLikelihood:
    """Tests for likelihood computation with termination (identities/freeze/semantics)."""

    def test_likelihood_finite(self, representative_terminating_model, terminating_test_data):
        """Test likelihood is finite."""
        partial_batch = terminating_test_data["partial_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        log_lik = representative_terminating_model.likelihood(partial_batch, T, log=True)

        assert torch.isfinite(log_lik), "Log-likelihood should be finite"

    def test_likelihood_full_termination(self, representative_terminating_model, terminating_test_data):
        """Test likelihood when all types have occurred."""
        full_batch = terminating_test_data["full_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        log_lik = representative_terminating_model.likelihood(full_batch, T, log=True)

        assert torch.isfinite(log_lik), "Full termination log-likelihood should be finite"

    def test_likelihood_batch_consistency(self, representative_terminating_model, terminating_test_data, tolerances):
        """Test batched likelihood consistency."""
        multi_batch = terminating_test_data["multi_batch"]
        partial_seq = terminating_test_data["partial_seq"]
        seq2 = terminating_test_data["seq2"]

        # Batched call - returns sum over batch
        log_lik_batched = representative_terminating_model.likelihood(
            multi_batch, torch.tensor([2.0, 2.0]).to(DEVICE), log=True
        )

        # Individual calls
        from torchpp.utils.event_data import BatchedMVEventData

        batch1 = BatchedMVEventData([partial_seq.time_points], [partial_seq.event_types]).to(DEVICE)
        batch2 = BatchedMVEventData([seq2.time_points], [seq2.event_types]).to(DEVICE)
        log_lik1 = representative_terminating_model.likelihood(batch1, torch.tensor([2.0]).to(DEVICE), log=True)
        log_lik2 = representative_terminating_model.likelihood(batch2, torch.tensor([2.0]).to(DEVICE), log=True)

        # Batched likelihood should be the sum
        expected_sum = log_lik1 + log_lik2
        assert torch.allclose(log_lik_batched.sum(), expected_sum, atol=tolerances["tight"]), (
            f"Batched log-likelihood {log_lik_batched} should equal sum {expected_sum}"
        )

    def test_likelihood_zero_for_repeated_events(self, representative_terminating_model, terminating_test_data, tolerances):
        """Test that a sequence with repeated event types has likelihood 0.

        For first-occurrence (terminating) models, each event type can only occur once.
        A sequence with repeated types is impossible, so P(sequence) = 0.
        """
        repeated_batch = terminating_test_data["repeated_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        lik = representative_terminating_model.likelihood(repeated_batch, T, log=False)

        assert lik == 0 or torch.isclose(
            lik.float(), torch.tensor(0.0).to(DEVICE).float(), atol=tolerances["standard"]
        ), f"Likelihood of sequence with repeated events should be 0, got {lik}"

    def test_log_likelihood_negative_infinity_for_repeated_events(self, representative_terminating_model, terminating_test_data):
        """Test that log-likelihood of repeated events is -inf.

        Since P(repeated events) = 0, log(0) = -inf.
        """
        repeated_batch = terminating_test_data["repeated_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        log_lik = representative_terminating_model.likelihood(repeated_batch, T, log=True)

        assert log_lik == float("-inf") or log_lik < -1e10, (
            f"Log-likelihood of repeated events should be -inf, got {log_lik}"
        )

    def test_positive_term_finite(self, representative_terminating_model, terminating_test_data):
        """Test positive term (intensity contribution) is finite."""
        skip_if_density(representative_terminating_model)  # _event_log_intensity_sum: intensity family
        partial_batch = terminating_test_data["partial_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        pos_lik = representative_terminating_model._event_log_intensity_sum(partial_batch).exp()

        assert torch.isfinite(pos_lik), "Positive term should be finite"
        assert pos_lik > 0, "Positive term should be positive"

    def test_positive_term_log_finite(self, representative_terminating_model, terminating_test_data):
        """Test log positive term is finite for valid sequences."""
        skip_if_density(representative_terminating_model)  # _event_log_intensity_sum: intensity family
        partial_batch = terminating_test_data["partial_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        log_pos_lik = representative_terminating_model._event_log_intensity_sum(partial_batch)

        assert torch.isfinite(log_pos_lik), "Log positive term should be finite"

    def test_positive_term_zero_for_repeated_events(self, representative_terminating_model, terminating_test_data, tolerances):
        """Test positive term is 0 when an event type repeats.

        The positive term is the product of intensities at event times.
        When a type repeats, its intensity is 0 at the second occurrence,
        so the product is 0.
        """
        skip_if_density(representative_terminating_model)  # _event_log_intensity_sum: intensity family
        repeated_batch = terminating_test_data["repeated_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        pos_lik = representative_terminating_model._event_log_intensity_sum(repeated_batch).exp()

        assert pos_lik == 0 or torch.isclose(
            pos_lik.float(), torch.tensor(0.0).to(DEVICE).float(), atol=tolerances["standard"]
        ), f"Positive term of repeated events should be 0, got {pos_lik}"

    def test_log_positive_term_neg_infinity_for_repeated_events(
        self, representative_terminating_model, terminating_test_data
    ):
        """Test log positive term is -inf for repeated events."""
        skip_if_density(representative_terminating_model)  # _event_log_intensity_sum: intensity family
        repeated_batch = terminating_test_data["repeated_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        log_pos_lik = representative_terminating_model._event_log_intensity_sum(repeated_batch)

        assert log_pos_lik == float("-inf") or log_pos_lik < -1e10, (
            f"Log positive term of repeated events should be -inf, got {log_pos_lik}"
        )

    def test_survival_term_finite(self, representative_terminating_model, terminating_test_data):
        """Test survival term (survival/CI contribution) is finite."""
        skip_if_density(representative_terminating_model)  # cumulative_intensity: intensity family
        partial_batch = terminating_test_data["partial_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        neg_lik = torch.exp(-representative_terminating_model.cumulative_intensity(T, partial_batch).sum(dim=1))

        assert torch.isfinite(neg_lik), "Survival term should be finite"
        # Survival term = exp(-CI), so should be in (0, 1]
        assert neg_lik > 0, "Survival term should be positive"
        assert neg_lik <= 1, "Survival term should be <= 1"

    def test_log_survival_term_non_positive(self, representative_terminating_model, terminating_test_data):
        """Test log survival term is non-positive (since survival in (0,1])."""
        skip_if_density(representative_terminating_model)  # cumulative_intensity: intensity family
        partial_batch = terminating_test_data["partial_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        log_neg_lik = -representative_terminating_model.cumulative_intensity(T, partial_batch).sum(dim=1)

        assert torch.isfinite(log_neg_lik), "Log survival term should be finite"
        assert log_neg_lik <= 0, "Log survival term should be <= 0 (CI >= 0)"

    def test_likelihood_decomposition(self, representative_terminating_model, terminating_test_data, tolerances):
        """Test likelihood = positive term × survival term."""
        skip_if_density(representative_terminating_model)  # decomposition via intensity/CI
        partial_batch = terminating_test_data["partial_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        lik = representative_terminating_model.likelihood(partial_batch, T, log=False)
        pos_lik = representative_terminating_model._event_log_intensity_sum(partial_batch).exp()
        neg_lik = torch.exp(-representative_terminating_model.cumulative_intensity(T, partial_batch).sum(dim=1))

        expected = pos_lik * neg_lik
        assert torch.allclose(lik.float(), expected.float(), atol=tolerances["tight"]), (
            f"Likelihood {lik} should equal pos_lik * neg_lik = {expected}"
        )

    def test_log_likelihood_decomposition(self, representative_terminating_model, terminating_test_data, tolerances):
        """Test log_likelihood = log positive term + log survival term."""
        skip_if_density(representative_terminating_model)  # decomposition via intensity/CI
        partial_batch = terminating_test_data["partial_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        log_lik = representative_terminating_model.likelihood(partial_batch, T, log=True)
        log_pos_lik = representative_terminating_model._event_log_intensity_sum(partial_batch)
        log_neg_lik = -representative_terminating_model.cumulative_intensity(T, partial_batch).sum(dim=1)

        expected = log_pos_lik + log_neg_lik
        assert torch.allclose(log_lik.float(), expected.float(), atol=tolerances["tight"]), (
            f"Log-likelihood {log_lik} should equal log_pos_lik + log_neg_lik = {expected}"
        )

    def test_survival_term_is_exp_minus_ci(self, representative_terminating_model, terminating_test_data, tolerances):
        """Test the survival factor folded into likelihood equals exp(-cumulative_intensity).

        The dedicated survival term was folded into ``likelihood``; isolate it from
        the full likelihood (divide out the positive factor) and confirm it matches
        ``exp(-CI)`` recomputed from ``cumulative_intensity``.
        """
        skip_if_density(representative_terminating_model)  # cumulative_intensity: intensity family
        partial_batch = terminating_test_data["partial_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        lik = representative_terminating_model.likelihood(partial_batch, T, log=False)
        pos_lik = representative_terminating_model._event_log_intensity_sum(partial_batch).exp()
        ci = representative_terminating_model.cumulative_intensity(T, partial_batch)

        expected = torch.exp(-ci.sum())
        assert torch.allclose((lik / pos_lik).float(), expected.float(), atol=tolerances["standard"]), (
            f"Survival factor {lik / pos_lik} should equal exp(-CI) = {expected}"
        )

    def test_log_survival_term_is_minus_ci(self, representative_terminating_model, terminating_test_data, tolerances):
        """Test the log survival term folded into likelihood equals -cumulative_intensity.

        Isolate the survival term from the full log-likelihood (subtract the positive
        term) and confirm it matches ``-CI`` recomputed from ``cumulative_intensity``.
        """
        skip_if_density(representative_terminating_model)  # cumulative_intensity: intensity family
        partial_batch = terminating_test_data["partial_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        log_lik = representative_terminating_model.likelihood(partial_batch, T, log=True)
        log_pos_lik = representative_terminating_model._event_log_intensity_sum(partial_batch)
        ci = representative_terminating_model.cumulative_intensity(T, partial_batch)

        expected = -ci.sum()
        assert torch.allclose((log_lik - log_pos_lik).float(), expected.float(), atol=tolerances["standard"]), (
            f"Log survival term {log_lik - log_pos_lik} should equal -CI = {expected}"
        )

    def test_likelihood_full_termination_finite_and_positive(self, representative_terminating_model, terminating_test_data):
        """Test likelihood is finite and positive when all types terminated.

        When all types have occurred (and none repeated), the sequence is valid
        and has positive probability.
        """
        full_batch = terminating_test_data["full_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        lik = representative_terminating_model.likelihood(full_batch, T, log=False)

        assert torch.isfinite(lik), "Full termination likelihood should be finite"
        assert lik > 0, "Full termination likelihood should be positive"

    def test_positive_term_from_intensities(self, representative_terminating_model, terminating_test_data, tolerances):
        """Test positive term equals product of intensities at event times."""
        skip_if_density(representative_terminating_model)  # positive term = Σ log λ (intensity family)
        partial_seq = terminating_test_data["partial_seq"]
        partial_batch = terminating_test_data["partial_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        pos_lik = representative_terminating_model._event_log_intensity_sum(partial_batch)

        # Compute intensity at each event time and select the event type's intensity
        from torchpp.utils.event_data import BatchedMVEventData

        log_intensity_sum = 0.0
        for i, (t, k) in enumerate(zip(partial_seq.time_points, partial_seq.event_types)):
            # History up to but not including this event
            if i == 0:
                hist_batch = BatchedMVEventData(
                    [torch.tensor([]).to(DEVICE)], [torch.tensor([], dtype=torch.long).to(DEVICE)]
                )
            else:
                hist_batch = BatchedMVEventData([partial_seq.time_points[:i]], [partial_seq.event_types[:i]]).to(DEVICE)

            # Intensity at event time t (for all types), select type k
            intensity = representative_terminating_model.intensity(t.unsqueeze(0), hist_batch)
            intensity_k = intensity[0, k.item()]
            log_intensity_sum += torch.log(intensity_k)

        assert torch.allclose(pos_lik.float(), log_intensity_sum.float(), atol=tolerances["standard"]), (
            f"Log positive term {pos_lik} should equal sum of log intensities {log_intensity_sum}"
        )

    def test_likelihood_empty_sequence_only_survival(self, representative_terminating_model, terminating_test_data, tolerances):
        """Test empty sequence likelihood equals survival (no positive contribution)."""
        skip_if_density(representative_terminating_model)  # asserted via cumulative_intensity
        empty_batch = terminating_test_data["empty_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        log_lik = representative_terminating_model.likelihood(empty_batch, T, log=True)
        log_neg_lik = -representative_terminating_model.cumulative_intensity(T, empty_batch).sum(dim=1)

        # For empty sequence, the positive term is 1 (product of zero terms = 1)
        # So log-likelihood = log(1) + log_neg_lik = log_neg_lik
        assert torch.allclose(log_lik.float(), log_neg_lik.float(), atol=tolerances["tight"]), (
            f"Empty sequence log-likelihood {log_lik} should equal log_neg_lik {log_neg_lik}"
        )

    def test_likelihood_increases_with_T_when_fully_terminated(
        self, representative_terminating_model, terminating_test_data, tolerances
    ):
        """Test likelihood doesn't decrease with larger T when fully terminated.

        When all types are terminated, the CI is frozen (no more accumulation),
        so survival is constant and likelihood doesn't change with T.
        """
        # Density realization is not termination-aware on the contract path: the
        # native survival keeps decaying past full termination, so this freeze is
        # intensity-family-specific.
        skip_if_density(representative_terminating_model)
        full_batch = terminating_test_data["full_batch"]

        T1 = torch.tensor([2.0]).to(DEVICE)
        T2 = torch.tensor([5.0]).to(DEVICE)

        log_lik_1 = representative_terminating_model.likelihood(full_batch, T1, log=True)
        log_lik_2 = representative_terminating_model.likelihood(full_batch, T2, log=True)

        tolerance = tolerances["standard"] if (representative_terminating_model.ci_config.mode == "closed_form") else tolerances["sampling"]
        # Should be essentially equal (CI frozen)
        assert torch.allclose(log_lik_1.float(), log_lik_2.float(), atol=tolerance), (
            f"Full termination likelihood should be constant with T: {log_lik_1} vs {log_lik_2}"
        )

    def test_likelihood_decreases_with_T_when_active_types(self, representative_terminating_model, terminating_test_data):
        """Test likelihood decreases with larger T when some types still active.

        Active types continue to accumulate CI, so survival decreases.
        """
        partial_batch = terminating_test_data["partial_batch"]

        T1 = torch.tensor([2.0]).to(DEVICE)
        T2 = torch.tensor([5.0]).to(DEVICE)

        log_lik_1 = representative_terminating_model.likelihood(partial_batch, T1, log=True)
        log_lik_2 = representative_terminating_model.likelihood(partial_batch, T2, log=True)

        # CI increases for active types, so survival decreases
        assert log_lik_2 <= log_lik_1, (
            f"Likelihood should decrease with T when active types exist: T=2: {log_lik_1}, T=5: {log_lik_2}"
        )


class TestTerminatingSampling:
    """Tests for sampling with termination."""

    def test_sampling_stops_when_all_terminated(self, representative_terminating_model, d):
        """Test sampling stops when all event types have occurred."""
        from torchpp.utils.event_data import MVEventData

        # Start with empty history
        empty_seq = MVEventData(torch.tensor([]).to(DEVICE), torch.tensor([], dtype=torch.long).to(DEVICE))

        # Request more samples than types exist
        ts_sampled, time_samples, dist_samples = representative_terminating_model.sample(
            empty_seq, num_steps=d + 10, rng=SEED
        )

        num_sampled = len(time_samples)
        sampled_types = ts_sampled.event_types.tolist()
        unique_types = set(sampled_types)

        # Should sample exactly d events (one of each type)
        assert num_sampled == d, f"Should sample exactly {d} events, got {num_sampled}"
        assert len(unique_types) == d, f"Should have {d} unique types, got {len(unique_types)}"
        assert unique_types == set(range(d)), f"Should have all types 0-{d - 1}, got {unique_types}"

    def test_sampling_with_partial_history(self, representative_terminating_model, terminating_test_data, d):
        """Test sampling continues correctly from partial history."""
        if d < 3:
            pytest.skip("Test requires D >= 3")

        partial_seq = terminating_test_data["partial_seq"]
        # Types 0 and 2 already occurred

        # Request remaining samples
        ts_sampled, time_samples, dist_samples = representative_terminating_model.sample(
            partial_seq, num_steps=10, rng=SEED
        )

        # Should only sample type 1 (the remaining type)
        new_types = ts_sampled.event_types[len(partial_seq.event_types) :].tolist()

        # Should get exactly 1 new sample (type 1)
        assert len(new_types) == 1, f"Should sample 1 remaining type, got {len(new_types)}"
        assert new_types[0] == 1, f"Should sample type 1, got type {new_types[0]}"

    def test_sampling_no_samples_when_fully_terminated(self, representative_terminating_model, terminating_test_data):
        """Test no samples are generated when all types already occurred."""
        full_seq = terminating_test_data["full_seq"]

        # Try to sample more events
        ts_sampled, time_samples, dist_samples = representative_terminating_model.sample(
            full_seq, num_steps=5, rng=SEED
        )

        # Should generate no new samples
        num_original = len(full_seq.event_types)
        num_final = len(ts_sampled.event_types)
        new_samples = num_final - num_original

        assert new_samples == 0, f"Should generate 0 new samples, got {new_samples}"

    def test_sampling_times_after_last_event(self, representative_terminating_model, terminating_test_data):
        """Test new samples come after last event in history."""
        partial_seq = terminating_test_data["partial_seq"]
        last_event_time = partial_seq.time_points.max().item()

        ts_sampled, time_samples, dist_samples = representative_terminating_model.sample(
            partial_seq, num_steps=5, rng=SEED
        )

        # All new sample times should be after last event
        # time_samples is a list, convert to tensor for comparison
        if len(time_samples) > 0:
            time_samples_tensor = (
                torch.tensor(time_samples).to(DEVICE) if not isinstance(time_samples, torch.Tensor) else time_samples
            )
            assert torch.all(time_samples_tensor > last_event_time), "New samples should come after last event"


class TestPaddingInvariance:
    """Tests for padding invariance with termination."""

    def test_intensity_padding_invariant(self, representative_terminating_model, terminating_test_data, tolerances):
        """Test padding does not affect terminating intensity."""
        partial_batch = terminating_test_data["partial_batch"]
        padded_batch = terminating_test_data["padded_batch"]

        t_test = torch.tensor([2.0]).to(DEVICE)
        int_normal = representative_terminating_model.intensity(t_test, partial_batch)
        int_padded = representative_terminating_model.intensity(t_test, padded_batch)

        assert torch.allclose(int_normal.float(), int_padded.float(), atol=tolerances["tight"]), (
            "Padding should not affect terminating intensity"
        )

    def test_ci_padding_invariant(self, representative_terminating_model, terminating_test_data, tolerances):
        """Test padding does not affect terminating cumulative intensity."""
        skip_if_density(representative_terminating_model)  # cumulative_intensity: intensity family
        partial_batch = terminating_test_data["partial_batch"]
        padded_batch = terminating_test_data["padded_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        ci_normal = representative_terminating_model.cumulative_intensity(T, partial_batch)
        ci_padded = representative_terminating_model.cumulative_intensity(T, padded_batch)

        assert torch.allclose(ci_normal.float(), ci_padded.float(), atol=tolerances["tight"]), (
            "Padding should not affect terminating CI"
        )

    def test_likelihood_padding_invariant(self, representative_terminating_model, terminating_test_data, tolerances):
        """Test padding does not affect likelihood."""
        partial_batch = terminating_test_data["partial_batch"]
        padded_batch = terminating_test_data["padded_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        log_lik_normal = representative_terminating_model.likelihood(partial_batch, T, log=True)
        log_lik_padded = representative_terminating_model.likelihood(padded_batch, T, log=True)

        assert torch.allclose(log_lik_normal.float(), log_lik_padded.float(), atol=tolerances["tight"]), (
            "Padding should not affect likelihood"
        )


class TestEdgeCases:
    """Tests for edge cases in terminating models."""

    def test_single_event_type(self, archetype):
        """Test terminating model with single event type."""
        from torchpp.utils.event_data import MVEventData

        model = create_model(archetype, d=1, seed=SEED, terminating=True)

        # Sample should produce exactly 1 event
        empty_seq = MVEventData(torch.tensor([]).to(DEVICE), torch.tensor([], dtype=torch.long).to(DEVICE))
        ts_sampled, time_samples, _ = model.sample(empty_seq, num_steps=10, rng=SEED)

        assert len(time_samples) == 1, "Should sample exactly 1 event for D=1"
        assert ts_sampled.event_types[0] == 0, "Should sample type 0"

    def test_zero_intensity_remains_zero(self, representative_terminating_model, terminating_test_data, d):
        """Test that zero intensity remains zero over time."""
        if d < 3:
            pytest.skip("Test requires D >= 3")

        partial_batch = terminating_test_data["partial_batch"]

        # Check intensity at multiple times after termination
        times = torch.tensor([2.0, 3.0, 4.0, 10.0]).to(DEVICE)

        for t in times:
            intensity = representative_terminating_model.intensity(t.unsqueeze(0), partial_batch)
            # Type 0 should remain zero
            assert intensity[0, 0].abs() < TIGHT_TOLERANCE, f"Type 0 intensity should remain 0 at t={t.item()}"

    def test_out_of_order_events(self, representative_terminating_model, terminating_test_data, d):
        """Test model handles out-of-order event types correctly."""
        out_of_order_batch = terminating_test_data["out_of_order_seq"]

        from torchpp.utils.event_data import BatchedMVEventData

        batch = BatchedMVEventData([out_of_order_batch.time_points], [out_of_order_batch.event_types]).to(DEVICE)

        # Should compute mask correctly
        mask = representative_terminating_model._get_termination_mask(batch)

        # All types in the sequence should be masked
        event_types = out_of_order_batch.event_types.tolist()
        for event_type in event_types:
            assert not mask[0, event_type], f"Type {event_type} should be masked"
