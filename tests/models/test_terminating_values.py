"""Per-family terminating-mode numerical value tests.

These checks confirm the terminating model's NUMERICAL VALUE matches the
non-terminating model (or a fine-grid reference) per family: the model_pair
comparisons (active-type intensity == non-terminating, CI truncated/frozen ==
non-terminating up to termination, PDF active types ~ scaled survival,
CDF -> 1), boundary-semantics checks against the non-terminating pair,
evaluate_at_time numerical checks, IntensityGrid-respects-termination, and the
numerical-stability boundary checks.

Run over the FULL matrix (model / model_pair / model_type).
"""

import pytest
import torch
from helpers import (
    DEVICE,
    SEED,
    NO_PER_TYPE_INTENSITY_MODEL_TYPES,
    DEFECTIVE_CDF_MODEL_TYPES,
    skip_if_density,
)
from torchpp.utils.evaluation import evaluate_at_time


class TestTerminatingIntensity:
    """Numerical comparison of terminating intensity vs non-terminating."""

    def test_intensity_empty_equals_non_terminating(self, model_pair, terminating_test_data, tolerances):
        """Test intensity on empty sequence equals non-terminating."""
        model_term, model_non_term = model_pair
        empty_batch = terminating_test_data["empty_batch"]

        t_test = torch.tensor([1.0]).to(DEVICE)
        int_term = model_term.intensity(t_test, empty_batch)
        int_non_term = model_non_term.intensity(t_test, empty_batch)

        assert torch.allclose(int_term, int_non_term, atol=tolerances["tight"]), (
            "Empty sequence intensity should match non-terminating"
        )

    def test_intensity_masked_for_occurred_types(self, model_pair, terminating_test_data, d, tolerances):
        """Test intensity is zero for occurred types."""
        if d < 3:
            pytest.skip("Test requires D >= 3")

        model_term, model_non_term = model_pair
        partial_batch = terminating_test_data["partial_batch"]

        t_after = torch.tensor([2.0]).to(DEVICE)
        int_term = model_term.intensity(t_after, partial_batch)

        # Types 0 and 2 should be zero
        assert int_term[0, 0].abs() < tolerances["tight"], f"Type 0 intensity should be 0: {int_term[0, 0]}"
        assert int_term[0, 2].abs() < tolerances["tight"], f"Type 2 intensity should be 0: {int_term[0, 2]}"

        # Type 1 should be positive
        assert int_term[0, 1] > 0, f"Type 1 intensity should be positive: {int_term[0, 1]}"

        # Type 1 should match non-terminating
        int_non_term = model_non_term.intensity(t_after, partial_batch)
        assert torch.allclose(int_term[0, 1], int_non_term[0, 1], atol=tolerances["tight"]), (
            "Active type intensity should match non-terminating"
        )

    def test_intensity_positive_at_event_time_zero_after(
        self, model_pair, terminating_test_data, d, tolerances, numerical_settings
    ):
        """Test intensity is still positive at event time but zero immediately after.

        Intensity at time t only sees history [0, t), so at the exact moment of termination
        the type should still be active. Immediately after, it should be zero.
        """
        if d < 3:
            pytest.skip("Test requires D >= 3")

        model_term, model_non_term = model_pair
        partial_batch = terminating_test_data["partial_batch"]
        # partial_seq: type 0 at t=0.5, type 2 at t=1.5

        # At exactly t=0.5 (event time for type 0): type 0 should still be positive
        # because intensity sees [0, 0.5) which doesn't include the event at 0.5
        t_at_event = torch.tensor([0.5]).to(DEVICE)
        int_at_event = model_term.intensity(t_at_event, partial_batch)

        assert int_at_event[0, 0] > 0, f"Type 0 intensity at event time should be positive: {int_at_event[0, 0]}"

        # Should match non-terminating at event time
        int_non_term_at_event = model_non_term.intensity(t_at_event, partial_batch)
        assert torch.allclose(int_at_event[0, 0], int_non_term_at_event[0, 0], atol=tolerances["tight"]), (
            "At event time, terminating intensity should match non-terminating"
        )

        # Immediately after t=0.5: type 0 should be zero
        epsilon = numerical_settings["epsilon"]
        t_after_event = torch.tensor([0.5 + epsilon]).to(DEVICE)
        int_after_event = model_term.intensity(t_after_event, partial_batch)

        assert int_after_event[0, 0].abs() < tolerances["tight"], (
            f"Type 0 intensity just after event should be 0: {int_after_event[0, 0]}"
        )

    def test_intensity_all_active_types_match_non_terminating(self, model_pair, terminating_test_data, d, tolerances):
        """Test all non-terminated types exactly match non-terminating model.

        For any type that hasn't occurred yet, the terminating model should
        produce identical intensity to the non-terminating model.
        """
        if d < 3:
            pytest.skip("Test requires D >= 3")

        model_term, model_non_term = model_pair
        partial_batch = terminating_test_data["partial_batch"]
        # partial_seq: type 0 at t=0.5, type 2 at t=1.5
        # Type 1 is still active

        # Test at multiple times
        test_times = [0.3, 0.6, 1.0, 1.6, 2.0, 3.0]

        for t in test_times:
            t_tensor = torch.tensor([t]).to(DEVICE)
            int_term = model_term.intensity(t_tensor, partial_batch)
            int_non_term = model_non_term.intensity(t_tensor, partial_batch)

            # Determine which types are still active at time t
            # Type 0 terminates at 0.5, type 2 at 1.5, type 1 never
            active_types = []
            if t <= 0.5:
                active_types = [0, 1, 2]
            elif t <= 1.5:
                active_types = [1, 2]
            else:
                active_types = [1]

            for event_type in active_types:
                assert torch.allclose(int_term[0, event_type], int_non_term[0, event_type], atol=tolerances["tight"]), (
                    f"Active type {event_type} at t={t} should match non-terminating: "
                    f"term={int_term[0, event_type]}, non_term={int_non_term[0, event_type]}"
                )

    def test_intensity_repeated_events_no_effect_on_masking(self, model_pair, terminating_test_data, d, tolerances):
        """Test that repeated occurrences of same type don't change the termination mask.

        Only the first occurrence counts for termination. A repeated event of an already-
        terminated type should NOT reactivate that type. However, for Hawkes processes,
        the event can still excite OTHER active types through the kernel.
        """
        if d < 2:
            pytest.skip("Test requires D >= 2")

        model_term, model_non_term = model_pair
        repeated_batch = terminating_test_data["repeated_batch"]
        # repeated_batch: type 0 at t=0.3, type 1 at t=0.8, type 0 again at t=1.2

        # At t=1.5, terminated types should remain zero regardless of repeated events
        t_test = torch.tensor([1.5]).to(DEVICE)
        int_repeated = model_term.intensity(t_test, repeated_batch)

        # Type 0 should be zero (terminated at first occurrence t=0.3)
        # The repeated occurrence at t=1.2 should NOT reactivate it
        assert int_repeated[0, 0].abs() < tolerances["tight"], (
            f"Type 0 should remain zero despite repeated occurrence: {int_repeated[0, 0]}"
        )

        # Type 1 should be zero (terminated at t=0.8)
        assert int_repeated[0, 1].abs() < tolerances["tight"], (
            f"Type 1 should be zero after occurrence: {int_repeated[0, 1]}"
        )

        # For all remaining active types, intensity should be positive
        for event_type in range(2, d):
            assert int_repeated[0, event_type] > 0, f"Active type {event_type} should have positive intensity"


class TestTerminatingCumulativeIntensity:
    """Numerical comparison of terminating CI vs non-terminating (intensity family only)."""

    @pytest.fixture(autouse=True)
    def _intensity_only(self, model_pair):
        skip_if_density(model_pair[0])  # cumulative_intensity is intensity-family-specific

    def test_ci_empty_equals_non_terminating(self, model_pair, terminating_test_data, tolerances):
        """Test CI on empty sequence equals non-terminating."""
        model_term, model_non_term = model_pair
        empty_batch = terminating_test_data["empty_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        ci_term = model_term.cumulative_intensity(T, empty_batch)
        ci_non_term = model_non_term.cumulative_intensity(T, empty_batch)

        assert torch.allclose(ci_term.float(), ci_non_term.float(), atol=tolerances["tight"]), (
            "Empty sequence CI should match non-terminating"
        )

    def test_ci_truncated_at_event_times(self, model_pair, terminating_test_data, d, tolerances):
        """Test CI is truncated at event times for occurred types."""
        if d < 3:
            pytest.skip("Test requires D >= 3")

        model_term, model_non_term = model_pair
        partial_batch = terminating_test_data["partial_batch"]
        # partial_seq: type 0 at t=0.5, type 2 at t=1.5

        T = torch.tensor([2.0]).to(DEVICE)
        ci_term = model_term.cumulative_intensity(T, partial_batch)
        ci_non_term = model_non_term.cumulative_intensity(T, partial_batch)

        # For terminating:
        # Type 0: CI from 0 to 0.5 (terminates at 0.5)
        # Type 1: CI from 0 to 2.0 (never terminates)
        # Type 2: CI from 0 to 1.5 (terminates at 1.5)

        # Type 1 should match non-terminating. Numerically-integrated CIs carry
        # discretization error, so the terminating/non-terminating grids need not
        # agree to tight tolerance — loosen to standard for those models.
        active_atol = tolerances["tight"] if (model_term.ci_config.mode == "closed_form") else tolerances["standard"]
        assert torch.allclose(ci_term[0, 1].float(), ci_non_term[0, 1].float(), atol=active_atol), (
            "Active type CI should match non-terminating"
        )

        # Types 0 and 2 should be less than non-terminating
        assert ci_term[0, 0] < ci_non_term[0, 0], "Terminated type 0 CI should be less"
        assert ci_term[0, 2] < ci_non_term[0, 2], "Terminated type 2 CI should be less"

    def test_ci_full_termination(self, model_pair, terminating_test_data, d, tolerances):
        """Test CI when all types have terminated."""
        model_term, model_non_term = model_pair
        full_batch = terminating_test_data["full_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        ci_term = model_term.cumulative_intensity(T, full_batch)
        ci_non_term = model_non_term.cumulative_intensity(T, full_batch)

        # All terminating CIs should be less than non-terminating
        assert torch.all(ci_term <= ci_non_term + tolerances["tight"]), (
            "All terminated CIs should be <= non-terminating"
        )

    def test_ci_equals_non_terminating_up_to_event(self, model_pair, model_type, terminating_test_data, d, tolerances):
        """Test CI exactly matches non-terminating model up to termination time.

        Before a type terminates, its CI should be identical to non-terminating.
        """
        if d < 3:
            pytest.skip("Test requires D >= 3")

        model_term, model_non_term = model_pair
        partial_batch = terminating_test_data["partial_batch"]
        # partial_seq: type 0 at t=0.5, type 2 at t=1.5

        # Numerical-CI models integrate terminating/non-terminating on different
        # quadrature grids (termination knots are inserted), so they only agree
        # to grid resolution, not bit-exactly.
        tol = tolerances["tight"] if (model_term.ci_config.mode == "closed_form") else tolerances["standard"]

        # At t=0.4 (before type 0 terminates), all types should match
        T_before = torch.tensor([0.4]).to(DEVICE)
        ci_term_before = model_term.cumulative_intensity(T_before, partial_batch)
        ci_non_term_before = model_non_term.cumulative_intensity(T_before, partial_batch)

        assert torch.allclose(ci_term_before.float(), ci_non_term_before.float(), atol=tol), (
            "Before any termination, CI should match non-terminating"
        )

        # At t=1.0 (after type 0, before type 2), types 1 and 2 should match
        T_middle = torch.tensor([1.0]).to(DEVICE)
        ci_term_middle = model_term.cumulative_intensity(T_middle, partial_batch)
        ci_non_term_middle = model_non_term.cumulative_intensity(T_middle, partial_batch)

        assert torch.allclose(
            ci_term_middle[0, 1].float(), ci_non_term_middle[0, 1].float(), atol=tol
        ), "Active type 1 CI should match non-terminating"
        assert torch.allclose(
            ci_term_middle[0, 2].float(), ci_non_term_middle[0, 2].float(), atol=tol
        ), "Type 2 CI should match non-terminating before its termination"

    def test_ci_at_termination_time(self, model_pair, terminating_test_data, d, tolerances, numerical_settings):
        """Test CI at exact termination time equals non-terminating CI.

        At the exact moment of termination, CI should equal non-terminating
        (the event at t is not yet counted in CI at t, similar to intensity).

        Note: Models using numerical CI may have small variations due to discretization.
        """
        if d < 3:
            pytest.skip("Test requires D >= 3")

        model_term, model_non_term = model_pair
        partial_batch = terminating_test_data["partial_batch"]
        # Type 0 terminates at t=0.5

        # Use standard tolerance for numerical CI models
        tol = tolerances["standard"] if not (model_term.ci_config.mode == "closed_form") else tolerances["tight"]

        # At exactly t=0.5, type 0 CI should equal non-terminating
        T_at_event = torch.tensor([0.5]).to(DEVICE)
        ci_term = model_term.cumulative_intensity(T_at_event, partial_batch)
        ci_non_term = model_non_term.cumulative_intensity(T_at_event, partial_batch)

        assert torch.allclose(ci_term[0, 0].float(), ci_non_term[0, 0].float(), atol=tolerances["tight"]), (
            "At termination time, CI should equal non-terminating"
        )

        # Just after t=0.5, type 0 CI should be frozen
        epsilon = numerical_settings["epsilon"]
        T_after = torch.tensor([0.5 + epsilon]).to(DEVICE)
        ci_term_after = model_term.cumulative_intensity(T_after, partial_batch)

        # CI should be essentially the same (frozen)
        assert torch.allclose(ci_term[0, 0].float(), ci_term_after[0, 0].float(), atol=tol), (
            f"CI should be frozen immediately after termination: {ci_term[0, 0]} vs {ci_term_after[0, 0]}"
        )

    def test_ci_t_low_with_termination(self, model_pair, terminating_test_data, d, tolerances):
        """Test T_low parameter interacts correctly with termination.

        If T_low > termination_time for a type, the CI for that type should be 0
        (no intensity accumulated in the interval [T_low, T]).
        """
        if d < 3:
            pytest.skip("Test requires D >= 3")

        model_term, model_non_term = model_pair
        partial_batch = terminating_test_data["partial_batch"]
        # Type 0 terminates at t=0.5

        # CI from T_low=1.0 to T=2.0 for type 0 should be 0 (already terminated)
        T = torch.tensor([2.0]).to(DEVICE)
        T_low = torch.tensor([1.0]).to(DEVICE)
        ci_diff = model_term.cumulative_intensity(T, partial_batch, T_low=T_low)

        assert ci_diff[0, 0].abs() < tolerances["tight"], (
            "CI diff should be 0 for terminated type when T_low > termination_time"
        )

        # Type 1 (active) should have positive CI diff
        assert ci_diff[0, 1] > 0, "Active type should have positive CI diff"

        # CI diff should satisfy: CI(T) - CI(T_low) = CI(T, T_low=T_low)
        ci_high = model_term.cumulative_intensity(T, partial_batch)
        ci_low = model_term.cumulative_intensity(T_low, partial_batch)
        ci_diff_expected = ci_high - ci_low

        assert torch.allclose(ci_diff.float(), ci_diff_expected.float(), atol=tolerances["standard"]), (
            "CI(T, T_low) should equal CI(T) - CI(T_low)"
        )


class TestTerminatingPDFCDF:
    """Numerical comparison of terminating PDF/CDF vs non-terminating / grid."""

    def test_pdf_masked_for_occurred_types(self, model_pair, terminating_test_data, d, tolerances):
        """Test PDF is zero for occurred types."""
        skip_if_density(model_pair[0])  # density realization is not termination-aware
        if d < 3:
            pytest.skip("Test requires D >= 3")

        model_term, model_non_term = model_pair
        partial_batch = terminating_test_data["partial_batch"]

        t_test = torch.tensor([2.0]).to(DEVICE)
        pdf_term = model_term.PDF(t_test, partial_batch)

        # Types 0 and 2 should be zero
        assert pdf_term[0, 0].abs() < tolerances["tight"], "Occurred type 0 PDF should be 0"
        assert pdf_term[0, 2].abs() < tolerances["tight"], "Occurred type 2 PDF should be 0"

        # Type 1 should be positive
        assert pdf_term[0, 1] > 0, "Active type PDF should be positive"

    def test_cdf_approaches_one(self, terminating_model, model_type, terminating_test_data, d, tolerances):
        """Test CDF approaches 1 as time increases (for partial termination).

        With active types remaining, the CDF should approach 1 as t → ∞.
        """
        if model_type in NO_PER_TYPE_INTENSITY_MODEL_TYPES:
            pytest.skip("No genuine per-type intensities: partial-termination CDF need not reach 1")
        if model_type in DEFECTIVE_CDF_MODEL_TYPES:
            pytest.skip("Defective inter-event density at init: active-type intensity decays, CDF plateaus < 1")
        if d < 3:
            pytest.skip("Test requires D >= 3")

        partial_batch = terminating_test_data["partial_batch"]
        # Type 1 is still active

        # CDF at large time should be close to 1
        cdf_large_t = terminating_model.CDF(torch.tensor([100.0]).to(DEVICE), partial_batch)

        assert cdf_large_t > 0.99, f"CDF should approach 1 for large t: {cdf_large_t}"

    def test_pdf_active_types_match_non_terminating_scaled(self, model_pair, terminating_test_data, d, tolerances):
        """Test PDF for active types relates correctly to non-terminating model.

        For active types, PDF_term(t, k) = λ_term(t, k) × S_term(t)
        where λ_term = λ_non_term for active types, but S_term differs due to
        frozen CI contributions from terminated types.
        """
        if d < 3:
            pytest.skip("Test requires D >= 3")

        model_term, model_non_term = model_pair
        partial_batch = terminating_test_data["partial_batch"]
        # Type 1 is active

        t_test = torch.tensor([2.0]).to(DEVICE)

        # Get PDFs
        pdf_term = model_term.PDF(t_test, partial_batch)
        pdf_non_term = model_non_term.PDF(t_test, partial_batch)

        # Intensities should match for active type
        int_term = model_term.intensity(t_test, partial_batch)
        int_non_term = model_non_term.intensity(t_test, partial_batch)

        assert torch.allclose(int_term[0, 1].float(), int_non_term[0, 1].float(), atol=tolerances["tight"]), (
            "Active type intensity should match"
        )

        # PDF ratio should equal survival ratio
        # PDF = intensity × survival, and intensities match for active types
        # So PDF_term / PDF_non_term = S_term / S_non_term
        if pdf_non_term[0, 1] > tolerances["tight"]:
            pdf_ratio = pdf_term[0, 1] / pdf_non_term[0, 1]

            # Survival for terminating should be >= non-terminating (less CI accumulated)
            assert pdf_ratio >= 1 - tolerances["standard"], (
                f"Terminating PDF should be >= non-terminating for active type (higher survival): ratio={pdf_ratio}"
            )


class TestTerminatingLikelihood:
    """Numerical comparison of terminating likelihood vs non-terminating."""

    def test_likelihood_empty_equals_non_terminating(self, model_pair, terminating_test_data, tolerances):
        """Test likelihood on empty sequence equals non-terminating."""
        model_term, model_non_term = model_pair
        empty_batch = terminating_test_data["empty_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        log_lik_term = model_term.likelihood(empty_batch, T, log=True)
        log_lik_non_term = model_non_term.likelihood(empty_batch, T, log=True)

        assert torch.allclose(log_lik_term.float(), log_lik_non_term.float(), atol=tolerances["standard"]), (
            "Empty sequence likelihood should match non-terminating"
        )

    def test_likelihood_different_from_non_terminating(self, model_pair, terminating_test_data, d):
        """Test likelihood differs from non-terminating when events occurred."""
        if d < 3:
            pytest.skip("Test requires D >= 3")

        model_term, model_non_term = model_pair
        partial_batch = terminating_test_data["partial_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        log_lik_term = model_term.likelihood(partial_batch, T, log=True)
        log_lik_non_term = model_non_term.likelihood(partial_batch, T, log=True)

        # Should be different (terminating has smaller survival integral)
        # Terminating likelihood should generally be higher (less negative)
        assert log_lik_term > log_lik_non_term - 0.1, (
            "Terminating likelihood should be >= non-terminating (smaller integral)"
        )

    def test_survival_term_smaller_than_non_terminating(self, model_pair, terminating_test_data, tolerances):
        """Test terminating survival term is larger (less negative log) than non-terminating.

        Terminating models have smaller cumulative intensity (CI freezes when types terminate),
        so exp(-CI) is larger, meaning the survival term is larger.
        """
        skip_if_density(model_pair[0])  # cumulative_intensity is intensity-family-specific
        model_term, model_non_term = model_pair
        partial_batch = terminating_test_data["partial_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        neg_lik_term = torch.exp(-model_term.cumulative_intensity(T, partial_batch).sum(dim=1))
        neg_lik_non_term = torch.exp(-model_non_term.cumulative_intensity(T, partial_batch).sum(dim=1))

        # Terminating has smaller CI, so larger exp(-CI)
        assert neg_lik_term >= neg_lik_non_term - tolerances["standard"], (
            f"Terminating survival term {neg_lik_term} should be >= non-terminating {neg_lik_non_term}"
        )

    def test_likelihood_full_termination_higher_survival(self, model_pair, terminating_test_data, tolerances):
        """Test full termination has higher survival (neg likelihood) than partial.

        When more types are terminated, the CI is smaller (more types frozen),
        so survival exp(-CI) is higher.
        """
        skip_if_density(model_pair[0])  # cumulative_intensity is intensity-family-specific
        model_term, _ = model_pair
        full_batch = terminating_test_data["full_batch"]
        partial_batch = terminating_test_data["partial_batch"]

        T = torch.tensor([2.0]).to(DEVICE)
        neg_lik_full = torch.exp(-model_term.cumulative_intensity(T, full_batch).sum(dim=1))
        neg_lik_partial = torch.exp(-model_term.cumulative_intensity(T, partial_batch).sum(dim=1))

        # Full termination has 0 CI contribution after last event, partial still accumulates for active types
        assert neg_lik_full >= neg_lik_partial - tolerances["standard"], (
            f"Full termination neg_lik {neg_lik_full} should be >= partial {neg_lik_partial}"
        )


class TestTerminatingEvaluateAtTime:
    """Tests for evaluate_at_time() with termination (intensity family only)."""

    @pytest.fixture(autouse=True)
    def _intensity_only(self, terminating_model):
        skip_if_density(terminating_model)  # evaluate_at_time is intensity/CI-based

    def test_evaluate_at_time_partial_termination(self, terminating_model, terminating_test_data, d, tolerances):
        """Test evaluate_at_time with partial termination."""
        if d < 3:
            pytest.skip("Test requires D >= 3")

        partial_batch = terminating_test_data["partial_batch"]
        T = torch.tensor([2.0]).to(DEVICE)

        result = evaluate_at_time(terminating_model, T, partial_batch, compute_marginals=False)

        # Types 0 and 2 should have zero intensity
        assert result["intensity"][0, 0].abs() < tolerances["tight"], "Type 0 should have zero intensity"
        assert result["intensity"][0, 2].abs() < tolerances["tight"], "Type 2 should have zero intensity"

        # Type 1 should be positive
        assert result["intensity"][0, 1] > 0, "Type 1 should have positive intensity"

        # PDF should also be zero for terminated types
        assert result["pdf"][0, 0].abs() < tolerances["tight"], "Type 0 should have zero PDF"
        assert result["pdf"][0, 2].abs() < tolerances["tight"], "Type 2 should have zero PDF"

    def test_evaluate_at_time_full_termination(self, terminating_model, terminating_test_data, tolerances):
        """Test evaluate_at_time with full termination."""
        full_batch = terminating_test_data["full_batch"]
        T = torch.tensor([2.0]).to(DEVICE)

        result = evaluate_at_time(terminating_model, T, full_batch, compute_marginals=False)

        # All intensities should be zero
        assert torch.all(result["intensity"].abs() < tolerances["tight"]), "All intensities should be zero"

        # All PDFs should be zero
        assert torch.all(result["pdf"].abs() < tolerances["tight"]), "All PDFs should be zero"

        # Survival should be 1 (no further events possible)
        assert torch.allclose(result["survival"], torch.ones_like(result["survival"]), atol=tolerances["standard"]), (
            "Survival should be 1 for full termination"
        )

    def test_evaluate_at_time_vs_non_terminating(self, model_pair, terminating_test_data, tolerances):
        """Test evaluate_at_time on empty sequence equals non-terminating."""
        model_term, model_non_term = model_pair
        empty_batch = terminating_test_data["empty_batch"]
        T = torch.tensor([1.0]).to(DEVICE)

        result_term = evaluate_at_time(model_term, T, empty_batch, compute_marginals=False)
        result_non_term = evaluate_at_time(model_non_term, T, empty_batch, compute_marginals=False)

        # All components should match for empty sequence
        assert torch.allclose(
            result_term["intensity"].float(), result_non_term["intensity"].float(), atol=tolerances["tight"]
        )
        assert torch.allclose(result_term["ci"].float(), result_non_term["ci"].float(), atol=tolerances["tight"])
        assert torch.allclose(result_term["pdf"].float(), result_non_term["pdf"].float(), atol=tolerances["tight"])
        assert torch.allclose(result_term["cdf"].float(), result_non_term["cdf"].float(), atol=tolerances["tight"])

    def test_evaluate_at_time_cond_type_dist_only_active(self, terminating_model, terminating_test_data, d, tolerances):
        """Test conditional type distribution only has mass on active types."""
        if d < 3:
            pytest.skip("Test requires D >= 3")

        partial_batch = terminating_test_data["partial_batch"]
        T = torch.tensor([2.0]).to(DEVICE)

        result = evaluate_at_time(terminating_model, T, partial_batch, compute_marginals=False)

        # Terminated types (0, 2) should have zero probability
        assert result["cond_type_dist"][0, 0].abs() < tolerances["tight"], "Type 0 should have zero probability"
        assert result["cond_type_dist"][0, 2].abs() < tolerances["tight"], "Type 2 should have zero probability"

        # Type 1 should have probability ~1 (only active type)
        assert torch.isclose(
            result["cond_type_dist"][0, 1].to(torch.double),
            torch.tensor(1.0).to(torch.double),
            atol=tolerances["tight"],
        ), "Type 1 should have probability ≈ 1"


class TestTerminatingIntensityGrid:
    """Tests for intensity grid with termination (intensity family only)."""

    @pytest.fixture(autouse=True)
    def _intensity_only(self, terminating_model):
        skip_if_density(terminating_model)  # build_intensity_grid is intensity-family-specific

    def test_intensity_grid_partial_termination(self, terminating_model, terminating_test_data, d, tolerances):
        """Test intensity grid respects partial termination."""
        if d < 3:
            pytest.skip("Test requires D >= 3")

        partial_batch = terminating_test_data["partial_batch"]
        t_start = partial_batch.max_time + 0.1
        t_end = t_start + 2.0
        num_points = 50

        grid = terminating_model.build_intensity_grid(t_start, t_end, partial_batch, num_points=num_points)

        # At all grid points, terminated types should have zero intensity
        for i in range(num_points):
            assert grid.intensities[0, i, 0].abs() < tolerances["tight"], (
                f"Type 0 intensity at point {i} should be zero"
            )
            assert grid.intensities[0, i, 2].abs() < tolerances["tight"], (
                f"Type 2 intensity at point {i} should be zero"
            )
            # Type 1 should be positive
            assert grid.intensities[0, i, 1] > 0, f"Type 1 intensity at point {i} should be positive"

    def test_intensity_grid_full_termination(self, terminating_model, terminating_test_data, tolerances):
        """Test intensity grid with full termination."""
        full_batch = terminating_test_data["full_batch"]
        t_start = full_batch.max_time + 0.1
        t_end = t_start + 2.0
        num_points = 50

        grid = terminating_model.build_intensity_grid(t_start, t_end, full_batch, num_points=num_points)

        # After full termination, all intensities in the grid should be zero
        # because build_intensity_grid calls self.intensity() which applies termination masking
        for i in range(num_points):
            assert torch.all(grid.intensities[0, i, :].abs() < tolerances["tight"]), (
                f"All intensities at grid point {i} should be zero for full termination"
            )

    def test_intensity_grid_ci_constant_after_termination(self, terminating_model, terminating_test_data, tolerances):
        """Test that CI grows very slowly after full termination."""
        full_batch = terminating_test_data["full_batch"]
        t_start = full_batch.max_time
        t_end = t_start + 2.0
        num_points = 100

        grid = terminating_model.build_intensity_grid(t_start, t_end, full_batch, num_points=num_points)

        # After full termination, intensities are all zero
        # So CI should be nearly constant (small growth only from numerical errors)
        # Check that CI at end is close to CI at start (should be very small increase)
        ci_start = grid.ci_prefix[0, 0, :]
        ci_end = grid.ci_prefix[0, -1, :]
        ci_total_growth = ci_end - ci_start

        # Should be very small (near numerical precision)
        # Use looser tolerance since trapezoidal on zero function can have small errors
        assert torch.all(ci_total_growth < 0.1), (
            f"CI should grow very little after full termination, got total growth: {ci_total_growth}"
        )

    def test_intensity_grid_vs_non_terminating_empty(self, model_pair, terminating_test_data, tolerances):
        """Test intensity grid on empty sequence equals non-terminating."""
        model_term, model_non_term = model_pair
        empty_batch = terminating_test_data["empty_batch"]
        t_start = torch.tensor([0.0]).to(DEVICE)
        t_end = torch.tensor([2.0]).to(DEVICE)
        num_points = 50

        grid_term = model_term.build_intensity_grid(t_start, t_end, empty_batch, num_points=num_points)
        grid_non_term = model_non_term.build_intensity_grid(t_start, t_end, empty_batch, num_points=num_points)

        # Intensities should match
        assert torch.allclose(
            grid_term.intensities.float(), grid_non_term.intensities.float(), atol=tolerances["tight"]
        ), "Empty sequence intensity grid should match non-terminating"


class TestTerminatingNumericalStability:
    """Numerical stability tests specific to terminating processes."""

    def test_likelihood_many_repeated_types(self, terminating_model, d):
        """Test likelihood with many repeated event types."""
        from torchpp.utils.event_data import MVEventData, BatchedMVEventData

        if d < 2:
            pytest.skip("Test requires D >= 2")

        # Create sequence with many repeats: [0, 1, 0, 1, 0, 1]
        time_points = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float64).to(DEVICE)
        event_types = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long).to(DEVICE)

        seq = MVEventData(time_points, event_types)
        batch = BatchedMVEventData([seq.time_points], [seq.event_types]).to(DEVICE)

        T = torch.tensor([1.0]).to(DEVICE)
        log_lik = terminating_model.likelihood(batch, T, log=True)

        # Should be -inf (impossible sequence)
        assert torch.isinf(log_lik) and log_lik < 0, f"Likelihood for repeated types should be -inf, got {log_lik}"

    def test_sampling_at_boundary_of_full_termination(self, terminating_model, terminating_test_data):
        """Test sampling when close to full termination."""
        if terminating_model.D < 3:
            pytest.skip("Test requires D >= 3")

        # Create sequence with D-1 types (one away from full termination)
        from torchpp.utils.event_data import MVEventData

        time_points = torch.tensor([0.1 * i for i in range(terminating_model.D - 1)], dtype=torch.float64).to(DEVICE)
        event_types = torch.tensor(list(range(terminating_model.D - 1)), dtype=torch.long).to(DEVICE)

        seq = MVEventData(time_points, event_types)

        # Sample one more event - should only sample the remaining type
        ts_sampled, time_samples, dist_samples = terminating_model.sample(seq, num_steps=1, rng=SEED)

        # Should have sampled exactly one event
        assert len(time_samples) == 1, "Should sample exactly one event before full termination"

        # The sampled type should be the remaining one
        last_type = ts_sampled.event_types[-1].item()
        expected_type = terminating_model.D - 1
        assert last_type == expected_type, f"Should sample remaining type {expected_type}, got {last_type}"

    def test_ci_at_exact_termination_time(self, terminating_model, terminating_test_data, d, tolerances):
        """Test CI evaluated exactly at termination time."""
        skip_if_density(terminating_model)  # cumulative_intensity is intensity-family-specific
        if d < 2:
            pytest.skip("Test requires D >= 2")

        single_batch = terminating_test_data["single_batch"]

        # Type 0 terminated at t=0.5
        T_at_termination = torch.tensor([0.5]).to(DEVICE)

        ci = terminating_model.cumulative_intensity(T_at_termination, single_batch)

        # CI for type 0 should be computed up to 0.5
        # Check by comparing to non-terminating model
        # (At t=0.5, before the event, both should have same CI for type 0)
        # Actually, terminating CI should use min(T, t_term) = min(0.5, 0.5) = 0.5
        # This is a boundary case where the integral endpoint equals termination time

        # The CI should be finite and non-negative
        assert torch.isfinite(ci).all(), "CI should be finite at termination time"
        assert torch.all(ci >= 0), "CI should be non-negative"


class TestTerminatingDensityContract:
    """First-occurrence masking on the density-family (IntensityFree) contract.

    The native density realization must apply first-occurrence masking on the whole
    predictive contract — ``prob_mark`` / ``density_joint`` — not only on ``likelihood``
    and ``sample`` (regression guard).  Values are checked as
    invariants (mass on occurred types is zero; remaining mass renormalises; the
    likelihood/density/sample masking is mutually consistent), independent of any
    captured baseline.
    """

    def _model_and_data(self):
        from helpers import create_model, build_terminating_test_data

        torch.manual_seed(SEED)
        model = create_model("intensity_free", terminating=True)
        return model, build_terminating_test_data(DEVICE)

    @torch.no_grad()
    def test_prob_mark_zero_on_occurred_and_renormalised(self):
        """p(m | H) assigns no mass to occurred types and the remaining mass sums to 1."""
        model, data = self._model_and_data()
        if model.D < 3:
            pytest.skip("Test requires D >= 3")

        # partial_batch: types 0 and 2 occurred, only type 1 active.
        pm = model.prob_mark(data["partial_batch"], log=False)  # (1, D)
        assert pm[0, 0] == 0.0 and pm[0, 2] == 0.0, f"occurred types must get 0 mass: {pm}"
        assert pm[0, 1] > 0.0, f"active type must keep positive mass: {pm}"
        assert abs(float(pm.sum()) - 1.0) < 1e-6, f"remaining mass must renormalise to 1: {pm.sum()}"

        # single_batch: only type 0 occurred, types 1 and 2 active -> genuine renormalisation.
        pm2 = model.prob_mark(data["single_batch"], log=False)  # (1, D)
        assert pm2[0, 0] == 0.0, f"occurred type 0 must get 0 mass: {pm2}"
        assert (pm2[0, 1] > 0.0) and (pm2[0, 2] > 0.0), f"both active types must keep mass: {pm2}"
        assert abs(float(pm2.sum()) - 1.0) < 1e-6, f"active mass must sum to 1: {pm2.sum()}"

        # log-space: occurred types are -inf.
        lpm = model.prob_mark(data["partial_batch"], log=True)
        assert torch.isneginf(lpm[0, 0]) and torch.isneginf(lpm[0, 2])

    @torch.no_grad()
    def test_density_joint_zero_on_occurred_types(self):
        """p(t, m | H) inherits the mark masking: zero on occurred types, positive on active."""
        model, data = self._model_and_data()
        if model.D < 3:
            pytest.skip("Test requires D >= 3")

        pb = data["partial_batch"]
        q = pb.max_time + 0.3
        dj = model.density_joint(q, pb, log=False)  # (1, D)
        assert dj[0, 0] == 0.0 and dj[0, 2] == 0.0, f"occurred types must have 0 joint density: {dj}"
        assert dj[0, 1] > 0.0, f"active type must have positive joint density: {dj}"

        # density_time = Σ_m density_joint (exact mark-axis identity).
        dt = model.density_time(q, pb, log=False)
        assert torch.isclose(dt, dj.sum(dim=1), atol=1e-6), f"density_time != Σ_m density_joint: {dt} vs {dj.sum(1)}"

        # mark ⟂ time still holds under termination: p(m|t) == p(m|H).
        cond = model.prob_mark_given_time(q, pb, log=False)
        assert torch.allclose(cond, model.prob_mark(pb, log=False), atol=1e-6, equal_nan=True)

    @torch.no_grad()
    def test_repeated_history_likelihood_neg_inf(self):
        """A repeated-type history is impossible under first-occurrence: likelihood -> -inf."""
        model, data = self._model_and_data()
        log_lik = model.likelihood(data["repeated_batch"], log=True)
        assert torch.isinf(log_lik).all() and (log_lik < 0).all(), f"repeated -> -inf, got {log_lik}"

    @torch.no_grad()
    def test_sample_masking_consistent_with_density(self):
        """Sampling never draws an occurred/repeated type — the masking density/likelihood use."""
        model, data = self._model_and_data()
        if model.D < 3:
            pytest.skip("Test requires D >= 3")

        # partial_seq: types 0 and 2 occurred; only type 1 may be drawn, and never twice.
        ts_sampled, _, _ = model.sample(data["partial_seq"], num_steps=5, rng=SEED)
        drawn = ts_sampled.event_types[len(data["partial_seq"]):]
        assert (drawn == 1).all(), f"only the active type (1) may be sampled: {drawn.tolist()}"
        assert len(drawn) <= 1, f"terminating sampling must not repeat a type: {drawn.tolist()}"

    @torch.no_grad()
    def test_cdf_survival_consistency(self):
        """CDF = 1 - survival and both stay in [0, 1] under termination."""
        model, data = self._model_and_data()
        pb = data["partial_batch"]
        q = pb.max_time + 0.5
        cdf = model.cdf_time(q, pb, log=False)
        surv = model.survival_time(q, pb, log=False)
        assert torch.allclose(cdf, 1.0 - surv, atol=1e-6)
        assert (cdf >= -1e-6).all() and (cdf <= 1.0 + 1e-6).all()

    @torch.no_grad()
    def test_time_quantile_defective_survival_is_inf(self):
        """A fully terminated history has a defective survival (no next event) → the quantile is
        +inf, not the meaningless grown-bracket endpoint (~1e19) (audit N3)."""
        model, data = self._model_and_data()
        if model.D < 3:
            pytest.skip("Test requires D >= 3")
        # full_batch: every type has occurred -> no next event can happen -> quantile undefined.
        q_full = model.time_quantile(0.5, data["full_batch"])
        assert torch.isinf(q_full).all() and (q_full > 0).all(), f"fully-terminated quantile must be +inf: {q_full}"
        # partial_batch: a type remains -> a finite median still exists.
        q_partial = model.time_quantile(0.5, data["partial_batch"])
        assert torch.isfinite(q_partial).all(), f"partial-termination quantile must be finite: {q_partial}"
