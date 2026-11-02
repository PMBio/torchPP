"""Per-backend neural-internal checks with simultaneous (tied-time) events.

These target the internal methods of the Transformer-based models and include
the analytical-vs-numerical CI regression for the double-counting bug, so they
stay fanned over the neural backends rather than the archetype set:
  - ``_event_intensities`` / ``_sequence_intensities``
  - ``_full_params``
  - ``_interval_ci`` / ``_raw_ci``
  - ``_get_params_at_time``

The test data in helpers provides two simultaneous-event sequences:
  - ``simul_batch``         – one pair of tied events in the middle of a sequence
  - ``simul_leading_batch`` – three tied events at the beginning
"""

import pytest
import torch
from helpers import DEVICE, create_model
from torchpp.utils.event_data import BatchedMVEventData, MVEventData


NEURAL_MODEL_TYPES = ["neural_poisson", "neural_log_linear", "neural_spline"]

D = 3
SEED = 42


@pytest.fixture(params=NEURAL_MODEL_TYPES)
def neural_model_type(request):
    return request.param


@pytest.fixture
def neural_model(neural_model_type):
    return create_model(neural_model_type, d=D, seed=SEED, terminating=False)


class TestNeuralSimultaneousEvents:
    """Tests targeting neural-model-specific methods."""

    # -- _sequence_intensities ------------------------------------------

    def test_sequence_log_intensities_shape(self, neural_model, test_data, d):
        """_event_intensities must return (B, L) for simul-event input."""
        simul_batch = test_data["simul_batch"]
        log_lam = neural_model._event_intensities(simul_batch)
        B, L = simul_batch.time_points.shape
        assert log_lam.shape == (B, L), f"_sequence_intensities shape {log_lam.shape} != ({B}, {L})"

    def test_sequence_log_intensities_finite(self, neural_model, test_data):
        """_event_intensities must be finite for simul-event sequences."""
        for key in ("simul_batch", "simul_leading_batch"):
            batch = test_data[key]
            log_lam = neural_model._event_intensities(batch)
            assert torch.all(torch.isfinite(log_lam)), (
                f"_sequence_intensities has non-finite values for {key}: {log_lam}"
            )

    def test_sequence_log_intensities_causal(self, neural_model, test_data, tolerances):
        """Simultaneous tied events are processed in sequence order (teacher-forcing).

        For simul_seq = [t=0.2 (type 0), t=1.0 (type 1), t=1.0 (type 2), t=2.3 (type 0)]:
        - Position 1 (first tied event, type=1)  uses context from pos 0 (after t=0.2).
        - Position 2 (second tied event, type=2) uses context from pos 1 (after t=1.0, type=1).
        This is correct causal teacher-forcing: each event in the sequence is predicted
        given all strictly preceding events in sequence order, even when times tie.
        """
        simul_batch = test_data["simul_batch"]
        lam = neural_model._event_intensities(simul_batch)

        # --- Check pos 1: context = after event 0 (t=0.2, type=0) ---
        seq_before_tie = MVEventData(
            torch.tensor([0.2], dtype=float).to(DEVICE), torch.tensor([0], dtype=int).to(DEVICE)
        )
        batch_before_tie = BatchedMVEventData([seq_before_tie.time_points], [seq_before_tie.event_types]).to(DEVICE)
        lam_pos1 = neural_model.intensity(torch.tensor([1.0]).to(DEVICE), batch_before_tie)  # (1, D)
        expected_type1 = lam_pos1[0, 1].item()
        lam_type1 = lam[0, 1].item()
        assert abs(lam_type1 - expected_type1) < tolerances["tight"], (
            f"λ for first tied event (pos 1) {lam_type1:.6f} != "
            f"intensity conditioned on pre-tie history {expected_type1:.6f}"
        )

        # --- Check pos 2: context = after events [t=0.2(0), t=1.0(1)] ---
        # _full_params[:, l, ...] for l >= 1 is _get_params(ts)[:, l-1, ...].
        # So full[:, 3, ...] == _get_params(ts)[:, 2, ...] (context after second tied event).
        full = neural_model._full_params(simul_batch)  # (B, L+1, ...)
        params = neural_model._get_params(ts=simul_batch)  # (B, L, ...)
        # column 3 of full == params[:, 2, ...]
        assert torch.allclose(full[:, 3, ...].float(), params[:, 2, ...].float(), atol=tolerances["tight"]), (
            "full[:, 3] must equal _get_params[:, 2] (context from second tied event)"
        )

    def test_sequence_log_intensities_padding_invariant(self, neural_model, test_data, tolerances):
        """Padding simul-event sequence must not change _event_intensities."""
        simul_seq = test_data["simul_seq"]
        padded = MVEventData(
            torch.cat([simul_seq.time_points, torch.tensor([3.0, 4.0]).to(DEVICE)]),
            torch.cat([simul_seq.event_types, torch.tensor([-1, -1], dtype=int).to(DEVICE)]),
        )
        batch_orig = BatchedMVEventData([simul_seq.time_points], [simul_seq.event_types]).to(DEVICE)
        batch_padded = BatchedMVEventData([padded.time_points], [padded.event_types]).to(DEVICE)

        ll_orig = neural_model._event_intensities(batch_orig)
        ll_padded = neural_model._event_intensities(batch_padded)

        # Valid positions must match; padding positions can differ
        L = simul_seq.time_points.shape[0]
        assert torch.allclose(ll_orig.float(), ll_padded[:, :L].float(), atol=tolerances["tight"]), (
            "_sequence_intensities must be padding-invariant for simul-event sequences"
        )

    # -- _get_params_at_time ------------------------------------------------

    def test_get_params_at_time_at_simul_time(self, neural_model, test_data, tolerances):
        """_get_params_at_time at the tie time must equal params conditioned only on prior events."""
        simul_batch = test_data["simul_batch"]
        # simul_seq: [0.2 (type 0), 1.0 (type 1), 1.0 (type 2), 2.3 (type 0)]
        # At t=1.0: only the t=0.2 event qualifies (strict <)
        seq_before = MVEventData(torch.tensor([0.2], dtype=float).to(DEVICE), torch.tensor([0], dtype=int).to(DEVICE))
        batch_before = BatchedMVEventData([seq_before.time_points], [seq_before.event_types]).to(DEVICE)

        neural_model.prepare_batch(simul_batch)
        params_at_tie = neural_model._get_params_at_time(torch.tensor([1.0]).to(DEVICE), simul_batch)
        neural_model.clear_batch()

        neural_model.prepare_batch(batch_before)
        params_before = neural_model._get_params_at_time(torch.tensor([1.0]).to(DEVICE), batch_before)
        neural_model.clear_batch()

        assert torch.allclose(params_at_tie.float(), params_before.float(), atol=tolerances["tight"]), (
            "_get_params_at_time at tie time must equal params from pre-tie context"
        )

    def test_get_params_at_time_after_simul_block(self, neural_model, test_data, tolerances):
        """_get_params_at_time just after the tie must use context from the LAST tied event."""
        simul_batch = test_data["simul_batch"]
        # simul_seq: [0.2 (0), 1.0 (1), 1.0 (2), 2.3 (0)]
        # At t=1.5: all events with time < 1.5 qualify → last is the second tied event at index 2
        # Compare with a sequence that explicitly has all three pre-t=1.5 events
        full_seq = MVEventData(
            torch.tensor([0.2, 1.0, 1.0], dtype=float).to(DEVICE),
            torch.tensor([0, 1, 2], dtype=int).to(DEVICE),
        )
        batch_full = BatchedMVEventData([full_seq.time_points], [full_seq.event_types]).to(DEVICE)

        neural_model.prepare_batch(simul_batch)
        params_simul = neural_model._get_params_at_time(torch.tensor([1.5]).to(DEVICE), simul_batch)
        neural_model.clear_batch()

        neural_model.prepare_batch(batch_full)
        params_full = neural_model._get_params_at_time(torch.tensor([1.5]).to(DEVICE), batch_full)
        neural_model.clear_batch()

        assert torch.allclose(params_simul.float(), params_full.float(), atol=tolerances["tight"]), (
            "_get_params_at_time after tie must use context from last tied event"
        )

    # -- _interval_ci -------------------------------------------------------

    def test_interval_ci_zero_width(self, neural_model, test_data):
        """_interval_ci over a zero-width interval [t, t] must be exactly zero."""
        simul_batch = test_data["simul_batch"]
        neural_model.prepare_batch(simul_batch)

        # Grab prior params (column 0 of full_params) for a single batch item
        full = neural_model._cached_full_params
        params = full[:1, 0, ...]  # (1, ...) — prior row

        t_tie = torch.tensor([1.0]).to(DEVICE)
        t_lo = t_tie.unsqueeze(1).expand(1, D)  # (1, D)
        t_hi = t_lo.clone()  # (1, D) — same as t_lo → zero-width

        ci = neural_model._interval_ci_from_params(params, t_lo, t_hi)
        assert torch.all(ci == 0), f"_interval_ci over [t,t] should be zero, got {ci}"

        neural_model.clear_batch()

    def test_interval_ci_non_negative(self, neural_model, test_data):
        """_interval_ci must be non-negative for any valid interval."""
        simul_batch = test_data["simul_batch"]
        neural_model.prepare_batch(simul_batch)

        full = neural_model._cached_full_params
        B = full.shape[0]
        params = full[:, 0, ...]  # prior row, (B, ...)

        t_lo = torch.full((B, D), 0.5).to(DEVICE)
        t_hi = torch.full((B, D), 1.5).to(DEVICE)
        ci = neural_model._interval_ci_from_params(params, t_lo, t_hi)

        assert torch.all(ci >= 0), f"_interval_ci must be non-negative, got {ci}"
        neural_model.clear_batch()

    # -- _full_params -------------------------------------------------------

    def test_full_params_shape(self, neural_model, test_data):
        """_full_params must return (B, L+1, ...) — one more column than _get_params."""
        simul_batch = test_data["simul_batch"]
        params = neural_model._get_params(ts=simul_batch)  # (B, L, ...)
        full = neural_model._full_params(simul_batch)  # (B, L+1, ...)

        assert full.shape[0] == params.shape[0], "Batch dim must match"
        assert full.shape[1] == params.shape[1] + 1, "_full_params must have L+1 columns"
        assert full.shape[2:] == params.shape[2:], "Trailing dims must match"

    def test_full_params_col0_is_prior(self, neural_model, test_data, tolerances):
        """Column 0 of _full_params must equal the learned prior from the head."""
        simul_batch = test_data["simul_batch"]
        full = neural_model._full_params(simul_batch)  # (B, L+1, ...)

        prior = neural_model._get_prior()  # (...)
        prior_cast = prior.to(device=full.device, dtype=full.dtype)
        # Column 0 should equal prior expanded to (B, ...)
        B = full.shape[0]
        prior_expanded = prior_cast.unsqueeze(0).expand(B, *prior_cast.shape)
        assert torch.allclose(full[:, 0, ...].float(), prior_expanded.float(), atol=tolerances["tight"]), (
            "Column 0 of _full_params must equal the learned prior"
        )

    def test_full_params_other_cols_match_get_params(self, neural_model, test_data, tolerances):
        """Columns 1..L of _full_params must equal _get_params(ts)."""
        simul_batch = test_data["simul_batch"]
        params = neural_model._get_params(ts=simul_batch)  # (B, L, ...)
        full = neural_model._full_params(simul_batch)  # (B, L+1, ...)

        assert torch.allclose(full[:, 1:, ...].float(), params.float(), atol=tolerances["tight"]), (
            "Columns 1..L of _full_params must equal _get_params"
        )

    def test_full_params_ties_are_finite(self, neural_model, test_data):
        """All positions of _full_params for a simul-event batch must be finite."""
        simul_batch = test_data["simul_batch"]
        full = neural_model._full_params(simul_batch)
        assert torch.all(torch.isfinite(full)), "_full_params must be finite for simultaneous-event batches"

    # -- raw_ci with simultaneous events ------------------------------------

    def test_raw_ci_with_simul_events_monotone(self, neural_model, test_data):
        """_raw_ci must be monotonically non-decreasing through a tie block."""
        simul_batch = test_data["simul_batch"]
        neural_model.prepare_batch(simul_batch)

        times = [0.5, 0.99, 1.0, 1.0 + 1e-5, 1.5, 2.5]
        ci_sums = []
        for t in times:
            T = torch.full((1, D), t, dtype=torch.float64).to(DEVICE)
            ci = neural_model._raw_ci(T, simul_batch)
            ci_sums.append(ci.sum().item())

        for i in range(len(ci_sums) - 1):
            assert ci_sums[i] <= ci_sums[i + 1] + 1e-6, (
                f"_raw_ci not monotone: CI({times[i]})={ci_sums[i]:.6f} > CI({times[i + 1]})={ci_sums[i + 1]:.6f}"
            )
        neural_model.clear_batch()

    def test_raw_ci_at_simul_time_matches_before(self, neural_model, test_data, tolerances):
        """_raw_ci at the exact tie time should equal _raw_ci just before it (no mass at a point)."""
        simul_batch = test_data["simul_batch"]
        neural_model.prepare_batch(simul_batch)

        T_at = torch.full((1, D), 1.0, dtype=torch.float64).to(DEVICE)
        T_just_before = torch.full((1, D), 1.0 - 1e-7, dtype=torch.float64).to(DEVICE)

        ci_at = neural_model._raw_ci(T_at, simul_batch).sum().item()
        ci_before = neural_model._raw_ci(T_just_before, simul_batch).sum().item()

        assert abs(ci_at - ci_before) < 1e-4, (
            f"_raw_ci should be continuous at tie time: CI(1.0)={ci_at:.8f} vs CI(1.0-ε)={ci_before:.8f}"
        )
        neural_model.clear_batch()

    # -- positive term (_event_log_intensity_sum) / likelihood --------------

    def test_positive_term_finite_with_simul(self, neural_model, test_data):
        """_event_log_intensity_sum must be finite on simul-event sequences."""
        for key in ("simul_batch", "simul_leading_batch"):
            batch = test_data[key]
            ll = neural_model._event_log_intensity_sum(batch)
            assert torch.all(torch.isfinite(ll)), f"_event_log_intensity_sum has non-finite values for {key}: {ll}"

    def test_likelihood_from_components_simul(self, neural_model, test_data, tolerances):
        """log_likelihood = _event_log_intensity_sum - CI(T), tested on simul data."""
        simul_batch = test_data["simul_batch"]
        T = simul_batch.max_time + 1.0
        log_ll = neural_model.likelihood(simul_batch, T, log=True)
        log_pos_ll = neural_model._event_log_intensity_sum(simul_batch)
        ci_T = neural_model.cumulative_intensity(T, simul_batch).sum(dim=1)
        expected = log_pos_ll - ci_T

        assert torch.allclose(log_ll.float(), expected.float(), atol=tolerances["tight"]), (
            f"log_likelihood ({log_ll}) != log_pos_lik - CI(T) ({expected}) for simul-event sequence"
        )

    def test_likelihood_batch_consistency_simul(self, neural_model, test_data, tolerances):
        """Batching simul-event with a normal sequence must not change individual likelihoods."""
        simul_batch = test_data["simul_batch"]
        normal_batch = test_data["batch_single"]
        mixed_batch = test_data["simul_multi_batch"]

        # Use a common T that covers both sequences
        T_common = torch.tensor([mixed_batch.max_time.max().item() + 1.0]).to(DEVICE).expand(2)
        T_simul = simul_batch.max_time + 1.0
        T_normal = normal_batch.max_time + 1.0

        ll_mixed = neural_model.likelihood(mixed_batch, T_common)
        ll_simul = neural_model.likelihood(simul_batch, T_simul)
        ll_normal = neural_model.likelihood(normal_batch, T_normal)

        assert torch.allclose(ll_mixed[0:1].float(), ll_simul.float(), atol=tolerances["tight"]), (
            "Batched likelihood[0] (simul) must match standalone simul likelihood"
        )
        assert torch.allclose(ll_mixed[1:2].float(), ll_normal.float(), atol=tolerances["tight"]), (
            "Batched likelihood[1] (normal) must match standalone normal likelihood"
        )

    # -- analytical CI vs numerical CI (regression for double-counting) -----

    def test_analytical_ci_matches_numerical_simul(self, neural_model, test_data):
        """Analytical CI must match numerical CI on sequences with simultaneous events.

        This is the regression test for the double-counting bug: when simultaneous
        events share the same causal column, their intervals must not be integrated
        multiple times.  The numerical CI (which integrates intensity() pointwise)
        is unaffected by the interval-segmentation logic and serves as ground truth.
        """
        from torchpp.utils import numerical_integration

        for key in ("simul_batch", "simul_leading_batch"):
            batch = test_data[key]
            T = batch.max_time + 1.0

            ci_analytical = neural_model.cumulative_intensity(T, batch)  # (B, D)
            with neural_model.fix_context(batch):
                ci_numerical = numerical_integration(
                    neural_model.intensity,
                    t_start=torch.zeros_like(T),
                    t_end=T,
                    ts=batch,
                    method="trapezoidal",
                    num_points=10000,
                )  # (B, D)

            assert torch.allclose(ci_analytical.float(), ci_numerical.float(), atol=1e-3), (
                f"Analytical CI does not match numerical CI for {key}. "
                f"Analytical: {ci_analytical}, Numerical: {ci_numerical}, "
                f"Diff: {ci_analytical - ci_numerical}"
            )
