"""Contract: simultaneous (tied-time) events for every model family.

Real-world data (e.g. health records) often records multiple events at the same
calendar date.  While a continuous-time TPP assigns probability zero to exact
ties, the model must still:

  1. Not produce NaN / Inf / non-positive values.
  2. Be padding-invariant when ties are padded.
  3. Respect the causal convention: intensity at time t conditions on events
     **strictly before** t — so all tied events at t are excluded when
     computing intensity AT t.
  4. Give a finite, non-negative likelihood.

These are relation/shape/API contracts that take the same code path for every
model, so they fan over the ARCHETYPE set rather than the full matrix.

The test data in helpers provides two simultaneous-event sequences:
  - ``simul_batch``         – one pair of tied events in the middle of a sequence
  - ``simul_leading_batch`` – three tied events at the beginning
"""

from types import SimpleNamespace

import torch
from helpers import DEVICE, skip_if_density
from torchpp.utils.event_data import BatchedMVEventData, MVEventData



def _padded_row(time_points, event_types):
    """A sequence with trailing ``-1`` padding as bare tensors (padding exists only inside a batch)."""
    return SimpleNamespace(time_points=time_points, event_types=event_types)

class TestSimultaneousEventsAllModels:
    """Checks that run for every model family via the ``representative_model`` fixture."""

    def test_intensity_finite_at_simul_time(self, representative_model, test_data):
        """Intensity at the exact tie time must be finite and positive."""
        model = representative_model
        simul_batch = test_data["simul_batch"]
        # Query exactly at the simultaneous time
        lam = model.intensity(torch.tensor([1.0]).to(DEVICE), simul_batch)
        assert torch.all(torch.isfinite(lam)), f"Non-finite intensity at tie time: {lam}"
        assert torch.all(lam > 0), f"Non-positive intensity at tie time: {lam}"

    def test_intensity_after_simul_time(self, representative_model, test_data):
        """Intensity after the tie block (conditioned on both tied events) must be finite/positive."""
        model = representative_model
        simul_batch = test_data["simul_batch"]
        lam = model.intensity(torch.tensor([1.5]).to(DEVICE), simul_batch)
        assert torch.all(torch.isfinite(lam)), f"Non-finite intensity after tie: {lam}"
        assert torch.all(lam > 0), f"Non-positive intensity after tie: {lam}"

    def test_intensity_causal_at_simul_time(self, representative_model, test_data, tolerances):
        """Intensity at tie time must equal intensity conditioned only on events *before* the tie."""
        model = representative_model
        simul_batch = test_data["simul_batch"]
        # simul_seq: events at t=0.2 (type 0), 1.0 (type 1), 1.0 (type 2), 2.3 (type 0)
        # Intensity at t=1.0 should equal intensity with only the t=0.2 event in history.
        seq_before_tie = MVEventData(
            torch.tensor([0.2], dtype=float).to(DEVICE), torch.tensor([0], dtype=int).to(DEVICE)
        )
        batch_before = BatchedMVEventData([seq_before_tie.time_points], [seq_before_tie.event_types]).to(DEVICE)

        lam_full = model.intensity(torch.tensor([1.0]).to(DEVICE), simul_batch)
        lam_before = model.intensity(torch.tensor([1.0]).to(DEVICE), batch_before)

        assert torch.allclose(lam_full.float(), lam_before.float(), atol=tolerances["tight"]), (
            "Intensity at tie time must condition only on events strictly before that time. "
            f"Got {lam_full} vs {lam_before}"
        )

    def test_ci_finite_through_simul_events(self, representative_model, test_data):
        """CI computed through the tie block must be finite and non-negative."""
        model = representative_model
        skip_if_density(model)  # cumulative_intensity is intensity-family-specific
        simul_batch = test_data["simul_batch"]
        # Query well beyond the tie block
        T = torch.tensor([[3.0, 3.0, 3.0]]).to(DEVICE)
        ci = model.cumulative_intensity(torch.tensor([3.0]).to(DEVICE), simul_batch)
        assert torch.all(torch.isfinite(ci)), f"Non-finite CI through tie block: {ci}"
        assert torch.all(ci >= 0), f"Negative CI through tie block: {ci}"

    def test_ci_monotone_through_simul_events(self, representative_model, test_data):
        """CI must be monotonically non-decreasing through the tie block."""
        model = representative_model
        skip_if_density(model)  # cumulative_intensity is intensity-family-specific
        simul_batch = test_data["simul_batch"]
        # Evaluate CI at a sequence of times spanning the tie at t=1.0
        times = [0.5, 0.9, 1.0, 1.0001, 1.5, 2.0]
        # Looser tol for numerical CI: the window grid wiggles ~1e-4 near the tie's intensity jump.
        tol = 1e-6 if (model.ci_config.mode == "closed_form") else 2e-4
        ci_values = [model.cumulative_intensity(torch.tensor([t]).to(DEVICE), simul_batch).sum().item() for t in times]
        for i in range(len(ci_values) - 1):
            assert ci_values[i] <= ci_values[i + 1] + tol, (
                f"CI not monotone at positions {i}/{i + 1}: "
                f"CI({times[i]})={ci_values[i]:.6f} > CI({times[i + 1]})={ci_values[i + 1]:.6f}"
            )

    def test_likelihood_finite_with_simul_events(self, representative_model, test_data):
        """Likelihood on a sequence containing simultaneous events must be finite."""
        model = representative_model
        simul_batch = test_data["simul_batch"]
        T = simul_batch.last_event_time + 1.0  # integration horizon beyond last event
        ll = model.likelihood(simul_batch, T)
        assert torch.all(torch.isfinite(ll)), f"Non-finite likelihood with tie: {ll}"

    def test_likelihood_finite_leading_simul_events(self, representative_model, test_data):
        """Likelihood on a sequence where the first events are simultaneous must be finite."""
        model = representative_model
        simul_leading_batch = test_data["simul_leading_batch"]
        T = simul_leading_batch.last_event_time + 1.0
        ll = model.likelihood(simul_leading_batch, T)
        assert torch.all(torch.isfinite(ll)), f"Non-finite likelihood with leading tie: {ll}"

    def test_intensity_padding_invariant_with_simul_events(self, representative_model, test_data, tolerances):
        """Padding a simul-event sequence with -1 events must not change intensity."""
        model = representative_model
        simul_seq = test_data["simul_seq"]
        # Add two padding events at the end
        padded = _padded_row(
            torch.cat([simul_seq.time_points, torch.tensor([3.0, 3.5]).to(DEVICE)]),
            torch.cat([simul_seq.event_types, torch.tensor([-1, -1], dtype=int).to(DEVICE)]),
        )
        batch_orig = BatchedMVEventData([simul_seq.time_points], [simul_seq.event_types]).to(DEVICE)
        batch_padded = BatchedMVEventData([padded.time_points], [padded.event_types]).to(DEVICE)

        lam_orig = model.intensity(torch.tensor([3.0]).to(DEVICE), batch_orig)
        lam_padded = model.intensity(torch.tensor([3.0]).to(DEVICE), batch_padded)

        assert torch.allclose(lam_orig.float(), lam_padded.float(), atol=tolerances["tight"]), (
            "Intensity should be padding-invariant even with simultaneous events"
        )

    def test_likelihood_padding_invariant_with_simul_events(self, representative_model, test_data, tolerances):
        """Padding must not change the likelihood of a simul-event sequence."""
        model = representative_model
        simul_seq = test_data["simul_seq"]
        padded = _padded_row(
            torch.cat([simul_seq.time_points, torch.tensor([3.0, 3.5]).to(DEVICE)]),
            torch.cat([simul_seq.event_types, torch.tensor([-1, -1], dtype=int).to(DEVICE)]),
        )
        batch_orig = BatchedMVEventData([simul_seq.time_points], [simul_seq.event_types]).to(DEVICE)
        batch_padded = BatchedMVEventData([padded.time_points], [padded.event_types]).to(DEVICE)

        # Use the same fixed T for both — padding must not change the likelihood.
        T = torch.tensor([4.0]).to(DEVICE)
        ll_orig = model.likelihood(batch_orig, T)
        ll_padded = model.likelihood(batch_padded, T)

        assert torch.allclose(ll_orig.float(), ll_padded.float(), atol=tolerances["tight"]), (
            "Likelihood must be padding-invariant even with simultaneous events"
        )

    def test_batch_consistency_mixed_simul_and_normal(self, representative_model, test_data, tolerances):
        """Batching a simul-event sequence with a normal sequence must give same result as separate calls."""
        model = representative_model
        simul_batch = test_data["simul_batch"]
        batch_single = test_data["batch_single"]  # seq1, no ties

        # Mixed batch: row 0 = simul_seq, row 1 = seq1
        mixed_batch = test_data["simul_multi_batch"]

        t_query = torch.tensor([3.0, 3.0]).to(DEVICE)
        lam_mixed = model.intensity(t_query, mixed_batch)

        lam_simul = model.intensity(torch.tensor([3.0]).to(DEVICE), simul_batch)
        lam_normal = model.intensity(torch.tensor([3.0]).to(DEVICE), batch_single)
        lam_expected = torch.cat([lam_simul, lam_normal], dim=0)

        assert torch.allclose(lam_mixed.float(), lam_expected.float(), atol=tolerances["tight"]), (
            "Batched intensity (mixed simul+normal) must match individual calls"
        )
