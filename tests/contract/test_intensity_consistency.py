"""Consistency tests for the three-level intensity hierarchy.

The public API provides three intensity methods, each of which may be
independently overridden by subclasses:

    _intensity_at_t(t, ts)        → (B, D)   single-time evaluation
    _sequence_intensities(ts)    → (B, L, D) all types at all event positions
    _event_intensities(ts)       → (B, L)   observed-type intensity per position

The default implementations form a chain:

    _sequence_intensities  calls  _intensity_at_t  once per L
    _event_intensities     calls  _sequence_intensities, then gathers

Subclasses can override any of the three for efficiency.  This test module
verifies that all three remain *mutually consistent*: their outputs must agree
regardless of which override path was taken.

These are universal relation/consistency contracts (identical code path for
every model), so they fan over the ARCHETYPE set (``representative_model``).
"""

import pytest
import torch
from helpers import DEVICE, skip_if_density
from torchpp.utils.event_data import BatchedMVEventData, MVEventData


# Every test here exercises the intensity hooks (_intensity_at_t / _sequence_intensities /
# _event_intensities / _event_log_intensity_sum), which a density-based model does not have
# Skip the whole module for the density archetype; its predictive contract is covered by
# test_model_contract.py.
@pytest.fixture(autouse=True)
def _require_intensity_family(representative_model):
    skip_if_density(representative_model)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(seq1, seq2=None):
    """Wrap one or two MVEventData sequences into a BatchedMVEventData."""
    if seq2 is None:
        return BatchedMVEventData([seq1.time_points], [seq1.event_types]).to(DEVICE)
    return BatchedMVEventData(
        [seq1.time_points, seq2.time_points],
        [seq1.event_types, seq2.event_types],
    ).to(DEVICE)


def _valid_mask(ts: BatchedMVEventData) -> torch.Tensor:
    """Return (B, L) bool mask of non-padding positions."""
    return ts.event_mask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seq1():
    return MVEventData(
        torch.tensor([0.2, 0.8, 2.0, 2.3], dtype=torch.float64).to(DEVICE),
        torch.tensor([0, 0, 1, 0], dtype=torch.long).to(DEVICE),
    )


@pytest.fixture
def seq2():
    return MVEventData(
        torch.tensor([0.1, 0.5, 1.2, 1.8, 2.5], dtype=torch.float64).to(DEVICE),
        torch.tensor([1, 0, 1, 0, 1], dtype=torch.long).to(DEVICE),
    )


# ---------------------------------------------------------------------------
# TestEventVsSequenceIntensities
# ---------------------------------------------------------------------------


class TestEventVsSequenceIntensities:
    """_event_intensities must agree with _sequence_intensities gathered at observed types."""

    def test_single_batch(self, representative_model, seq1, tolerances):
        """Single-sequence batch: _event_intensities == gather(_sequence_intensities, event_types)."""
        model = representative_model
        ts = _make_batch(seq1)
        lam_event = model._event_intensities(ts)  # (1, L)
        lam_all = model._sequence_intensities(ts)  # (1, L, D)

        et = ts.event_types_idx_safe.unsqueeze(-1)  # (1, L, 1)
        lam_gathered = lam_all.gather(2, et).squeeze(-1)  # (1, L)

        valid = _valid_mask(ts)  # (1, L)
        assert torch.allclose(
            lam_event[valid].float(),
            lam_gathered[valid].float(),
            atol=tolerances["tight"],
        ), (
            f"_event_intensities != gather(_sequence_intensities) at valid positions.\n"
            f"  _event_intensities: {lam_event[valid]}\n"
            f"  gathered:           {lam_gathered[valid]}"
        )

    def test_multi_batch(self, representative_model, seq1, seq2, tolerances):
        """Multi-sequence batch: consistency holds for every element."""
        model = representative_model
        ts = _make_batch(seq1, seq2)
        lam_event = model._event_intensities(ts)  # (2, L)
        lam_all = model._sequence_intensities(ts)  # (2, L, D)

        et = ts.event_types_idx_safe.unsqueeze(-1)
        lam_gathered = lam_all.gather(2, et).squeeze(-1)

        valid = _valid_mask(ts)
        assert torch.allclose(
            lam_event[valid].float(),
            lam_gathered[valid].float(),
            atol=tolerances["tight"],
        ), "_event_intensities != gather(_sequence_intensities) in multi-batch"

    def test_empty_batch(self, representative_model, tolerances):
        """Empty sequence: both return zero-sized tensors of the right shape."""
        model = representative_model
        empty = MVEventData(torch.tensor([]).to(DEVICE), torch.tensor([], dtype=torch.long).to(DEVICE))
        ts = _make_batch(empty)

        lam_event = model._event_intensities(ts)
        lam_all = model._sequence_intensities(ts)

        assert lam_event.shape == (1, 0), f"_event_intensities empty shape: {lam_event.shape}"
        assert lam_all.shape == (1, 0, model.D), f"_sequence_intensities empty shape: {lam_all.shape}"

    def test_with_padding(self, representative_model, seq1, tolerances):
        """Padding positions may differ; valid positions must still agree."""
        model = representative_model
        padded = MVEventData(
            torch.cat([seq1.time_points, torch.tensor([3.0, 4.0], dtype=torch.float64).to(DEVICE)]),
            torch.cat([seq1.event_types, torch.tensor([-1, -1], dtype=torch.long).to(DEVICE)]),
        )
        ts = _make_batch(padded)
        lam_event = model._event_intensities(ts)
        lam_all = model._sequence_intensities(ts)

        et = ts.event_types_idx_safe.unsqueeze(-1)
        lam_gathered = lam_all.gather(2, et).squeeze(-1)

        valid = _valid_mask(ts)
        assert torch.allclose(
            lam_event[valid].float(),
            lam_gathered[valid].float(),
            atol=tolerances["tight"],
        ), "_event_intensities != gather(_sequence_intensities) when sequence has padding"


# ---------------------------------------------------------------------------
# TestSequenceVsRawIntensity
# ---------------------------------------------------------------------------


class TestSequenceVsRawIntensity:
    """_sequence_intensities[:, l, :] must equal _intensity_at_t(t[:, l], ts) at each position."""

    def test_each_position_single_batch(self, representative_model, seq1, d, tolerances):
        """For every event position, the slice of _sequence_intensities equals _intensity_at_t."""
        model = representative_model
        ts = _make_batch(seq1)
        has_cache = hasattr(model, "prepare_batch")
        if has_cache:
            model.prepare_batch(ts)
        try:
            lam_seq = model._sequence_intensities(ts)  # (1, L, D)
            B, L, _ = lam_seq.shape

            for l in range(L):
                t_l = ts.time_points[:, l]  # (B,)
                lam_raw = model._intensity_at_t(t_l, ts)  # (B, D)
                assert torch.allclose(
                    lam_seq[:, l, :].float(),
                    lam_raw.float(),
                    atol=tolerances["tight"],
                ), (
                    f"_sequence_intensities[:, {l}, :] != _intensity_at_t at position {l}.\n"
                    f"  _sequence_intensities: {lam_seq[:, l, :]}\n"
                    f"  _intensity_at_t:        {lam_raw}"
                )
        finally:
            if has_cache:
                model.clear_batch()

    def test_each_position_multi_batch(self, representative_model, seq1, seq2, tolerances):
        """Consistency holds for every element in a multi-sequence batch."""
        model = representative_model
        ts = _make_batch(seq1, seq2)
        has_cache = hasattr(model, "prepare_batch")
        if has_cache:
            model.prepare_batch(ts)
        try:
            lam_seq = model._sequence_intensities(ts)  # (2, L, D)
            B, L, _ = lam_seq.shape

            for l in range(L):
                t_l = ts.time_points[:, l]
                lam_raw = model._intensity_at_t(t_l, ts)
                assert torch.allclose(
                    lam_seq[:, l, :].float(),
                    lam_raw.float(),
                    atol=tolerances["tight"],
                ), f"_sequence_intensities[:, {l}, :] != _intensity_at_t in multi-batch at position {l}"
        finally:
            if has_cache:
                model.clear_batch()


# ---------------------------------------------------------------------------
# TestEventIntensitiesVsRawIntensity
# ---------------------------------------------------------------------------


class TestEventIntensitiesVsRawIntensity:
    """_event_intensities[b, l] must equal _intensity_at_t(t[b,l], ts)[b, type[b,l]]."""

    def test_single_batch(self, representative_model, seq1, tolerances):
        """Direct comparison per (b, l) position."""
        model = representative_model
        ts = _make_batch(seq1)
        has_cache = hasattr(model, "prepare_batch")
        if has_cache:
            model.prepare_batch(ts)
        try:
            lam_event = model._event_intensities(ts)  # (1, L)
            B, L = ts.time_points.shape

            valid = _valid_mask(ts)
            for l in range(L):
                if not valid[0, l]:
                    continue
                t_l = ts.time_points[:, l]  # (B,)
                lam_raw = model._intensity_at_t(t_l, ts)  # (B, D)
                etype = ts.event_types[0, l].item()
                expected = lam_raw[0, etype]

                assert torch.isclose(
                    lam_event[0, l].float(),
                    expected.float(),
                    atol=tolerances["tight"],
                ), (
                    f"_event_intensities[0, {l}]={lam_event[0, l]:.6f} != "
                    f"_intensity_at_t[0,{etype}]={expected:.6f} at position {l}"
                )
        finally:
            if has_cache:
                model.clear_batch()

    def test_multi_batch(self, representative_model, seq1, seq2, tolerances):
        """Same check over both sequences in a batched call."""
        model = representative_model
        ts = _make_batch(seq1, seq2)
        has_cache = hasattr(model, "prepare_batch")
        if has_cache:
            model.prepare_batch(ts)
        try:
            lam_event = model._event_intensities(ts)  # (2, L)
            B, L = ts.time_points.shape
            valid = _valid_mask(ts)

            for b in range(B):
                for l in range(L):
                    if not valid[b, l]:
                        continue
                    t_l = ts.time_points[:, l]
                    lam_raw = model._intensity_at_t(t_l, ts)  # (B, D)
                    etype = ts.event_types[b, l].item()
                    expected = lam_raw[b, etype]

                    assert torch.isclose(
                        lam_event[b, l].float(),
                        expected.float(),
                        atol=tolerances["tight"],
                    ), (
                        f"_event_intensities[{b},{l}]={lam_event[b, l]:.6f} != _intensity_at_t[{b},{etype}]={expected:.6f}"
                    )
        finally:
            if has_cache:
                model.clear_batch()


# ---------------------------------------------------------------------------
# TestIntensityValues
# ---------------------------------------------------------------------------


class TestIntensityValues:
    """Sanity checks on the values returned by all three methods."""

    def test_intensity_at_t_positive(self, representative_model, seq1, d):
        """_intensity_at_t must return strictly positive values."""
        model = representative_model
        ts = _make_batch(seq1)
        t = ts.time_points[:, 0]
        lam = model._intensity_at_t(t, ts)
        assert lam.shape == (1, d)
        assert torch.all(lam > 0), f"_intensity_at_t has non-positive values: {lam}"

    def test_intensity_at_t_finite(self, representative_model, seq1, d):
        """_intensity_at_t must be finite."""
        model = representative_model
        ts = _make_batch(seq1)
        t = ts.time_points[:, -1]
        lam = model._intensity_at_t(t, ts)
        assert torch.all(torch.isfinite(lam)), f"_intensity_at_t has non-finite values: {lam}"

    def test_sequence_intensities_shape(self, representative_model, seq1, d):
        """_sequence_intensities must return (B, L, D)."""
        model = representative_model
        ts = _make_batch(seq1)
        B, L = ts.time_points.shape
        lam = model._sequence_intensities(ts)
        assert lam.shape == (B, L, d), f"_sequence_intensities shape {lam.shape} != ({B},{L},{d})"

    def test_sequence_intensities_positive_at_valid(self, representative_model, seq1):
        """_sequence_intensities must be positive at valid (non-padding) positions."""
        model = representative_model
        ts = _make_batch(seq1)
        lam = model._sequence_intensities(ts)  # (1, L, D)
        valid = _valid_mask(ts)  # (1, L)
        valid_lam = lam[valid]  # (n_valid, D)
        assert torch.all(valid_lam > 0), (
            f"_sequence_intensities has non-positive values at valid positions: {valid_lam}"
        )

    def test_sequence_intensities_finite_at_valid(self, representative_model, seq1):
        """_sequence_intensities must be finite at valid positions."""
        model = representative_model
        ts = _make_batch(seq1)
        lam = model._sequence_intensities(ts)
        valid = _valid_mask(ts)
        assert torch.all(torch.isfinite(lam[valid])), "_sequence_intensities has non-finite values at valid positions"

    def test_event_intensities_shape(self, representative_model, seq1):
        """_event_intensities must return (B, L)."""
        model = representative_model
        ts = _make_batch(seq1)
        B, L = ts.time_points.shape
        lam = model._event_intensities(ts)
        assert lam.shape == (B, L), f"_event_intensities shape {lam.shape} != ({B},{L})"

    def test_event_intensities_positive_at_valid(self, representative_model, seq1):
        """_event_intensities must be positive at valid (non-padding) positions."""
        model = representative_model
        ts = _make_batch(seq1)
        lam = model._event_intensities(ts)
        valid = _valid_mask(ts)
        assert torch.all(lam[valid] > 0), f"_event_intensities has non-positive values at valid positions: {lam[valid]}"

    def test_event_intensities_finite_at_valid(self, representative_model, seq1):
        """_event_intensities must be finite at valid positions."""
        model = representative_model
        ts = _make_batch(seq1)
        lam = model._event_intensities(ts)
        valid = _valid_mask(ts)
        assert torch.all(torch.isfinite(lam[valid])), "_event_intensities has non-finite values at valid positions"


# ---------------------------------------------------------------------------
# TestPaddingInvariance
# ---------------------------------------------------------------------------


class TestPaddingInvariance:
    """Adding padding to a sequence must not change results at valid positions."""

    def _padded(self, seq: MVEventData) -> MVEventData:
        return MVEventData(
            torch.cat([seq.time_points, torch.tensor([99.0, 100.0], dtype=seq.time_points.dtype).to(DEVICE)]),
            torch.cat([seq.event_types, torch.tensor([-1, -1], dtype=seq.event_types.dtype).to(DEVICE)]),
        )

    def test_event_intensities_padding_invariant(self, representative_model, seq1, tolerances):
        """_event_intensities at original positions must not change when padding is added."""
        model = representative_model
        ts_orig = _make_batch(seq1)
        ts_padded = _make_batch(self._padded(seq1))

        lam_orig = model._event_intensities(ts_orig)
        lam_padded = model._event_intensities(ts_padded)

        L = ts_orig.time_points.shape[1]
        assert torch.allclose(
            lam_orig.float(),
            lam_padded[:, :L].float(),
            atol=tolerances["tight"],
        ), "_event_intensities changed when padding was added"

    def test_sequence_intensities_padding_invariant(self, representative_model, seq1, tolerances):
        """_sequence_intensities at original positions must not change when padding is added."""
        model = representative_model
        ts_orig = _make_batch(seq1)
        ts_padded = _make_batch(self._padded(seq1))

        lam_orig = model._sequence_intensities(ts_orig)
        lam_padded = model._sequence_intensities(ts_padded)

        L = ts_orig.time_points.shape[1]
        valid = _valid_mask(ts_orig)  # (1, L)
        assert torch.allclose(
            lam_orig[:, :L, :][valid].float(),
            lam_padded[:, :L, :][valid].float(),
            atol=tolerances["tight"],
        ), "_sequence_intensities changed when padding was added"

    def test_intensity_at_t_padding_invariant(self, representative_model, seq1, tolerances):
        """_intensity_at_t must be unchanged when the history batch is padded."""
        model = representative_model
        ts_orig = _make_batch(seq1)
        ts_padded = _make_batch(self._padded(seq1))

        t = ts_orig.time_points[:, -1]  # evaluate at last real event time
        lam_orig = model._intensity_at_t(t, ts_orig)
        lam_padded = model._intensity_at_t(t, ts_padded)

        assert torch.allclose(
            lam_orig.float(),
            lam_padded.float(),
            atol=tolerances["tight"],
        ), "_intensity_at_t changed when the history had extra padding"


# ---------------------------------------------------------------------------
# TestPositiveLikelihoodConsistency
# ---------------------------------------------------------------------------


class TestPositiveLikelihoodConsistency:
    """_event_log_intensity_sum must equal sum(log(_event_intensities)) at valid positions."""

    def test_positive_term_matches_event_intensities(self, representative_model, seq1, tolerances):
        """_event_log_intensity_sum must equal Σ log(λ_{m_i}(t_i)) over valid events."""
        model = representative_model
        ts = _make_batch(seq1)
        pos_ll = model._event_log_intensity_sum(ts)  # (1,)

        lam = model._event_intensities(ts)  # (1, L)
        valid = _valid_mask(ts)  # (1, L)
        expected = torch.log(lam.clamp(min=1e-12)).masked_fill(~valid, 0.0).sum(dim=1)  # (1,)

        assert torch.allclose(
            pos_ll.float(),
            expected.float(),
            atol=tolerances["tight"],
        ), f"_event_log_intensity_sum={pos_ll} != Σ log(event_intensities)={expected}"

    def test_positive_term_multi_batch(self, representative_model, seq1, seq2, tolerances):
        """Same check for a multi-sequence batch."""
        model = representative_model
        ts = _make_batch(seq1, seq2)
        pos_ll = model._event_log_intensity_sum(ts)  # (2,)

        lam = model._event_intensities(ts)
        valid = _valid_mask(ts)
        expected = torch.log(lam.clamp(min=1e-12)).masked_fill(~valid, 0.0).sum(dim=1)

        assert torch.allclose(
            pos_ll.float(),
            expected.float(),
            atol=tolerances["tight"],
        ), "_event_log_intensity_sum != Σ log(event_intensities) in multi-batch"
