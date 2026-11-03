"""Stage-2 regression: ``BatchedMVEventData`` uses ``batch_size=(B,)`` so that
``T_obs`` and ``covariates`` are tensordict entries carried automatically by
clone / slice / index / ``.to`` — where the old ``(B, L)`` layout silently
dropped them.  ``seq_lengths`` is derived from the padding mask (read-only)."""

import pytest
import torch

from torchpp.utils.event_data import BatchedMVEventData, MVEventData
from tests.helpers import DEVICE


def _make_batch(device=torch.device("cpu")) -> BatchedMVEventData:
    # Three sequences with distinct real-event counts (2, 3, 1).
    time_points = [
        torch.tensor([0.1, 0.2], dtype=torch.float64),
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64),
    ]
    event_types = [
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
    ]
    covariates = {
        "sex": [torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1.0])],
        "age": [torch.tensor([0.3]), torch.tensor([0.7]), torch.tensor([0.5])],
    }
    T_obs = [5.0, 6.0, 7.0]
    return BatchedMVEventData(time_points, event_types, covariates=covariates, T_obs=T_obs).to(device)


def test_batch_size_is_rank_one():
    b = _make_batch()
    assert tuple(b.batch_size) == (3,)
    assert tuple(b.time_points.shape) == (3, 3)  # padded to L=3
    assert b.seq_lengths == [2, 3, 1]
    assert torch.allclose(b.T_obs, torch.tensor([5.0, 6.0, 7.0], dtype=b.T_obs.dtype))
    assert set(b.covariates.keys()) == {"sex", "age"}


def test_clone_preserves_entries():
    b = _make_batch()
    c = b.clone()
    assert torch.allclose(c.T_obs, b.T_obs)
    assert c.seq_lengths == [2, 3, 1]
    assert torch.allclose(c.covariates["sex"], torch.tensor([[1.0], [0.0], [1.0]]))
    assert torch.allclose(c.covariates["age"], torch.tensor([[0.3], [0.7], [0.5]]))
    # Independent copy: mutating the clone must not touch the original.
    c["time_points"] = c["time_points"] * 2.0
    assert not torch.allclose(c.time_points, b.time_points)


def test_slice_preserves_and_slices_entries():
    b = _make_batch()
    s = b[:2]
    assert tuple(s.batch_size) == (2,)
    assert torch.allclose(s.T_obs, torch.tensor([5.0, 6.0], dtype=s.T_obs.dtype))
    assert s.seq_lengths == [2, 3]
    assert torch.allclose(s.covariates["sex"], torch.tensor([[1.0], [0.0]]))
    assert torch.allclose(s.covariates["age"], torch.tensor([[0.3], [0.7]]))


def test_index_preserves_and_indexes_entries():
    b = _make_batch()
    i = b[1]
    assert tuple(i.batch_size) == ()
    assert float(i.T_obs) == 6.0
    # 0-dim batch → real_counts is a scalar, so seq_lengths is an int.
    assert int(i.real_counts) == 3
    assert i.seq_lengths == 3
    assert torch.allclose(i.covariates["sex"], torch.tensor([0.0]))
    assert torch.allclose(i.covariates["age"], torch.tensor([0.7]))


def test_to_device_moves_all_entries():
    b = _make_batch()
    moved = b.to(DEVICE)
    assert moved.T_obs.device.type == DEVICE.type
    assert moved.time_points.device.type == DEVICE.type
    assert moved.covariates["sex"].device.type == DEVICE.type
    assert moved.seq_lengths == [2, 3, 1]
    assert torch.allclose(moved.T_obs.cpu(), torch.tensor([5.0, 6.0, 7.0], dtype=moved.T_obs.dtype))


def test_to_cuda_moves_all_entries():
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA not available")
    b = _make_batch()
    cu = b.to("cuda")
    assert cu.T_obs.is_cuda
    assert cu.covariates["sex"].is_cuda
    assert cu.time_points.is_cuda
    assert cu.seq_lengths == [2, 3, 1]


def test_rescale_time_scales_t_obs_entry():
    b = _make_batch()
    r = b.rescale_time(2.0)
    assert torch.allclose(r.T_obs, b.T_obs * 2.0)
    assert torch.allclose(r.time_points, b.time_points * 2.0)
    # covariates (static features) are unchanged, carried by clone.
    assert torch.allclose(r.covariates["sex"], b.covariates["sex"])
    assert r.seq_lengths == [2, 3, 1]
    # factor == 1.0 is the zero-copy fast path.
    assert b.rescale_time(1.0) is b


def test_round_trip_via_to_sequences():
    b = _make_batch()
    seqs = b.to_sequences()
    assert len(seqs) == 3
    assert [len(s.time_points) for s in seqs] == [2, 3, 1]
    assert torch.allclose(seqs[1].time_points, torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64))
    assert torch.equal(seqs[1].event_types, torch.tensor([0, 1, 2], dtype=torch.long))
    assert float(seqs[1].T_obs) == 6.0
    assert torch.allclose(seqs[1].covariates["age"], torch.tensor([0.7]))


def test_iter_is_unpadded_and_matches_to_sequences():
    b = _make_batch()
    via_iter = list(b)
    via_method = b.to_sequences()
    assert len(via_iter) == len(via_method) == 3
    # Each decollated sequence holds only its real events (no -1 padding types).
    for seq, n in zip(via_iter, b.seq_lengths):
        assert len(seq.time_points) == n
        assert int((seq.event_types == -1).sum()) == 0
    # __iter__ and to_sequences agree row-by-row.
    for a, c in zip(via_iter, via_method):
        assert torch.allclose(a.time_points, c.time_points)
        assert torch.equal(a.event_types, c.event_types)
        assert float(a.T_obs) == float(c.T_obs)
        for k in a.covariates:
            assert torch.allclose(a.covariates[k], c.covariates[k])


def test_from_sequences_reconstructs_batch():
    b = _make_batch()
    rt = BatchedMVEventData.from_sequences(b.to_sequences())
    assert tuple(rt.batch_size) == tuple(b.batch_size)
    assert rt.seq_lengths == b.seq_lengths
    assert torch.allclose(rt.time_points, b.time_points)
    assert torch.equal(rt.event_types, b.event_types)
    assert torch.allclose(rt.T_obs, b.T_obs)
    assert set(rt.covariates.keys()) == set(b.covariates.keys())
    for k in b.covariates.keys():
        assert torch.allclose(rt.covariates[k], b.covariates[k])


def test_from_sequences_raises_on_covariate_key_mismatch():
    t = torch.tensor([0.1], dtype=torch.float64)
    e = torch.tensor([0], dtype=torch.long)
    a = MVEventData(t.clone(), e.clone(), covariates={"sex": torch.tensor([1.0])})
    # Extra key on the second sequence must raise (silent-drop regression).
    b = MVEventData(t.clone(), e.clone(), covariates={"sex": torch.tensor([0.0]), "age": torch.tensor([0.5])})
    with pytest.raises(ValueError, match="same covariate keys"):
        BatchedMVEventData.from_sequences([a, b])
    # Missing covariates entirely on one sequence also raises.
    c = MVEventData(t.clone(), e.clone())
    with pytest.raises(ValueError, match="same covariate keys"):
        BatchedMVEventData.from_sequences([a, c])


def test_mv_event_to_batch_round_trip():
    seq = _make_batch().to_sequences()[1]  # length-3 sequence with covariates + T_obs
    batch = seq.to_batch()
    assert tuple(batch.batch_size) == (1,)
    back = batch.to_sequences()
    assert len(back) == 1
    got = back[0]
    assert torch.allclose(got.time_points, seq.time_points)
    assert torch.equal(got.event_types, seq.event_types)
    assert float(got.T_obs) == float(seq.T_obs)
    for k in seq.covariates:
        assert torch.allclose(got.covariates[k], seq.covariates[k])
