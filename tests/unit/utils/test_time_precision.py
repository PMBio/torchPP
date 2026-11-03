"""Regression tests for float64 time precision on the save/load + batching path.

Data preparation can make event times strictly increasing by adding a tiny forward
jitter (e.g. ``1e-6`` days) computed in float64.  For multi-day timestamps the
float32 ULP exceeds that jitter, so casting the stored times to float32 rounds it
away — simultaneous events (Δt = 0) reappear and every history-dependent model's
likelihood becomes order-sensitive.  Such data must be stored as float64.

These tests pin the contract that the jitter survives the save/load +
:class:`BatchedMVEventData` path in float64, and demonstrate (as a meaningfulness
check) that the same data in float32 collapses.
"""

import os
import tempfile

import torch

from torchpp.utils.event_data import MVEventData, BatchedMVEventData


# Multi-day near-tie timestamps in the regime where float32 collapse is exact:
# each pair is separated by the data-prep jitter of 1e-6 days.  At t >= 32 the
# float32 spacing exceeds 1e-6, so the second event of a pair rounds onto the
# first (Δt -> 0); float64 keeps the full 1e-6 gap.
_NEAR_TIE_TIMES = [2.0, 2.0 + 1e-6, 50.0, 50.0 + 1e-6, 200.0, 200.0 + 1e-6]
_EVENT_TYPES = [0, 1, 2, 0, 1, 2]
_T_OBS = 210.0


def _roundtrip_gaps(dtype: torch.dtype) -> torch.Tensor:
    """Δt the model sees after save/load + batching, for times stored as ``dtype``."""
    seq = MVEventData(
        time_points=torch.tensor(_NEAR_TIE_TIMES, dtype=dtype),
        event_types=torch.tensor(_EVENT_TYPES, dtype=torch.long),
        T_obs=_T_OBS,
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "seq.pt")
        torch.save([seq], path)
        loaded = torch.load(path, weights_only=False)[0]
    batch = BatchedMVEventData(mv_events=[loaded])
    tp = batch.time_points[0]
    return tp[1:] - tp[:-1]


def test_float64_times_preserve_jitter():
    """float64 storage keeps every jittered gap ~1e-6 through save/load + batching."""
    gaps = _roundtrip_gaps(torch.float64)
    assert gaps.dtype == torch.float64  # batcher inherits the stored dtype
    # No gap collapses; each near-tie pair retains at least half the 1e-6 jitter.
    assert (gaps > 5e-7).all(), gaps.tolist()
    assert int((gaps == 0).sum()) == 0


def test_float32_times_collapse_jitter():
    """float32 storage rounds the jitter away — the bug this fix removes.

    Included so the float64 assertion above is meaningful: the same data in
    float32 produces exact Δt = 0 gaps for the multi-day near-tie pairs.
    """
    gaps = _roundtrip_gaps(torch.float32)
    assert int((gaps == 0).sum()) >= 1, gaps.tolist()


def test_model_sees_nonzero_gaps_in_float64():
    """A history-dependent closed-form model gets a finite likelihood on float64
    near-tie data (no log(0) from a collapsed Δt)."""
    from torchpp.tpp.hawkes import SplineBaselineSplineKernelHawkesProcess

    seq = MVEventData(
        time_points=torch.tensor(_NEAR_TIE_TIMES, dtype=torch.float64),
        event_types=torch.tensor(_EVENT_TYPES, dtype=torch.long),
        T_obs=_T_OBS,
    )
    batch = BatchedMVEventData(mv_events=[seq])
    model = SplineBaselineSplineKernelHawkesProcess(
        D=3, baseline_num_knots=5, kernel_num_knots=5,
        baseline_delta_t=1.0, kernel_delta_t=1.0, seed=0,
    )
    ll = model.likelihood(batch)
    assert ll.dtype == torch.float64
    assert torch.isfinite(ll).all()
