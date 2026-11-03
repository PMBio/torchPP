"""Model-fanned intensity-access cascade tests.

These pin the multi-query primitive of ``base_tpp.py`` across the
full model matrix:

- the flat / structured hooks form a self-consistent cascade — a batched flat
  multi-query equals the per-query flat evaluated one query at a time, and a
  structured grid equals its flattened flat evaluation (the "override ≡
  loop-default" gate);
- ``_sequence_intensities`` is the P=1 structured case and is bit-identical
  (tight-tol) to the per-event ``_intensities_at_queries`` evaluation;
- the AttNHP float64 tests exercise the batched multi-query hooks
  (flat-override ≡ per-query loop, structured-override ≡ default, chunk invariance).

The conditioning tie-break at an exact event time is strictly-before / left-limit;
all query points used here are interval interiors or event times where the
flat and structured conventions coincide, so the comparisons are well-posed.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch

from torchpp.tpp.base_tpp import IntensityBasedTPP
from torchpp.modules.nn.attnhp import AttNHPEncoder
from torchpp.tpp.neural import AttNHPProcess
from torchpp.utils.event_data import BatchedMVEventData

from helpers import create_model, MODEL_TYPES, DEVICE, skip_if_density


def _fix(model, ts):
    """Cache encoder params for neural models; a no-op for classical ones."""
    return model.fix_context(ts) if hasattr(model, "fix_context") else nullcontext()


def _batch(device="cpu"):
    # No simultaneous events; distinct interval interiors per sequence.
    return BatchedMVEventData(
        [torch.tensor([0.5, 1.2, 2.0, 3.1], dtype=torch.float64), torch.tensor([0.3, 0.9, 1.5], dtype=torch.float64)],
        [torch.tensor([0, 2, 1, 0]), torch.tensor([1, 1, 2])],
    ).to(device)


# Interior query times, one strictly inside each interval, shared across the tests.
_Q = [[0.7, 1.6, 2.5, 3.4], [0.4, 1.1, 1.8, 2.2]]


# ---------------------------------------------------------------------------
# Cascade consistency across the whole model matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_flat_multiquery_matches_per_query(model_type):
    """``_intensities_at_queries`` over Q times equals the per-query stack (one Q=1 call each).

    The real content is AttNHP, whose flat override batches the attention re-runs; the
    per-query route re-runs attention one query at a time, so this pins batching invariance.
    """
    m = create_model(model_type, terminating=False)
    skip_if_density(m)  # density models have no intensity hooks
    ts = _batch(DEVICE)
    q = torch.tensor(_Q, dtype=torch.float64, device=DEVICE)
    with _fix(m, ts), torch.no_grad():
        multi = m._intensities_at_queries(q, ts)  # (B, Q, D)
        per = torch.stack(
            [m._intensities_at_queries(q[:, k].unsqueeze(1), ts).squeeze(1) for k in range(q.shape[1])], dim=1
        )  # (B, Q, D)
    assert multi.shape == per.shape
    assert torch.allclose(multi, per, rtol=1e-4, atol=1e-5), (model_type, (multi - per).abs().max().item())


@pytest.mark.parametrize("terminating", [False, True])
@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_public_multiquery_intensity_matches_loop(model_type, terminating):
    """Public ``intensity((B, Q))`` equals the per-query ``intensity((B,))`` loop.

    Covers the multi-query API and its ``(B, Q, D)`` termination mask:
    the mask must reduce to the single-time mask at each query, including for terminating
    processes (queries here span times before and after events).
    """
    m = create_model(model_type, terminating=terminating)
    ts = _batch(DEVICE)
    q = torch.tensor(_Q, dtype=torch.float64, device=DEVICE)
    with _fix(m, ts), torch.no_grad():
        multi = m.intensity(q, ts)  # (B, Q, D)
        loop = torch.stack([m.intensity(q[:, k], ts) for k in range(q.shape[1])], dim=1)  # (B, Q, D)
    assert multi.shape == loop.shape
    assert torch.allclose(multi, loop, rtol=1e-4, atol=1e-5), (
        model_type,
        terminating,
        (multi - loop).abs().max().item(),
    )


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_structured_matches_flat_in_interval_interiors(model_type):
    """``_interval_intensities`` row ``i`` equals the flat hook at points inside ``I_i``.

    The structured hook conditions row ``i`` on ``H_i``.  In the *interior*
    of interval ``I_i`` the flat hook (strictly-before) sees exactly ``H_i`` too, so
    they must agree — this is the well-posed override≡loop check that covers both the
    query-conditioned overrides (AttNHP, base default) and the row-conditioned ones
    (NHP's state-decay fold, which differs from flat only at the interval boundaries).
    """
    m = create_model(model_type, terminating=False)
    skip_if_density(m)  # density models have no intensity hooks / _interval_bounds
    ts = _batch(DEVICE)
    T = ts.last_event_time + 0.5
    lo, hi = m._interval_bounds(ts, None, T)  # (B, L+1) per-interval bounds
    fr = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64, device=DEVICE)  # strictly interior
    grid = lo.unsqueeze(-1) + (hi - lo).unsqueeze(-1) * fr  # (B, L+1, P) interior query times
    B, R = lo.shape
    P = fr.shape[0]
    with _fix(m, ts), torch.no_grad():
        struct = m._interval_intensities(grid, ts)  # (B, R, P, D)
        flat = m._intensities_at_queries(grid.reshape(B, R * P), ts).reshape(B, R, P, m.D)
    mask = (hi - lo) > 1e-9  # real (positive-width) intervals only
    assert torch.allclose(struct[mask], flat[mask], rtol=1e-4, atol=1e-5), (
        model_type,
        (struct[mask] - flat[mask]).abs().max().item(),
    )


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_sequence_intensities_is_p1_structured(model_type):
    """``_sequence_intensities`` equals the per-event ``_intensities_at_queries`` loop.

    This follows from the left-limit convention: the right endpoint of interval ``I_l`` is event
    ``t_l``, conditioned on the strictly-before history ``H_l``. The vectorised path must match
    the loop bit for bit.
    """
    m = create_model(model_type, terminating=False)
    skip_if_density(m)  # density models have no intensity hooks
    ts = _batch(DEVICE)
    with _fix(m, ts), torch.no_grad():
        seq = m._sequence_intensities(ts)  # (B, L, D)
        loop = torch.stack(
            [
                m._intensities_at_queries(ts.time_points[:, l].unsqueeze(1), ts).squeeze(1)
                for l in range(ts.time_points.shape[1])
            ],
            dim=1,
        )  # (B, L, D)
    assert seq.shape == loop.shape
    assert torch.allclose(seq, loop, rtol=1e-4, atol=1e-5), (model_type, (seq - loop).abs().max().item())


# ---------------------------------------------------------------------------
# AttNHP — batched multi-query hooks (float64, exact)
# ---------------------------------------------------------------------------


def _attnhp_f64(seed=0):
    torch.manual_seed(seed)
    m = AttNHPProcess(AttNHPEncoder(3, d_model=16, d_time=8, n_layers=2, n_head=2), seed=seed)
    m.ci_config = replace(m.ci_config, n_points=50)
    return m.double()


def test_attnhp_flat_override_matches_per_query_f64():
    m = _attnhp_f64()
    ts = _batch()
    q = torch.tensor([[0.7, 1.6, 2.5, 3.4], [0.4, 1.1, 1.8, 2.2]], dtype=torch.float64)
    with m.fix_context(ts), torch.no_grad():
        override = m._intensities_at_queries(q, ts)  # batched multi-query
        loop = torch.stack(
            [m._intensities_at_queries(q[:, k].unsqueeze(1), ts).squeeze(1) for k in range(q.shape[1])], dim=1
        )  # per-query flat at Q=1
    assert torch.allclose(override, loop, rtol=0, atol=1e-12), (override - loop).abs().max().item()


def test_attnhp_structured_override_matches_default_f64():
    m = _attnhp_f64()
    ts = _batch()
    q = torch.tensor([[0.7, 1.6, 2.5, 3.4], [0.4, 1.1, 1.8, 2.2]], dtype=torch.float64)
    grid = q.unsqueeze(1).expand(2, 2, 4).contiguous()
    with m.fix_context(ts), torch.no_grad():
        override = m._interval_intensities(grid, ts)
        default = IntensityBasedTPP._interval_intensities(m, grid, ts)
    assert torch.allclose(override, default, rtol=0, atol=1e-12), (override - default).abs().max().item()


def test_attnhp_structured_chunk_invariance_f64():
    m = _attnhp_f64()
    ts = _batch()
    q = torch.tensor([[0.7, 1.6, 2.5, 3.4], [0.4, 1.1, 1.8, 2.2]], dtype=torch.float64)
    grid = q.unsqueeze(1).expand(2, 2, 4).contiguous()
    with m.fix_context(ts), torch.no_grad():
        m.quadrature_chunk_size = None
        full = m._interval_intensities(grid, ts)
        m.quadrature_chunk_size = 3
        chunked = m._interval_intensities(grid, ts)
    assert torch.equal(full, chunked)  # chunking only changes the fold grouping, not the math
