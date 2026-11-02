"""Tests for the NonStationarySplineKernelModule.

Covers:
1. Parameter / output shapes and softplus positivity.
2. Stationary-limit reduction: with all absolute-time slices equal, every path
   (intensity, cumulative_intensity, event_intensities) equals SplineKernelModule.
3. Intensity <-> CI self-consistency (analytic vs fine numerical integration) for a
   genuinely time-varying config.
4. Non-stationarity: heights are the linear interpolation along absolute time.
5. Terminating-mode likelihood and finite gradients on the new parameters.
"""

import pytest
import torch

from torchpp.modules.intensity import (
    SplineKernelModule,
    NonStationarySplineKernelModule,
)
from torchpp.tpp.hawkes import SplineBaselineNSSplineKernelHawkesProcess
from torchpp.utils.event_data import BatchedMVEventData
from torchpp.utils.math import interaction_spline_knot_gaps

D = 4
K_NS = 5
K_DT = 6
SEED = 7


def _gaps():
    return interaction_spline_knot_gaps(4.0, K_DT, schedule="geometric", power=2.0)


@pytest.fixture
def ts():
    seqs_t = [
        torch.tensor([0.15, 0.6, 1.3, 2.1, 3.0, 4.2], dtype=torch.float64),
        torch.tensor([0.3, 0.9, 1.1, 2.8, 3.9], dtype=torch.float64),
    ]
    seqs_k = [torch.tensor([0, 1, 2, 3, 0, 1]), torch.tensor([2, 0, 3, 1, 2])]
    return BatchedMVEventData(seqs_t, seqs_k)


@pytest.fixture
def T():
    return torch.tensor([5.0, 5.0], dtype=torch.float64)


def _ns(seed=SEED, **kw):
    return NonStationarySplineKernelModule(
        D, num_knots_ns=K_NS, num_knots_kernel=K_DT, delta_t_ns=1.25, delta_t_kernel=_gaps(), seed=seed, **kw
    ).double()


# =============================================================================
# Shapes & positivity
# =============================================================================


def test_param_and_output_shapes(ts, T):
    ns = _ns()
    h, knot_locs = ns.transform_params()
    assert h.shape == (K_NS, D, D, K_DT + 1)  # +1 zero tail on the Δt axis
    assert knot_locs.shape == (K_DT + 1,)
    assert (h[:, :, :, -1] == 0).all()  # tail heights are zero
    assert ns.ns_knot_locs.shape == (K_NS,)

    lam = ns.intensity(torch.tensor([2.0, 2.0], dtype=torch.float64), ts)
    assert lam.shape == (2, D)
    ci = ns.cumulative_intensity(T, ts)
    assert ci.shape == (2, D)
    ei = ns.event_intensities(ts)
    assert ei.shape == (2, ts.time_points.shape[1])


def test_positivity(ts):
    ns = _ns()
    with torch.no_grad():
        ns.log_h_knots.copy_(torch.randn(ns.log_h_knots.shape) * 1.0)
    h, _ = ns.transform_params()
    assert (h >= 0).all()
    lam = ns.intensity(torch.tensor([2.0, 2.0], dtype=torch.float64), ts)
    assert (lam >= 0).all()


def test_fresh_init_starts_stationary():
    """Stationary-replicated init: all absolute-time slices are identical."""
    ns = _ns()
    assert torch.equal(ns.log_h_knots, ns.log_h_knots[0:1].expand_as(ns.log_h_knots))


# =============================================================================
# Stationary-limit reduction
# =============================================================================


def test_stationary_reduction(ts, T):
    """All K_ns slices equal -> exactly the validated SplineKernelModule."""
    stat = SplineKernelModule(D, num_knots=K_DT, delta_t=_gaps(), seed=SEED).double()
    ns = _ns()
    with torch.no_grad():
        ns.log_h_knots.copy_(stat.log_h_knots.unsqueeze(0).expand(K_NS, -1, -1, -1))
        ns.log_tail_delta.copy_(stat.log_tail_delta)

    for q in torch.linspace(0.1, 4.9, 12):
        qb = q.expand(2).double()
        assert torch.allclose(ns.intensity(qb, ts), stat.intensity(qb, ts), rtol=0, atol=1e-12)

    assert torch.allclose(ns.cumulative_intensity(T, ts), stat.cumulative_intensity(T, ts), rtol=0, atol=1e-12)
    # per-type T
    T2 = torch.stack([T + 0.1 * d for d in range(D)], dim=1)
    assert torch.allclose(ns.cumulative_intensity(T2, ts), stat.cumulative_intensity(T2, ts), rtol=0, atol=1e-12)
    assert torch.allclose(ns.event_intensities(ts), stat.event_intensities(ts), rtol=0, atol=1e-12)


# =============================================================================
# CI self-consistency
# =============================================================================


def test_ci_matches_numerical_integration(ts):
    """Analytic CI over a smooth post-last-event interval == ∫ intensity dt."""
    ns = _ns(seed=11)
    with torch.no_grad():  # genuinely time-varying
        ns.log_h_knots.copy_(torch.randn(ns.log_h_knots.shape, generator=torch.Generator().manual_seed(3)) * 0.8)

    ta, tb, n = 4.25, 4.55, 40001  # strictly after the last event (max t = 4.2)
    Ta = torch.full((2,), ta, dtype=torch.float64)
    Tb = torch.full((2,), tb, dtype=torch.float64)
    analytic = ns.cumulative_intensity(Tb, ts) - ns.cumulative_intensity(Ta, ts)

    grid = torch.linspace(ta, tb, n, dtype=torch.float64)
    vals = torch.stack([ns.intensity(grid[i].expand(2), ts) for i in range(n)], dim=0)
    numeric = torch.trapezoid(vals, grid, dim=0)
    rel = (analytic - numeric).abs().max() / numeric.abs().max().clamp(min=1e-30)
    assert rel < 1e-6, rel.item()


# =============================================================================
# Non-stationarity
# =============================================================================


def test_interpolation_is_linear_in_absolute_time(ts):
    ns = _ns(seed=11)
    with torch.no_grad():
        ns.log_h_knots.copy_(torch.randn(ns.log_h_knots.shape, generator=torch.Generator().manual_seed(3)) * 0.8)
        h, _ = ns.transform_params()
        locs = ns.ns_knot_locs
        t_mid = (locs[1] + locs[2]) / 2.0
        frac = ((t_mid - locs[1]) / (locs[2] - locs[1])).item()
        expected = (1 - frac) * h[1, 0, 2, :] + frac * h[2, 0, 2, :]
        got = ns._interp_dt_heights_for_pairs(t_mid.reshape(1), torch.tensor([0]), torch.tensor([2]))[0]
        assert torch.allclose(got, expected, rtol=0, atol=1e-12)

        # early vs late trigger differ
        early = ns._interp_dt_heights_for_pairs(locs[0:1], torch.tensor([0]), torch.tensor([2]))[0]
        late = ns._interp_dt_heights_for_pairs(locs[-1:], torch.tensor([0]), torch.tensor([2]))[0]
        assert (early - late).abs().max() > 1e-3


# =============================================================================
# Terminating + gradients (composite)
# =============================================================================


def test_terminating_and_gradients(ts, T):
    model = SplineBaselineNSSplineKernelHawkesProcess(
        D, num_knots_baseline=8, num_knots_ns=K_NS, num_knots_kernel=K_DT,
        delta_t_baseline=0.5, delta_t_ns=1.25, delta_t_kernel=_gaps(), seed=5, terminating=True,
    ).double()
    # terminating data: each type at most once
    seqs_t = [torch.tensor([0.2, 1.0, 2.5, 3.8], dtype=torch.float64)]
    seqs_k = [torch.tensor([0, 1, 2, 3])]
    ts_term = BatchedMVEventData(seqs_t, seqs_k)
    ll = model.likelihood(ts_term, torch.tensor([5.0], dtype=torch.float64), log=True)
    assert torch.isfinite(ll).all()

    model2 = SplineBaselineNSSplineKernelHawkesProcess(
        D, num_knots_baseline=8, num_knots_ns=K_NS, num_knots_kernel=K_DT,
        delta_t_baseline=0.5, delta_t_ns=1.25, delta_t_kernel=_gaps(), seed=5,
    ).double()
    loss = -model2.likelihood(ts, T, log=True).sum()
    loss.backward()
    assert model2.kernel.log_h_knots.grad is not None
    assert torch.isfinite(model2.kernel.log_h_knots.grad).all()
    assert model2.kernel.log_h_knots.grad.abs().max() > 0
    assert torch.isfinite(model2.kernel.log_tail_delta.grad).all()
