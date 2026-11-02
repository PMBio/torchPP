"""Direct unit tests for baseline intensity modules.

Constructs each baseline module directly from small explicit params (no model
factory) and checks its genuine contract:

- ``ConstantBaselineModule``: time-independent intensity; CI = mu * T.
- ``LinearBaselineModule``: log-linear intensity exp(mu + slope*t); closed-form
  CI matches a fine trapezoid, including the slope ~= 0 Taylor guard.
- ``SplineBaselineModule``: linear-spline intensity; piecewise-linear with kinks
  at knots; analytical CI matches numerical trapezoid.
- ``PiecewiseConstantBaselineModule``: zero-order-hold intensity (step); CI is
  piecewise-linear with kinks at knots.

All output-shape and non-negativity contracts are checked, plus
``event_intensities`` shapes / padding handling.
"""

import pytest
import torch
from helpers import DEVICE

from torchpp.modules.intensity import (
    ConstantBaselineModule,
    ConstantBaselineParams,
    LinearBaselineModule,
    LinearBaselineParams,
    SplineBaselineModule,
    SplineBaselineParams,
    PiecewiseConstantBaselineModule,
    PiecewiseConstantBaselineParams,
)
from torchpp.utils.event_data import MVEventData, BatchedMVEventData


D = 3
TOL = 1e-5


@pytest.fixture
def batch():
    """Two unequal-length sequences (one padded) for shape / padding checks."""
    seq1 = MVEventData(
        torch.tensor([0.2, 0.8, 2.0, 2.3], dtype=torch.float64).to(DEVICE),
        torch.tensor([0, 1, 2, 0], dtype=torch.long).to(DEVICE),
    )
    seq2 = MVEventData(
        torch.tensor([0.1, 0.5, 1.2, 1.8, 2.5], dtype=torch.float64).to(DEVICE),
        torch.tensor([1, 0, 2, 0, 1], dtype=torch.long).to(DEVICE),
    )
    return BatchedMVEventData(
        [seq1.time_points, seq2.time_points],
        [seq1.event_types, seq2.event_types],
    ).to(DEVICE)


def _numerical_ci(module, ts, T_scalar, n=4000, t0=0.0):
    """Trapezoid of intensity over [t0, T] on a fine grid. Returns (1, D)."""
    grid = torch.linspace(t0, T_scalar, n, dtype=torch.float64).to(DEVICE)
    lam = torch.stack([module.intensity(grid[i : i + 1], ts) for i in range(n)], dim=0).squeeze(1)  # (n, D)
    return torch.trapezoid(lam, grid, dim=0).unsqueeze(0)  # (1, D)


# =============================================================================
# ConstantBaselineModule
# =============================================================================


class TestConstantBaseline:
    def _module(self):
        params = ConstantBaselineParams(mu=torch.tensor([0.5, 1.5, 3.0]))
        return ConstantBaselineModule(D=D, params=params).to(DEVICE)

    def test_intensity_shape_and_value(self, batch):
        m = self._module()
        t = torch.tensor([0.0, 5.0], dtype=torch.float64).to(DEVICE)
        lam = m.intensity(t, batch)
        assert lam.shape == (2, D)
        assert (lam > 0).all()
        # softplus(inverse_softplus(mu)) == mu
        expected = torch.tensor([0.5, 1.5, 3.0]).to(DEVICE)
        assert torch.allclose(lam[0].double(), expected.double(), atol=1e-5)

    def test_intensity_independent_of_time(self, batch):
        m = self._module()
        lam0 = m.intensity(torch.tensor([0.0]).to(DEVICE), batch)
        lam1 = m.intensity(torch.tensor([99.0]).to(DEVICE), batch)
        assert torch.allclose(lam0, lam1, atol=TOL)

    def test_cumulative_intensity_linear_in_T(self, batch):
        m = self._module()
        mu = torch.tensor([0.5, 1.5, 3.0]).to(DEVICE)
        T = torch.tensor([4.0, 2.0], dtype=torch.float64).to(DEVICE)
        ci = m.cumulative_intensity(T, batch)
        assert ci.shape == (2, D)
        assert torch.allclose(ci[0].double(), (mu * 4.0).double(), atol=1e-5)
        assert torch.allclose(ci[1].double(), (mu * 2.0).double(), atol=1e-5)

    def test_ci_matches_numerical(self, batch):
        m = self._module()
        single = BatchedMVEventData(
            [batch.time_points[0]], [batch.event_types[0]]
        ).to(DEVICE)
        ci = m.cumulative_intensity(torch.tensor([3.0], dtype=torch.float64).to(DEVICE), single)
        ci_num = _numerical_ci(m, single, 3.0)
        assert torch.allclose(ci.double(), ci_num, atol=1e-3)

    def test_event_intensities(self, batch):
        m = self._module()
        ei = m.event_intensities(batch)
        B, L = batch.time_points.shape
        assert ei.shape == (B, L)
        assert (ei[batch.event_mask] > 0).all()
        assert (ei[~batch.event_mask] == 0).all()


# =============================================================================
# LinearBaselineModule
# =============================================================================


class TestLinearBaseline:
    def _module(self, slope):
        params = LinearBaselineParams(
            mu=torch.tensor([-0.5, 0.0, 0.3]),
            slope=torch.tensor(slope),
        )
        return LinearBaselineModule(params=params, D=D).to(DEVICE)

    def test_intensity_loglinear_form(self, batch):
        m = self._module([0.2, -0.1, 0.0])
        t = torch.tensor([1.0, 2.0], dtype=torch.float64).to(DEVICE)
        lam = m.intensity(t, batch)
        assert lam.shape == (2, D)
        assert (lam > 0).all()
        mu = torch.tensor([-0.5, 0.0, 0.3]).to(DEVICE)
        slope = torch.tensor([0.2, -0.1, 0.0]).to(DEVICE)
        expected = torch.exp(mu.unsqueeze(0) + slope.unsqueeze(0) * t.unsqueeze(1))
        assert torch.allclose(lam.double(), expected.double(), atol=1e-5)

    def test_ci_matches_numerical_nonzero_slope(self, batch):
        m = self._module([0.4, -0.3, 0.2])
        single = BatchedMVEventData([batch.time_points[0]], [batch.event_types[0]]).to(DEVICE)
        ci = m.cumulative_intensity(torch.tensor([2.5], dtype=torch.float64).to(DEVICE), single)
        ci_num = _numerical_ci(m, single, 2.5)
        assert torch.allclose(ci.double(), ci_num, atol=1e-3)

    def test_ci_near_zero_slope_guard(self, batch):
        """Near-zero slope hits the Taylor branch; CI should approach mu*T."""
        m = self._module([1e-10, 0.0, -1e-10])
        single = BatchedMVEventData([batch.time_points[0]], [batch.event_types[0]]).to(DEVICE)
        T = torch.tensor([3.0], dtype=torch.float64).to(DEVICE)
        ci = m.cumulative_intensity(T, single)
        assert torch.isfinite(ci).all()
        mu = torch.tensor([-0.5, 0.0, 0.3]).to(DEVICE)
        expected = (torch.exp(mu) * 3.0).double()
        assert torch.allclose(ci[0].double(), expected, atol=1e-4)
        # And matches a fine numerical trapezoid.
        ci_num = _numerical_ci(m, single, 3.0)
        assert torch.allclose(ci.double(), ci_num, atol=1e-3)

    def test_event_intensities(self, batch):
        m = self._module([0.2, -0.1, 0.0])
        ei = m.event_intensities(batch)
        B, L = batch.time_points.shape
        assert ei.shape == (B, L)
        assert (ei[batch.event_mask] > 0).all()
        assert (ei[~batch.event_mask] == 0).all()

    def test_unconstrained_rejected(self):
        with pytest.raises(ValueError):
            LinearBaselineModule(D=D, constrained=False)


# =============================================================================
# SplineBaselineModule
# =============================================================================


class TestSplineBaseline:
    NUM_KNOTS = 6
    DELTA_T = 0.5  # knots at 0.0, 0.5, ..., 2.5

    def _module(self):
        # Distinct per-type, per-knot heights so the spline genuinely varies.
        h = torch.tensor(
            [
                [1.0, 2.0, 1.5, 0.8, 1.2, 0.6],
                [0.5, 0.7, 1.0, 1.4, 1.1, 0.9],
                [2.0, 1.8, 1.6, 1.4, 1.2, 1.0],
            ]
        )
        params = SplineBaselineParams(
            h_knots=h,
            delta_ts=torch.full((self.NUM_KNOTS,), self.DELTA_T),
        )
        # First delta_ts entry must be 0 (knots start at t=0).
        params.delta_ts = params.delta_ts.clone()
        params.delta_ts[0] = 0.0
        return SplineBaselineModule(D=D, num_knots=self.NUM_KNOTS, delta_t=self.DELTA_T, params=params).to(DEVICE)

    def test_intensity_shape_positive(self, batch):
        m = self._module()
        t = torch.tensor([0.3, 1.7], dtype=torch.float64).to(DEVICE)
        lam = m.intensity(t, batch)
        assert lam.shape == (2, D)
        assert (lam > 0).all()

    def test_intensity_linear_interpolation_at_midpoint(self, batch):
        """Between two knots the linear spline equals the height average."""
        m = self._module()
        h = m.get_heights()[0]  # (D, K), constrained positive
        # Midpoint of the first segment [0, 0.5].
        t = torch.tensor([0.25], dtype=torch.float64).to(DEVICE)
        lam = m.intensity(t, batch)  # history-independent; batch is shape context only
        mid = 0.5 * (h[:, 0] + h[:, 1])
        assert torch.allclose(lam[0].double(), mid.double(), atol=1e-5)

    def test_ci_matches_numerical(self):
        m = self._module()
        seq = MVEventData(
            torch.tensor([0.4], dtype=torch.float64).to(DEVICE),
            torch.tensor([0], dtype=torch.long).to(DEVICE),
        )
        ts = BatchedMVEventData([seq.time_points], [seq.event_types]).to(DEVICE)
        T = torch.tensor([2.5], dtype=torch.float64).to(DEVICE)
        ci = m.cumulative_intensity(T, ts)
        ci_num = _numerical_ci(m, ts, 2.5)
        assert torch.allclose(ci.double(), ci_num, atol=1e-3)

    def test_event_intensities(self, batch):
        m = self._module()
        ei = m.event_intensities(batch)
        B, L = batch.time_points.shape
        assert ei.shape == (B, L)
        assert (ei[batch.event_mask] > 0).all()
        assert (ei[~batch.event_mask] == 0).all()


# =============================================================================
# PiecewiseConstantBaselineModule
# =============================================================================


class TestPiecewiseConstantBaseline:
    NUM_KNOTS = 5
    DELTA_T = 0.5  # bins [0,.5), [.5,1), [1,1.5), [1.5,2), [2, inf)

    def _module(self):
        h = torch.tensor(
            [
                [1.0, 2.0, 0.5, 1.5, 0.8],
                [0.3, 0.6, 0.9, 1.2, 1.5],
                [2.0, 1.0, 2.0, 1.0, 2.0],
            ]
        )
        params = PiecewiseConstantBaselineParams(
            h_knots=h,
            delta_ts=torch.full((self.NUM_KNOTS,), self.DELTA_T),
        )
        params.delta_ts = params.delta_ts.clone()
        params.delta_ts[0] = 0.0
        return PiecewiseConstantBaselineModule(
            D=D, num_knots=self.NUM_KNOTS, delta_t=self.DELTA_T, params=params
        ).to(DEVICE)

    def test_intensity_constant_within_bin(self, batch):
        """Intensity is the left-knot height anywhere inside a bin (step)."""
        m = self._module()
        h = m.get_heights()[0]  # (D, K)
        # Two points inside the second bin [0.5, 1.0): should be identical.
        lam_a = m.intensity(torch.tensor([0.6], dtype=torch.float64).to(DEVICE), batch)
        lam_b = m.intensity(torch.tensor([0.9], dtype=torch.float64).to(DEVICE), batch)
        assert torch.allclose(lam_a, lam_b, atol=TOL)
        assert torch.allclose(lam_a[0].double(), h[:, 1].double(), atol=1e-5)

    def test_intensity_steps_at_knot(self, batch):
        """Value jumps across a knot (zero-order hold to next height)."""
        m = self._module()
        h = m.get_heights()[0]
        lam_before = m.intensity(torch.tensor([0.49], dtype=torch.float64).to(DEVICE), batch)
        lam_after = m.intensity(torch.tensor([0.51], dtype=torch.float64).to(DEVICE), batch)
        assert torch.allclose(lam_before[0].double(), h[:, 0].double(), atol=1e-5)
        assert torch.allclose(lam_after[0].double(), h[:, 1].double(), atol=1e-5)

    def test_ci_piecewise_linear_kink(self):
        """CI is piecewise-linear: its slope changes across a knot.

        Within a bin the CI grows at the (constant) bin height; the slope jumps
        at the knot, so the centred second difference around a knot is non-zero
        while inside a bin it is ~0.
        """
        m = self._module()
        seq = MVEventData(
            torch.tensor([0.1], dtype=torch.float64).to(DEVICE),
            torch.tensor([0], dtype=torch.long).to(DEVICE),
        )
        ts = BatchedMVEventData([seq.time_points], [seq.event_types]).to(DEVICE)

        def ci_at(t):
            return m.cumulative_intensity(torch.tensor([t], dtype=torch.float64).to(DEVICE), ts)[0]

        eps = 1e-3
        # Inside bin [0.5, 1.0): CI is linear -> second difference ~ 0.
        inside = ci_at(0.7 + eps) - 2 * ci_at(0.7) + ci_at(0.7 - eps)
        assert torch.allclose(inside, torch.zeros_like(inside), atol=1e-6)
        # Across the knot at t=1.0: heights differ (h[:,1] vs h[:,2]) -> kink.
        h = m.get_heights()[0]
        differs = (h[:, 1] - h[:, 2]).abs() > 1e-3
        across = ci_at(1.0 + eps) - 2 * ci_at(1.0) + ci_at(1.0 - eps)
        assert (across[differs].abs() > 1e-7).all()

    def test_ci_matches_rectangle_sum(self):
        """Analytical CI matches the closed-form rectangle accumulation."""
        m = self._module()
        h = m.get_heights()[0]  # (D, K)
        seq = MVEventData(
            torch.tensor([0.1], dtype=torch.float64).to(DEVICE),
            torch.tensor([0], dtype=torch.long).to(DEVICE),
        )
        ts = BatchedMVEventData([seq.time_points], [seq.event_types]).to(DEVICE)
        # T = 1.3 -> full bins [0,.5) and [.5,1) plus 0.3 of bin [1,1.5).
        T = torch.tensor([1.3], dtype=torch.float64).to(DEVICE)
        ci = m.cumulative_intensity(T, ts)
        expected = (h[:, 0] * 0.5 + h[:, 1] * 0.5 + h[:, 2] * 0.3).double()
        assert torch.allclose(ci[0].double(), expected, atol=1e-5)

    def test_event_intensities(self, batch):
        m = self._module()
        ei = m.event_intensities(batch)
        B, L = batch.time_points.shape
        assert ei.shape == (B, L)
        assert (ei[batch.event_mask] > 0).all()
        assert (ei[~batch.event_mask] == 0).all()
