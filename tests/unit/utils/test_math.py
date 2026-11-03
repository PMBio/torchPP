"""Direct unit tests for the self-contained numerical helpers in ``torchpp.utils.math``.

These exercise the primitives on synthetic tensors only — no models. They cover
the spline/piecewise-constant interpolation+integration+inversion machinery, the
``inverse_softplus`` numerical guard, the knot-location/gap generators, and the
closed-form ``loglinear_cumulative_intensity``.

Numerical-integration helpers (trapezoidal / monte-carlo) live in
``test_integration.py`` because they need a ``BatchedMVEventData`` scaffold.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from helpers import DEVICE

from torchpp.utils.math import (
    LinearSpline,
    PiecewiseConstant,
    baseline_spline_knot_locs,
    interaction_spline_knot_gaps,
    inverse_softplus,
    loglinear_cumulative_intensity,
)


# ----------------------------------------------------------------------------
# inverse_softplus
# ----------------------------------------------------------------------------


class TestInverseSoftplus:
    def test_round_trips_softplus(self):
        """inverse_softplus(softplus(x)) recovers x across a wide range."""
        x = torch.linspace(-5.0, 30.0, 200, device=DEVICE)
        y = F.softplus(x)
        recovered = inverse_softplus(y)
        assert torch.allclose(recovered, x, atol=1e-4)

    def test_stable_for_large_input(self):
        """The branch for large x stays finite where log(expm1(x)) would overflow."""
        x = torch.tensor([50.0, 88.0, 100.0, 200.0], device=DEVICE)
        out = inverse_softplus(x)
        assert torch.isfinite(out).all()
        # For large x, softplus(x) ≈ x so inverse_softplus(x) ≈ x.
        assert torch.allclose(out, x, atol=1e-3)


# ----------------------------------------------------------------------------
# Knot generators
# ----------------------------------------------------------------------------


class TestKnotGenerators:
    def test_uniform_locs_equidistant_and_endpoints(self):
        """Uniform schedule: first knot at 0, last at T, equal gaps."""
        locs = baseline_spline_knot_locs(T=4.0, K=5, schedule="uniform")
        assert locs.shape == (5,)
        assert locs[0].item() == pytest.approx(0.0)
        assert locs[-1].item() == pytest.approx(4.0)
        gaps = locs[1:] - locs[:-1]
        assert torch.allclose(gaps, torch.full_like(gaps, 1.0))

    def test_geometric_locs_denser_near_zero(self):
        """Geometric schedule: first at 0, last at T, gaps grow monotonically."""
        locs = baseline_spline_knot_locs(T=4.0, K=6, schedule="geometric", power=2.0)
        assert locs[0].item() == pytest.approx(0.0)
        assert locs[-1].item() == pytest.approx(4.0)
        gaps = locs[1:] - locs[:-1]
        # Power>1 packs knots near zero, so successive gaps increase.
        assert (gaps[1:] > gaps[:-1]).all()

    def test_geometric_power_one_equals_uniform(self):
        """power=1 collapses the geometric schedule onto the uniform one."""
        geo = baseline_spline_knot_locs(T=3.0, K=7, schedule="geometric", power=1.0)
        uni = baseline_spline_knot_locs(T=3.0, K=7, schedule="uniform")
        assert torch.allclose(geo, uni, atol=1e-6)

    def test_locs_invalid_schedule_raises(self):
        with pytest.raises(ValueError):
            baseline_spline_knot_locs(T=1.0, K=3, schedule="bogus")

    def test_uniform_gaps_equal_and_sum_to_T(self):
        """Uniform gaps: K-1 equal widths that tile [0, T]."""
        gaps = interaction_spline_knot_gaps(T=4.0, K=5, schedule="uniform")
        assert gaps.shape == (4,)
        assert torch.allclose(gaps, torch.full_like(gaps, 1.0))
        assert gaps.sum().item() == pytest.approx(4.0)

    def test_geometric_gaps_reconstruct_locations(self):
        """Cumulative geometric gaps (with implicit 0) reproduce the knot locs."""
        T, K = 5.0, 6
        gaps = interaction_spline_knot_gaps(T=T, K=K, schedule="geometric", power=2.0)
        assert gaps.shape == (K - 1,)
        assert gaps.sum().item() == pytest.approx(T)
        locs = baseline_spline_knot_locs(T=T, K=K, schedule="geometric", power=2.0)
        reconstructed = torch.cat([torch.zeros(1), torch.cumsum(gaps, 0)])
        assert torch.allclose(reconstructed, locs, atol=1e-5)

    def test_gaps_invalid_schedule_raises(self):
        with pytest.raises(ValueError):
            interaction_spline_knot_gaps(T=1.0, K=3, schedule="bogus")


# ----------------------------------------------------------------------------
# loglinear_cumulative_intensity
# ----------------------------------------------------------------------------


class TestLoglinearCumulativeIntensity:
    def test_matches_closed_form_nonzero_slope(self):
        """∫₀ᵀ exp(mu + b·t) dt = exp(mu)·(exp(bT)-1)/b for b≠0."""
        mu = torch.tensor([0.5, -1.0], device=DEVICE)
        slope = torch.tensor([0.7, -0.3], device=DEVICE)
        T = torch.tensor([[2.0, 3.0]], device=DEVICE)
        out = loglinear_cumulative_intensity(mu, slope, T)
        expected = torch.exp(mu) * (torch.exp(slope * T) - 1) / slope
        assert torch.allclose(out, expected, atol=1e-5)

    def test_matches_numerical_integral(self):
        """Closed form agrees with a fine trapezoidal integral of exp(mu+b·t)."""
        mu = torch.tensor([0.2], device=DEVICE)
        slope = torch.tensor([0.5], device=DEVICE)
        T_end = 2.5
        out = loglinear_cumulative_intensity(mu, slope, torch.tensor([[T_end]], device=DEVICE))
        grid = torch.linspace(0.0, T_end, 20000, device=DEVICE)
        vals = torch.exp(mu + slope * grid)
        numeric = torch.trapezoid(vals, grid)
        assert out.item() == pytest.approx(numeric.item(), rel=1e-4)

    def test_taylor_guard_near_zero_slope(self):
        """For |b|→0 the Taylor branch reduces to exp(mu)·T and stays smooth."""
        mu = torch.tensor([0.0], device=DEVICE)
        T = torch.tensor([[3.0]], device=DEVICE)
        # Exactly zero slope → integral is exp(mu)*T.
        zero = loglinear_cumulative_intensity(mu, torch.tensor([0.0], device=DEVICE), T)
        assert zero.item() == pytest.approx(3.0, abs=1e-6)
        # A tiny slope should be continuous with the zero-slope value.
        tiny = loglinear_cumulative_intensity(mu, torch.tensor([1e-10], device=DEVICE), T)
        assert tiny.item() == pytest.approx(3.0, abs=1e-5)
        assert torch.isfinite(tiny).all()


# ----------------------------------------------------------------------------
# LinearSpline
# ----------------------------------------------------------------------------


class TestLinearSpline:
    def _knots(self):
        # Single dimension, knots at 0,1,2,3 with heights 1,2,2,5.
        x = torch.tensor([0.0, 1.0, 2.0, 3.0], device=DEVICE)
        y = torch.tensor([[1.0, 2.0, 2.0, 5.0]], device=DEVICE)
        return x, y

    def test_interpolate_linear_between_knots(self):
        """Values interpolate linearly inside each segment."""
        x, y = self._knots()
        t = torch.tensor([0.0, 0.5, 1.5, 2.5], device=DEVICE)
        out = LinearSpline.interpolate(x, y, t).squeeze(1)
        # seg0 slope 1: at 0.5 -> 1.5; seg1 slope 0: at 1.5 -> 2; seg2 slope 3: at 2.5 -> 3.5
        expected = torch.tensor([1.0, 1.5, 2.0, 3.5], device=DEVICE)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_interpolate_flat_extrapolation(self):
        """Beyond the last knot the value is held flat at the last height."""
        x, y = self._knots()
        t = torch.tensor([3.0, 4.0, 10.0], device=DEVICE)
        out = LinearSpline.interpolate(x, y, t).squeeze(1)
        assert torch.allclose(out, torch.full_like(out, 5.0), atol=1e-6)

    def test_interpolate_before_first_knot_raises(self):
        x, y = self._knots()
        t = torch.tensor([-0.1], device=DEVICE)
        with pytest.raises(ValueError):
            LinearSpline.interpolate(x, y, t)

    def test_integrate_matches_trapezoid_area(self):
        """Integral equals analytic trapezoid area of the linear pieces."""
        x, y = self._knots()
        # ∫₀² = area seg0 (trap 1->2 over width1 = 1.5) + seg1 (rect 2*1 = 2) = 3.5
        out = LinearSpline.integrate(x, y, torch.tensor([2.0], device=DEVICE)).squeeze(1)
        assert out.item() == pytest.approx(3.5, abs=1e-6)
        # Mid-segment: ∫₀^2.5 = 3.5 + ∫₂^2.5 (slope3 start2) = 3.5 + (2*0.5 + 0.5*3*0.25) = 3.5+1.375
        out2 = LinearSpline.integrate(x, y, torch.tensor([2.5], device=DEVICE)).squeeze(1)
        assert out2.item() == pytest.approx(4.875, abs=1e-6)

    def test_integrate_monotone_in_t(self):
        """With positive heights the integral is non-decreasing in t."""
        x, y = self._knots()
        t = torch.linspace(0.0, 3.0, 50, device=DEVICE)
        out = LinearSpline.integrate(x, y, t).squeeze(1)
        diffs = out[1:] - out[:-1]
        assert (diffs >= -1e-6).all()

    def test_inverse_integral_round_trips(self):
        """inverse_integral recovers t from u = ∫₀ᵗ for positive heights."""
        x, y = self._knots()  # all heights > 0
        t_query = torch.tensor([[0.3, 1.2, 2.4]], device=DEVICE).T.reshape(3, 1)
        u = LinearSpline.integrate(x, y, t_query.squeeze(1))
        t_back = LinearSpline.inverse_integral(x, y, u)
        assert torch.allclose(t_back, t_query, atol=1e-5)

    def test_inverse_integral_handles_zero_slope_segment(self):
        """Round-trip works for a query landing in a flat (zero-slope) segment."""
        x, y = self._knots()  # segment [1,2] has slope 0
        t_query = torch.tensor([1.6], device=DEVICE)
        u = LinearSpline.integrate(x, y, t_query)  # (1,1)
        t_back = LinearSpline.inverse_integral(x, y, u)
        assert t_back.item() == pytest.approx(1.6, abs=1e-5)

    def test_inverse_integral_round_trips_negative_heights(self):
        """Round-trip holds for all-negative heights (descending integral)."""
        x, y = self._knots()
        y_neg = -y  # strictly negative, permitted by the assertion
        t_query = torch.tensor([[0.3, 1.2, 2.4]], device=DEVICE).T.reshape(3, 1)
        u = LinearSpline.integrate(x, y_neg, t_query.squeeze(1))
        t_back = LinearSpline.inverse_integral(x, y_neg, u)
        assert torch.allclose(t_back, t_query, atol=1e-5)


# ----------------------------------------------------------------------------
# PiecewiseConstant
# ----------------------------------------------------------------------------


class TestPiecewiseConstant:
    def _knots(self):
        x = torch.tensor([0.0, 1.0, 2.0, 3.0], device=DEVICE)
        y = torch.tensor([[2.0, 3.0, 1.0, 4.0]], device=DEVICE)
        return x, y

    def test_interpolate_zero_order_hold(self):
        """Value equals the left knot height of the containing segment."""
        x, y = self._knots()
        t = torch.tensor([0.0, 0.9, 1.0, 1.5, 2.7], device=DEVICE)
        out = PiecewiseConstant.interpolate(x, y, t).squeeze(1)
        expected = torch.tensor([2.0, 2.0, 3.0, 3.0, 1.0], device=DEVICE)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_interpolate_flat_extrapolation(self):
        x, y = self._knots()
        t = torch.tensor([3.0, 5.0], device=DEVICE)
        out = PiecewiseConstant.interpolate(x, y, t).squeeze(1)
        assert torch.allclose(out, torch.full_like(out, 4.0), atol=1e-6)

    def test_interpolate_before_first_knot_raises(self):
        x, y = self._knots()
        with pytest.raises(ValueError):
            PiecewiseConstant.interpolate(x, y, torch.tensor([-0.5], device=DEVICE))

    def test_integrate_sum_of_rectangles(self):
        """Integral is the cumulative sum of rectangle areas."""
        x, y = self._knots()
        # ∫₀² = 2*1 + 3*1 = 5 ; ∫₀^2.5 = 5 + 1*0.5 = 5.5
        out2 = PiecewiseConstant.integrate(x, y, torch.tensor([2.0], device=DEVICE)).squeeze(1)
        assert out2.item() == pytest.approx(5.0, abs=1e-6)
        out25 = PiecewiseConstant.integrate(x, y, torch.tensor([2.5], device=DEVICE)).squeeze(1)
        assert out25.item() == pytest.approx(5.5, abs=1e-6)

    def test_integrate_monotone_in_t(self):
        x, y = self._knots()
        t = torch.linspace(0.0, 3.0, 40, device=DEVICE)
        out = PiecewiseConstant.integrate(x, y, t).squeeze(1)
        diffs = out[1:] - out[:-1]
        assert (diffs >= -1e-6).all()

    def test_inverse_integral_round_trips(self):
        """inverse_integral recovers t from u = ∫₀ᵗ (positive heights)."""
        x, y = self._knots()
        t_query = torch.tensor([[0.4, 1.7, 2.3]], device=DEVICE).reshape(3, 1)
        u = PiecewiseConstant.integrate(x, y, t_query.squeeze(1))
        t_back = PiecewiseConstant.inverse_integral(x, y, u)
        assert torch.allclose(t_back, t_query, atol=1e-5)

    def test_inverse_integral_multi_dim(self):
        """Round-trip holds across multiple independent dimensions at once."""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0], device=DEVICE)
        y = torch.tensor([[2.0, 3.0, 1.0, 4.0], [1.0, 1.0, 2.0, 2.0]], device=DEVICE)  # (D=2, K=4)
        t_query = torch.tensor([[1.4, 2.2]], device=DEVICE)  # (B=1, D=2)
        u = PiecewiseConstant.integrate(x, y, t_query)
        t_back = PiecewiseConstant.inverse_integral(x, y, u)
        assert torch.allclose(t_back, t_query, atol=1e-5)

    def test_inverse_integral_round_trips_negative_heights(self):
        """Round-trip holds for all-negative heights (descending integral)."""
        x, y = self._knots()
        y_neg = -y  # strictly negative, permitted by the assertion
        t_query = torch.tensor([[0.4, 1.7, 2.3]], device=DEVICE).reshape(3, 1)
        u = PiecewiseConstant.integrate(x, y_neg, t_query.squeeze(1))
        t_back = PiecewiseConstant.inverse_integral(x, y_neg, u)
        assert torch.allclose(t_back, t_query, atol=1e-5)
