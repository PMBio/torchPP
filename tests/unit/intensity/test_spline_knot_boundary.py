"""Regression tests for spline/piecewise knot-boundary handling.

Pins two failure modes that used to be silent:

1. ``_get_knot_info`` computed ``searchsorted - 1`` without guarding ``k = -1``,
   so a query *before the first knot* wrapped around to the LAST knot via
   Python negative indexing (interpolate returned the last knot height;
   integrate returned garbage or crashed in gather).

2. The tensor-``delta_t`` / ``params`` construction branches of the baseline
   modules build knot locations via ``cumsum`` — without a leading 0 entry the
   first knot silently lands at ``delta_ts[0]`` instead of t=0, exposing every
   query in ``[0, delta_ts[0])`` to failure mode 1.

Both now raise ``ValueError`` (at query time and construction time
respectively).
"""

import pytest
import torch

from torchpp.modules.intensity import (
    PiecewiseConstantBaselineModule,
    PiecewiseConstantBaselineParams,
    SplineBaselineModule,
    SplineBaselineParams,
)
from torchpp.utils.math import LinearSpline, PiecewiseConstant


X_KNOTS = torch.tensor([1.0, 2.0, 3.0])
Y_KNOTS = torch.tensor([[2.0, 4.0, 6.0]])


class TestKnotInfoBeforeFirstKnot:
    """Queries before the first knot must raise, not wrap around."""

    def test_linear_spline_interpolate_raises(self):
        with pytest.raises(ValueError, match="precedes the first knot"):
            LinearSpline.interpolate(X_KNOTS, Y_KNOTS, torch.tensor([0.5]))

    def test_linear_spline_integrate_raises(self):
        with pytest.raises(ValueError, match="precedes the first knot"):
            LinearSpline.integrate(X_KNOTS, Y_KNOTS, torch.tensor([0.5]))

    def test_piecewise_constant_interpolate_raises(self):
        with pytest.raises(ValueError, match="precedes the first knot"):
            PiecewiseConstant.interpolate(X_KNOTS, Y_KNOTS, torch.tensor([0.5]))

    def test_valid_queries_unaffected(self):
        # On-knot, mid-segment, and beyond-last-knot queries still work.
        t = torch.tensor([1.0, 1.5, 10.0])
        vals = LinearSpline.interpolate(X_KNOTS, Y_KNOTS, t)
        assert torch.allclose(vals.squeeze(), torch.tensor([2.0, 3.0, 6.0]))
        vals_pc = PiecewiseConstant.interpolate(X_KNOTS, Y_KNOTS, t)
        assert torch.allclose(vals_pc.squeeze(), torch.tensor([2.0, 2.0, 6.0]))

    def test_query_at_first_knot_is_valid(self):
        vals = LinearSpline.interpolate(X_KNOTS, Y_KNOTS, torch.tensor([1.0]))
        assert torch.allclose(vals.squeeze(), torch.tensor(2.0))


class TestKnotConstructionStartsAtZero:
    """cumsum-built knot locations must start at t=0 or fail at construction."""

    def test_spline_params_without_leading_zero_raises(self):
        params = SplineBaselineParams(
            h_knots=torch.ones(2, 3),
            delta_ts=torch.tensor([0.5, 0.5, 0.5]),  # first knot at 0.5, not 0
        )
        with pytest.raises(ValueError, match="first knot must lie at t=0"):
            SplineBaselineModule(D=2, num_knots=3, delta_t=0.5, params=params)

    def test_spline_params_with_leading_zero_works(self):
        params = SplineBaselineParams(
            h_knots=torch.ones(2, 3),
            delta_ts=torch.tensor([0.0, 0.5, 0.5]),
        )
        module = SplineBaselineModule(D=2, num_knots=3, delta_t=0.5, params=params, constrained=False)
        assert module.knot_locs[0].item() == 0.0
        assert torch.allclose(module.knot_locs, torch.tensor([0.0, 0.5, 1.0]))
        # The t=0..first-gap region is now well-defined (this used to hit k=-1).
        (h,) = module.transform_params()
        vals = LinearSpline.interpolate(module.knot_locs, h, torch.tensor([0.1]))
        assert torch.isfinite(vals).all()

    def test_spline_tensor_delta_t_without_leading_zero_raises(self):
        with pytest.raises(ValueError, match="first knot must lie at t=0"):
            SplineBaselineModule(D=2, num_knots=3, delta_t=torch.tensor([0.5, 0.5, 0.5]))

    def test_piecewise_params_without_leading_zero_raises(self):
        params = PiecewiseConstantBaselineParams(
            h_knots=torch.ones(2, 3),
            delta_ts=torch.tensor([0.5, 0.5, 0.5]),
        )
        with pytest.raises(ValueError, match="first knot must lie at t=0"):
            PiecewiseConstantBaselineModule(D=2, num_knots=3, delta_t=0.5, params=params)

    def test_piecewise_params_with_leading_zero_works(self):
        params = PiecewiseConstantBaselineParams(
            h_knots=torch.ones(2, 3),
            delta_ts=torch.tensor([0.0, 0.5, 0.5]),
        )
        module = PiecewiseConstantBaselineModule(D=2, num_knots=3, delta_t=0.5, params=params, constrained=False)
        assert torch.allclose(module.knot_locs, torch.tensor([0.0, 0.5, 1.0]))

    def test_scalar_delta_t_unaffected(self):
        # The production path (scalar spacing) builds knots from 0 and is untouched.
        module = SplineBaselineModule(D=2, num_knots=4, delta_t=0.5, seed=0)
        assert module.knot_locs[0].item() == 0.0
        assert torch.allclose(module.knot_locs, torch.tensor([0.0, 0.5, 1.0, 1.5]))
