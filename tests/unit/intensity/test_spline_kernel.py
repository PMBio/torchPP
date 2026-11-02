"""
Tests for the SplineKernelModule.

Tests cover:
1. Basic shape and positivity checks
2. Consistency between intensity(), cumulative_intensity(), and
   event_intensities()
3. Equivalence with ExpKernelModule when the spline knot heights are set
   to match α·β·exp(-β·Δt) at the knot locations (approximate, since linear
   interpolation of an exponential is not exact)
4. Gradient flow
"""

import pytest
import torch
from helpers import DEVICE

from torchpp.modules.intensity import (
    SplineKernelModule,
    SplineKernelParams,
    ExpKernelModule,
)
from torchpp.utils.event_data import MVEventData, BatchedMVEventData


# =============================================================================
# Constants & Fixtures
# =============================================================================

D = 3
SEED = 42
NUM_KNOTS = 15
DELTA_T = 0.3  # knots at 0.0, 0.3, 0.6, ..., 4.2
TOLERANCE = 1e-5


@pytest.fixture
def kernel():
    """Create a SplineKernelModule with random params."""
    torch.manual_seed(SEED)
    return SplineKernelModule(D=D, num_knots=NUM_KNOTS, delta_t=DELTA_T, seed=SEED).to(DEVICE)


@pytest.fixture
def test_data():
    """Create test event data."""
    seq1 = MVEventData(
        torch.tensor([0.2, 0.8, 2.0, 2.3], dtype=torch.float64).to(DEVICE),
        torch.tensor([0, 1, 2, 0], dtype=torch.long).to(DEVICE),
    )
    seq2 = MVEventData(
        torch.tensor([0.1, 0.5, 1.2, 1.8, 2.5], dtype=torch.float64).to(DEVICE),
        torch.tensor([1, 0, 2, 0, 1], dtype=torch.long).to(DEVICE),
    )
    return {
        "seq1": seq1,
        "seq2": seq2,
        "batch_single": BatchedMVEventData([seq1.time_points], [seq1.event_types]).to(DEVICE),
        "batch_multi": BatchedMVEventData(
            [seq1.time_points, seq2.time_points],
            [seq1.event_types, seq2.event_types],
        ).to(DEVICE),
    }


# =============================================================================
# Kernel Module Tests — Basic Properties
# =============================================================================


class TestSplineKernelBasics:
    """Test basic properties of SplineKernelModule."""

    def test_param_shapes(self, kernel):
        """Transform params should return (D, D, K+1) heights and (K+1,) knot locs."""
        h, knot_locs = kernel.transform_params()
        assert h.shape == (D, D, NUM_KNOTS + 1)  # K regular + 1 zero tail
        assert knot_locs.shape == (NUM_KNOTS + 1,)
        # Last height column must be zero (the tail knot)
        assert (h[:, :, -1] == 0).all()
        # Tail knot must be strictly beyond the last regular knot
        assert knot_locs[-1] > knot_locs[-2]

    def test_param_positivity(self, kernel):
        """Regular knot heights must be positive (softplus constraint)."""
        h, _ = kernel.transform_params()
        assert (h[:, :, :-1] > 0).all()  # all except tail

    def test_knot_locations(self, kernel):
        """Knot locations should be uniformly spaced on Δt axis."""
        expected = torch.arange(NUM_KNOTS, dtype=torch.float32, device=DEVICE) * DELTA_T
        assert torch.allclose(kernel.knot_locs, expected, atol=1e-6)

    def test_intensity_shape(self, kernel, test_data):
        """Intensity should return (B, D)."""
        ts = test_data["batch_multi"]
        t = torch.tensor([3.0, 3.5], dtype=torch.float64).to(DEVICE)
        intensity = kernel.intensity(t, ts)
        assert intensity.shape == (2, D)

    def test_intensity_positive(self, kernel, test_data):
        """Kernel intensity should be non-negative (positive spline knots)."""
        ts = test_data["batch_multi"]
        t = torch.tensor([3.0, 3.5], dtype=torch.float64).to(DEVICE)
        intensity = kernel.intensity(t, ts)
        assert (intensity >= 0).all()

    def test_ci_shape(self, kernel, test_data):
        """Cumulative intensity should return (B, D)."""
        ts = test_data["batch_multi"]
        T = torch.tensor([4.0, 3.5], dtype=torch.float64).to(DEVICE)
        ci = kernel.cumulative_intensity(T, ts)
        assert ci.shape == (2, D)

    def test_ci_positive(self, kernel, test_data):
        """Cumulative intensity should be non-negative."""
        ts = test_data["batch_multi"]
        T = torch.tensor([4.0, 3.5], dtype=torch.float64).to(DEVICE)
        ci = kernel.cumulative_intensity(T, ts)
        assert (ci >= 0).all()

    def test_ci_monotonic(self, kernel, test_data):
        """CI at T2 > T1 should be >= CI at T1."""
        ts = test_data["batch_single"]
        T1 = torch.tensor([2.5]).to(DEVICE)
        T2 = torch.tensor([4.0]).to(DEVICE)
        ci1 = kernel.cumulative_intensity(T1, ts)
        ci2 = kernel.cumulative_intensity(T2, ts)
        assert (ci2 >= ci1 - 1e-6).all()

    def test_pli_shape(self, kernel, test_data):
        """PLI should return (B, N)."""
        ts = test_data["batch_multi"]
        pli = kernel.event_intensities(ts)
        B, N = ts.time_points.shape
        assert pli.shape == (B, N)

    def test_pli_non_negative(self, kernel, test_data):
        """PLI values should be non-negative."""
        ts = test_data["batch_multi"]
        pli = kernel.event_intensities(ts)
        assert (pli >= 0).all()

    def test_pli_first_event_zero(self, kernel, test_data):
        """First event in each sequence should have zero kernel PLI (no past events)."""
        ts = test_data["batch_multi"]
        pli = kernel.event_intensities(ts)
        assert pli[0, 0] == 0.0
        assert pli[1, 0] == 0.0

    def test_empty_events(self, kernel):
        """Intensity should be zero with no past events."""
        ts = BatchedMVEventData(
            [torch.tensor([], dtype=torch.float64).to(DEVICE)],
            [torch.tensor([], dtype=torch.long).to(DEVICE)],
        )
        t = torch.tensor([1.0]).to(DEVICE)
        intensity = kernel.intensity(t, ts)
        assert (intensity == 0).all()


# =============================================================================
# Consistency Tests — PLI vs intensity() loop
# =============================================================================


class TestSplineKernelConsistency:
    """Test that PLI matches intensity() evaluated at each event time."""

    def test_pli_matches_intensity_loop_single(self, kernel, test_data):
        """PLI should match calling intensity() at each event time and gathering."""
        ts = test_data["batch_single"]
        B, N = ts.time_points.shape

        pli = kernel.event_intensities(ts)

        # Compute via intensity loop (the base class default method)
        intensities = torch.stack([kernel.intensity(ts.time_points[:, l], ts=ts) for l in range(N)], dim=1)  # (B, N, D)
        event_types = ts.event_types_idx_safe.unsqueeze(-1)
        loop_pli = torch.gather(intensities, dim=2, index=event_types).squeeze(-1)
        loop_pli[~ts.event_mask] = 0.0

        assert torch.allclose(pli, loop_pli, atol=TOLERANCE), f"PLI vs loop mismatch:\npli={pli}\nloop={loop_pli}"

    def test_pli_matches_intensity_loop_multi(self, kernel, test_data):
        """PLI should match intensity loop for multi-batch data."""
        ts = test_data["batch_multi"]
        B, N = ts.time_points.shape

        pli = kernel.event_intensities(ts)

        intensities = torch.stack([kernel.intensity(ts.time_points[:, l], ts=ts) for l in range(N)], dim=1)
        event_types = ts.event_types_idx_safe.unsqueeze(-1)
        loop_pli = torch.gather(intensities, dim=2, index=event_types).squeeze(-1)
        loop_pli[~ts.event_mask] = 0.0

        assert torch.allclose(pli, loop_pli, atol=TOLERANCE), f"PLI vs loop mismatch:\npli={pli}\nloop={loop_pli}"

    def test_pli_padded_events_zero(self, kernel, test_data):
        """PLI should be zero for padded events in batched data."""
        ts = test_data["batch_multi"]
        pli = kernel.event_intensities(ts)

        # batch_multi has sequences of length 4 and 5
        # The shorter one (seq1, length 4) should have zero PLI at position 4
        padding_mask = ~ts.event_mask
        assert (pli[padding_mask] == 0).all()


# =============================================================================
# CI Consistency — numerical vs analytical
# =============================================================================


class TestSplineKernelCIConsistency:
    """Verify analytical CI against numerical integration."""

    def test_ci_vs_numerical(self, kernel, test_data):
        """Analytical CI should match numerical trapezoidal integration."""
        ts = test_data["batch_single"]
        T = torch.tensor([3.5]).to(DEVICE)

        # Analytical
        ci_analytical = kernel.cumulative_intensity(T, ts)

        # Numerical (trapezoidal rule)
        n_steps = 1000
        t_grid = torch.linspace(0.01, T.item(), n_steps).to(DEVICE)
        dt = t_grid[1] - t_grid[0]

        intensities = torch.stack([kernel.intensity(t_grid[i : i + 1], ts) for i in range(n_steps)], dim=0)
        # (n_steps, 1, D) -> (n_steps, D)
        intensities = intensities.squeeze(1)
        ci_numerical = torch.trapezoid(intensities, t_grid, dim=0).unsqueeze(0)  # (1, D)

        assert torch.allclose(ci_analytical, ci_numerical, atol=0.05), (
            f"CI analytical vs numerical:\nanalytical={ci_analytical}\nnumerical={ci_numerical}\n"
            f"diff={ci_analytical - ci_numerical}"
        )

    def test_spline_kernel_integral_finite_in_tail_segment(self, kernel):
        """Integrating to an infinite horizon must stay finite, not NaN.

        Regression guard for the 0 * inf = NaN trap at the zero-height TAIL knot.
        SplineKernelModule.transform_params() appends a zero-height sentinel knot
        at the end of the support, so the final ("tail") segment of the spline
        has height 0 and slope 0. When a query Δt lands in / beyond that segment
        (here Δt = T - t_i = inf, a legitimate unbounded-horizon integration),
        the active-segment math in _cumulative_intensity evaluates

            h_k * offset      and      s_k / 2 * offset**2

        with h_k == 0.0, s_k == 0.0 and offset == inf. Plain multiplication gives
        0 * inf = NaN. The torch.where(h_k != 0.0, ...) / torch.where(s_k != 0.0, ...)
        guards at intensity.py ~1460-1461 skip those products when the factor is
        exactly zero, keeping the result finite. The tail-knot design is permanent,
        so this test FAILS (NaN) if the guards are ever removed.

        Because the kernel is compactly supported, the integral to +inf equals the
        total area under the spline (sum of trapezoidal segment areas), which we
        also check against a hand-computed value.
        """
        # Single type-0 event at t=0; integrate the whole kernel to an infinite horizon.
        seq = MVEventData(
            torch.tensor([0.0], dtype=torch.float64).to(DEVICE),
            torch.tensor([0], dtype=torch.long).to(DEVICE),
        )
        ts = BatchedMVEventData([seq.time_points], [seq.event_types]).to(DEVICE)
        T = torch.full((1, D), float("inf"), dtype=torch.float64).to(DEVICE)

        ci = kernel.cumulative_intensity(T, ts)

        # The load-bearing assertion: no NaN/inf leaking from 0 * inf at the tail knot.
        assert torch.isfinite(ci).all(), f"CI to infinite horizon is not finite (tail-knot 0*inf guard removed?): {ci}"
        assert (ci >= 0).all(), f"CI must be non-negative: {ci}"

        # Hand-computed expectation: integral to +inf == total trapezoidal area
        # under the spline for receiver d triggered by the single type-0 event.
        h, knot_locs = kernel.transform_params()  # (D, D, K+1), (K+1,)
        seg_widths = knot_locs[1:] - knot_locs[:-1]  # (K,)
        total_area = ((h[:, :, :-1] + h[:, :, 1:]) / 2.0 * seg_widths).sum(dim=2)  # (D_recv, D_trig)
        expected = total_area[:, 0].to(torch.float64).unsqueeze(0)  # (1, D), trigger type 0
        assert torch.allclose(ci, expected, atol=1e-6), f"CI to inf vs total kernel area:\nci={ci}\nexpected={expected}"

    def test_ci_with_termination(self, kernel, test_data):
        """CI with termination_times should give smaller values."""
        ts = test_data["batch_single"]
        T = torch.tensor([4.0]).to(DEVICE)

        ci_full = kernel.cumulative_intensity(T, ts)

        # Terminate type 0 at t=1.0, type 1 at t=2.0, type 2 at t=3.0
        term_times = torch.tensor([[1.0, 2.0, 3.0]]).to(DEVICE)
        T = torch.minimum(T.unsqueeze(1).expand_as(term_times), term_times)
        ci_term = kernel.cumulative_intensity(T, ts)

        assert (ci_term <= ci_full + 1e-6).all(), (
            f"CI with termination should be <= CI without:\nfull={ci_full}\nterm={ci_term}"
        )


# =============================================================================
# Gradient Flow Tests
# =============================================================================


class TestSplineKernelGradients:
    """Test that gradients flow through all methods."""

    def test_intensity_gradient(self, kernel, test_data):
        """Intensity should have gradients w.r.t. parameters."""
        ts = test_data["batch_single"]
        t = torch.tensor([3.0], dtype=torch.float64).to(DEVICE)

        intensity = kernel.intensity(t, ts)
        loss = intensity.sum()
        loss.backward()

        assert kernel.log_h_knots.grad is not None
        assert kernel.log_h_knots.grad.abs().sum() > 0

    def test_ci_gradient(self, kernel, test_data):
        """Cumulative intensity should have gradients."""
        kernel.zero_grad()
        ts = test_data["batch_single"]
        T = torch.tensor([4.0], dtype=torch.float64).to(DEVICE)

        ci = kernel.cumulative_intensity(T, ts)
        loss = ci.sum()
        loss.backward()

        assert kernel.log_h_knots.grad is not None
        assert kernel.log_h_knots.grad.abs().sum() > 0

    def test_pli_gradient(self, kernel, test_data):
        """PLI should have gradients."""
        kernel.zero_grad()
        ts = test_data["batch_single"]

        pli = kernel.event_intensities(ts)
        loss = pli.sum()
        loss.backward()

        assert kernel.log_h_knots.grad is not None
        assert kernel.log_h_knots.grad.abs().sum() > 0


# =============================================================================
# Spline Kernel with Params Dataclass
# =============================================================================


class TestSplineKernelParams:
    """Test creating a kernel from explicit SplineKernelParams."""

    def test_from_params(self, test_data):
        """Kernel created from params should match the original."""
        # Create original kernel
        kernel1 = SplineKernelModule(D=D, num_knots=NUM_KNOTS, delta_t=DELTA_T, seed=SEED).to(DEVICE)
        h1, _ = kernel1.transform_params()
        h1_regular = h1[:, :, :-1].detach().clone()  # strip the zero tail

        # Create from params
        params = SplineKernelParams(
            h_knots=h1_regular,
            delta_ts=torch.full((NUM_KNOTS,), DELTA_T).to(DEVICE),
        )
        kernel2 = SplineKernelModule(D=D, num_knots=NUM_KNOTS, delta_t=DELTA_T, params=params, seed=99).to(DEVICE)
        h2, _ = kernel2.transform_params()

        assert torch.allclose(h1, h2, atol=1e-6)

    def test_params_round_trip(self, test_data):
        """Kernel from params should produce same intensity."""
        kernel1 = SplineKernelModule(D=D, num_knots=NUM_KNOTS, delta_t=DELTA_T, seed=SEED).to(DEVICE)
        h1, _ = kernel1.transform_params()
        h1_regular = h1[:, :, :-1].detach().clone()  # strip the zero tail

        params = SplineKernelParams(
            h_knots=h1_regular,
            delta_ts=torch.full((NUM_KNOTS,), DELTA_T).to(DEVICE),
        )
        kernel2 = SplineKernelModule(D=D, num_knots=NUM_KNOTS, delta_t=DELTA_T, params=params, seed=99).to(DEVICE)

        ts = test_data["batch_single"]
        t = torch.tensor([3.0], dtype=torch.float64).to(DEVICE)

        int1 = kernel1.intensity(t, ts)
        int2 = kernel2.intensity(t, ts)

        assert torch.allclose(int1, int2, atol=1e-5)


# =============================================================================
# Approximate Equivalence with ExpKernel
# =============================================================================


class TestSplineVsExpKernelApprox:
    """
    When spline knot heights are set to match α·β·exp(-β·Δt) at knot locations,
    the spline kernel should approximate the exponential kernel.

    This is approximate because linear interpolation of an exponential is not exact.
    With enough knots and small spacing, the approximation should be good.
    """

    @pytest.fixture
    def exp_kernel(self):
        """Create a reference exponential kernel."""
        return ExpKernelModule(D=D, seed=SEED).to(DEVICE)

    @pytest.fixture
    def matching_spline_kernel(self, exp_kernel):
        """Create a spline kernel whose knot heights match exp kernel at knot locations."""
        alpha, beta = exp_kernel.transform_params()
        alpha, beta = alpha.detach(), beta.detach()

        # Fine knots for good approximation
        fine_num_knots = 50
        fine_dt = 0.1
        knot_locs = torch.arange(fine_num_knots, dtype=torch.float32, device=DEVICE) * fine_dt

        # φ_{d,j}(Δt) = α_{d,j} · β_{d,j} · exp(-β_{d,j} · Δt)
        # h_knots[d, j, k] = α[d,j] · β[d,j] · exp(-β[d,j] · knot_locs[k])
        h_knots = (alpha * beta).unsqueeze(2) * torch.exp(
            -beta.unsqueeze(2) * knot_locs.unsqueeze(0).unsqueeze(0)
        )  # (D, D, K)

        params = SplineKernelParams(
            h_knots=h_knots,
            delta_ts=torch.full((fine_num_knots,), fine_dt).to(DEVICE),
        )
        return SplineKernelModule(D=D, num_knots=fine_num_knots, delta_t=fine_dt, params=params, seed=SEED).to(DEVICE)

    def test_intensity_approx(self, exp_kernel, matching_spline_kernel, test_data):
        """Spline kernel intensity should approximate exp kernel intensity."""
        ts = test_data["batch_single"]
        t = torch.tensor([2.5], dtype=torch.float64).to(DEVICE)

        int_exp = exp_kernel.intensity(t, ts)
        int_spline = matching_spline_kernel.intensity(t, ts)

        # Relative tolerance of ~5% (linear approx of exponential)
        assert torch.allclose(int_exp, int_spline, rtol=0.05, atol=1e-4), (
            f"Intensity mismatch:\nexp={int_exp}\nspline={int_spline}\n"
            f"rel_diff={(int_exp - int_spline).abs() / int_exp.abs()}"
        )

    def test_ci_approx(self, exp_kernel, matching_spline_kernel, test_data):
        """Spline kernel CI should approximate exp kernel CI."""
        ts = test_data["batch_single"]
        T = torch.tensor([3.0], dtype=torch.float64).to(DEVICE)

        ci_exp = exp_kernel.cumulative_intensity(T, ts)
        ci_spline = matching_spline_kernel.cumulative_intensity(T, ts)

        assert torch.allclose(ci_exp, ci_spline, rtol=0.05, atol=1e-3), (
            f"CI mismatch:\nexp={ci_exp}\nspline={ci_spline}\nrel_diff={(ci_exp - ci_spline).abs() / ci_exp.abs()}"
        )

    def test_pli_approx(self, exp_kernel, matching_spline_kernel, test_data):
        """Spline kernel PLI should approximate exp kernel PLI."""
        ts = test_data["batch_single"]

        pli_exp = exp_kernel.event_intensities(ts)
        pli_spline = matching_spline_kernel.event_intensities(ts)

        # Only compare non-zero entries (first event is always 0)
        mask = pli_exp.abs() > 1e-8

        assert torch.allclose(pli_exp[mask], pli_spline[mask], rtol=0.05, atol=1e-4), (
            f"PLI mismatch:\nexp={pli_exp}\nspline={pli_spline}"
        )
