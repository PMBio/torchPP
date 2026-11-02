"""Tests for SplineKernelModule and PiecewiseConstantKernelModule initialisation.

Checks that:
1. Knot heights follow a clearly exponentially decaying profile (h[0] >> h[-1]).
2. The total kernel integral (branching ratio per pair) is close to the target 0.5.
3. The decay timescale is consistent with the knot range (1–5 decay lengths fit).
4. The dataset_stats path uses beta = 1/mean_inter_event_time and still produces
   a proper exponential shape.
5. Both uniform and geometric knot spacing work correctly.
"""

import pytest
import torch
from torchpp.modules.intensity import PiecewiseConstantKernelModule, SplineKernelModule
from torchpp.utils.event_data import DatasetStats
from torchpp.utils.math import LinearSpline


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _spline_integral(kernel: SplineKernelModule, recv: int, trig: int) -> float:
    """Integrate the spline kernel φ_{recv,trig} from 0 to tail_end."""
    h_knots, knot_locs = kernel.transform_params()
    selected_h = h_knots[recv, trig, :].unsqueeze(0)  # (1, K+1)
    tail_end = knot_locs[-1]
    t = tail_end.unsqueeze(0)  # (1,)
    integral = LinearSpline.integrate(knot_locs, selected_h, t)  # (1, 1)
    return float(integral[0, 0].detach())


def _pwc_integral(kernel: PiecewiseConstantKernelModule, recv: int, trig: int) -> float:
    """Integrate the PWC kernel φ_{recv,trig} from 0 to tail_end."""
    h_knots, _ = kernel.transform_params()
    regular_integral = float((h_knots[recv, trig, :-2] * kernel.bin_widths).sum().detach())
    tail_height = float(h_knots[recv, trig, -2].detach())  # tail has same height as last regular bin
    tail_width = float((kernel.tail_end - kernel.tail_start).detach())
    return regular_integral + tail_height * tail_width


def _decay_ratio(heights: torch.Tensor) -> float:
    """h[0] / h[-1] — should be >> 1 for a clearly decaying profile."""
    h = heights.detach()
    return float(h[0]) / (float(h[-1]) + 1e-12)


def _make_dataset_stats(D: int, mean_iet: float, terminating: bool = True) -> DatasetStats:
    return DatasetStats(
        D=D,
        mean_first_occ_time=torch.full((D,), mean_iet),
        mean_rate=torch.full((D,), 1.0 / mean_iet),
        mean_inter_event_time=mean_iet,
        type_distribution=torch.full((D,), 1.0 / D),
        mean_events_per_sequence=5.0,
        mean_sequence_time=1.0,
        sequence_time_p0=0.1,
        sequence_time_p50=0.5,
        sequence_time_p75=0.7,
        sequence_time_p90=0.85,
        sequence_time_p95=0.9,
        sequence_time_p100=1.0,
        is_terminating=terminating,
    )


# ---------------------------------------------------------------------------
# SplineKernelModule
# ---------------------------------------------------------------------------


class TestSplineKernelInit:
    """Verify SplineKernelModule random initialisation properties."""

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_heights_are_positive(self, seed):
        """All regular knot heights must be positive."""
        k = SplineKernelModule(D=4, num_knots=8, delta_t=0.1, seed=seed)
        h, _ = k.transform_params()
        regular_h = h[:, :, :-1]  # exclude tail knot (height=0)
        assert (regular_h > 0).all(), "All regular knot heights should be positive"

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_heights_are_exponentially_decaying(self, seed):
        """h[0] should be substantially larger than h[-2] (the last non-zero knot).

        With the fixed beta ~ Uniform[1,5] / T_max, the ratio h[0]/h[-2]
        should be at least e^1 ≈ 2.7 (one decay length) for all pairs.
        """
        k = SplineKernelModule(D=4, num_knots=8, delta_t=0.1, seed=seed)
        h, _ = k.transform_params()
        for d in range(4):
            for j in range(4):
                ratio = _decay_ratio(h[d, j, :-1])  # exclude zero tail knot
                assert ratio > 2.5, (
                    f"Pair ({d},{j}) seed={seed}: h[0]/h[-2]={ratio:.2f} < 2.5 — not a clearly decaying exponential"
                )

    def test_branching_ratio_close_to_target(self):
        """Total integral of each pair should be close to the target branching ratio.

        For the no-dataset_stats path the alpha_weight matrix is spectral-normalised
        to 0.5, so the largest eigenvalue of the integral matrix should be ~0.5.
        Individual pairs will vary, but the mean should be in a reasonable range.
        """
        D = 4
        k = SplineKernelModule(D=D, num_knots=10, delta_t=0.1, seed=0)
        integrals = torch.zeros(D, D)
        for r in range(D):
            for t in range(D):
                integrals[r, t] = _spline_integral(k, r, t)

        max_eig = float(torch.max(torch.abs(torch.linalg.eigvals(integrals))).real)
        assert 0.1 < max_eig < 2.0, f"Spectral radius of integral matrix = {max_eig:.3f}, expected in (0.1, 2.0)"

    @pytest.mark.parametrize("num_knots,delta_t", [(6, 0.05), (10, 0.2), (12, 0.1)])
    def test_decay_within_knot_range(self, num_knots, delta_t):
        """Beta should be scaled to T_max so the decay is visible within the knot range.

        Concretely, h[0] / h[K//2] should be > 1.5 for all pairs (visible half-decay).
        """
        k = SplineKernelModule(D=3, num_knots=num_knots, delta_t=delta_t, seed=7)
        h, _ = k.transform_params()
        mid = num_knots // 2
        for d in range(3):
            for j in range(3):
                ratio = float(h[d, j, 0].detach()) / (float(h[d, j, mid].detach()) + 1e-12)
                assert ratio > 1.5, (
                    f"num_knots={num_knots}, delta_t={delta_t}, pair ({d},{j}): h[0]/h[mid]={ratio:.2f} < 1.5"
                )

    def test_dataset_stats_beta_from_mean_iet(self):
        """With dataset_stats, beta = 1/mean_IET. If T_max >= 3/beta the ratio
        if T_max >= 3/beta (i.e., delta_t*num_knots >= 3*mean_IET).
        """
        D = 3
        mean_iet = 0.1  # natural timescale
        stats = _make_dataset_stats(D, mean_iet)

        # Choose knot range to cover 3 decay lengths: T_max = 3 * mean_IET = 0.3
        k = SplineKernelModule(D=D, num_knots=6, delta_t=0.05, seed=0, dataset_stats=stats)
        h, _ = k.transform_params()
        # After 3 decay lengths: h should drop by factor e^3 ≈ 20
        for d in range(D):
            for j in range(D):
                ratio = _decay_ratio(h[d, j, :-1])
                assert ratio > 5.0, f"dataset_stats path, pair ({d},{j}): h[0]/h[-2]={ratio:.2f} < 5.0"

    def test_geometric_knot_spacing(self):
        """Geometric spacing should also produce a decaying init."""
        # Geometric spacings: 0.02, 0.04, 0.08, 0.16, 0.32 (5 gaps → 6 knots)
        spacings = torch.tensor([0.02, 0.04, 0.08, 0.16, 0.32])
        k = SplineKernelModule(D=3, num_knots=6, delta_t=spacings, seed=5)
        h, locs = k.transform_params()
        assert locs.shape[0] == 7, "Should have K+1=7 knot locations (6 regular + tail)"
        for d in range(3):
            for j in range(3):
                ratio = _decay_ratio(h[d, j, :-1])
                assert ratio > 2.5, f"Geometric spacing, pair ({d},{j}): ratio={ratio:.2f} < 2.5"


# ---------------------------------------------------------------------------
# PiecewiseConstantKernelModule
# ---------------------------------------------------------------------------


class TestPWCKernelInit:
    """Verify PiecewiseConstantKernelModule random initialisation properties."""

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_heights_are_positive(self, seed):
        """All regular bin heights must be positive."""
        k = PiecewiseConstantKernelModule(D=4, num_knots=8, delta_t=0.1, seed=seed)
        h, _ = k.transform_params()
        regular_h = h[:, :, :-2]  # exclude tail and zero-beyond slots
        assert (regular_h > 0).all(), "All regular bin heights should be positive"

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_heights_are_exponentially_decaying(self, seed):
        """h[0] / h[-1] (last regular bin) should be >> 1 for all pairs."""
        k = PiecewiseConstantKernelModule(D=4, num_knots=8, delta_t=0.1, seed=seed)
        h, _ = k.transform_params()
        for d in range(4):
            for j in range(4):
                ratio = _decay_ratio(h[d, j, :-2])
                assert ratio > 2.5, f"Pair ({d},{j}) seed={seed}: h[0]/h[-1]={ratio:.2f} < 2.5"

    def test_branching_ratio_close_to_target(self):
        """Spectral radius of the bin-integral matrix should be ~0.5."""
        D = 4
        k = PiecewiseConstantKernelModule(D=D, num_knots=10, delta_t=0.1, seed=0)
        integrals = torch.zeros(D, D)
        for r in range(D):
            for t in range(D):
                integrals[r, t] = _pwc_integral(k, r, t)

        max_eig = float(torch.max(torch.abs(torch.linalg.eigvals(integrals))).real)
        assert 0.1 < max_eig < 2.0, f"Spectral radius of integral matrix = {max_eig:.3f}, expected in (0.1, 2.0)"

    @pytest.mark.parametrize("num_knots,delta_t", [(6, 0.05), (10, 0.2), (12, 0.1)])
    def test_decay_within_bin_range(self, num_knots, delta_t):
        """h[0] / h[K//2] > 1.5 so the half-decay is visible."""
        k = PiecewiseConstantKernelModule(D=3, num_knots=num_knots, delta_t=delta_t, seed=7)
        h, _ = k.transform_params()
        mid = num_knots // 2
        for d in range(3):
            for j in range(3):
                ratio = float(h[d, j, 0].detach()) / (float(h[d, j, mid].detach()) + 1e-12)
                assert ratio > 1.5, (
                    f"num_knots={num_knots}, delta_t={delta_t}, pair ({d},{j}): h[0]/h[mid]={ratio:.2f} < 1.5"
                )

    def test_dataset_stats_beta_from_mean_iet(self):
        """With dataset_stats, kernel should strongly decay if T_max >= 3/beta."""
        D = 3
        mean_iet = 0.1
        stats = _make_dataset_stats(D, mean_iet)

        k = PiecewiseConstantKernelModule(D=D, num_knots=6, delta_t=0.05, seed=0, dataset_stats=stats)
        h, _ = k.transform_params()
        for d in range(D):
            for j in range(D):
                ratio = _decay_ratio(h[d, j, :-2])
                assert ratio > 5.0, f"dataset_stats path, pair ({d},{j}): ratio={ratio:.2f} < 5.0"

    def test_tail_height_equals_last_regular_bin(self):
        """The tail bin height must equal the last regular bin height (constant extrapolation)."""
        k = PiecewiseConstantKernelModule(D=3, num_knots=6, delta_t=0.1, seed=0)
        h, _ = k.transform_params()
        # Slot K (index -2) = tail, should equal slot K-1 (index -3)
        torch.testing.assert_close(h[:, :, -2], h[:, :, -3], msg="Tail slot must equal last regular bin height")

    def test_zero_beyond_tail(self):
        """The beyond-tail slot must be exactly zero."""
        k = PiecewiseConstantKernelModule(D=3, num_knots=6, delta_t=0.1, seed=0)
        h, _ = k.transform_params()
        assert (h[:, :, -1] == 0).all(), "Beyond-tail slot must be zero"
