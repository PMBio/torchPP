"""Direct unit tests for kernel intensity modules.

Constructs each kernel module directly (no model factory) and checks its genuine
contract:

- ``ExpKernelModule``: intensity = Σ_{t_i < t} Σ_j α·β·exp(−β·Δt); strictly
  causal (no contribution from the present or future); closed-form cumulative
  intensity Λ = Σ_{t_i < T} α·(1 − exp(−β·(T − t_i))) matching a hand-computed
  value and a fine numerical trapezoid.
- ``PiecewiseConstantKernelModule``: step kernel over Δt; shapes, non-negativity,
  causality, and analytical CI matching the closed-form rectangle accumulation.

``SplineKernelModule`` is covered separately in test_spline_kernel.py.
"""

import pytest
import torch
from helpers import DEVICE

from torchpp.modules.intensity import (
    ExpKernelModule,
    ExpKernelParams,
    PiecewiseConstantKernelModule,
    PiecewiseConstantKernelParams,
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


@pytest.fixture
def single():
    """A single sequence for analytical / numerical CI checks."""
    seq = MVEventData(
        torch.tensor([0.2, 0.8, 2.0, 2.3], dtype=torch.float64).to(DEVICE),
        torch.tensor([0, 1, 2, 0], dtype=torch.long).to(DEVICE),
    )
    return BatchedMVEventData([seq.time_points], [seq.event_types]).to(DEVICE)


# =============================================================================
# ExpKernelModule
# =============================================================================


class TestExpKernel:
    def _module(self):
        # Explicit, well-conditioned alpha/beta so the analytical form is exact.
        alpha = torch.tensor(
            [
                [0.3, 0.1, 0.2],
                [0.2, 0.4, 0.1],
                [0.1, 0.2, 0.3],
            ]
        )
        beta = torch.tensor(
            [
                [1.0, 1.5, 2.0],
                [2.0, 1.0, 1.5],
                [1.5, 2.0, 1.0],
            ]
        )
        return ExpKernelModule(D=D, params=ExpKernelParams(alpha=alpha, beta=beta)).to(DEVICE)

    def test_intensity_shape_positive(self, batch):
        m = self._module()
        t = torch.tensor([3.0, 3.5], dtype=torch.float64).to(DEVICE)
        lam = m.intensity(t, batch)
        assert lam.shape == (2, D)
        assert (lam >= 0).all()

    def test_intensity_matches_explicit_sum(self, single):
        """λ_d(t) = Σ_{t_i < t} α[d, type_i]·β[d, type_i]·exp(−β·(t − t_i))."""
        m = self._module()
        alpha, beta = m.transform_params()
        alpha, beta = alpha.double(), beta.double()

        times = single.time_points[0].double()  # (N,)
        types = single.event_types[0]  # (N,)
        t_query = 1.5
        t = torch.tensor([t_query], dtype=torch.float64).to(DEVICE)

        lam = m.intensity(t, single)[0].double()  # (D,)

        expected = torch.zeros(D, dtype=torch.float64).to(DEVICE)
        for i in range(times.shape[0]):
            if times[i] < t_query:  # strictly past
                j = types[i].item()
                dt = t_query - times[i]
                expected += alpha[:, j] * beta[:, j] * torch.exp(-beta[:, j] * dt)
        assert torch.allclose(lam, expected, atol=1e-6)

    def test_causal_no_self_or_future_contribution(self):
        """An event at exactly t contributes nothing (Δt = 0 is not 'past')."""
        m = self._module()
        seq = MVEventData(
            torch.tensor([1.0], dtype=torch.float64).to(DEVICE),
            torch.tensor([0], dtype=torch.long).to(DEVICE),
        )
        ts = BatchedMVEventData([seq.time_points], [seq.event_types]).to(DEVICE)
        # At the event time itself: not strictly past -> zero.
        lam_at = m.intensity(torch.tensor([1.0], dtype=torch.float64).to(DEVICE), ts)
        assert torch.allclose(lam_at, torch.zeros_like(lam_at), atol=TOL)
        # Before the event: also zero (no past events).
        lam_before = m.intensity(torch.tensor([0.5], dtype=torch.float64).to(DEVICE), ts)
        assert torch.allclose(lam_before, torch.zeros_like(lam_before), atol=TOL)
        # Just after the event: strictly positive.
        lam_after = m.intensity(torch.tensor([1.0001], dtype=torch.float64).to(DEVICE), ts)
        assert (lam_after > 0).all()

    def test_empty_history_zero(self):
        m = self._module()
        ts = BatchedMVEventData(
            [torch.tensor([], dtype=torch.float64).to(DEVICE)],
            [torch.tensor([], dtype=torch.long).to(DEVICE)],
        )
        lam = m.intensity(torch.tensor([1.0]).to(DEVICE), ts)
        assert (lam == 0).all()

    def test_ci_closed_form(self, single):
        """Λ_d(T) = Σ_{t_i < T} α·(1 − exp(−β·(T − t_i))) over past events."""
        m = self._module()
        alpha, beta = m.transform_params()
        alpha, beta = alpha.double(), beta.double()

        times = single.time_points[0].double()
        types = single.event_types[0]
        T_val = 3.0
        T = torch.tensor([T_val], dtype=torch.float64).to(DEVICE)

        ci = m.cumulative_intensity(T, single)[0].double()

        expected = torch.zeros(D, dtype=torch.float64).to(DEVICE)
        for i in range(times.shape[0]):
            if times[i] < T_val:
                j = types[i].item()
                dt = T_val - times[i]
                expected += alpha[:, j] * (1.0 - torch.exp(-beta[:, j] * dt))
        assert torch.allclose(ci, expected, atol=1e-6)

    def test_ci_matches_numerical(self, single):
        m = self._module()
        T_val = 3.0
        T = torch.tensor([T_val], dtype=torch.float64).to(DEVICE)
        ci = m.cumulative_intensity(T, single).double()

        # Fine trapezoid of intensity over [0, T].
        n = 6000
        grid = torch.linspace(0.0, T_val, n, dtype=torch.float64).to(DEVICE)
        lam = torch.stack([m.intensity(grid[i : i + 1], single) for i in range(n)], dim=0).squeeze(1)  # (n, D)
        ci_num = torch.trapezoid(lam.double(), grid, dim=0).unsqueeze(0)
        assert torch.allclose(ci, ci_num, atol=1e-3)

    def test_ci_monotonic(self, single):
        m = self._module()
        ci1 = m.cumulative_intensity(torch.tensor([1.5]).to(DEVICE), single)
        ci2 = m.cumulative_intensity(torch.tensor([3.0]).to(DEVICE), single)
        assert (ci2 >= ci1 - 1e-6).all()

    def test_event_intensities(self, batch):
        m = self._module()
        ei = m.event_intensities(batch)
        B, L = batch.time_points.shape
        assert ei.shape == (B, L)
        assert (ei >= 0).all()
        # First event in each sequence has no past -> zero kernel contribution.
        assert ei[0, 0] == 0.0
        assert ei[1, 0] == 0.0
        # Padded entries are zero.
        assert (ei[~batch.event_mask] == 0).all()


# =============================================================================
# PiecewiseConstantKernelModule
# =============================================================================


class TestPiecewiseConstantKernel:
    NUM_KNOTS = 4
    DELTA_T = 0.5  # bins [0,.5), [.5,1), [1,1.5), [1.5,2); tail then zero

    def _module(self):
        # Decaying heights per (recv, trig) pair so the kernel is non-trivial.
        h = torch.zeros(D, D, self.NUM_KNOTS)
        for i in range(D):
            for j in range(D):
                base = 0.5 + 0.1 * (i + j)
                h[i, j] = base * torch.tensor([1.0, 0.7, 0.4, 0.2])
        params = PiecewiseConstantKernelParams(
            h_knots=h,
            delta_ts=torch.full((self.NUM_KNOTS,), self.DELTA_T),
        )
        return PiecewiseConstantKernelModule(
            D=D, num_knots=self.NUM_KNOTS, delta_t=self.DELTA_T, params=params
        ).to(DEVICE)

    def test_intensity_shape_positive(self, batch):
        m = self._module()
        t = torch.tensor([1.0, 1.3], dtype=torch.float64).to(DEVICE)
        lam = m.intensity(t, batch)
        assert lam.shape == (2, D)
        assert (lam >= 0).all()

    def test_intensity_step_within_bin(self):
        """Kernel value is constant within a Δt bin (zero-order hold)."""
        m = self._module()
        h, _ = m.transform_params()  # (D, D, K+2)
        seq = MVEventData(
            torch.tensor([0.0], dtype=torch.float64).to(DEVICE),
            torch.tensor([0], dtype=torch.long).to(DEVICE),
        )
        ts = BatchedMVEventData([seq.time_points], [seq.event_types]).to(DEVICE)
        # Single trigger at t=0; query Δt = 0.6 and 0.9, both in bin [0.5, 1.0).
        lam_a = m.intensity(torch.tensor([0.6], dtype=torch.float64).to(DEVICE), ts)
        lam_b = m.intensity(torch.tensor([0.9], dtype=torch.float64).to(DEVICE), ts)
        assert torch.allclose(lam_a, lam_b, atol=TOL)
        # Equals the second-bin height for trigger type 0 (all receiver dims).
        assert torch.allclose(lam_a[0].double(), h[:, 0, 1].double(), atol=1e-5)

    def test_causal(self):
        """No contribution at or before the trigger time."""
        m = self._module()
        seq = MVEventData(
            torch.tensor([1.0], dtype=torch.float64).to(DEVICE),
            torch.tensor([0], dtype=torch.long).to(DEVICE),
        )
        ts = BatchedMVEventData([seq.time_points], [seq.event_types]).to(DEVICE)
        lam_at = m.intensity(torch.tensor([1.0], dtype=torch.float64).to(DEVICE), ts)
        assert torch.allclose(lam_at, torch.zeros_like(lam_at), atol=TOL)
        lam_after = m.intensity(torch.tensor([1.2], dtype=torch.float64).to(DEVICE), ts)
        assert (lam_after > 0).all()

    def test_ci_rectangle_sum(self):
        """Λ_d(T) accumulates rectangle areas of the step kernel over Δt."""
        m = self._module()
        h, _ = m.transform_params()  # (D, D, K+2)
        seq = MVEventData(
            torch.tensor([0.0], dtype=torch.float64).to(DEVICE),
            torch.tensor([0], dtype=torch.long).to(DEVICE),
        )
        ts = BatchedMVEventData([seq.time_points], [seq.event_types]).to(DEVICE)
        # T = 1.3 -> Δt = 1.3: full bins [0,.5) and [.5,1) plus 0.3 of [1,1.5).
        T = torch.full((1, D), 1.3, dtype=torch.float64).to(DEVICE)
        ci = m.cumulative_intensity(T, ts)[0].double()
        expected = (h[:, 0, 0] * 0.5 + h[:, 0, 1] * 0.5 + h[:, 0, 2] * 0.3).double()
        assert torch.allclose(ci, expected, atol=1e-5)

    def test_ci_accepts_1d_T(self, single):
        """A 1-D ``(B,)`` T is normalised to ``(B, D)`` like the base class / sibling kernels."""
        m = self._module()
        ci_1d = m.cumulative_intensity(torch.tensor([1.5]).to(DEVICE), single)
        ci_2d = m.cumulative_intensity(torch.full((1, D), 1.5).to(DEVICE), single)
        assert ci_1d.shape == (1, D)
        assert torch.allclose(ci_1d, ci_2d, atol=1e-6)

    def test_ci_monotonic(self, single):
        m = self._module()
        ci1 = m.cumulative_intensity(torch.tensor([1.0]).to(DEVICE), single)
        ci2 = m.cumulative_intensity(torch.tensor([2.0]).to(DEVICE), single)
        assert (ci2 >= ci1 - 1e-6).all()

    def test_event_intensities(self, batch):
        m = self._module()
        ei = m.event_intensities(batch)
        B, L = batch.time_points.shape
        assert ei.shape == (B, L)
        assert (ei >= 0).all()
        assert ei[0, 0] == 0.0
        assert (ei[~batch.event_mask] == 0).all()
