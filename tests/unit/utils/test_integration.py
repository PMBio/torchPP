"""Direct unit tests for the numerical-integration helpers in ``torchpp.utils.math``.

``trapezoidal_integration``, ``monte_carlo_trapezoidal_integration``,
``monte_carlo_integration`` and the ``numerical_integration`` dispatcher all take
an ``intensity_func(t, ts) -> (B, M, D)`` — a single multi-query pass over the
``(B, M)`` grid — and a ``BatchedMVEventData`` scaffold. We pass intensities that
ignore ``ts`` so the integral has a known closed form, and build a minimal batched
event container only to satisfy the signature (the helpers read its batch size and
device).
"""

import pytest
import torch
from helpers import DEVICE

from torchpp.utils.event_data import MVEventData, BatchedMVEventData
from torchpp.utils.math import (
    monte_carlo_integration,
    monte_carlo_trapezoidal_integration,
    numerical_integration,
    trapezoidal_integration,
)


def _make_ts(B: int) -> BatchedMVEventData:
    """A minimal B-sequence batch; the integration helpers only use its shape/device."""
    seqs = [
        MVEventData(
            time_points=torch.tensor([0.1, 0.5], device=DEVICE),
            event_types=torch.tensor([0, 1], device=DEVICE),
        )
        for _ in range(B)
    ]
    return BatchedMVEventData(mv_events=seqs)


def _const_intensity(value=2.0, D=2):
    """Constant intensity: multi-query ``(B, M) -> (B, M, D)`` -> integral = value*(t_end-t_start)."""

    def f(t, ts):
        return torch.full((*t.shape, D), value, device=t.device, dtype=t.dtype)

    return f


def _linear_intensity(D=1):
    """intensity = t, broadcast across D: multi-query ``(B, M) -> (B, M, D)`` -> ∫_{a}^{b} t dt = (b²-a²)/2."""

    def f(t, ts):
        return t.unsqueeze(-1).expand(*t.shape, D).clone()

    return f


class TestTrapezoidal:
    def test_constant_integrand_exact(self):
        """Trapezoid of a constant equals value * interval, per type."""
        B, D = 3, 2
        ts = _make_ts(B)
        t_start = torch.zeros(B, device=DEVICE)
        t_end = torch.full((B,), 4.0, device=DEVICE)
        out = trapezoidal_integration(_const_intensity(2.0, D), t_start, t_end, ts, num_points=50)
        assert out.shape == (B, D)
        assert torch.allclose(out, torch.full_like(out, 8.0), atol=1e-4)

    def test_linear_integrand_matches_closed_form(self):
        """Trapezoid of t over [0, T] equals T²/2 (exact for a fine grid)."""
        B = 2
        ts = _make_ts(B)
        t_start = torch.zeros(B, device=DEVICE)
        t_end = torch.full((B,), 3.0, device=DEVICE)
        out = trapezoidal_integration(_linear_intensity(1), t_start, t_end, ts, num_points=200).squeeze(1)
        assert torch.allclose(out, torch.full_like(out, 4.5), atol=1e-3)

    def test_non_negative_and_increases_with_t_end(self):
        """A non-negative integrand gives non-negative integrals that grow with t_end."""
        B, D = 2, 1
        ts = _make_ts(B)
        t_start = torch.zeros(B, device=DEVICE)
        small = trapezoidal_integration(_const_intensity(1.5, D), t_start, torch.full((B,), 1.0, device=DEVICE), ts)
        large = trapezoidal_integration(_const_intensity(1.5, D), t_start, torch.full((B,), 5.0, device=DEVICE), ts)
        assert (small >= 0).all()
        assert (large > small).all()


class TestMonteCarlo:
    def test_mc_matches_trapezoid_for_smooth_integrand(self):
        """Seeded MC of a smooth integrand approximates the trapezoidal value."""
        B, D = 1, 1
        ts = _make_ts(B)
        t_start = torch.zeros(B, device=DEVICE)
        t_end = torch.full((B,), 3.0, device=DEVICE)
        trap = trapezoidal_integration(_linear_intensity(D), t_start, t_end, ts, num_points=500)
        rng = torch.Generator(device=DEVICE).manual_seed(0)
        mc = monte_carlo_integration(_linear_intensity(D), t_start, t_end, ts, num_points=20000, rng=rng)
        assert mc.item() == pytest.approx(trap.item(), rel=0.05)

    def test_mc_non_negative(self):
        B, D = 2, 2
        ts = _make_ts(B)
        rng = torch.Generator(device=DEVICE).manual_seed(1)
        out = monte_carlo_integration(
            _const_intensity(2.0, D), torch.zeros(B, device=DEVICE), torch.full((B,), 2.0, device=DEVICE), ts,
            num_points=100, rng=rng,
        )
        assert (out >= 0).all()
        assert torch.allclose(out, torch.full_like(out, 4.0), atol=1e-5)  # constant -> exact regardless of samples

    def test_mc_trapezoidal_constant_exact(self):
        """MC-trapezoidal of a constant is exact (constant cancels the random nodes)."""
        B, D = 2, 1
        ts = _make_ts(B)
        rng = torch.Generator(device=DEVICE).manual_seed(2)
        out = monte_carlo_trapezoidal_integration(
            _const_intensity(3.0, D), torch.zeros(B, device=DEVICE), torch.full((B,), 2.0, device=DEVICE), ts,
            num_points=50, rng=rng,
        )
        assert torch.allclose(out, torch.full_like(out, 6.0), atol=1e-4)


class TestDispatcher:
    def test_dispatch_methods_agree_on_constant(self):
        B, D = 1, 1
        ts = _make_ts(B)
        t_start = torch.zeros(B, device=DEVICE)
        t_end = torch.full((B,), 2.0, device=DEVICE)
        rng = torch.Generator(device=DEVICE).manual_seed(3)
        for method in ("trapezoidal", "mc_trapezoidal", "monte_carlo"):
            out = numerical_integration(
                _const_intensity(2.0, D), t_start, t_end, ts, method=method, num_points=50, rng=rng
            )
            assert torch.allclose(out, torch.full_like(out, 4.0), atol=1e-4), method

    def test_dispatch_unknown_method_raises(self):
        ts = _make_ts(1)
        with pytest.raises(ValueError):
            numerical_integration(
                _const_intensity(), torch.zeros(1, device=DEVICE), torch.ones(1, device=DEVICE), ts, method="nope"
            )
