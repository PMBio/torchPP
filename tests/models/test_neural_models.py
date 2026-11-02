"""Per-backend CI-value trust-chain checks for the continuous-time neural TPPs.

NHP / AttNHP model-specific checks the parameterised suite cannot express:
intensity positivity, causal-masking consistency (query-at-event matches the
sequence intensities), numerical CI vs a trapezoid reference, and the
per-interval survival quadrature feeding the likelihood's survival term.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from torchpp.modules.nn.encoder import ContTimeLSTMEncoder
from torchpp.modules.nn.attnhp import AttNHPEncoder
from torchpp.tpp.neural import NHPProcess, AttNHPProcess
from torchpp.utils.event_data import BatchedMVEventData


def _batch():
    return BatchedMVEventData(
        [torch.tensor([0.5, 1.2, 2.0, 3.1], dtype=torch.float64), torch.tensor([0.3, 0.9, 1.5], dtype=torch.float64)],
        [torch.tensor([0, 2, 1, 0]), torch.tensor([1, 1, 2])],
    )


def _nhp(d=3, seed=0):
    torch.manual_seed(seed)
    m = NHPProcess(ContTimeLSTMEncoder(d, hidden_dim=16), seed=seed)
    m.ci_config = replace(m.ci_config, n_points=50)
    return m


def _attnhp(d=3, seed=0):
    torch.manual_seed(seed)
    m = AttNHPProcess(AttNHPEncoder(d, d_model=16, d_time=8, n_layers=1, n_head=2), seed=seed)
    m.ci_config = replace(m.ci_config, n_points=50)
    return m


class TestContinuousTimeNeuralModels:
    """NHP and AttNHP: intensity positivity, causal-masking consistency, numerical CI."""

    @pytest.mark.parametrize("build", [_nhp, _attnhp], ids=["nhp", "attnhp"])
    def test_intensity_positive_and_finite(self, build):
        m = build()
        ts = _batch()
        with m.fix_context(ts):
            lam = m.intensity(torch.tensor([2.5, 1.2], dtype=torch.float64), ts)
            assert torch.isfinite(lam).all() and (lam > 0).all()
            assert torch.isfinite(m.likelihood(ts)).all()

    @pytest.mark.parametrize("build", [_nhp, _attnhp], ids=["nhp", "attnhp"])
    def test_query_at_event_matches_sequence(self, build):
        """_intensity_at_t at an event time uses strictly-before history, so it must
        equal _sequence_intensities at that position (validates causal masking)."""
        m = build()
        ts = _batch()
        with m.fix_context(ts):
            seq = m._sequence_intensities(ts)  # (B, L, D)
            for l in range(ts.time_points.shape[1]):
                lam = m._intensity_at_t(ts.time_points[:, l], ts)  # (B, D)
                assert torch.allclose(lam, seq[:, l, :], atol=1e-5)

    @pytest.mark.parametrize("build", [_nhp, _attnhp], ids=["nhp", "attnhp"])
    def test_numerical_ci_matches_trapezoid(self, build):
        """cumulative_intensity (numerical) ~ trapezoid of intensity over [t_last, T]."""
        m = build()
        ts = _batch()
        with m.fix_context(ts):
            t_last = ts.max_time
            T = t_last + 0.7
            ci = m.cumulative_intensity(T, ts, T_low=t_last).sum(1)  # (B,)
            grid = torch.linspace(0, 1, 400, dtype=torch.float64)
            ts_grid = t_last.unsqueeze(1) + grid.unsqueeze(0) * (T - t_last).unsqueeze(1)  # (B, G)
            lam = torch.stack(
                [m.intensity(ts_grid[:, g], ts).sum(1) for g in range(ts_grid.shape[1])], dim=1
            )  # (B, G)
            ref = torch.trapezoid(lam, ts_grid, dim=1)  # (B,)
            assert torch.allclose(ci, ref, rtol=0.05, atol=1e-3), (ci, ref)

    @pytest.mark.parametrize("build", [_nhp, _attnhp], ids=["nhp", "attnhp"])
    def test_per_interval_survival_quadrature(self, build):
        """Per-interval survival CI (the likelihood's survival term) matches a fine knotted reference."""
        m = build()
        ts = _batch()
        T = ts.max_time + 0.5
        with m.fix_context(ts):
            assert m.ci_config.mode == "numerical" and m.ci_config.placement == "interval"
            # The survival CI now flows through the unified per-interval dispatch
            # (cumulative_intensity -> _interval_intensities); there is no separate
            # _raw_ci / analytical oracle for these intensity-numerical models.
            ci = m.cumulative_intensity(T, ts).sum(1)  # (B,)

            # Reference: fine per-interval trapezoid via the public intensity.
            ref = torch.zeros_like(ci)
            tp = ts.time_points
            for b in range(tp.shape[0]):
                knots = [0.0] + [float(t) for t in tp[b][ts.event_mask[b]]] + [float(T[b])]
                for lo, hi in zip(knots[:-1], knots[1:]):
                    if hi <= lo:
                        continue
                    g = torch.linspace(lo, hi, 200, dtype=torch.float64)
                    # offset interior points to stay inside the open interval
                    g[0] = g[0] + 1e-9
                    lam = torch.stack([m.intensity(g[k].expand(tp.shape[0]), ts).sum(1)[b] for k in range(len(g))])
                    ref[b] += torch.trapezoid(lam, g)

            assert torch.allclose(ci, ref, rtol=0.02, atol=1e-3), (ci, ref)
