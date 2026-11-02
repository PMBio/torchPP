"""CI-construction equivalence matrix.

Three distinct kinds of check — kept separate because they hold to different
standards:

1. **Closed-form exact** — ``Σ_i _interval_ci ≡ _raw_ci`` for every closed-form
   family (the deprecated ``_raw_ci`` is the oracle).  The per-interval
   decomposition over ``interval_bounds`` must reconstruct the global analytic CI
   up to float re-association.
2. **Same-scheme exact** — forcing a ``CIConfig`` to a model's *prior* scheme
   reproduces the prior path's numbers exactly (global → ``_numerical_cumulative_intensity``;
   interval survival → ``_raw_ci``).
3. **Cross-scheme / statistical** — global vs interval (vs closed-form) differ only
   by quadrature error (quantified, not a failure); ``mc``/``mc_trapezoidal`` match
   the deterministic rule within Monte-Carlo tolerance at a fixed seed.

Models are run in float64 so the exact checks have a meaningful (re-association)
tolerance.
"""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from torchpp.tpp.base_tpp import CIConfig
from torchpp.utils.event_data import BatchedMVEventData

from helpers import create_model, MODEL_TYPES, DEVICE, skip_if_density


def _fix(m, ts):
    return m.fix_context(ts) if hasattr(m, "fix_context") else nullcontext()


def _batch():
    return BatchedMVEventData(
        [torch.tensor([0.5, 1.2, 2.0, 3.1], dtype=torch.float64), torch.tensor([0.3, 0.9, 1.5], dtype=torch.float64)],
        [torch.tensor([0, 2, 1, 0]), torch.tensor([1, 1, 2])],
    ).to(DEVICE)


def _model(model_type):
    """Model (native float32 heads, float64 batch — the standard setup) + T.

    Neural heads manage their own float32/float64 casting internally, so we do not
    ``.double()`` the model; the float64 batch keeps the time arithmetic precise.
    """
    m = create_model(model_type, terminating=False)
    m.eval()
    ts = _batch()
    return m, ts, ts.max_time + 0.5


# ---------------------------------------------------------------------------
# 1. Closed-form exact: Σ _interval_ci ≡ _raw_ci
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_interval_ci_sums_to_raw_ci(model_type):
    m, ts, T = _model(model_type)
    skip_if_density(m)  # density models have no CIConfig / _interval_ci
    if m.ci_config.mode != "closed_form":
        pytest.skip(f"{model_type} is not a closed-form family")
    with _fix(m, ts), torch.no_grad():
        lo, hi = m.interval_bounds(ts, None, T)  # (B, L+1) over [0, T]
        try:
            per_interval = m._interval_ci(ts, lo, hi).sum(dim=1)  # (B, D)
        except NotImplementedError:
            pytest.skip(f"{model_type} provides only a global closed-form CI (no per-interval hook)")
        oracle = m.analytical_cumulative_intensity(T, ts)  # = _raw_ci(T), the global form
    assert torch.allclose(per_interval, oracle, rtol=1e-6, atol=1e-8), (
        model_type,
        (per_interval - oracle).abs().max().item(),
    )


# ---------------------------------------------------------------------------
# 2. Same-scheme exact: forcing the prior scheme reproduces the prior path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_global_numerical_matches_legacy_path(model_type):
    """``cumulative_intensity`` with a global CIConfig == ``_numerical_cumulative_intensity``
    with identical (deterministic) nodes — exact, for every model."""
    m, ts, T = _model(model_type)
    skip_if_density(m)  # density models have no cumulative_intensity
    T_low = ts.max_time
    cfg = CIConfig(mode="numerical", placement="global", rule="trapezoidal", n_points=64)
    with _fix(m, ts), torch.no_grad():
        new = m.cumulative_intensity(T, ts, T_low=T_low, ci_config=cfg)
        legacy = m._numerical_cumulative_intensity(T, ts, T_low=T_low, method="trapezoidal", num_points=64)
    assert torch.equal(new, legacy), (model_type, (new - legacy).abs().max().item())


@pytest.mark.parametrize("model_type", ["nhp", "attnhp"])
def test_interval_survival_quadrature_converged(model_type):
    """The intensity-numerical per-interval survival is a converged trapezoid.

    These models have no closed-form / ``_raw_ci`` oracle — their CI *is* the
    event-knotted grid over ``_interval_intensities`` (NHP's state-decay fold /
    AttNHP's batch-fold).  Check the quadrature has converged: refining the node
    count barely moves the survival.  (Absolute correctness vs a fine reference is
    covered in test_new_models::test_per_interval_survival_quadrature.)
    """
    m, ts, T = _model(model_type)
    coarse = CIConfig(mode="numerical", placement="interval", rule="trapezoidal", n_points=50)
    fine = CIConfig(mode="numerical", placement="interval", rule="trapezoidal", n_points=400)
    with m.fix_context(ts), torch.no_grad():
        ci_coarse = m.cumulative_intensity(T, ts, ci_config=coarse)
        ci_fine = m.cumulative_intensity(T, ts, ci_config=fine)
    assert torch.allclose(ci_coarse, ci_fine, rtol=1e-2, atol=1e-3), (ci_coarse - ci_fine).abs().max().item()


# ---------------------------------------------------------------------------
# 3a. Cross-scheme: global vs interval converge to the closed form (quadrature)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", ["neural_poisson", "neural_thp", "hawkes"])
def test_numerical_converges_to_closed_form(model_type):
    """Both numerical placements approach the closed form as n_points grows; report
    the (expected) cross-scheme quadrature gap."""
    m, ts, T = _model(model_type)
    T_low = ts.max_time
    with _fix(m, ts), torch.no_grad():
        exact = m.cumulative_intensity(T, ts, T_low=T_low, ci_config=CIConfig("closed_form")).sum()
        glob = m.cumulative_intensity(
            T, ts, T_low=T_low, ci_config=CIConfig("numerical", placement="global", rule="trapezoidal", n_points=400)
        ).sum()
        intv = m.cumulative_intensity(
            T, ts, T_low=T_low, ci_config=CIConfig("numerical", placement="interval", rule="trapezoidal", n_points=400)
        ).sum()
    assert torch.allclose(glob, exact, rtol=1e-3, atol=1e-3), (model_type, "global", (glob - exact).item())
    assert torch.allclose(intv, exact, rtol=1e-3, atol=1e-3), (model_type, "interval", (intv - exact).item())


# ---------------------------------------------------------------------------
# 3b. Statistical: mc / mc_trapezoidal match the deterministic rule within MC error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", ["mc", "mc_trapezoidal"])
def test_mc_rules_match_trapezoidal_within_tolerance(rule):
    """A fixed-seed MC estimate of the interval CI is close to the trapezoid value;
    distinct from the exact tests — this is statistical."""
    m, ts, T = _model("nhp")
    T_low = ts.max_time
    torch.manual_seed(0)
    with m.fix_context(ts), torch.no_grad():
        det = m.cumulative_intensity(
            T, ts, T_low=T_low, ci_config=CIConfig("numerical", placement="interval", rule="trapezoidal", n_points=200)
        ).sum()
        mc = m.cumulative_intensity(
            T, ts, T_low=T_low, ci_config=CIConfig("numerical", placement="interval", rule=rule, n_points=2000)
        ).sum()
    assert torch.allclose(mc, det, rtol=0.05, atol=1e-2), (rule, (mc - det).item())
