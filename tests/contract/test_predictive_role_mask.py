"""Sequence-representation role mask: the intensity-family *predictive* contract
excludes context types, consistently with the likelihood's masked survival.

``IntensityBasedTPP._likelihood_scaled`` sums the survival Σ_d Λ_d over predictable
type-columns only.  The public predictive API must agree: ``survival_time`` / ``cdf_time``
/ ``density_joint`` (via ``_windowed_total_ci``) and the grid marginals ``prob_mark`` /
``time_mean`` / ``time_quantile`` (grids grown on the predictable types) all drop context columns.

Under the default all-True mask every masking site is a no-op — the bit-identity is the
regression guard (Gate: nothing else may move).  These tests exercise the *active* mask.
"""

import pytest
import torch

from helpers import create_model, SEED, DEVICE
from torchpp.utils.event_data import BatchedMVEventData

# Intensity-family models spanning the CI dispatch: closed-form (poisson/hawkes),
# numerical global (inhibitive), grid/neural (nhp).
_MODELS = ["poisson", "const_exp_hawkes", "inhibitive_const_exp_hawkes", "nhp"]

_CTX = 2  # context/read-only type


def _batch():
    times = [
        torch.tensor([0.2, 0.8, 1.5], dtype=torch.float64, device=DEVICE),
        torch.tensor([0.1, 1.2, 2.0], dtype=torch.float64, device=DEVICE),
    ]
    types = [
        torch.tensor([0, 1, _CTX], dtype=torch.long, device=DEVICE),
        torch.tensor([_CTX, 0, 1], dtype=torch.long, device=DEVICE),
    ]
    return BatchedMVEventData(times, types).to(DEVICE)


def _set_ctx(model, k):
    mask = torch.ones(model.D, dtype=torch.bool, device=model.predictable_mask.device)
    mask[k] = False
    model.predictable_mask.copy_(mask)  # in-place: preserves the Hawkes kernel alias


@pytest.mark.parametrize("model_type", _MODELS)
def test_survival_uses_masked_columns(model_type):
    """``survival_time`` must integrate only predictable columns (== likelihood survival)."""
    torch.manual_seed(SEED)
    model = create_model(model_type, terminating=False).to(DEVICE).eval()
    ts = _batch()
    _set_ctx(model, _CTX)
    t = torch.full((ts.time_points.shape[0],), 3.0, dtype=torch.float64, device=DEVICE)

    with torch.no_grad():
        surv_log = model.survival_time(t, ts, log=True)  # (B,)
        ci = model.cumulative_intensity(t, ts, T_low=ts.last_event_time)  # (B, D)
    predictable = model.predictable_mask
    masked = -ci[:, predictable].sum(dim=1)  # predictable-only
    all_cols = -ci.sum(dim=1)  # the pre-fix (leaky) survival

    assert torch.allclose(surv_log, masked, atol=1e-9), f"{model_type}: survival != masked survival"
    # The context column carries real mass, so masking must actually change the value.
    assert not torch.allclose(surv_log, all_cols, atol=1e-6), f"{model_type}: masking was inert (test premise)"
    # cdf_time is 1 - S, so it inherits the same masking.
    with torch.no_grad():
        cdf = model.cdf_time(t, ts, log=False)
    assert torch.allclose(cdf, -torch.expm1(surv_log), atol=1e-9), f"{model_type}: cdf inconsistent with survival"


@pytest.mark.parametrize("model_type", _MODELS)
def test_prob_mark_zeros_context(model_type):
    """``prob_mark`` gives ~0 mass to a context type and renormalizes over predictable."""
    torch.manual_seed(SEED)
    model = create_model(model_type, terminating=False).to(DEVICE).eval()
    ts = _batch()
    _set_ctx(model, _CTX)

    with torch.no_grad():
        pm = model.prob_mark(ts, log=False)  # (B, D)
    assert torch.allclose(pm[:, _CTX], torch.zeros_like(pm[:, _CTX]), atol=1e-6), f"{model_type}: ctx mark not ~0"
    assert torch.allclose(pm.sum(dim=1), torch.ones(pm.shape[0], dtype=pm.dtype, device=pm.device), atol=1e-4), (
        f"{model_type}: predictable mark mass does not renormalize to 1"
    )


@pytest.mark.parametrize("model_type", _MODELS)
def test_all_true_mask_is_inert(model_type):
    """Every predictive quantity is bit-identical between the freshly-built (all-True)
    model and an explicit all-True stamp — the masking must be a no-op by default."""
    torch.manual_seed(SEED)
    model = create_model(model_type, terminating=False).to(DEVICE).eval()
    ts = _batch()
    t = torch.full((ts.time_points.shape[0],), 3.0, dtype=torch.float64, device=DEVICE)

    with torch.no_grad():
        base = {
            "surv": model.survival_time(t, ts, log=True),
            "cdf": model.cdf_time(t, ts, log=True),
            "pm": model.prob_mark(ts, log=True),
            "tm": model.time_mean(ts),
            "tq": model.time_quantile(0.5, ts),
        }
        model.predictable_mask.copy_(torch.ones(model.D, dtype=torch.bool, device=model.predictable_mask.device))
        after = {
            "surv": model.survival_time(t, ts, log=True),
            "cdf": model.cdf_time(t, ts, log=True),
            "pm": model.prob_mark(ts, log=True),
            "tm": model.time_mean(ts),
            "tq": model.time_quantile(0.5, ts),
        }
    for k in base:
        assert torch.equal(base[k], after[k]), f"{model_type}: {k} moved under an all-True stamp (not inert)"
