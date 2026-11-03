"""Native → internal time conversion keeps tied events and their targets apart.

Event times and query times must be scaled with the same operation and dtype; otherwise a
history event tied with its target can look strictly earlier at ``time_scale != 1``.
"""

import pytest
import torch

from torchpp import create_model
from torchpp.utils.event_data import MVEventData, scored_event_mask
from torchpp.utils.evaluation import evaluate_tpp

D = 4
MODELS = [
    "poisson",
    "const_exp_hawkes",
    "spline_spline_hawkes",
    "inhibitive_const_exp_hawkes",
    "neural_spline",
    "thp",
    "nhp",
    "intensity_free",
]
CLOSED_FORM = ["poisson", "const_exp_hawkes", "spline_spline_hawkes", "neural_spline", "intensity_free"]


def _sequences(dtype, n=8, seed=1, base=125.0, ties=True):
    """Sequences at large absolute times (with ties by default); half observed beyond the last event."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        L = 4 + int(torch.randint(0, 10, (1,), generator=g))
        types = torch.randint(0, D, (L,), generator=g)
        t = base + torch.cumsum(-torch.log(torch.rand(L, generator=g, dtype=torch.float64)) * 2.5, 0)
        if ties:
            t[2] = t[1]
            if L > 6:
                t[5] = t[4]
        t = t.to(torch.float32).to(torch.float64)  # identical values in both dtypes
        T_obs = float(t[-1]) + (1.7 if i % 2 == 0 else 0.0)
        out.append(MVEventData(t.to(dtype), types, T_obs=T_obs))
    return out


def _model(model_type, time_scale):
    model = create_model(model_type, D, seed=0)
    model.time_scale.fill_(time_scale)
    return model.eval()


@pytest.mark.parametrize("time_scale", [1.0, 2.9])
@pytest.mark.parametrize("model_type", MODELS)
def test_float32_evaluation_matches_float64(model_type, time_scale):
    model = _model(model_type, time_scale)
    with torch.no_grad():
        r32 = evaluate_tpp(model, _sequences(torch.float32), compute_marginals=False)
        r64 = evaluate_tpp(model, _sequences(torch.float64), compute_marginals=False)
    tol = 2e-5 if model_type == "intensity_free" else 1e-5  # tied events: τ ≈ 0, LL ≈ −34
    for key in ("joint_ll", "marginal_time_ll", "cond_type_ll", "ll_per_event"):
        assert abs(r32[key] - r64[key]) <= tol, (key, r32[key], r64[key])


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("time_scale", [1.0, 2.9])
@pytest.mark.parametrize("model_type", CLOSED_FORM)
def test_next_event_densities_add_up_to_the_likelihood(model_type, time_scale, dtype, ties):
    """Σ log p(t_i, m_i | prefix) + log S(T_obs | full history) == likelihood."""
    if ties and model_type == "intensity_free":
        pytest.skip("IntensityFree scores a tied event from the history strictly before it, "
                    "while its predictive density conditions on the whole prefix")
    model = _model(model_type, time_scale)
    roles = model.token_roles
    with torch.no_grad():
        for seq in _sequences(dtype, ties=ties):
            scored = scored_event_mask(seq.to_batch(), model.predictable_mask, roles.has_origin)[0]
            total = torch.zeros((), dtype=torch.float64)
            for i in torch.nonzero(scored).flatten().tolist():
                prefix = MVEventData(seq.time_points[:i], seq.event_types[:i])
                log_p = model.density_joint(seq.time_points[i : i + 1], prefix, log=True)
                total = total + log_p[0, seq.event_types[i]].double()
            T = torch.tensor([seq.T_obs], dtype=torch.float64)
            history = MVEventData(seq.time_points, seq.event_types)  # window (t_last, T_obs]
            total = total + model.survival_time(T, history, log=True)[0].double()
            ll = model.likelihood(seq)[0].double()
            assert torch.isclose(total, ll, rtol=1e-5, atol=1e-4), (float(total), float(ll))
