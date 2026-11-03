"""Origin/encoder consistency for neural models and the IntensityFree hazard at event times."""

import copy

import pytest
import torch

import torchpp as t
from torchpp import TokenRoles

NEURAL_TYPES = ["neural_poisson", "neural_spline", "thp", "nhp", "attnhp", "intensity_free"]
# Heads that map zero parameters to zero intensity cannot score from the zeros no-history column.
ZERO_COLUMN_TYPES = ["neural_poisson", "neural_spline", "neural_piecewise_constant"]
# Heads whose zeros column is a usable default state: a later has_origin=True stays allowed.
DEFAULT_STATE_TYPES = ["neural_log_linear", "thp", "nhp", "attnhp", "intensity_free"]


def _seq():
    return t.MVEventData(torch.tensor([0.1, 0.3, 0.7]), torch.tensor([0, 1, 0]), T_obs=1.0)


@pytest.mark.parametrize("model_type", ZERO_COLUMN_TYPES)
def test_origin_without_origin_encoder_raises(model_type):
    model = t.create_model(model_type, D=2, seed=0)
    model.token_roles = TokenRoles(has_origin=True)
    with pytest.raises(ValueError, match="encoder built with has_origin=True"):
        with torch.no_grad():
            model.likelihood(_seq())


@pytest.mark.parametrize("model_type", DEFAULT_STATE_TYPES)
def test_origin_set_later_uses_the_default_state(model_type):
    model = t.create_model(model_type, D=2, seed=0)
    model.has_origin = True
    with torch.no_grad():
        ll = model.likelihood(_seq())
    assert torch.isfinite(ll).all() and ll.item() > -50.0


@pytest.mark.parametrize("model_type", NEURAL_TYPES + ["neural_piecewise_constant"])
def test_origin_built_by_create_model_is_finite(model_type):
    model = t.create_model(model_type, D=2, seed=0, token_roles=TokenRoles(has_origin=True))
    with torch.no_grad():
        ll = model.likelihood(_seq())
    assert torch.isfinite(ll).all() and ll.item() > -50.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_intensity_free_hazard_at_event_times_is_scale_equivariant(dtype):
    s = 2.9
    base = t.create_model("intensity_free", D=2, seed=0).eval()
    scaled = copy.deepcopy(base)
    scaled.time_scale.fill_(s)

    times = torch.tensor([0.4, 0.9, 1.7, 2.6], dtype=dtype) * s
    seq = t.MVEventData(times, torch.tensor([0, 1, 1, 0]))
    seq_internal = t.MVEventData((times * (1.0 / s)).to(dtype), seq.event_types)
    queries = times[1:]  # exactly at event times: history must be strictly before each query

    with torch.no_grad():
        native = scaled.intensity(queries.unsqueeze(0), t.BatchedMVEventData(mv_events=[seq]))
        reference = base.intensity((queries * (1.0 / s)).to(dtype).unsqueeze(0),
                                   t.BatchedMVEventData(mv_events=[seq_internal])) / s
    torch.testing.assert_close(native, reference, rtol=1e-5, atol=0.0)
    assert (native > 1e-12).all()
