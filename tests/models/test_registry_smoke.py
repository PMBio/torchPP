"""Registry smoke test: every model in the matrix constructs and runs.

The contract layer runs relation/shape checks over a 6-archetype set, not the
full matrix. This single fan over all of ``MODEL_TYPES`` is the safety net that
keeps every registered model — including any newly added one — exercised end to
end: it must construct (terminating and not) and produce one finite likelihood.
A model added to the registry without a ``create_model`` branch fails here loudly.
"""

import pytest
import torch

from helpers import MODEL_TYPES, DEVICE, create_model, build_test_data, build_terminating_test_data


@pytest.mark.parametrize("model_type", MODEL_TYPES)
@pytest.mark.parametrize("terminating", [False, True])
def test_constructs_and_one_finite_likelihood(model_type, terminating):
    model = create_model(model_type, terminating=terminating)
    # Terminating mode needs a sequence with each type at most once.
    if terminating:
        ts = build_terminating_test_data(DEVICE)["partial_batch"]
    else:
        ts = build_test_data(DEVICE)["batch_single"]
    T = torch.tensor([3.0], dtype=torch.float64).to(DEVICE)
    ll = model.likelihood(ts, T, log=True)
    assert ll.shape == (1,)
    assert torch.isfinite(ll).all()
