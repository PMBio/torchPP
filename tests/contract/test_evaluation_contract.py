"""Contract test for the ``evaluate_tpp`` metric surface.

Pins the emitted metric-key set so a future accidental rename (or a half-done
one) fails a fast test instead of surfacing as a blank column in a downstream
results table. No metric key may end in ``_days`` — the model reports native units, and ``days`` is only correct for
some datasets, so the keys are unit-neutral.
"""

import pytest
import torch

from helpers import D, build_test_data, create_model
from torchpp.utils.evaluation import evaluate_tpp

# Core metrics always emitted, plus the four (renamed) time-error keys.
_REQUIRED_KEYS = {
    "marginal_time_ll",
    "cond_type_ll",
    "joint_ll",
    "cond_top1_accuracy",
    "cond_top5_accuracy",
    "num_predictions",
}
_TIME_ERROR_KEYS = {"time_mae_mean", "time_mae_median", "time_rmse_mean", "time_rmse_median"}


@pytest.mark.parametrize("model_type", ["const_exp_hawkes", "intensity_free"])
def test_evaluate_tpp_metric_key_contract(model_type):
    """evaluate_tpp emits the pinned key set, in native units (no ``*_days`` keys)."""
    model = create_model(model_type, terminating=False).to("cpu")
    data = build_test_data(torch.device("cpu"))
    seqs = [data["seq1"], data["seq2"]]  # ≥2 events each → valid prediction items

    metrics = evaluate_tpp(model, seqs, num_event_types=D, device="cpu", compute_marginals=True)
    keys = set(metrics)

    assert _REQUIRED_KEYS <= keys, f"missing required metrics: {_REQUIRED_KEYS - keys}"
    assert _TIME_ERROR_KEYS <= keys, f"missing time-error metrics: {_TIME_ERROR_KEYS - keys}"
    stale = {k for k in keys if k.endswith("_days")}
    assert not stale, f"native-unit metrics must not carry a '_days' suffix: {stale}"
