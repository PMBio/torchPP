"""Python training API (``fit``) and self-describing checkpoints (``save_model`` / ``load_model``)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from torchpp.training import create_model, default_fit_config, fit, load_model, save_model
from torchpp.training.config import FitConfig
from torchpp.training.dataset_registry import get_data_root
from torchpp.utils.event_data import BatchedMVEventData, MVEventData

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _toy_sequences(n: int, D: int = 2, seed: int = 0) -> list[MVEventData]:
    g = torch.Generator().manual_seed(seed)
    seqs = []
    for _ in range(n):
        L = int(torch.randint(3, 12, (1,), generator=g))
        times = torch.cumsum(torch.rand(L, generator=g, dtype=torch.float64) + 0.05, 0)
        types = torch.randint(0, D, (L,), generator=g)
        seqs.append(MVEventData(times, types, T_obs=float(times[-1]) + 0.5))
    return seqs


def test_fit_trains_and_restores_best():
    model = create_model("poisson", D=2, terminating=False, seed=0)
    config = default_fit_config(num_steps=20, learning_rate=0.1, eval_freq=5, batch_size=8)
    result = fit(model, _toy_sequences(40), _toy_sequences(10, seed=1), config, device="cpu")

    assert result.steps == 20
    assert len(result.train_ll) == 20
    assert len(result.val_ll) == 5  # steps 0, 5, 10, 15 and the last step
    assert result.best_val_ll == max(result.val_ll)
    assert result.val_ll[-1] > result.val_ll[0]


def test_fit_without_validation():
    model = create_model("poisson", D=2, terminating=False, seed=0)
    config = default_fit_config(num_steps=3, batch_size=8)
    before = repr(config)
    result = fit(model, _toy_sequences(20), config=config, device="cpu")
    assert result.steps == 3 and result.val_ll == [] and result.best_val_ll is None
    assert repr(config) == before  # fit works on a copy


def test_fit_rejects_unset_required_fields():
    model = create_model("poisson", D=2, terminating=False, seed=0)
    with pytest.raises(ValueError, match="Unset required config fields"):
        fit(model, _toy_sequences(4), config=FitConfig(model_type="poisson"), device="cpu")


def _batch(D: int = 2) -> BatchedMVEventData:
    return BatchedMVEventData(mv_events=_toy_sequences(4, D=D, seed=3))


@pytest.mark.parametrize("model_type", ["const_exp_hawkes", "inhibitive_const_exp_hawkes", "neural_spline"])
def test_checkpoint_round_trip(tmp_path, model_type):
    arch = {"head": {"num_bins": 5}} if model_type == "neural_spline" else None
    model = create_model(model_type, D=2, arch=arch, terminating=False, seed=0)
    path = save_model(tmp_path / "m.pth", model, model_type=model_type, arch=arch, D=2, seed=0)

    loaded = load_model(path)
    model.eval()
    batch = _batch()
    sd, sd_loaded = model.state_dict(), loaded.state_dict()
    assert sd.keys() == sd_loaded.keys() and all(torch.equal(sd[k], sd_loaded[k]) for k in sd)
    with torch.no_grad():
        torch.manual_seed(0)  # the inhibitive model's default quadrature draws random nodes
        ll = model.likelihood(batch)
        torch.manual_seed(0)
        assert torch.equal(ll, loaded.likelihood(batch))


def test_save_model_unwraps_compiled_models(tmp_path):
    class _Compiled(torch.nn.Module):  # what torch.compile returns: the model under ``_orig_mod``
        def __init__(self, inner):
            super().__init__()
            self._orig_mod = inner

    model = create_model("const_exp_hawkes", D=2, terminating=False, seed=0)
    path = save_model(tmp_path / "m.pth", _Compiled(model), model_type="const_exp_hawkes", D=2)
    assert not any(k.startswith("_orig_mod.") for k in torch.load(path, weights_only=True)["state_dict"])
    load_model(path)


def test_load_model_arch_override(tmp_path):
    model = create_model("inhibitive_const_exp_hawkes", D=2, terminating=False, seed=0)
    path = save_model(tmp_path / "m.pth", model, model_type="inhibitive_const_exp_hawkes", D=2)
    assert load_model(path, ci={"n_points": 17}).ci_config.n_points == 17


def test_create_model_loads_both_formats(tmp_path):
    model = create_model("const_exp_hawkes", D=2, terminating=False, seed=0)
    bare = tmp_path / "bare.pth"
    torch.save(model.state_dict(), bare)
    full = save_model(tmp_path / "full.pth", model, model_type="const_exp_hawkes", D=2)

    for path in (bare, full):
        create_model("const_exp_hawkes", D=2, terminating=False, load_path=path)
    with pytest.raises(ValueError, match="holds a 'const_exp_hawkes' model"):
        create_model("poisson", D=2, terminating=False, load_path=full)
    with pytest.raises(ValueError, match="only a state_dict"):
        load_model(bare)


@pytest.mark.skipif(not (get_data_root() / "benchmarks" / "taxi").exists(), reason="taxi benchmark data not downloaded")
def test_evaluate_cli_needs_only_model_path(tmp_path):
    model = create_model("poisson", D=10, terminating=False, seed=0)
    path = save_model(
        tmp_path / "poisson_taxi.pth", model, model_type="poisson", D=10,
        dataset_type="benchmark", dataset={"dataset_name": "taxi", "benchmark_root": "benchmarks"},
    )
    subprocess.run(
        [sys.executable, str(_REPO_ROOT / "evaluate.py"), "-c", "default_eval", f"model_path={path}",
         "eval.compute_marginal_type=false", "eval.limit_test_size=20", "device=cpu"],
        check=True, capture_output=True, cwd=tmp_path,
    )
    out = json.loads(path.with_suffix(".eval.json").read_text())
    assert out["model_type"] == "poisson" and out["n_test_sequences"] == 20
    assert math.isfinite(out["metrics"]["joint_ll"])


def test_fit_trains_on_the_models_device_by_default():
    """With config.device="auto", fit keeps the model where it is (no silent move to CUDA)."""
    model = create_model("poisson", D=2, seed=0)
    result = fit(model, _toy_sequences(8), None, default_fit_config(num_steps=2, eval_freq=1))
    assert all(p.device.type == "cpu" for p in result.model.parameters())
