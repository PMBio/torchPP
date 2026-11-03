"""Public API conventions: keyword-only arguments, ``Sample``, exact checkpoints, ``register_model``,
and ``CIConfig`` / ``SamplingConfig`` through constructors, YAML and checkpoints."""

from __future__ import annotations

import inspect

import pytest
import torch

import torchpp
from torchpp.tpp.base_tpp import CIConfig, SamplingConfig
from torchpp.training import create_model, load_model, register_model, save_model
from torchpp.training.config import FitConfig
from torchpp.training.config_utils import ConfigError, load_config
from torchpp.training.registry import MODEL_ALIASES, MODEL_REGISTRY, VALID_MODEL_TYPES
from torchpp.utils.event_data import BatchedMVEventData, MVEventData

_LOG_METHODS = (
    "likelihood",
    "density_joint",
    "density_time",
    "prob_mark",
    "prob_mark_given_time",
    "density_time_given_mark",
    "survival_time",
    "cdf_time",
)


def _batch() -> BatchedMVEventData:
    seqs = [
        MVEventData(torch.tensor([0.1, 0.4, 0.9, 1.5]), torch.tensor([0, 1, 2, 0])),
        MVEventData(torch.tensor([0.2, 0.7]), torch.tensor([2, 1])),
    ]
    return BatchedMVEventData(mv_events=seqs)


@pytest.mark.parametrize("model_type", list(MODEL_REGISTRY))
def test_log_and_constructor_arguments_are_keyword_only(model_type):
    cls = MODEL_REGISTRY[model_type].cls
    for name in _LOG_METHODS:
        param = inspect.signature(getattr(cls, name)).parameters["log"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, f"{cls.__name__}.{name}"
    params = list(inspect.signature(cls.__init__).parameters.values())[1:]
    positional = [p.name for p in params if p.kind is p.POSITIONAL_OR_KEYWORD]
    assert positional in (["D"], ["encoder"]), f"{cls.__name__}: {positional}"
    names = {p.name for p in params}
    assert "sampling_config" in names
    assert ("ci_config" in names) == issubclass(cls, torchpp.IntensityBasedTPP)


def test_create_model_takes_keywords_after_model_type_and_d():
    with pytest.raises(TypeError):
        create_model("poisson", 2, None)  # noqa


def test_sample_returns_named_tuple():
    model = create_model("const_exp_hawkes", D=3, seed=0)
    result = model.sample(MVEventData(torch.tensor([0.1, 0.5]), torch.tensor([0, 1])), num_steps=2, rng=0)
    assert isinstance(result, torchpp.Sample)
    sequence, times, mark_probs = result
    assert sequence is result.sequence and len(times) == len(mark_probs) == 2
    assert torch.equal(sequence.time_points[-2:], torch.cat(times))


def test_direct_model_round_trips_with_non_parameter_settings(tmp_path):
    model = torchpp.InhibitiveSplineBaselineSplineKernelHawkesProcess(
        3,
        baseline_num_knots=4,
        kernel_num_knots=3,
        baseline_delta_t=0.5,
        kernel_delta_t=0.3,
        clamping_fn="relu",
        ci_config=CIConfig("numerical", "interval", "trapezoidal", 37),
        sampling_config=SamplingConfig(mode="thinning"),
    )
    loaded = load_model(save_model(tmp_path / "m.pth", model))
    assert type(loaded) is type(model)
    assert loaded.ci_config == model.ci_config
    assert loaded.sampling_config == model.sampling_config
    assert loaded._init_args["clamping_fn"] == "relu"
    with torch.no_grad():
        assert torch.equal(loaded.likelihood(_batch()), model.likelihood(_batch()))


@pytest.mark.parametrize("model_type", ["spline_spline_hawkes", "thp", "nhp", "attnhp", "intensity_free", "neural_spline"])
def test_create_model_round_trip_restores_the_configs(tmp_path, model_type):
    model = create_model(model_type, D=3, seed=0, arch={"ci": {"n_points": 9}} if model_type != "intensity_free" else None)
    model.eval()
    loaded = load_model(save_model(tmp_path / "m.pth", model))
    assert loaded.model_type == model_type and loaded.run_name == model.run_name
    assert getattr(loaded, "ci_config", None) == getattr(model, "ci_config", None)
    with torch.no_grad():
        assert torch.equal(loaded.likelihood(_batch()), model.likelihood(_batch()))


def test_load_refuses_classes_outside_torchpp_that_are_not_imported(tmp_path):
    path = save_model(tmp_path / "m.pth", create_model("poisson", D=2, seed=0))
    ckpt = torch.load(path, weights_only=True)
    ckpt["class"] = "not_a_module_xyz:Model"
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match="import not_a_module_xyz"):
        load_model(path)


@pytest.fixture
def registered_model():
    @register_model("tmp_registered_poisson", description="test model", aliases=("tmp_alias",))
    class TmpPoisson(torchpp.PoissonProcess):
        pass

    yield TmpPoisson
    MODEL_REGISTRY.pop("tmp_registered_poisson")
    VALID_MODEL_TYPES.remove("tmp_registered_poisson")
    MODEL_ALIASES.pop("tmp_alias")


def test_registered_model_works_everywhere(tmp_path, registered_model):
    model = create_model("tmp_alias", D=2, seed=0)
    assert type(model) is registered_model and model.run_name == "tmp_registered_poisson"
    loaded = load_model(save_model(tmp_path / "m.pth", model))
    assert type(loaded) is registered_model
    cfg = load_config(
        FitConfig, ["-c", "default_fit", "-c", "data/taxi", "model_type=tmp_registered_poisson", "training_preset=classical"]
    )
    assert cfg.model_type == "tmp_registered_poisson"
    with pytest.raises(ValueError, match="already registered"):
        register_model("tmp_registered_poisson", registered_model)
    with pytest.raises(TypeError):
        register_model("not_a_model", int)


def test_arch_ci_and_sampling_from_yaml():
    cfg = load_config(
        FitConfig,
        ["-c", "default_fit", "-c", "model/thp", "-c", "data/taxi", "arch.ci.n_points=7", "arch.sampling.mode=thinning"],
    )
    model = create_model(cfg.model_type, D=3, arch=cfg.arch, seed=0)
    assert model.ci_config.n_points == 7 and model.sampling_config.mode == "thinning"
    with pytest.raises(ConfigError, match="chunk_size"):
        load_config(FitConfig, ["-c", "default_fit", "-c", "model/attnhp", "-c", "data/taxi", "arch.ci.chunk_sz=3"])
    with pytest.raises(ValueError, match="has no architecture field 'arch.ci'"):
        create_model("intensity_free", D=3, arch={"ci": {"n_points": 7}})  # density-based: no cumulative intensity


def test_model_module_registers_a_model_for_the_cli(tmp_path, monkeypatch):
    (tmp_path / "tmp_user_models.py").write_text(
        "import torchpp\n\n"
        "@torchpp.register_model('tmp_cli_poisson')\n"
        "class CliPoisson(torchpp.PoissonProcess):\n"
        "    pass\n"
    )
    monkeypatch.chdir(tmp_path)
    try:
        cfg = load_config(
            FitConfig,
            ["-c", "default_fit", "-c", "data/taxi", "model_module=tmp_user_models", "model_type=tmp_cli_poisson",
             "training_preset=classical"],
            configs_dir=torchpp.__file__.rsplit("/torchpp/", 1)[0] + "/configs",
        )
        assert cfg.model_type == "tmp_cli_poisson"
    finally:
        MODEL_REGISTRY.pop("tmp_cli_poisson", None)
        if "tmp_cli_poisson" in VALID_MODEL_TYPES:
            VALID_MODEL_TYPES.remove("tmp_cli_poisson")
