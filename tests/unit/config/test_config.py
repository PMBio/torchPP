"""
Tests for the config loading and validation system.

Covers:
  - MISSING field detection (required fields not provided → error)
  - YAML loading via load_config() with --config / -c flags
  - defaults: list expansion (recursive YAML composition)
  - key=value CLI overrides (dotted path merging)
  - Type coercion and nested dataclass construction
  - device="auto" resolution
  - arch.seed=None randomisation
  - model_type validation against VALID_MODEL_TYPES
  - Unknown key rejection
  - StandaloneEvalConfig loading
  - Every real config preset file can be parsed without error (schema compatibility)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import MissingMandatoryValue, OmegaConf

from torchpp.training.arch import IntensityArchConfig, SplineBaselineSplineKernelArchConfig
from torchpp.training.config import (
    VALID_MODEL_TYPES,
    BatchingConfig,
    CompileConfig,
    EvalConfig,
    FitConfig,
    ProfilingConfig,
    SaveConfig,
    StandaloneEvalConfig,
    TrainingConfig,
    WandbConfig,
)
from torchpp.training.config_utils import (
    ConfigError,
    _dataclass_to_omegaconf,
    _omegaconf_to_dataclass,
    _resolve_config_path,
    _split_argv,
    load_config,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


@pytest.fixture()
def configs_dir() -> Path:
    return CONFIGS_DIR


@pytest.fixture()
def minimal_fit_argv() -> list[str]:
    """Minimal CLI argv that satisfies all MISSING fields for FitConfig."""
    return [
        "-c",
        "default_fit",
        "-c",
        "model/poisson",
        "-c",
        "data/taxi",
        # default_fit sets batching, eval and wandb; model/poisson supplies model_type and
        # its training preset; the data preset fills the dataset dict — together all
        # MISSING fields are covered.
    ]


# ---------------------------------------------------------------------------
# _split_argv
# ---------------------------------------------------------------------------


class TestSplitArgv:
    def test_config_long_form(self):
        paths, overrides, print_flag, _ = _split_argv(["--config", "foo.yaml"])
        assert paths == ["foo.yaml"]
        assert overrides == []
        assert print_flag is False

    def test_config_short_form(self):
        paths, overrides, print_flag, _ = _split_argv(["-c", "foo.yaml", "-c", "bar.yaml"])
        assert paths == ["foo.yaml", "bar.yaml"]

    def test_config_inline_equals(self):
        paths, overrides, _, _ = _split_argv(["--config=foo.yaml"])
        assert paths == ["foo.yaml"]

    def test_overrides(self):
        _, overrides, _, _ = _split_argv(["training.num_steps=100", "wandb.enable=true"])
        assert "training.num_steps=100" in overrides
        assert "wandb.enable=true" in overrides

    def test_print_flag(self):
        _, _, print_flag, _ = _split_argv(["--print"])
        assert print_flag is True

    def test_print_short(self):
        _, _, print_flag, _ = _split_argv(["-p"])
        assert print_flag is True

    def test_mixed(self):
        paths, overrides, print_flag, _ = _split_argv(
            [
                "-c",
                "model/poisson",
                "training.num_steps=10",
                "--print",
            ]
        )
        assert paths == ["model/poisson"]
        assert "training.num_steps=10" in overrides
        assert print_flag is True

    def test_unknown_flag_raises(self):
        with pytest.raises(ConfigError, match="Unknown option '--unknown-flag'"):
            _split_argv(["--unknown-flag", "-c", "poisson.yaml"])

    def test_token_without_equals_raises(self):
        with pytest.raises(ConfigError, match="missing '='"):
            _split_argv(["training.num_steps", "50"])

    def test_help_flag(self):
        assert _split_argv(["--help"])[3] is True


# ---------------------------------------------------------------------------
# _resolve_config_path
# ---------------------------------------------------------------------------


class TestResolveConfigPath:
    def test_short_name_resolves(self, configs_dir):
        path = _resolve_config_path("model/poisson", configs_dir)
        assert path.exists()
        assert path.suffix == ".yaml"

    def test_short_name_with_extension(self, configs_dir):
        path = _resolve_config_path("model/poisson.yaml", configs_dir)
        assert path.exists()

    def test_absolute_path(self, configs_dir):
        abs_path = configs_dir / "model" / "poisson.yaml"
        result = _resolve_config_path(str(abs_path), configs_dir)
        assert result == abs_path

    def test_missing_file_raises(self, configs_dir):
        with pytest.raises(FileNotFoundError, match="nonexistent_model"):
            _resolve_config_path("model/nonexistent_model", configs_dir)


# ---------------------------------------------------------------------------
# _dataclass_to_omegaconf
# ---------------------------------------------------------------------------


class TestDataclassToOmegaconf:
    def test_dataset_is_empty_dict_default(self):
        # The dataset payload is an untyped dict (validated at runtime by the
        # dataset registry), so it defaults to an empty mapping rather than
        # carrying MISSING typed fields.
        cfg = _dataclass_to_omegaconf(FitConfig)
        assert OmegaConf.to_container(cfg.dataset) == {}
        assert cfg.dataset_type == "benchmark"

    def test_fitconfig_model_type_missing(self):
        cfg = _dataclass_to_omegaconf(FitConfig)
        assert OmegaConf.is_missing(cfg, "model_type")

    def test_fitconfig_device_default(self):
        cfg = _dataclass_to_omegaconf(FitConfig)
        assert cfg.device == "auto"

    def test_training_missing_fields(self):
        cfg = _dataclass_to_omegaconf(TrainingConfig)
        for field in ("num_steps", "learning_rate", "eval_freq", "scheduler_step_size", "scheduler_gamma"):
            assert OmegaConf.is_missing(cfg, field), f"{field} should be MISSING"

    def test_batching_batch_size_missing(self):
        cfg = _dataclass_to_omegaconf(BatchingConfig)
        assert OmegaConf.is_missing(cfg, "batch_size")

    def test_eval_limit_test_size_missing(self):
        cfg = _dataclass_to_omegaconf(EvalConfig)
        assert OmegaConf.is_missing(cfg, "limit_test_size")

    def test_non_dataclass_raises(self):
        with pytest.raises(TypeError):
            _dataclass_to_omegaconf(int)


# ---------------------------------------------------------------------------
# _omegaconf_to_dataclass
# ---------------------------------------------------------------------------


class TestOmegaconfToDataclass:
    def test_roundtrip_nested_config(self):
        cfg = OmegaConf.create({"num_buckets": 7, "batch_size": 16})
        dc = _omegaconf_to_dataclass(cfg, BatchingConfig)
        assert isinstance(dc, BatchingConfig)
        assert dc.num_buckets == 7 and dc.batch_size == 16

    def test_missing_field_raises(self):
        base = _dataclass_to_omegaconf(BatchingConfig)
        # batch_size is still MISSING — should raise
        with pytest.raises(MissingMandatoryValue):
            _omegaconf_to_dataclass(base, BatchingConfig)


# ---------------------------------------------------------------------------
# load_config — MISSING field errors
# ---------------------------------------------------------------------------


class TestLoadConfigMissingFields:
    def test_no_args_raises_missing(self):
        """Calling load_config with no arguments must fail due to MISSING fields."""
        with pytest.raises((MissingMandatoryValue, Exception)):
            load_config(FitConfig, argv=[])

    def test_model_type_missing_raises(self):
        """Config without model_type must fail with a meaningful error."""
        argv = [
            "-c",
            "default_fit",
            # No -c model/... so model_type stays MISSING
        ]
        with pytest.raises((MissingMandatoryValue, Exception)):
            load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)

    def test_standalone_eval_no_args_raises(self):
        with pytest.raises((MissingMandatoryValue, Exception)):
            load_config(StandaloneEvalConfig, argv=[], configs_dir=CONFIGS_DIR)


# ---------------------------------------------------------------------------
# load_config — successful loading
# ---------------------------------------------------------------------------


class TestLoadConfigSuccess:
    def test_minimal_fit_config(self, minimal_fit_argv):
        cfg = load_config(FitConfig, argv=minimal_fit_argv, configs_dir=CONFIGS_DIR)
        assert isinstance(cfg, FitConfig)
        assert cfg.model_type == "poisson"
        assert cfg.dataset_type == "benchmark"

    def test_device_auto_resolved(self, minimal_fit_argv):
        cfg = load_config(FitConfig, argv=minimal_fit_argv, configs_dir=CONFIGS_DIR)
        assert cfg.device in ("cuda", "cpu"), f"Unexpected device: {cfg.device}"

    def test_seed_none_randomised(self, minimal_fit_argv):
        """arch.seed=None must be replaced with a random integer at load time."""
        cfg = load_config(FitConfig, argv=minimal_fit_argv, configs_dir=CONFIGS_DIR)
        assert isinstance(cfg.arch.seed, int)
        assert 0 <= cfg.arch.seed <= 2**31 - 1

    def test_seed_explicit_preserved(self, minimal_fit_argv):
        argv = minimal_fit_argv + ["arch.seed=99"]
        cfg = load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)
        assert cfg.arch.seed == 99

    def test_sub_configs_are_dataclasses(self, minimal_fit_argv):
        cfg = load_config(FitConfig, argv=minimal_fit_argv, configs_dir=CONFIGS_DIR)
        assert isinstance(cfg.dataset, dict)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.batching, BatchingConfig)
        assert isinstance(cfg.arch, IntensityArchConfig)  # poisson: only the shared fields
        assert isinstance(cfg.eval, EvalConfig)
        assert isinstance(cfg.wandb, WandbConfig)
        assert isinstance(cfg.save, SaveConfig)
        assert isinstance(cfg.profiling, ProfilingConfig)
        assert isinstance(cfg.compile, CompileConfig)

    def test_cli_override_wins(self, minimal_fit_argv):
        argv = minimal_fit_argv + ["training.num_steps=999"]
        cfg = load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)
        assert cfg.training.num_steps == 999

    def test_second_config_wins_over_first(self):
        """Later -c flags override earlier ones."""
        argv = [
            "-c",
            "default_fit",
            "-c",
            "model/poisson",
            "-c",
            "model/const_exp_hawkes",  # overwrites model_type
        ]
        cfg = load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)
        assert cfg.model_type == "const_exp_hawkes"

    def test_data_from_yaml(self, minimal_fit_argv):
        cfg = load_config(FitConfig, argv=minimal_fit_argv, configs_dir=CONFIGS_DIR)
        # data/taxi.yaml fills the dataset dict (dataset_name=taxi).
        assert cfg.dataset.get("dataset_name") == "taxi"

    def test_eval_limit_test_size_set(self, minimal_fit_argv):
        cfg = load_config(FitConfig, argv=minimal_fit_argv, configs_dir=CONFIGS_DIR)
        # default_fit sets limit_test_size=1000
        assert cfg.eval.limit_test_size is not None


# ---------------------------------------------------------------------------
# load_config — invalid inputs
# ---------------------------------------------------------------------------


class TestLoadConfigInvalidInputs:
    def test_unknown_model_type_raises(self, minimal_fit_argv):
        argv = minimal_fit_argv + ["model_type=totally_fake_model_xyz"]
        with pytest.raises(ValueError, match="Unknown model_type"):
            load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)

    def test_missing_config_file_raises(self, minimal_fit_argv):
        argv = minimal_fit_argv + ["-c", "model/does_not_exist"]
        with pytest.raises(FileNotFoundError):
            load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)


# ---------------------------------------------------------------------------
# defaults: list composition
# ---------------------------------------------------------------------------


class TestDefaultsComposition:
    def test_model_preset_selects_family_training(self):
        """A model preset's training_preset supplies its family's training defaults."""
        classical = load_config(FitConfig, argv=["-c", "default_fit", "-c", "model/poisson"], configs_dir=CONFIGS_DIR)
        neural = load_config(FitConfig, argv=["-c", "default_fit", "-c", "model/thp"], configs_dir=CONFIGS_DIR)
        assert classical.training.learning_rate == 0.1
        assert neural.training.learning_rate == 0.01
        assert classical.training.scheduler_type == neural.training.scheduler_type == "linear_warmup_cosine"

    def test_every_model_preset_has_a_training_preset(self):
        for path in sorted((CONFIGS_DIR / "model").glob("*.yaml")):
            cfg = load_config(FitConfig, argv=["-c", "default_fit", "-c", f"model/{path.stem}"], configs_dir=CONFIGS_DIR)
            assert cfg.training.learning_rate in (0.1, 0.01), path.stem

    def test_training_preset_precedence(self):
        """model default < any -c training/... (before or after the model) < key=value."""
        base = ["-c", "default_fit", "-c", "model/thp"]
        before = load_config(FitConfig, argv=["-c", "default_fit", "-c", "training/smoke", "-c", "model/thp"], configs_dir=CONFIGS_DIR)
        after = load_config(FitConfig, argv=base + ["-c", "training/smoke"], configs_dir=CONFIGS_DIR)
        cli = load_config(FitConfig, argv=base + ["-c", "training/smoke", "training.num_steps=7"], configs_dir=CONFIGS_DIR)
        assert before.training.num_steps == after.training.num_steps == 10
        assert before.training.learning_rate == 0.01  # smoke keeps the family learning rate
        assert cli.training.num_steps == 7

    def test_training_preset_can_be_switched_on_the_command_line(self):
        cfg = load_config(FitConfig, argv=["-c", "default_fit", "-c", "model/thp", "training_preset=classical"], configs_dir=CONFIGS_DIR)
        assert cfg.training.learning_rate == 0.1

    def test_model_type_without_preset_uses_its_family(self):
        cfg = load_config(FitConfig, argv=["-c", "default_fit", "model_type=thp"], configs_dir=CONFIGS_DIR)
        assert cfg.training.learning_rate == 0.01
        alias = load_config(FitConfig, argv=["-c", "default_fit", "model_type=hawkes"], configs_dir=CONFIGS_DIR)
        assert alias.training.learning_rate == 0.1

    def test_training_preset_is_ignored_for_evaluation(self):
        cfg = load_config(
            StandaloneEvalConfig, argv=["-c", "default_eval", "-c", "model/thp", "model_path=x.pth"], configs_dir=CONFIGS_DIR
        )
        assert cfg.model_type == "thp"

    def test_data_preset_populates_dataset(self):
        # default_fit supplies no data; a data/ preset fills the dataset dict.
        argv = ["-c", "default_fit", "-c", "model/poisson", "-c", "data/taxi"]
        cfg = load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)
        assert cfg.dataset_type == "benchmark"
        assert cfg.dataset.get("dataset_name") == "taxi"

    def test_default_fit_composes_batching(self):
        argv = ["-c", "default_fit", "-c", "model/poisson"]
        cfg = load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)
        assert cfg.batching.batch_size is not None

    def test_defaults_override_order(self):
        """A -c flag after default_fit should override values set by default_fit's defaults:."""
        argv = [
            "-c",
            "default_fit",
            "-c",
            "model/poisson",
            "-c",
            "training/smoke",  # num_steps=10, overrides the family preset's 40000
        ]
        cfg = load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)
        assert cfg.training.num_steps == 10

    def test_eval_defaults_composition(self):
        """default_eval.yaml uses defaults: and should produce a complete config."""
        argv = [
            "-c",
            "default_eval",
            "model_path=models/fake.pth",
            "model_type=poisson",
        ]
        cfg = load_config(StandaloneEvalConfig, argv=argv, configs_dir=CONFIGS_DIR)
        assert cfg.model_type == "poisson"
        assert cfg.model_path == "models/fake.pth"
        assert cfg.eval.limit_test_size is not None


# ---------------------------------------------------------------------------
# StandaloneEvalConfig
# ---------------------------------------------------------------------------


class TestStandaloneEvalConfig:
    def test_loads_successfully(self):
        argv = [
            "-c",
            "default_eval",
            "model_path=models/my_model.pth",
            "model_type=const_exp_hawkes",
        ]
        cfg = load_config(StandaloneEvalConfig, argv=argv, configs_dir=CONFIGS_DIR)
        assert isinstance(cfg, StandaloneEvalConfig)
        assert cfg.model_path == "models/my_model.pth"
        assert cfg.model_type == "const_exp_hawkes"

    def test_missing_model_path_raises(self):
        argv = ["-c", "default_eval", "model_type=poisson"]
        with pytest.raises((MissingMandatoryValue, Exception)):
            load_config(StandaloneEvalConfig, argv=argv, configs_dir=CONFIGS_DIR)

    def test_model_type_optional(self):
        """model_type is read from the checkpoint; it is only required for a bare state_dict."""
        argv = ["-c", "default_eval", "model_path=models/my_model.pth"]
        cfg = load_config(StandaloneEvalConfig, argv=argv, configs_dir=CONFIGS_DIR)
        assert cfg.model_type is None

    def test_unknown_model_type_raises(self):
        argv = ["-c", "default_eval", "model_path=models/my_model.pth", "model_type=not_a_model"]
        with pytest.raises(ValueError):
            load_config(StandaloneEvalConfig, argv=argv, configs_dir=CONFIGS_DIR)

    def test_device_resolved(self):
        argv = [
            "-c",
            "default_eval",
            "model_path=x.pth",
            "model_type=poisson",
        ]
        cfg = load_config(StandaloneEvalConfig, argv=argv, configs_dir=CONFIGS_DIR)
        assert cfg.device in ("cuda", "cpu")

    def test_only_required_sub_configs(self):
        """StandaloneEvalConfig should have dataset, arch, eval but not training."""
        argv = ["-c", "default_eval", "model_path=x.pth", "model_type=poisson"]
        cfg = load_config(StandaloneEvalConfig, argv=argv, configs_dir=CONFIGS_DIR)
        assert isinstance(cfg.dataset, dict)
        assert isinstance(cfg.arch, dict)  # architecture overrides, applied to the checkpoint's model
        assert isinstance(cfg.eval, EvalConfig)
        assert not hasattr(cfg, "training")
        assert not hasattr(cfg, "batching")


# ---------------------------------------------------------------------------
# All model YAML presets are schema-compatible with FitConfig
# ---------------------------------------------------------------------------


def _iter_model_presets() -> list[str]:
    """Return all short names of configs/model/*.yaml presets."""
    return [f"model/{p.stem}" for p in (CONFIGS_DIR / "model").glob("*.yaml")]


@pytest.mark.parametrize("model_preset", _iter_model_presets())
def test_model_preset_loads_cleanly(model_preset: str):
    """Every model config preset must load without error alongside default_fit."""
    argv = ["-c", "default_fit", "-c", model_preset]
    cfg = load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)
    assert cfg.model_type in VALID_MODEL_TYPES, (
        f"{model_preset} sets model_type='{cfg.model_type}' which is not in VALID_MODEL_TYPES"
    )


# ---------------------------------------------------------------------------
# All training YAML presets are schema-compatible
# ---------------------------------------------------------------------------


def _iter_training_presets() -> list[str]:
    return [f"training/{p.stem}" for p in (CONFIGS_DIR / "training").glob("*.yaml")]


@pytest.mark.parametrize("training_preset", _iter_training_presets())
def test_training_preset_loads_cleanly(training_preset: str):
    """Every training preset must load without error alongside default_fit + poisson."""
    argv = ["-c", "default_fit", "-c", "model/poisson", "-c", training_preset]
    cfg = load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)
    assert cfg.training.num_steps > 0


# ---------------------------------------------------------------------------
# All data YAML presets are schema-compatible
# ---------------------------------------------------------------------------


def _iter_data_presets() -> list[str]:
    return [f"data/{p.stem}" for p in (CONFIGS_DIR / "data").glob("*.yaml")]


@pytest.mark.parametrize("data_preset", _iter_data_presets())
def test_data_preset_loads_cleanly(data_preset: str):
    """Every data preset must select a dataset_type and fill the dataset dict.

    The dataset payload is validated at runtime by the dataset registry, so the
    required keys differ per type (e.g. benchmark: dataset_name).
    Here we only assert the preset selects a type and populates a non-empty dict.
    """
    argv = ["-c", "default_fit", "-c", "model/poisson", "-c", data_preset]
    cfg = load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)
    assert isinstance(cfg.dataset_type, str) and cfg.dataset_type
    assert isinstance(cfg.dataset, dict) and cfg.dataset


# ---------------------------------------------------------------------------
# All eval YAML presets are schema-compatible
# ---------------------------------------------------------------------------


def _iter_eval_presets() -> list[str]:
    return [f"eval/{p.stem}" for p in (CONFIGS_DIR / "eval").glob("*.yaml")]


@pytest.mark.parametrize("eval_preset", _iter_eval_presets())
def test_eval_preset_loads_cleanly(eval_preset: str):
    """Every eval preset must satisfy EvalConfig (including limit_test_size)."""
    argv = ["-c", "default_fit", "-c", "model/poisson", "-c", eval_preset]
    cfg = load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)
    assert isinstance(cfg.eval, EvalConfig)


def _iter_all_presets() -> list[str]:
    return sorted(str(p.relative_to(CONFIGS_DIR).with_suffix("")) for p in CONFIGS_DIR.rglob("*.yaml"))


@pytest.mark.parametrize("preset", _iter_all_presets())
def test_every_shipped_preset_loads(preset: str):
    """Every YAML under configs/ resolves against the schema (no unknown keys)."""
    if preset == "default_eval":
        load_config(StandaloneEvalConfig, argv=["-c", preset, "model_path=x.pth"], configs_dir=CONFIGS_DIR)
    else:
        argv = ["-c", "default_fit", "-c", "model/poisson", "-c", preset]
        load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)


# ---------------------------------------------------------------------------
# VALID_MODEL_TYPES completeness
# ---------------------------------------------------------------------------


class TestValidModelTypes:
    def test_all_model_presets_in_valid_types(self):
        """Every model/*.yaml that sets model_type must be in VALID_MODEL_TYPES."""
        missing = []
        for preset_path in (CONFIGS_DIR / "model").glob("*.yaml"):
            raw = OmegaConf.load(preset_path)
            model_type = OmegaConf.select(raw, "model_type", default=None)
            if model_type is not None and model_type not in VALID_MODEL_TYPES:
                missing.append(f"{preset_path.name}: model_type={model_type!r}")
        assert not missing, (
            "The following model presets have model_type values not in VALID_MODEL_TYPES:\n" + "\n".join(missing)
        )

    def test_valid_model_types_no_duplicates(self):
        assert len(VALID_MODEL_TYPES) == len(set(VALID_MODEL_TYPES)), "VALID_MODEL_TYPES contains duplicates"

    def test_valid_model_types_non_empty(self):
        assert len(VALID_MODEL_TYPES) > 0


# ---------------------------------------------------------------------------
# Seed randomisation reproducibility
# ---------------------------------------------------------------------------


class TestSeedBehaviour:
    def test_two_loads_different_seeds(self, minimal_fit_argv):
        """Two loads with seed=None should (almost always) produce different seeds."""
        cfg1 = load_config(FitConfig, argv=minimal_fit_argv, configs_dir=CONFIGS_DIR)
        cfg2 = load_config(FitConfig, argv=minimal_fit_argv, configs_dir=CONFIGS_DIR)
        # There is a ~1/(2^31) chance this fails spuriously — acceptable.
        assert cfg1.arch.seed != cfg2.arch.seed, (
            "Two loads with seed=None produced the same seed (very unlikely unless broken)"
        )

    def test_explicit_seed_preserved(self, minimal_fit_argv):
        argv = minimal_fit_argv + ["arch.seed=12345"]
        cfg1 = load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)
        cfg2 = load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)
        assert cfg1.arch.seed == 12345
        assert cfg2.arch.seed == 12345


# ---------------------------------------------------------------------------
# Per-model architecture sections
# ---------------------------------------------------------------------------


class TestPerModelArch:
    def test_model_sections_are_parsed(self, minimal_fit_argv):
        argv = minimal_fit_argv + ["-c", "model/spline_spline_hawkes", "arch.kernel.num_knots=8"]
        cfg = load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)
        assert isinstance(cfg.arch, SplineBaselineSplineKernelArchConfig)
        assert cfg.arch.kernel.num_knots == 8 and cfg.arch.baseline.num_knots == 5

    def test_field_of_another_model_is_rejected(self, minimal_fit_argv):
        with pytest.raises(ConfigError, match="has no architecture field 'arch.encoder'.*Valid fields"):
            load_config(FitConfig, argv=minimal_fit_argv + ["arch.encoder.n_layers=2"], configs_dir=CONFIGS_DIR)

    def test_typo_suggests_the_field(self, minimal_fit_argv):
        argv = minimal_fit_argv + ["-c", "model/thp", "arch.encoder.n_layer=2"]
        with pytest.raises(ConfigError, match="Did you mean 'arch.encoder.n_layers'"):
            load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)

    def test_wrong_type_is_rejected(self, minimal_fit_argv):
        argv = minimal_fit_argv + ["-c", "model/spline_poisson", "arch.baseline.num_knots=abc"]
        with pytest.raises(ConfigError, match="arch.baseline.num_knots"):
            load_config(FitConfig, argv=argv, configs_dir=CONFIGS_DIR)

    def test_every_model_type_has_an_arch_config(self):
        import dataclasses

        from torchpp.training.arch import ArchConfig, IntensityArchConfig, build_arch
        from torchpp.tpp.base_tpp import IntensityBasedTPP
        from torchpp.training.registry import MODEL_REGISTRY

        for name, spec in MODEL_REGISTRY.items():
            arch = build_arch(name)
            assert isinstance(arch, ArchConfig), name
            assert isinstance(arch, IntensityArchConfig) == issubclass(spec.cls, IntensityBasedTPP), name
            assert dataclasses.asdict(build_arch(name, dataclasses.asdict(arch))) == dataclasses.asdict(arch)

