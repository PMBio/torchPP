"""
Config loading utilities for TorchPP.

Supports three layers, merged in order (later layers win):

1. Defaults defined in dataclass structured configs (only fields that have
   explicit default values — ``MISSING`` fields are left unset and must be
   supplied by one of the later layers).
2. One or more YAML config files (``--config path``, repeatable, short-form
   ``-c`` also accepted).
3. Positional ``key=value`` CLI overrides (dotted paths via OmegaConf).

Usage in a script::

    from torchpp.training.config_utils import load_config
    from torchpp.training.config import FitConfig

    config = load_config(FitConfig, sys.argv[1:])

The loader accepts a mix of styles, e.g.::

    # Full path
    python train.py --config configs/model/spline_exp_hawkes.yaml \\
        training.num_steps=1000

    # Short paths — resolved relative to ``configs/``, extension optional
    python train.py -c default_fit -c model/spline_exp_hawkes \\
        -c data/taxi wandb.enable=true

Resolution order for each ``--config`` / ``-c`` token:
  1. Raw path as given (absolute, or relative from CWD with extension).
  2. ``<configs_dir>/<value>``
  3. ``<configs_dir>/<value>.yaml``
  4. ``<configs_dir>/<value>.yml``

``defaults:`` list inside a YAML file (Hydra-style composition)::

    # configs/default_fit.yaml
    defaults:
      - wandb/disabled

Each entry in ``defaults:`` is resolved with the same rules above and merged
left-to-right; the file's own keys are merged on top (file always wins).
Entries are resolved recursively, so ``defaults:`` lists can be nested.

A model preset may set ``training_preset: <name>`` (e.g. ``classical`` or ``neural``).
``training/<name>.yaml`` is then merged before all ``-c`` files, so an explicit
``-c training/...`` or ``training.*=...`` override always wins, regardless of order.

After merging all layers, the result is validated against the dataclass schema:
``MISSING`` fields that were never set, unknown keys (with a "did you mean"
suggestion), values of the wrong type, and unknown ``dataset.*`` keys of datasets
that declare their keys all raise a :class:`ConfigError`.
"""

from __future__ import annotations

import dataclasses
import random
import os
import sys
import typing
from pathlib import Path
from typing import Any, Optional, Sequence, Type, TypeVar

from omegaconf import DictConfig, OmegaConf

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Invalid command line or configuration."""


def _split_argv(argv: Sequence[str]) -> tuple[list[str], list[str], bool, bool]:
    """Split *argv* into (config_paths, key=value overrides, print_and_exit, show_help).

    ``--config <path>`` and ``-c <path>`` may be repeated; ``--print`` / ``-p`` prints the
    merged configuration and exits; ``--help`` / ``-h`` prints usage. Any other token
    must be a ``key=value`` override.
    """
    config_paths: list[str] = []
    overrides: list[str] = []
    print_and_exit = False
    show_help = False

    it = iter(argv)
    for tok in it:
        if tok in ("--config", "-c"):
            path = next(it, None)
            if path is None:
                raise ConfigError(f"{tok} needs a preset name or a YAML path.")
            config_paths.append(path)
        elif tok.startswith("--config="):
            config_paths.append(tok.split("=", 1)[1])
        elif tok in ("--print", "-p"):
            print_and_exit = True
        elif tok in ("--help", "-h"):
            show_help = True
        elif tok.startswith("-"):
            raise ConfigError(f"Unknown option '{tok}'. Options: -c/--config, -p/--print, -h/--help.")
        elif "=" not in tok:
            raise ConfigError(f"Expected a key=value override, got '{tok}' (missing '='?).")
        else:
            overrides.append(tok)

    return config_paths, overrides, print_and_exit, show_help


def _usage(configs_dir: Path, description: Optional[str] = None) -> str:
    script = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "<script>.py"
    lines = [f"usage: python {script} [-c PRESET ...] [key=value ...] [--print]", ""]
    if description:
        lines += [description, ""]
    lines += [
        "  -c, --config PRESET   YAML preset (name under configs/ or a path); repeatable, later wins",
        "  key=value             override a field, e.g. training.num_steps=1000",
        "  -p, --print           print the merged configuration and exit",
        "",
        f"Presets in {configs_dir}:",
    ]
    if configs_dir.is_dir():
        for sub in sorted(d for d in configs_dir.rglob("*") if d.is_dir() and any(d.glob("*.y*ml"))):
            names = sorted(f.stem for f in sub.glob("*.y*ml"))
            lines.append(f"  {sub.relative_to(configs_dir).as_posix()}/: {', '.join(names)}")
        top = sorted(f.stem for f in configs_dir.glob("*.y*ml"))
        if top:
            lines.append(f"  (top level): {', '.join(top)}")
    return "\n".join(lines)


def _check_keys(cfg: Any, dataclass_cls: type, prefix: str = "") -> None:
    """Raise for keys that are not fields of *dataclass_cls* (recursively)."""
    import dataclasses
    import difflib

    hints = typing.get_type_hints(dataclass_cls)
    names = [f.name for f in dataclasses.fields(dataclass_cls)]
    for key in cfg:
        if key not in names:
            close = difflib.get_close_matches(str(key), names, n=1)
            hint = f" Did you mean '{prefix}{close[0]}'?" if close else ""
            where = f"section '{prefix[:-1]}'" if prefix else "the top level"
            raise ConfigError(f"Unknown key '{prefix}{key}' in {where}.{hint}")
        sub = hints.get(key)
        value = cfg[key]
        if dataclasses.is_dataclass(sub) and isinstance(value, DictConfig):
            _check_keys(value, sub, f"{prefix}{key}.")


def _check_type(value: Any, annotation: Any, key: str) -> None:
    """Raise when a scalar config value does not match its annotation."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Union:
        if value is None and type(None) in args:
            return
        options = [a for a in args if a is not type(None)]
        errors = []
        for a in options:
            try:
                _check_type(value, a, key)
                return
            except ConfigError as e:
                errors.append(e)
        raise errors[0]
    if origin is typing.Literal:
        if value not in args:
            raise ConfigError(f"'{key}' must be one of {list(args)}, got {value!r}.")
        return
    if annotation is bool:
        ok = isinstance(value, bool)
    elif annotation is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif annotation is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif annotation is str:
        ok = isinstance(value, str)
    else:
        return  # containers and other types are not checked here
    if not ok:
        raise ConfigError(f"'{key}' must be {annotation.__name__}, got {value!r}.")


def _resolve_config_path(raw: str, configs_dir: Path) -> Path:
    """Resolve a raw ``--config`` token to an existing file path.

    Tries in order:
      1. ``raw`` as-is (absolute or explicit relative path).
      2. ``configs_dir / raw``
      3. ``configs_dir / raw + ".yaml"``
      4. ``configs_dir / raw + ".yml"``

    Raises ``FileNotFoundError`` listing all tried paths if nothing resolves.
    """
    candidates = [
        Path(raw),
        configs_dir / raw,
        configs_dir / (raw + ".yaml"),
        configs_dir / (raw + ".yml"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    import difflib

    known = sorted(str(f.relative_to(configs_dir).with_suffix("")) for f in configs_dir.rglob("*.y*ml")) if configs_dir.is_dir() else []
    close = difflib.get_close_matches(raw, known, n=1)
    hint = f" Did you mean '{close[0]}'?" if close else " See --help for the presets."
    raise FileNotFoundError(f"Preset '{raw}' not found under {configs_dir}.{hint}")


def _load_yaml_with_defaults(yaml_path: Path, configs_dir: Path) -> DictConfig:
    """Load a YAML file and recursively expand any top-level ``defaults:`` list.

    The ``defaults:`` list (if present) contains short config paths resolved
    via :func:`_resolve_config_path`.  Each entry is loaded and merged
    left-to-right; the current file's own keys are merged on top so the file
    always wins over its defaults.
    """
    raw_cfg: DictConfig = OmegaConf.load(yaml_path)  # type: ignore[assignment]
    defaults = OmegaConf.select(raw_cfg, "defaults", default=None)
    if defaults is None:
        return raw_cfg

    from omegaconf import ListConfig

    if not isinstance(defaults, (list, tuple, ListConfig)):
        raise ValueError(f"{yaml_path}: 'defaults' must be a list, got {type(defaults).__name__}")

    merged: DictConfig = OmegaConf.create({})
    for entry in defaults:
        entry_path = _resolve_config_path(str(entry), configs_dir)
        entry_cfg = _load_yaml_with_defaults(entry_path, configs_dir)  # recursive
        merged = OmegaConf.merge(merged, entry_cfg)

    # Strip the 'defaults' key then merge the file's own keys on top.
    own_keys = [k for k in raw_cfg if k != "defaults"]
    file_cfg: DictConfig = OmegaConf.masked_copy(raw_cfg, own_keys)  # type: ignore[arg-type]
    return OmegaConf.merge(merged, file_cfg)


def _dataclass_to_omegaconf(dataclass_cls: type) -> DictConfig:
    """Convert a dataclass (possibly nested) to an OmegaConf DictConfig.

    Fields whose default is ``MISSING`` are left as ``OmegaConf.MISSING``
    (``???``) in the resulting config, which ensures they must be filled before
    the config can be used.  Fields with actual defaults are seeded with those
    values.
    """
    import dataclasses
    from omegaconf import MISSING as OC_MISSING

    if not dataclasses.is_dataclass(dataclass_cls):
        raise TypeError(f"{dataclass_cls} is not a dataclass")

    result: dict = {}
    for f in dataclasses.fields(dataclass_cls):
        from omegaconf import MISSING as _MISSING

        if f.default is _MISSING and f.default_factory is dataclasses.MISSING:  # type: ignore[misc]
            result[f.name] = OC_MISSING
        elif f.default is not dataclasses.MISSING:
            from omegaconf import MISSING as _OMISSING
            if f.default is _OMISSING:
                result[f.name] = OC_MISSING
            elif dataclasses.is_dataclass(f.default) or dataclasses.is_dataclass(type(f.default)):
                result[f.name] = _dataclass_to_omegaconf(type(f.default))
            else:
                result[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            factory_result = f.default_factory()  # type: ignore[misc]
            if dataclasses.is_dataclass(factory_result):
                result[f.name] = _dataclass_to_omegaconf(type(factory_result))
            else:
                result[f.name] = factory_result

    return OmegaConf.create(result)  # type: ignore[return-value]


def _omegaconf_to_dataclass(cfg: DictConfig, dataclass_cls: Type[T]) -> T:
    """Instantiate a (possibly nested) dataclass from a DictConfig.

    Raises ``MissingMandatoryValue`` for any ``MISSING`` fields that were never
    filled and :class:`ConfigError` for values of the wrong type.
    """
    import dataclasses
    from omegaconf import MISSING as OC_MISSING

    raw: dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[assignment]

    def _build(value: Any, cls: type, prefix: str) -> Any:
        hints = typing.get_type_hints(cls)
        fields = {}
        for f in dataclasses.fields(cls):
            if f.name not in value:
                continue
            ann, v = hints[f.name], value[f.name]
            if dataclasses.is_dataclass(ann) and isinstance(v, dict):
                fields[f.name] = _build(v, ann, f"{prefix}{f.name}.")
            else:
                _check_type(v, ann, f"{prefix}{f.name}")
                fields[f.name] = v
        return cls(**fields)

    return _build(raw, dataclass_cls, "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _import_user_module(key: str, name: str) -> None:
    """Import a user module that registers datasets or models; the working directory is importable."""
    import importlib

    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    try:
        importlib.import_module(name)
    except ImportError as exc:
        raise ConfigError(f"{key}={name!r} could not be imported: {exc}") from exc


def load_config(
    model_cls: Type[T],
    argv: Optional[Sequence[str]] = None,
    configs_dir: str | Path = "configs",
    description: Optional[str] = None,
) -> T:
    """Build a validated config of type *model_cls* from defaults + YAML + CLI.

    Three-layer merge (later layers win):

    1. Dataclass field defaults (``MISSING`` fields left unset).
    2. YAML files supplied via ``--config`` / ``-c`` flags (in order).
    3. ``key=value`` CLI overrides (dotted paths).

    After merging, any ``MISSING`` field that was never provided triggers a
    ``MissingMandatoryValue`` error with a clear message naming the field.

    Args:
        model_cls:    A dataclass class (e.g. ``FitConfig``).
        argv:         Argument list to parse (defaults to ``sys.argv[1:]``).
        configs_dir:  Root directory for resolving short ``--config`` paths.
        description:  One-line description of the calling script, shown by ``--help``.
                      Defaults to ``"configs"`` relative to CWD.

    Returns:
        A fully resolved and populated instance of *model_cls*.

    Raises:
        ``omegaconf.MissingMandatoryValue``: A required field was never set.
        ``FileNotFoundError``: A ``--config`` path cannot be resolved.
        ConfigError: A command-line token is malformed, a key does not exist in the schema,
            or a value has the wrong type.
    """
    if argv is None:
        argv = sys.argv[1:]

    configs_root = Path(configs_dir)
    config_paths, overrides, print_and_exit, show_help = _split_argv(argv)
    if show_help:
        print(_usage(configs_root, description))
        sys.exit(0)

    # Layer 1: dataclass defaults (MISSING where no default exists).
    base_cfg: DictConfig = _dataclass_to_omegaconf(model_cls)

    file_cfgs = [
        _load_yaml_with_defaults(_resolve_config_path(p, configs_root), configs_root) for p in config_paths
    ]
    cli_cfg: Optional[DictConfig] = OmegaConf.from_dotlist(overrides) if overrides else None  # type: ignore[assignment]

    # A model preset names its family's training defaults (``training_preset: neural``).
    # They are merged before every -c file, so an explicit -c training/... or
    # training.*=... always wins, whatever the order on the command line.
    training_preset = None
    for layer in file_cfgs + ([cli_cfg] if cli_cfg is not None else []):
        if "training_preset" in layer:
            training_preset = layer.pop("training_preset")
    has_training = "training" in {f.name for f in dataclasses.fields(model_cls)}
    if training_preset is None and has_training:
        # model_type given without its preset (e.g. model_type=thp): use the preset's family.
        model_type = None
        for layer in file_cfgs + ([cli_cfg] if cli_cfg is not None else []):
            model_type = OmegaConf.select(layer, "model_type", default=model_type)
        if isinstance(model_type, str):
            from torchpp.training.registry import MODEL_ALIASES

            model_preset = configs_root / "model" / f"{MODEL_ALIASES.get(model_type, model_type)}.yaml"
            if model_preset.is_file():
                training_preset = OmegaConf.load(model_preset).get("training_preset")
    if training_preset and has_training:
        preset_path = _resolve_config_path(f"training/{training_preset}", configs_root)
        base_cfg = OmegaConf.merge(base_cfg, _load_yaml_with_defaults(preset_path, configs_root))

    # Layer 2: YAML files merged in order, later files win.
    for file_cfg in file_cfgs:
        base_cfg = OmegaConf.merge(base_cfg, file_cfg)

    # Layer 3: key=value CLI overrides.
    if cli_cfg is not None:
        base_cfg = OmegaConf.merge(base_cfg, cli_cfg)

    # Resolve device="auto" and randomise seed=None before constructing the
    # dataclass so the dataclass receives final values.
    if OmegaConf.select(base_cfg, "device", default=None) == "auto":
        import torch
        OmegaConf.update(base_cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")

    if has_training and OmegaConf.select(base_cfg, "arch.seed", default=None) is None:
        OmegaConf.update(base_cfg, "arch.seed", random.randint(0, 2**31 - 1), force_add=True)

    _check_keys(base_cfg, model_cls)
    for key in ("dataset_module", "model_module"):
        module = OmegaConf.select(base_cfg, key, default=None)
        if module:
            _import_user_module(key, module)
    dataset_type = OmegaConf.select(base_cfg, "dataset_type", default=None)
    dataset_cfg = OmegaConf.select(base_cfg, "dataset", default=None)
    if isinstance(dataset_type, str) and dataset_cfg is not None:
        from torchpp.training.dataset_registry import validate_dataset_cfg

        validate_dataset_cfg(dataset_type, OmegaConf.to_container(dataset_cfg))

    # Construction also validates: MISSING fields raise here with a clear error.
    cfg = _omegaconf_to_dataclass(base_cfg, model_cls)

    from torchpp.training.registry import resolve_model_type

    model_type = getattr(cfg, "model_type", None)
    if model_type is not None:
        resolve_model_type(model_type)  # raises ValueError for unknown names
    arch = getattr(cfg, "arch", None)
    if isinstance(arch, dict):
        from torchpp.tpp.base_tpp import CIConfig, SamplingConfig
        from torchpp.training.arch import build_arch
        from torchpp.training.model_factory import _config_overrides

        try:
            _config_overrides(CIConfig, arch.get("ci") or {}, "arch.ci")
            _config_overrides(SamplingConfig, arch.get("sampling") or {}, "arch.sampling")
            if has_training and model_type is not None:
                cfg.arch = build_arch(model_type, arch)  # only this model's fields are accepted
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

    if print_and_exit:
        print_config(cfg)
        sys.exit(0)

    return cfg  # type: ignore[return-value]


def print_config(cfg: Any, title: str = "Configuration") -> None:
    """Pretty-print a dataclass config (grouped by sub-config).

    The ``dataset`` dict field is rendered as a sub-section labelled with the
    dataset type so it reads like a named config group rather than a raw dict.
    """
    import dataclasses

    width = 60
    print("=" * width)
    print(f"{title}:")
    print("=" * width)

    def _print_section(obj: Any, indent: str) -> None:
        for sf in dataclasses.fields(obj):
            value = getattr(obj, sf.name)
            if dataclasses.is_dataclass(value):
                print(f"{indent}{sf.name}:")
                _print_section(value, indent + "  ")
            else:
                print(f"{indent}{sf.name}: {value}")

    def _print_obj(obj: Any, prefix: str = "") -> None:
        if dataclasses.is_dataclass(obj):
            for f in dataclasses.fields(obj):
                value = getattr(obj, f.name)
                full_key = f"{prefix}{f.name}"
                if dataclasses.is_dataclass(value):
                    print(f"\n  [{f.name}]")
                    _print_section(value, "    ")
                elif f.name == "dataset" and isinstance(value, dict):
                    dataset_type = getattr(obj, "dataset_type", "?")
                    print(f"\n  [dataset]  # type: {dataset_type}")
                    if value:
                        for k, v in value.items():
                            print(f"    {k}: {v}")
                    else:
                        print("    (no parameters set)")
                else:
                    print(f"  {full_key}: {value}")
        else:
            print(f"  {prefix}: {obj}")

    _print_obj(cfg)
    print("=" * width)
