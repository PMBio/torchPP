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
        -c training/very_long wandb.enable=true

Resolution order for each ``--config`` / ``-c`` token:
  1. Raw path as given (absolute, or relative from CWD with extension).
  2. ``<configs_dir>/<value>``
  3. ``<configs_dir>/<value>.yaml``
  4. ``<configs_dir>/<value>.yml``

``defaults:`` list inside a YAML file (Hydra-style composition)::

    # configs/default_fit.yaml
    defaults:
      - data/ukb_real_full
      - batching/bucket_default_small
      - training/very_long
      - wandb/disabled
      - eval/normal

Each entry in ``defaults:`` is resolved with the same rules above and merged
left-to-right; the file's own keys are merged on top (file always wins).
Entries are resolved recursively, so ``defaults:`` lists can be nested.

After merging all layers, the result is validated: all ``MISSING`` fields that
were never set produce a clear ``MissingMandatoryValue`` error, and unknown
keys (typos) are caught by OmegaConf's struct mode before the dataclass is
constructed.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Type, TypeVar

from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_argv(argv: Sequence[str]) -> tuple[list[str], list[str], bool]:
    """Split *argv* into (config_paths, key=value overrides, print_and_exit).

    ``--config <path>`` and ``-c <path>`` may be repeated.
    ``--print`` / ``-p`` sets the print-and-exit flag.
    Everything else is treated as a ``key=value`` dotlist override or ignored.
    """
    config_paths: list[str] = []
    overrides: list[str] = []
    print_and_exit: bool = False

    it = iter(argv)
    for tok in it:
        if tok in ("--config", "-c"):
            path = next(it, None)
            if path is not None:
                config_paths.append(path)
        elif tok.startswith("--config="):
            config_paths.append(tok.split("=", 1)[1])
        elif tok in ("--print", "-p"):
            print_and_exit = True
        elif tok.startswith("-") and "=" not in tok:
            # Skip silently so callers can layer their own flag parsing on top.
            pass
        else:
            overrides.append(tok)

    return config_paths, overrides, print_and_exit


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
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Config file not found for '{raw}'.  Tried:\n  {tried}")


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

    Raises ``MissingMandatoryValue`` for any ``MISSING`` fields that were
    never filled.  Raises ``omegaconf.errors.ConfigAttributeError`` for
    unknown keys (struct-mode check).
    """
    import dataclasses
    from omegaconf import MISSING as OC_MISSING

    raw: dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[assignment]

    field_types = {f.name: f.type for f in dataclasses.fields(dataclass_cls)}

    def _convert(value: Any, annotation: Any) -> Any:
        import typing
        # Forward references can show up as raw strings under PEP 563.
        if isinstance(annotation, str):
            import sys
            annotation = eval(annotation, sys.modules[dataclass_cls.__module__].__dict__)

        origin = getattr(annotation, "__origin__", None)
        if origin is type(None):
            return value
        if origin is not None:
            # Generic alias (Optional[X], List[X], ...) — leave the value as-is
            # rather than trying to recurse into a dataclass.
            return value

        if dataclasses.is_dataclass(annotation) and isinstance(value, dict):
            sub_fields = {f.name: f.type for f in dataclasses.fields(annotation)}
            converted = {k: _convert(v, sub_fields.get(k, type(v))) for k, v in value.items()}
            return annotation(**converted)
        return value

    resolved_fields = {}
    for f in dataclasses.fields(dataclass_cls):
        if f.name in raw:
            resolved_fields[f.name] = _convert(raw[f.name], f.type)

    return dataclass_cls(**resolved_fields)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    model_cls: Type[T],
    argv: Optional[Sequence[str]] = None,
    configs_dir: str | Path = "configs",
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
                      Defaults to ``"configs"`` relative to CWD.

    Returns:
        A fully resolved and populated instance of *model_cls*.

    Raises:
        ``omegaconf.MissingMandatoryValue``: A required field was never set.
        ``FileNotFoundError``: A ``--config`` path cannot be resolved.
        ``omegaconf.OmegaConfBaseException``: A CLI override is malformed or
            references a key that does not exist in the schema.
    """
    if argv is None:
        argv = sys.argv[1:]

    configs_root = Path(configs_dir)
    config_paths, overrides, print_and_exit = _split_argv(argv)

    # Layer 1: dataclass defaults (MISSING where no default exists).
    base_cfg: DictConfig = _dataclass_to_omegaconf(model_cls)

    # Layer 2: YAML files merged in order, later files win.
    for config_path in config_paths:
        yaml_path = _resolve_config_path(config_path, configs_root)
        file_cfg: DictConfig = _load_yaml_with_defaults(yaml_path, configs_root)
        base_cfg = OmegaConf.merge(base_cfg, file_cfg)

    # Layer 3: key=value CLI overrides.
    if overrides:
        cli_cfg: DictConfig = OmegaConf.from_dotlist(overrides)  # type: ignore[assignment]
        base_cfg = OmegaConf.merge(base_cfg, cli_cfg)

    # Resolve device="auto" and randomise seed=None before constructing the
    # dataclass so the dataclass receives final values.
    if OmegaConf.select(base_cfg, "device", default=None) == "auto":
        import torch
        OmegaConf.update(base_cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")

    seed_path = "arch.seed"
    arch_seed = OmegaConf.select(base_cfg, seed_path, default="__sentinel__")
    if arch_seed is None:
        OmegaConf.update(base_cfg, seed_path, random.randint(0, 2**31 - 1))

    # Construction also validates: MISSING fields raise here with a clear error.
    cfg = _omegaconf_to_dataclass(base_cfg, model_cls)

    from torchpp.training.config import VALID_MODEL_TYPES
    model_type = getattr(cfg, "model_type", None)
    if model_type is not None and model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Valid choices: {', '.join(VALID_MODEL_TYPES)}"
        )

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

    def _print_obj(obj: Any, prefix: str = "") -> None:
        if dataclasses.is_dataclass(obj):
            for f in dataclasses.fields(obj):
                value = getattr(obj, f.name)
                full_key = f"{prefix}{f.name}"
                if dataclasses.is_dataclass(value):
                    print(f"\n  [{f.name}]")
                    for sf in dataclasses.fields(value):
                        print(f"    {sf.name}: {getattr(value, sf.name)}")
                elif f.name == "dataset" and isinstance(value, dict):
                    dataset_type = getattr(obj, "dataset_type", "?")
                    print(f"\n  [dataset]  # type: {dataset_type}")
                    if value:
                        for k, v in value.items():
                            print(f"    {k}: {v}")
                    else:
                        print("    (empty — using registered defaults)")
                else:
                    print(f"  {full_key}: {value}")
        else:
            print(f"  {prefix}: {obj}")

    _print_obj(cfg)
    print("=" * width)
