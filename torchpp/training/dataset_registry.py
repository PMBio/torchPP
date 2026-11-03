"""
Dataset registry for TorchPP training scripts.

Registering a new dataset
--------------------------
Decorate the :class:`~torchpp.data.base_dataset.TPPDataset` subclass with
:func:`register_dataset` and implement the :meth:`load` classmethod::

    from torchpp.training.dataset_registry import register_dataset
    from torchpp.data.base_dataset import TPPDataset

    @register_dataset("my_dataset")
    class MyDataset(TPPDataset):
        # Dataset-intrinsic properties — no config needed.
        terminating = False
        default_token_roles = TokenRoles()  # origin regime, context and ignored types

        @classmethod
        def load(cls, cfg: dict, root: Path, split: str) -> "MyDataset":
            # cfg contains only the operational keys from the YAML / CLI.
            # split is one of "train", "val", "test".
            ...

Then add a YAML preset in ``configs/data/`` that populates the ``dataset:``
key with operational parameters (paths, size limits, etc.).  No changes to
:class:`~torchpp.training.config.FitConfig` are needed.

At runtime the training script loads a split via::

    from torchpp.training.dataset_registry import load_split

    dataset = load_split(dataset_type, cfg_dict, root, split="train")
    D = dataset.num_event_types
    terminating = dataset.terminating    # class-level attribute
    roles = dataset.token_roles          # default roles plus per-run overrides

``dataset.token_roles`` applies any per-run ``dataset.context_types`` /
``dataset.ignored_types`` override to the class-level ``default_token_roles``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

_TORCHPPDATA_ENV = "TORCHPPDATA"
_DEFAULT_DATA_ROOT = Path.home() / "data" / "torchpp"


def get_data_root() -> Path:
    """Return the root directory for all TorchPP datasets.

    Reads the ``TORCHPPDATA`` environment variable.  Falls back to
    ``~/data/torchpp/`` when unset.
    """
    return Path(os.environ.get(_TORCHPPDATA_ENV, _DEFAULT_DATA_ROOT)).expanduser()

# ---------------------------------------------------------------------------
# Shared operational config
# ---------------------------------------------------------------------------

# Keys accepted for every dataset.  ``context_types`` / ``ignored_types`` override the
# dataset's token roles on the returned instance.  ``limit_{train,val,test}_size`` are forwarded to loaders that declare
# them in ``cfg_keys`` (they subset before their own preprocessing) and applied by the
# registry, as the first ``n`` sequences, for every other dataset.
_LIMIT_KEYS = {"train": "limit_train_size", "val": "limit_val_size", "test": "limit_test_size"}
_COMMON_KEYS = {"context_types", "ignored_types", *_LIMIT_KEYS.values()}
_REGISTRY_KEYS = {"context_types", "ignored_types"}


# ---------------------------------------------------------------------------
# Registry infrastructure
# ---------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    """Registry entry for one dataset type.

    Attributes:
        loader_class:   The :class:`~torchpp.data.base_dataset.TPPDataset`
                        subclass that handles this dataset type.  It must
                        implement the :meth:`load` classmethod.
        valid_cfg_keys: Set of parameter names (beyond the shared keys) that
                        the loader accepts in its ``cfg`` dict.  Used for
                        early validation of YAML/CLI input.
    """

    loader_class: Type[Any]
    valid_cfg_keys: frozenset[str]


# Maps dataset_type string → DatasetSpec.
DATASET_REGISTRY: Dict[str, DatasetSpec] = {}


def register_dataset(name: str) -> Callable[[Type[Any]], Type[Any]]:
    """Class decorator that registers a :class:`~torchpp.data.base_dataset.TPPDataset`
    subclass under *name* in :data:`DATASET_REGISTRY`.

    The decorated class must:

    * Implement a ``load(cls, cfg: dict, root: Path, split: str)`` classmethod
      that constructs and returns a dataset instance.
    * Declare a class-level ``terminating: bool`` attribute.
    * Optionally declare ``default_token_roles: TokenRoles`` (origin regime, context and
      ignored types); the default is no origin and every type predictable.
    * Optionally declare ``cfg_keys: frozenset[str]``, the accepted ``dataset.*`` keys;
      unknown keys then raise when the configuration is loaded.

    Args:
        name: Unique registry key (e.g. ``"benchmark"``).

    Raises:
        ValueError: If *name* is already registered.

    Example::

        @register_dataset("my_dataset")
        class MyDataset(TPPDataset):
            terminating = True
            default_token_roles = TokenRoles(has_origin=True, ignored_types=range(11))

            @classmethod
            def load(cls, cfg, root, split): ...
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        if name in DATASET_REGISTRY:
            raise ValueError(
                f"Dataset '{name}' is already registered.  Use a different name or remove the existing registration."
            )
        DATASET_REGISTRY[name] = DatasetSpec(
            loader_class=cls,
            valid_cfg_keys=frozenset(getattr(cls, "cfg_keys", frozenset())),
        )
        return cls

    return decorator


def validate_dataset_cfg(dataset_type: str, cfg: dict) -> None:
    """Raise ``ValueError`` for keys the dataset does not declare in ``cfg_keys``.

    Datasets that declare no ``cfg_keys`` accept any key.
    """
    import difflib

    if dataset_type not in DATASET_REGISTRY:
        close = difflib.get_close_matches(str(dataset_type), sorted(DATASET_REGISTRY), n=1)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        valid = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(
            f"Unknown dataset_type '{dataset_type}'.{hint} Registered types: {valid}. "
            "A dataset defined in your own module needs dataset_module=<module>."
        )
    known = DATASET_REGISTRY[dataset_type].valid_cfg_keys
    if not known:
        return
    for key in cfg:
        if key in _COMMON_KEYS or key in known:
            continue
        close = difflib.get_close_matches(str(key), sorted(known | _COMMON_KEYS), n=1)
        hint = f" Did you mean 'dataset.{close[0]}'?" if close else ""
        raise ValueError(
            f"Unknown key 'dataset.{key}' for dataset_type='{dataset_type}'.{hint} Valid keys: {sorted(known)}."
        )


def load_split(dataset_type: str, cfg: dict, root: Path, split: str) -> Any:
    """Load one split of a registered dataset.

    This is the single entry point used by training scripts.  It:

    1. Looks up the loader class for *dataset_type*.
    2. Validates the cfg dict for unrecognised keys.
    3. Applies the ``context_types`` / ``ignored_types`` overrides from *cfg* to the
       returned instance's ``token_roles`` (or leaves the class defaults in place).
    4. Delegates construction to ``loader_class.load(cfg, root, split)``.

    Args:
        dataset_type: Registered name.
        cfg:          Raw dataset parameter dict from the merged config.
        root:         Data root (``$TORCHPPDATA``); relative paths in *cfg* are resolved
                      against this.
        split:        One of ``"train"``, ``"val"``, ``"test"``.

    Returns:
        A :class:`~torchpp.data.base_dataset.TPPDataset` instance whose
        ``token_roles`` reflect any override in *cfg*.

    Raises:
        ValueError: If *dataset_type* is not registered or *cfg* has keys the dataset does
            not declare in ``cfg_keys``.
    """
    if dataset_type not in DATASET_REGISTRY:
        valid = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset_type '{dataset_type}'.  Registered types: {valid}")

    spec = DATASET_REGISTRY[dataset_type]

    # The token-role keys are handled here; the rest is forwarded to the loader.
    role_overrides = {k: cfg[k] for k in _REGISTRY_KEYS if cfg.get(k) is not None}
    declared = spec.valid_cfg_keys
    loader_cfg = {
        k: v for k, v in cfg.items() if k not in _REGISTRY_KEYS and (k not in _LIMIT_KEYS.values() or k in declared)
    }

    validate_dataset_cfg(dataset_type, cfg)
    dataset = spec.loader_class.load(loader_cfg, root, split)
    limit_key = _LIMIT_KEYS.get(split)
    limit = cfg.get(limit_key) if limit_key else None
    if limit is not None and limit > 0 and limit_key not in declared and len(dataset) > limit:
        from torchpp.data.base_dataset import LimitedDataset

        dataset = LimitedDataset(dataset, limit)

    if role_overrides:
        dataset.token_roles = dataset.token_roles.replace(**{k: tuple(v) for k, v in role_overrides.items()})

    return dataset
