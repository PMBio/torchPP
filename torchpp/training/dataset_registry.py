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
        terminating: bool = False
        default_ignore_indices: list[int] = []

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
    terminating = dataset.terminating        # class-level attribute
    ignore_indices = dataset.ignore_indices  # effective list (override or default)

``ignore_indices`` on the instance reflects any per-run override that was set
via the ``dataset.ignore_indices`` YAML/CLI key; it falls back to the class-
level ``default_ignore_indices`` when no override is provided.
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

# Keys that are handled by the registry itself before forwarding the config
# dict to the loader.  ``ignore_indices`` is applied to the returned instance;
# split-size limits are read directly by each loader from the full cfg dict,
# so they are *not* stripped here.
_COMMON_KEYS = {"ignore_indices"}


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
    * Declare a class-level ``default_ignore_indices: list[int]`` attribute.

    Args:
        name: Unique registry key (e.g. ``"ukb"``).

    Raises:
        ValueError: If *name* is already registered.

    Example::

        @register_dataset("ukb")
        class UKBDataset(TPPDataset):
            terminating = True
            default_ignore_indices = list(range(11))

            @classmethod
            def load(cls, cfg, root, split): ...
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        if name in DATASET_REGISTRY:
            raise ValueError(
                f"Dataset '{name}' is already registered.  Use a different name or remove the existing registration."
            )
        import inspect

        try:
            sig = inspect.signature(cls.load)
            loader_keys = frozenset(sig.parameters) - {"cls", "cfg", "root", "split"}
        except (AttributeError, ValueError):
            loader_keys = frozenset()

        DATASET_REGISTRY[name] = DatasetSpec(
            loader_class=cls,
            valid_cfg_keys=loader_keys,
        )
        return cls

    return decorator


def load_split(dataset_type: str, cfg: dict, root: Path, split: str) -> Any:
    """Load one split of a registered dataset.

    This is the single entry point used by training scripts.  It:

    1. Looks up the loader class for *dataset_type*.
    2. Validates the cfg dict for unrecognised keys.
    3. Applies the ``ignore_indices`` override from *cfg* onto the returned
       instance (or leaves the class default in place).
    4. Delegates construction to ``loader_class.load(cfg, root, split)``.

    Args:
        dataset_type: Registered name.
        cfg:          Raw dataset parameter dict from the merged config.
        root:         Repository root; relative paths in *cfg* are resolved
                      against this.
        split:        One of ``"train"``, ``"val"``, ``"test"``.

    Returns:
        A :class:`~torchpp.data.base_dataset.TPPDataset` instance whose
        ``ignore_indices`` attribute reflects any override in *cfg*.

    Raises:
        ValueError: If *dataset_type* is not registered.
        TypeError:  If *cfg* contains keys unknown to the loader.
    """
    if dataset_type not in DATASET_REGISTRY:
        valid = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset_type '{dataset_type}'.  Registered types: {valid}")

    spec = DATASET_REGISTRY[dataset_type]

    # Common keys (e.g. ignore_indices) are handled here; the rest is forwarded
    # to the loader.
    ignore_indices_override: Optional[List[int]] = cfg.get("ignore_indices", None)
    loader_cfg = {k: v for k, v in cfg.items() if k not in _COMMON_KEYS}

    known = spec.valid_cfg_keys
    if known:  # only validate when the loader declared its accepted keys
        unknown = set(loader_cfg) - known
        if unknown:
            raise TypeError(
                f"Unknown keys for dataset_type='{dataset_type}': {sorted(unknown)}.  Valid keys: {sorted(known)}"
            )

    dataset = spec.loader_class.load(loader_cfg, root, split)

    if ignore_indices_override is not None:
        dataset._ignore_indices_override = list(ignore_indices_override)

    return dataset
