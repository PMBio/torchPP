"""Dataset and loading utilities for EasyTPP benchmark datasets.

Datasets are stored as ``$TORCHPPDATA/easytpp/<name>/{train,dev,test}.pt``, where each
``.pt`` file is a ``list[MVEventData]`` saved with :func:`torch.save`.

The files are produced by ``scripts/data_prep/download_easytpp_datasets.py``.

Dataset registry
----------------
name            num_event_types
taxi            10
stackoverflow   22
retweet          3
amazon          16
taobao          17
earthquake       7
volcano          1

Time conventions
----------------
EasyTPP raw timestamps (``time_since_start``) carry a per-sequence offset
relative to an external reference point — **not** to the first event.  This
offset is meaningless for our models and is removed at load time.

Preprocessing per sequence:

1. **Remove offset** — subtract the first timestamp so that inter-event
   times are preserved and the sequence starts at 0.
2. **Place first event at mean inter-event time** — shift the whole
   sequence by the training-set mean Δt so the first event sits at a
   positive, representative time (avoids a point mass at t = 0).
3. **Scale to O(1)** — divide by the mean sequence duration (computed
   *after* the shift) so a typical sequence spans ≈ 1.0.

The ``time_scale_factor`` property returns the divisor used in step 3 so
that evaluation metrics (LL Jacobian correction, MAE/RMSE unscaling) are
computed correctly.
"""

from pathlib import Path
from typing import ClassVar, List, Optional, cast

import torch

from ..utils.event_data import MVEventData
from .base_dataset import TPPDataset
from ..training.dataset_registry import register_dataset


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

#: Maps dataset short-name → number of event types.
EASYTPP_NUM_TYPES: dict[str, int] = {
    "taxi": 10,
    "stackoverflow": 22,
    "retweet": 3,
    "amazon": 16,
    "taobao": 17,
    "earthquake": 7,
    "volcano": 1,
}

#: Maps dataset short-name → mean inter-event time on the training set
#: (computed after offset removal).  Used to place the first event at a
#: positive time instead of t = 0.
EASYTPP_MEAN_IET: dict[str, float] = {
    "taxi": 0.2246,
    "stackoverflow": 0.8891,
    "retweet": 13.3252,
    "amazon": 0.5218,
    "taobao": 0.8081,
    "earthquake": 1.2976,
    "volcano": 817.2235,
}

#: Maps dataset short-name → time-scale factor (mean sequence duration on the
#: training set, *after* the mean-IET shift).  Times are divided by this value
#: at load time so that the typical sequence spans ≈ 1.0 in scaled units.
EASYTPP_TIME_SCALE: dict[str, float] = {
    "taxi": 8.3182,
    "stackoverflow": 57.4299,
    "retweet": 547.4165,
    "amazon": 23.3154,
    "taobao": 46.7464,
    "earthquake": 21.3504,
    "volcano": 12062.2188,
}


#: Per-(easytpp_root, dataset_name) overrides of (EASYTPP_MEAN_IET, EASYTPP_TIME_SCALE) for data roots that hold a
#: different release of a dataset than the one the defaults above were computed on. Same rule (pooled mean train
#: gap after offset removal; mean train duration t_last - t0 + MEAN_IET), rounded to 4 decimals, computed with
#: scripts/data_prep/compute_easytpp_time_constants.py (which also reproduces the defaults for taobao/taxi/amazon).
#: The defaults ("taobao" = the 17-type Google Drive release, 0.8081 / 46.7464) are unchanged.
EASYTPP_ROOT_OVERRIDES: dict[tuple[str, str], tuple[float, float]] = {
    # HuggingFace easytpp/taobao, float64 (scripts/data_prep/download_easytpp_datasets.py --json_dir);
    # train split: 0.046346 / 2.619704
    ("easytpp_hf_f64", "taobao"): (0.0463, 2.6197),
}


def easytpp_time_constants(dataset_name: str, easytpp_root: Optional[str] = None) -> tuple[float, float]:
    """(mean_iet, time_scale) for a dataset, honouring :data:`EASYTPP_ROOT_OVERRIDES`."""
    if easytpp_root is not None and (easytpp_root, dataset_name) in EASYTPP_ROOT_OVERRIDES:
        return EASYTPP_ROOT_OVERRIDES[(easytpp_root, dataset_name)]
    return EASYTPP_MEAN_IET.get(dataset_name, 0.0), EASYTPP_TIME_SCALE.get(dataset_name, 1.0)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


@register_dataset("easytpp")
class EasyTPPDataset(TPPDataset):
    """A ``torch.utils.data.Dataset`` over a single EasyTPP split.

    Loads a ``.pt`` file containing a ``list[MVEventData]`` (produced by
    ``scripts/data_prep/download_easytpp_datasets.py``) and exposes it as a dataset.

    Unlike :class:`~torchpp.data.ukb_data.UKBDataset`, which lazily parses a
    binary ``numpy.memmap``, this class loads the entire split into memory on
    construction (the EasyTPP datasets are small).

    Args:
        data_dir:    Path to the per-dataset directory
                     (e.g. ``easytpp/taxi``).
        split:       One of ``"train"``, ``"dev"``, or ``"test"``.
        dataset_name: Short name used for registry look-up (e.g. ``"taxi"``).
                     Required if ``num_event_types`` is not supplied directly.
        num_event_types: Explicit event-type count.  If ``None``, the count is
                     looked up from :data:`EASYTPP_NUM_TYPES` using
                     ``dataset_name``, or inferred from the loaded data when
                     the name is unrecognised.
        limit_size:  If set, only the first *limit_size* sequences are retained.
    """

    # Dataset-intrinsic properties — not user-configurable per run.
    terminating: ClassVar[bool] = False
    """EasyTPP datasets contain repeating event types; terminating mode is off."""

    default_ignore_indices: ClassVar[List[int]] = []
    """All event types are valid prediction targets by default."""

    @classmethod
    def load(cls, cfg: dict, root: Path, split: str) -> "EasyTPPDataset":
        """Construct an :class:`EasyTPPDataset` from a flat config dict.

        Called by :func:`~torchpp.training.dataset_registry.load_split`.

        Expected cfg keys:

        * ``dataset_name`` — short name, e.g. ``"taxi"`` (required).
        * ``easytpp_root`` — root dir containing per-dataset sub-folders.
          Default: ``"easytpp"``.
        """
        dataset_name: str = cfg.get("dataset_name", "")
        if not dataset_name:
            raise ValueError(
                "dataset.dataset_name is not set for dataset_type='easytpp'.  "
                "Set it via YAML or CLI (e.g. dataset.dataset_name=taxi)."
            )
        easytpp_root: str = cfg.get("easytpp_root", "easytpp")
        dataset_dir = root / easytpp_root / dataset_name

        # EasyTPP uses "dev" as the validation split name on disk.
        file_split = "dev" if split == "val" else split

        # Resolve limit_size for this split.
        limit_key = f"limit_{split}_size" if split != "val" else "limit_val_size"
        limit_size: Optional[int] = cfg.get(limit_key, None)

        return cls(
            data_dir=dataset_dir,
            split=file_split,
            dataset_name=dataset_name,
            limit_size=limit_size,
            easytpp_root=easytpp_root,
        )

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        dataset_name: str,
        limit_size: Optional[int] = None,
        easytpp_root: Optional[str] = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._split = split
        self._dataset_name = dataset_name

        path = self._data_dir / f"{split}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"EasyTPP split file not found: {path}\n"
                "Run  python scripts/data_prep/download_easytpp_datasets.py  to download datasets."
            )

        sequences: list[MVEventData] = cast(list[MVEventData], torch.load(path, weights_only=False))

        if limit_size is not None and limit_size > 0:
            sequences = sequences[:limit_size]

        # --- Time preprocessing (see module docstring) ---
        # 1. Remove per-sequence offset so first event is at t=0
        # 2. Shift by mean inter-event time so first event is at a positive t
        # 3. Scale by mean sequence duration so typical T ≈ 1.0
        mean_iet, tsf = easytpp_time_constants(dataset_name, easytpp_root)
        self._mean_iet = mean_iet
        for seq in sequences:
            tp = cast(torch.Tensor, seq["time_points"])
            tp = tp - tp[0] + mean_iet  # steps 1 + 2
            seq["time_points"] = tp / tsf  # step 3
        self._time_scale_factor = tsf

        self._sequences = sequences

        # Resolve num_event_types
        if dataset_name in EASYTPP_NUM_TYPES:
            self._num_event_types = EASYTPP_NUM_TYPES[dataset_name]
        else:
            raise ValueError(
                f"Unknown EasyTPP dataset '{dataset_name}'. Known datasets: {list(EASYTPP_NUM_TYPES.keys())}."
            )

    # ------------------------------------------------------------------ #
    # TPPDataset interface                                                 #
    # ------------------------------------------------------------------ #

    @property
    def num_event_types(self) -> int:
        return self._num_event_types

    @property
    def time_scale_factor(self) -> float:
        """Divisor applied to raw times (mean sequence duration)."""
        return self._time_scale_factor

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> MVEventData:
        return self._sequences[idx]

    def __repr__(self) -> str:
        name = self._dataset_name or str(self._data_dir.name)
        return (
            f"EasyTPPDataset(dataset={name!r}, split={self._split!r}, "
            f"n_sequences={len(self)}, D={self._num_event_types})"
        )


def load_easytpp_sequences(
    data_dir: str | Path,
    split: str,
    limit_size: Optional[int] = None,
) -> list[MVEventData]:
    """Load an EasyTPP split from a ``.pt`` file.

    Args:
        data_dir:   Path to the dataset directory (e.g. ``easytpp/taxi``).
        split:      One of ``"train"``, ``"dev"``, or ``"test"``.
        limit_size: If set, return only the first *limit_size* sequences.

    Returns:
        List of :class:`~torchpp.utils.event_data.MVEventData` objects.
    """
    data_dir = Path(data_dir)
    path = data_dir / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"EasyTPP split file not found: {path}\n"
            "Run  python scripts/data_prep/download_easytpp_datasets.py  to download datasets."
        )

    sequences: list[MVEventData] = cast(list[MVEventData], torch.load(path, weights_only=False))

    if limit_size is not None and limit_size > 0:
        sequences = sequences[:limit_size]

    return sequences


def num_event_types_for(
    dataset_name: str,
    data_dir: Optional[str | Path] = None,
) -> int:
    """Return the number of event types for a named EasyTPP dataset.

    Falls back to scanning the actual data if the name is not in the registry.
    """
    if dataset_name in EASYTPP_NUM_TYPES:
        return EASYTPP_NUM_TYPES[dataset_name]

    if data_dir is not None:
        seqs = load_easytpp_sequences(data_dir, "train", limit_size=100)
        if seqs:
            return max(int(seq.event_types.max().item()) for seq in seqs) + 1  # type: ignore[union-attr]

    raise ValueError(
        f"Unknown EasyTPP dataset '{dataset_name}'. "
        f"Known datasets: {list(EASYTPP_NUM_TYPES.keys())}. "
        "Pass data_dir to infer num_event_types from the data."
    )
