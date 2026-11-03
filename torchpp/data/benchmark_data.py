"""Dataset and loading utilities for the standard TPP benchmark datasets.

Datasets are stored as ``$TORCHPPDATA/benchmarks/<name>/{train,dev,test}.pt``, where each
``.pt`` file is a ``list[MVEventData]`` saved with :func:`torch.save`.

The files are produced by ``scripts/data_prep/download_benchmark_datasets.py``.

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
The raw timestamps (``time_since_start``) carry a per-sequence offset
relative to an external reference point — **not** to the first event.  This
offset is meaningless for our models and is removed at load time.

Preprocessing per sequence (raw native units):

1. **Remove offset** — subtract the first timestamp so inter-event times
   are preserved and the sequence starts at 0.
2. **Place first event at mean inter-event time** — shift by this split's
   mean Δt so the first event sits at a representative positive time
   (avoids a point mass at t = 0).

Times are emitted in **raw native units** — no dataset-level scaling. The model
owns time normalisation via its ``time_scale`` buffer (derived from the training
mean inter-event time).
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
BENCHMARK_NUM_TYPES: dict[str, int] = {
    "taxi": 10,
    "stackoverflow": 22,
    "retweet": 3,
    "amazon": 16,
    "taobao": 17,
    "earthquake": 7,
    "volcano": 1,
}

# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


@register_dataset("benchmark")
class BenchmarkDataset(TPPDataset):
    """A ``torch.utils.data.Dataset`` over a single split of a benchmark dataset.

    Loads a ``.pt`` file containing a ``list[MVEventData]`` (produced by
    ``scripts/data_prep/download_benchmark_datasets.py``) and exposes it as a dataset.

    The entire split is loaded into memory on construction (the benchmark
    datasets are small).

    Args:
        data_dir:    Path to the per-dataset directory
                     (e.g. ``benchmarks/taxi``).
        split:       One of ``"train"``, ``"dev"``, or ``"test"``.
        dataset_name: Short name, e.g. ``"taxi"``; sets the number of event types via
                     :data:`BENCHMARK_NUM_TYPES`.
        limit_size:  If set, only the first *limit_size* sequences are retained.
    """

    # Dataset-intrinsic properties — not user-configurable per run.
    terminating: ClassVar[bool] = False
    """The benchmark datasets contain repeating event types; terminating mode is off."""

    cfg_keys = frozenset({"dataset_name", "benchmark_root", "limit_train_size", "limit_val_size", "limit_test_size"})

    @classmethod
    def load(cls, cfg: dict, root: Path, split: str) -> "BenchmarkDataset":
        """Construct a :class:`BenchmarkDataset` from a flat config dict.

        Called by :func:`~torchpp.training.dataset_registry.load_split`.

        Expected cfg keys:

        * ``dataset_name`` — short name, e.g. ``"taxi"`` (required).
        * ``benchmark_root`` — directory under ``$TORCHPPDATA`` with one sub-folder per
          dataset. Default: ``"benchmarks"``.
        """
        dataset_name: str = cfg.get("dataset_name", "")
        if not dataset_name:
            raise ValueError(
                "dataset.dataset_name is not set for dataset_type='benchmark'.  "
                "Set it via YAML or CLI (e.g. dataset.dataset_name=taxi)."
            )
        benchmark_root: str = cfg.get("benchmark_root", "benchmarks")
        dataset_dir = root / benchmark_root / dataset_name

        # The validation split is stored as "dev" on disk.
        file_split = "dev" if split == "val" else split

        # Resolve limit_size for this split.
        limit_key = f"limit_{split}_size" if split != "val" else "limit_val_size"
        limit_size: Optional[int] = cfg.get(limit_key, None)

        return cls(
            data_dir=dataset_dir,
            split=file_split,
            dataset_name=dataset_name,
            limit_size=limit_size,
        )

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        dataset_name: str,
        limit_size: Optional[int] = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._split = split
        self._dataset_name = dataset_name

        path = self._data_dir / f"{split}.pt"
        if not path.exists():
            if dataset_name not in BENCHMARK_NUM_TYPES:
                raise ValueError(
                    f"Unknown benchmark dataset '{dataset_name}'. Known datasets: {sorted(BENCHMARK_NUM_TYPES)}."
                )
            raise FileNotFoundError(
                f"Benchmark split file not found: {path} (data root $TORCHPPDATA={self._data_dir.parent.parent}). "
                "Run  python scripts/data_prep/download_benchmark_datasets.py  to download the datasets."
            )

        sequences: list[MVEventData] = cast(list[MVEventData], torch.load(path, weights_only=False))

        if limit_size is not None and limit_size > 0:
            sequences = sequences[:limit_size]

        # --- Time preprocessing (raw native units) ---
        # The sequences have no meaningful time origin, so one is synthesised:
        #   1. Remove the per-sequence offset  → first event at t=0.
        #   2. Shift by this split's mean inter-event time → first event at a
        #      representative positive time, avoiding a point mass at t=0.
        # Times stay in native units; the model owns normalisation via its ``time_scale``.
        offset_removed = [cast(torch.Tensor, seq["time_points"]) - cast(torch.Tensor, seq["time_points"])[0]
                          if cast(torch.Tensor, seq["time_points"]).numel() > 0
                          else cast(torch.Tensor, seq["time_points"]) for seq in sequences]
        gaps = [tp[1:] - tp[:-1] for tp in offset_removed if tp.numel() > 1]
        mean_iet = float(torch.cat(gaps).mean()) if gaps else 1.0
        for seq, tp in zip(sequences, offset_removed):
            seq["time_points"] = tp + mean_iet

        self._sequences = sequences

        # Resolve num_event_types
        if dataset_name in BENCHMARK_NUM_TYPES:
            self._num_event_types = BENCHMARK_NUM_TYPES[dataset_name]
        else:
            raise ValueError(
                f"Unknown benchmark dataset '{dataset_name}'. Known datasets: {list(BENCHMARK_NUM_TYPES.keys())}."
            )

    # ------------------------------------------------------------------ #
    # TPPDataset interface                                                 #
    # ------------------------------------------------------------------ #

    @property
    def num_event_types(self) -> int:
        return self._num_event_types

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> MVEventData:
        return self._sequences[idx]

    def __repr__(self) -> str:
        name = self._dataset_name or str(self._data_dir.name)
        return (
            f"BenchmarkDataset(dataset={name!r}, split={self._split!r}, "
            f"n_sequences={len(self)}, D={self._num_event_types})"
        )


def load_benchmark_sequences(
    data_dir: str | Path,
    split: str,
    limit_size: Optional[int] = None,
) -> list[MVEventData]:
    """Load a benchmark split from a ``.pt`` file.

    Args:
        data_dir:   Path to the dataset directory (e.g. ``benchmarks/taxi``).
        split:      One of ``"train"``, ``"dev"``, or ``"test"``.
        limit_size: If set, return only the first *limit_size* sequences.

    Returns:
        List of :class:`~torchpp.utils.event_data.MVEventData` objects.
    """
    data_dir = Path(data_dir)
    path = data_dir / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark split file not found: {path}\n"
            "Run  python scripts/data_prep/download_benchmark_datasets.py  to download datasets."
        )

    sequences: list[MVEventData] = cast(list[MVEventData], torch.load(path, weights_only=False))

    if limit_size is not None and limit_size > 0:
        sequences = sequences[:limit_size]

    return sequences


def num_event_types_for(
    dataset_name: str,
    data_dir: Optional[str | Path] = None,
) -> int:
    """Return the number of event types for a named benchmark dataset.

    Falls back to scanning the actual data if the name is not in the registry.
    """
    if dataset_name in BENCHMARK_NUM_TYPES:
        return BENCHMARK_NUM_TYPES[dataset_name]

    if data_dir is not None:
        seqs = load_benchmark_sequences(data_dir, "train", limit_size=100)
        if seqs:
            return max(int(seq.event_types.max().item()) for seq in seqs) + 1  # type: ignore[union-attr]

    raise ValueError(
        f"Unknown benchmark dataset '{dataset_name}'. "
        f"Known datasets: {list(BENCHMARK_NUM_TYPES.keys())}. "
        "Pass data_dir to infer num_event_types from the data."
    )
