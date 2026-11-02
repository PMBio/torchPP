"""Dataset and loading utilities for preprocessed MIMIC-IV benchmarks.

Datasets are stored as ``$TORCHPPDATA/mimic_iv/<dataset_name>/{train,dev,test}.pt``,
where each split file contains a ``list[MVEventData]`` saved with :func:`torch.save`.

The companion ``metadata.pt`` file stores the number of event types,
the time-scale factor used to normalise times, and task-specific label metadata.

The files are produced by ``scripts/data_prep/prepare_mimic_iv_datasets.py``.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, cast

import torch

from ..utils.event_data import MVEventData
from .base_dataset import TPPDataset
from ..training.dataset_registry import register_dataset


def _count_event_types(sequences: List[MVEventData], num_event_types: int) -> torch.Tensor:
    counts = torch.zeros(num_event_types, dtype=torch.long)
    for seq in sequences:
        if len(seq.event_types) == 0:
            continue
        event_types = seq.event_types.long()
        valid = event_types[(event_types >= 0) & (event_types < num_event_types)]
        if valid.numel() > 0:
            counts += torch.bincount(valid, minlength=num_event_types)
    return counts


@lru_cache(maxsize=None)
def _icu_topk_spec(data_dir: str, top_k_event_types: int) -> Dict[str, Any]:
    dataset_dir = Path(data_dir)
    metadata = cast(Dict[str, Any], torch.load(dataset_dir / "metadata.pt", weights_only=False))
    raw_num_event_types = int(metadata["num_event_types"])
    if top_k_event_types <= 0:
        raise ValueError(f"top_k_event_types must be positive, got {top_k_event_types}.")

    train_sequences = cast(List[MVEventData], torch.load(dataset_dir / "train.pt", weights_only=False))
    event_counts = _count_event_types(train_sequences, raw_num_event_types)
    ranked_type_ids = torch.argsort(event_counts, descending=True, stable=True)

    if top_k_event_types >= raw_num_event_types:
        identity = torch.arange(raw_num_event_types, dtype=torch.long)
        return {
            "old_to_new": identity,
            "raw_num_event_types": raw_num_event_types,
            "effective_num_event_types": raw_num_event_types,
            "retained_raw_type_ids": ranked_type_ids.tolist(),
            "other_type_id": None,
            "retained_event_coverage": 1.0,
        }

    kept_raw_type_ids = ranked_type_ids[:top_k_event_types]
    old_to_new = torch.full((raw_num_event_types,), top_k_event_types, dtype=torch.long)
    old_to_new[kept_raw_type_ids] = torch.arange(top_k_event_types, dtype=torch.long)

    total_events = int(event_counts.sum().item())
    kept_events = int(event_counts[kept_raw_type_ids].sum().item())
    retained_event_coverage = float(kept_events / total_events) if total_events else 1.0
    return {
        "old_to_new": old_to_new,
        "raw_num_event_types": raw_num_event_types,
        "effective_num_event_types": top_k_event_types + 1,
        "retained_raw_type_ids": kept_raw_type_ids.tolist(),
        "other_type_id": top_k_event_types,
        "retained_event_coverage": retained_event_coverage,
    }


def _apply_type_mapping(sequence: MVEventData, old_to_new: torch.Tensor) -> MVEventData:
    mapped_event_types = sequence.event_types.clone()
    valid = (mapped_event_types >= 0) & (mapped_event_types < old_to_new.numel())
    mapped_event_types[valid] = old_to_new[mapped_event_types[valid].long()]
    return MVEventData(
        time_points=sequence.time_points,
        event_types=mapped_event_types,
        covariates=sequence.covariates,
        T_obs=sequence.T_obs,
    )


def _build_topk_metadata(metadata: Dict[str, Any], spec: Dict[str, Any], top_k_event_types: int) -> Dict[str, Any]:
    transformed = deepcopy(metadata)
    retained_raw_type_ids = [int(x) for x in spec["retained_raw_type_ids"]]
    retained_itemids = [int(metadata["type_id_to_itemid"][raw_type_id]) for raw_type_id in retained_raw_type_ids]
    retained_itemid_metadata = {
        str(itemid): deepcopy(metadata.get("itemid_metadata", {}).get(str(itemid), {})) for itemid in retained_itemids
    }

    transformed["raw_num_event_types"] = int(spec["raw_num_event_types"])
    transformed["num_event_types"] = int(spec["effective_num_event_types"])
    transformed["top_k_event_types"] = int(top_k_event_types)
    transformed["top_k_retained_raw_type_ids"] = retained_raw_type_ids
    transformed["top_k_retained_itemids"] = retained_itemids
    transformed["top_k_retained_event_coverage"] = float(spec["retained_event_coverage"])
    transformed["other_type_id"] = spec["other_type_id"]
    transformed["event_type_transform"] = "top_k_plus_other"

    if spec["other_type_id"] is None:
        transformed["type_id_to_itemid"] = retained_itemids
        transformed["itemid_metadata"] = retained_itemid_metadata
        return transformed

    transformed["type_id_to_itemid"] = retained_itemids + [-1]
    retained_itemid_metadata["-1"] = {
        "label": "OTHER",
        "abbreviation": "OTHER",
        "category": "OTHER",
        "linksto": "derived",
    }
    transformed["itemid_metadata"] = retained_itemid_metadata
    return transformed


@register_dataset("mimic_iv")
class MIMICIVDataset(TPPDataset):
    """A ``torch.utils.data.Dataset`` over one preprocessed MIMIC-IV split.

    The supported dataset names are currently:

    * ``"hospital_flow"``       — care-unit transfer trajectories within one admission
    * ``"icu_interventions"``   — ICU intervention streams built from inputs and procedures
    """

    terminating: ClassVar[bool] = False
    default_ignore_indices: ClassVar[List[int]] = []

    @classmethod
    def load(cls, cfg: dict, root: Path, split: str) -> "MIMICIVDataset":
        dataset_name: str = cfg.get("dataset_name", "")
        if not dataset_name:
            raise ValueError(
                "dataset.dataset_name is not set for dataset_type='mimic_iv'. "
                "Set it via YAML or CLI (e.g. dataset.dataset_name=hospital_flow)."
            )

        mimic_root: str = cfg.get("mimic_root", "mimic_iv")
        dataset_dir = root / mimic_root / dataset_name

        file_split = "dev" if split == "val" else split
        limit_key = f"limit_{split}_size" if split != "val" else "limit_val_size"
        limit_size: Optional[int] = cfg.get(limit_key, None)
        top_k_event_types = cfg.get("top_k_event_types", cfg.get("top_k_types", cfg.get("top_k", None)))

        return cls(
            data_dir=dataset_dir,
            split=file_split,
            dataset_name=dataset_name,
            limit_size=limit_size,
            top_k_event_types=top_k_event_types,
        )

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        dataset_name: str,
        limit_size: Optional[int] = None,
        top_k_event_types: Optional[int] = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._split = split
        self._dataset_name = dataset_name
        self._top_k_event_types = int(top_k_event_types) if top_k_event_types is not None else None

        split_path = self._data_dir / f"{split}.pt"
        metadata_path = self._data_dir / "metadata.pt"
        if not split_path.exists():
            raise FileNotFoundError(
                f"MIMIC-IV split file not found: {split_path}\n"
                "Run  python scripts/data_prep/prepare_mimic_iv_datasets.py  to create the datasets."
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"MIMIC-IV metadata file not found: {metadata_path}\n"
                "Run  python scripts/data_prep/prepare_mimic_iv_datasets.py  to create the datasets."
            )

        self._metadata: Dict = cast(Dict, torch.load(metadata_path, weights_only=False))
        self._sequences: List[MVEventData] = cast(List[MVEventData], torch.load(split_path, weights_only=False))

        if self._top_k_event_types is not None:
            if self._dataset_name != "icu_interventions":
                raise ValueError(
                    "top_k_event_types is currently only supported for dataset.dataset_name='icu_interventions'."
                )

            spec = _icu_topk_spec(str(self._data_dir), self._top_k_event_types)
            if spec["other_type_id"] is not None:
                self._sequences = [_apply_type_mapping(seq, spec["old_to_new"]) for seq in self._sequences]
            self._metadata = _build_topk_metadata(self._metadata, spec, self._top_k_event_types)

        if limit_size is not None and limit_size > 0:
            self._sequences = self._sequences[:limit_size]

        self._num_event_types = int(self._metadata["num_event_types"])
        self._time_scale_factor = float(self._metadata["time_scale_factor"])

    @property
    def num_event_types(self) -> int:
        return self._num_event_types

    @property
    def time_scale_factor(self) -> float:
        return self._time_scale_factor

    @property
    def metadata(self) -> Dict:
        return self._metadata

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> MVEventData:
        return self._sequences[idx]

    def __repr__(self) -> str:
        variant = ""
        if self._top_k_event_types is not None:
            variant = f", top_k_event_types={self._top_k_event_types}"
        return (
            f"MIMICIVDataset(dataset={self._dataset_name!r}, split={self._split!r}, "
            f"n_sequences={len(self)}, D={self._num_event_types}{variant})"
        )


def load_mimic_iv_sequences(
    data_dir: str | Path,
    split: str,
    limit_size: Optional[int] = None,
    top_k_event_types: Optional[int] = None,
) -> list[MVEventData]:
    """Load a preprocessed MIMIC-IV split from a ``.pt`` file."""
    data_dir = Path(data_dir)
    path = data_dir / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"MIMIC-IV split file not found: {path}\n"
            "Run  python scripts/data_prep/prepare_mimic_iv_datasets.py  to create the datasets."
        )

    metadata_path = data_dir / "metadata.pt"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"MIMIC-IV metadata file not found: {metadata_path}\n"
            "Run  python scripts/data_prep/prepare_mimic_iv_datasets.py  to create the datasets."
        )

    metadata = cast(Dict[str, Any], torch.load(metadata_path, weights_only=False))
    sequences = cast(list[MVEventData], torch.load(path, weights_only=False))
    if top_k_event_types is not None:
        if metadata.get("dataset_name") != "icu_interventions":
            raise ValueError("top_k_event_types is currently only supported for the icu_interventions dataset.")
        spec = _icu_topk_spec(str(data_dir), int(top_k_event_types))
        if spec["other_type_id"] is not None:
            sequences = [_apply_type_mapping(seq, spec["old_to_new"]) for seq in sequences]
    if limit_size is not None and limit_size > 0:
        sequences = sequences[:limit_size]
    return sequences
