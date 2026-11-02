"""Shared helpers for ``train.py`` and ``evaluate.py``.

This module collects the dataset loading, dataset-statistics caching, collate
function construction, dataloader assembly, save-path resolution and W&B
initialisation utilities used by the training and evaluation entry points.
The training and evaluation scripts themselves contain only the high-level
control flow.

All helpers take primitive arguments (paths, ints, floats) or the relevant
config sub-objects (``BatchingConfig``, ``DatasetStatsConfig``, …) rather than
the full top-level config, so they can be reused by both ``FitConfig`` and
``StandaloneEvalConfig`` driven scripts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, cast

import torch
from torch.utils.data import DataLoader

from torchpp.data.base_dataset import TPPDataset
from torchpp.training.config import (
    BatchingConfig,
    DatasetStatsConfig,
    SaveConfig,
    WandbConfig,
)
from torchpp.training.dataset_registry import load_split
from torchpp.utils.batching import (
    BucketBatchSampler,
    TruncationStrategy,
    collate_batch_with_truncation,
)
from torchpp.utils.event_data import (
    BatchedMVEventData,
    DatasetStats,
    MVEventData,
)

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_train_val_datasets(
    dataset_type: str,
    dataset_cfg: dict,
    root: Path,
) -> tuple[TPPDataset, TPPDataset, int]:
    """Load train and validation splits via the dataset registry.

    Returns ``(dataset_train, dataset_val, D)`` where ``D`` is the number of
    event types reported by the training dataset.
    """
    cfg = dict(dataset_cfg)

    print(f"\nLoading train split ({dataset_type})…")
    dataset_train = load_split(dataset_type, cfg, root, split="train")
    D = dataset_train.num_event_types
    print(f"  {len(dataset_train)} sequences loaded, {D} event types")

    print("Loading val split…")
    dataset_val = load_split(dataset_type, cfg, root, split="val")
    print(f"  {len(dataset_val)} sequences loaded")

    return dataset_train, dataset_val, D


def load_test_dataset(
    dataset_type: str,
    dataset_cfg: dict,
    root: Path,
    limit_test_size: Optional[int] = None,
) -> TPPDataset:
    """Load the test split via the dataset registry.

    ``limit_test_size`` is injected into the dataset cfg so any registry-aware
    loader can apply it (e.g. ``UKBDataset.load`` reads ``limit_test_size``).
    """
    cfg = dict(dataset_cfg)
    if limit_test_size is not None:
        cfg["limit_test_size"] = limit_test_size

    print("Loading test split…")
    dataset = load_split(dataset_type, cfg, root, split="test")
    print(f"  {len(dataset)} sequences loaded")
    return dataset


def print_dataset_stats(stats: DatasetStats, dataset: TPPDataset) -> None:
    """Print a summary of *stats* with units derived from the dataset's
    ``time_scale_factor``.

    Datasets whose scale factor differs from 1 (e.g. UKB, where 1 scaled unit
    ≈ 80 years) get a converted natural-unit display alongside the raw value.
    """
    mean_raw = stats.mean_first_occ_time.mean().item()
    tsf = dataset.time_scale_factor

    if tsf != 1.0:
        print(f"  mean first-occurrence time (all types): {mean_raw:.4f} scaled (= {mean_raw * tsf:.0f} natural units)")
        print(
            f"  mean inter-event time (global):         "
            f"{stats.mean_inter_event_time:.4f} "
            f"(= {stats.mean_inter_event_time * tsf:.1f} natural units)"
        )
        print(f"  mean events per sequence:               {stats.mean_events_per_sequence:.2f}")
        print(
            f"  mean sequence duration:                 "
            f"{stats.mean_sequence_time:.4f} scaled "
            f"(= {stats.mean_sequence_time * tsf:.0f} natural units)"
        )
        k = min(5, stats.type_distribution.numel())
        topk = stats.type_distribution.topk(k)
        print(
            f"  top-{k} event types by frequency:         types {topk.indices.tolist()} "
            f"({[f'{v:.3f}' for v in topk.values.tolist()]})"
        )
    else:
        print(f"  mean first-occurrence time (all types): {mean_raw:.4f} (raw units)")
        print(f"  mean inter-event time (global):         {stats.mean_inter_event_time:.4f}")
        print(f"  mean events per sequence:               {stats.mean_events_per_sequence:.2f}")
        print(f"  mean sequence duration:                 {stats.mean_sequence_time:.4f}")


# ---------------------------------------------------------------------------
# Sequence truncation for evaluation
# ---------------------------------------------------------------------------


def truncate_sequences_for_eval(
    sequences: Sequence[MVEventData],
    max_length: Optional[int],
    strategy: str,
    seed: int,
) -> list[MVEventData]:
    """Apply a training-style truncation policy to a list of evaluation sequences."""
    if max_length is None:
        return list(sequences)

    if strategy == "right":
        return [TruncationStrategy.right_truncate(seq, max_length) for seq in sequences]
    if strategy == "left":
        return [TruncationStrategy.left_truncate(seq, max_length) for seq in sequences]
    if strategy == "random":
        return [TruncationStrategy.random_sample(seq, max_length, seed) for seq in sequences]
    raise ValueError(f"Unknown truncation strategy: {strategy}")


# ---------------------------------------------------------------------------
# Dataset statistics — caching layer
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Return a deterministically serialisable representation of *value*."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _dataset_stats_cache_path(
    root: Path,
    dataset_type: str,
    dataset_cfg: dict,
    dataset: TPPDataset,
    D: int,
    stats_cfg: DatasetStatsConfig,
) -> Path:
    """Return the cache path for the current dataset-stats configuration."""
    fingerprint = {
        "version": 2,  # bump whenever DatasetStats gains fields, to invalidate old caches
        "dataset_type": dataset_type,
        "dataset_class": dataset.__class__.__name__,
        "dataset_cfg": _json_safe(dict(dataset_cfg)),
        "num_event_types": D,
        "terminating": bool(dataset.terminating),
        "num_sequences": len(dataset),
        "stats_max_sequences": stats_cfg.max_sequences,
        "stats_subsample_seed": stats_cfg.subsample_seed,
    }
    digest = hashlib.sha1(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(stats_cfg.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = root / cache_dir
    return cache_dir / f"{dataset_type}_{digest}.pt"


def _dataset_stats_from_state(state: dict[str, Any]) -> DatasetStats:
    # Tolerate stale caches: keep only known fields (defaults fill any keys added
    # after the cache was written; unknown keys are dropped).
    valid = {f.name for f in dataclasses.fields(DatasetStats)}
    return DatasetStats(**{k: v for k, v in state.items() if k in valid})


def _dataset_stats_to_state(stats: DatasetStats) -> dict[str, Any]:
    return {field.name: getattr(stats, field.name) for field in dataclasses.fields(DatasetStats)}


@contextmanager
def _dataset_stats_lock(lock_path: Path):
    """Serialize cache writes so parallel jobs do not recompute the same stats."""
    if fcntl is None:
        yield
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _select_sequences_for_dataset_stats(
    dataset: TPPDataset,
    max_sequences: Optional[int],
    seed: int,
) -> tuple[Sequence[MVEventData], int]:
    """Return either the full dataset or a deterministic subsample for stats."""
    if max_sequences is None or max_sequences <= 0 or len(dataset) <= max_sequences:
        return cast(Sequence[MVEventData], dataset), len(dataset)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_sequences].tolist()
    indices.sort()
    return [dataset[idx] for idx in indices], max_sequences


def load_dataset_stats(
    stats_cfg: DatasetStatsConfig,
    root: Path,
    dataset_train: TPPDataset,
    D: int,
    dataset_type: str,
    dataset_cfg: dict,
) -> Optional[DatasetStats]:
    """Load cached dataset stats or compute them from the configured sample."""
    if not stats_cfg.enable:
        print("Skipping dataset statistics for model initialisation (stats.enable=false).")
        return None

    cache_path = _dataset_stats_cache_path(root, dataset_type, dataset_cfg, dataset_train, D, stats_cfg)
    cache_enabled = stats_cfg.cache

    if cache_enabled and cache_path.exists() and not stats_cfg.force_recompute:
        print(f"Loading cached dataset statistics from: {cache_path}")
        state = torch.load(cache_path, map_location="cpu")
        return _dataset_stats_from_state(state)

    def _compute_stats() -> DatasetStats:
        sequences, num_used = _select_sequences_for_dataset_stats(
            dataset_train,
            max_sequences=stats_cfg.max_sequences,
            seed=stats_cfg.subsample_seed,
        )
        if num_used < len(dataset_train):
            print(
                "Computing dataset statistics for model initialisation "
                f"from {num_used}/{len(dataset_train)} sampled training sequences "
                f"(seed={stats_cfg.subsample_seed})…"
            )
        else:
            print("Computing dataset statistics for model initialisation from the full training set…")
        return DatasetStats.from_sequences(
            sequences,
            D,
            T_obs=None,
            terminating=dataset_train.terminating,
        )

    if not cache_enabled:
        return _compute_stats()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with _dataset_stats_lock(lock_path):
        if cache_path.exists() and not stats_cfg.force_recompute:
            print(f"Loading cached dataset statistics from: {cache_path}")
            state = torch.load(cache_path, map_location="cpu")
            return _dataset_stats_from_state(state)

        dataset_stats = _compute_stats()
        torch.save(_dataset_stats_to_state(dataset_stats), cache_path)
        print(f"Cached dataset statistics at: {cache_path}")
        return dataset_stats


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def collate_batch(mv_l: list[MVEventData]) -> BatchedMVEventData:
    """Standard collate function without truncation."""
    if not mv_l:
        return BatchedMVEventData([], [])
    return BatchedMVEventData(mv_events=mv_l)


def make_collate_fns(batching: BatchingConfig, seed: int):
    """Return ``(collate_fn_train, collate_fn_val)`` based on truncation settings."""
    if batching.max_sequence_length is not None:
        _max_len = batching.max_sequence_length
        _trunc = batching.truncation_strategy

        def collate_fn_train(batch):
            return collate_batch_with_truncation(batch, max_length=_max_len, truncation_strategy=_trunc, seed=seed)

        def collate_fn_val(batch):
            return collate_batch_with_truncation(batch, max_length=_max_len, truncation_strategy=_trunc, seed=seed)
    else:

        def collate_fn_train(batch):
            return collate_batch(batch)

        def collate_fn_val(batch):
            return collate_batch(batch)

    return collate_fn_train, collate_fn_val


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------


def make_train_val_dataloaders(
    dataset_train: TPPDataset,
    dataset_val: TPPDataset,
    batching: BatchingConfig,
    seed: int,
    collate_fn_train,
    collate_fn_val,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders honouring the batching config."""
    if batching.use_bucket_batching:
        print(f"\nBucket batching: {batching.num_buckets} buckets")
        if batching.dynamic_batch_size:
            print("  Dynamic batch sizing enabled (constant total events per batch)")

        train_sampler = BucketBatchSampler(
            data_source=dataset_train,
            batch_size=batching.batch_size,
            shuffle=True,
            drop_last=False,
            num_buckets=batching.num_buckets,
            dynamic_batch_size=batching.dynamic_batch_size,
            target_total_events=batching.target_total_events,
            seed=seed,
        )
        val_sampler = BucketBatchSampler(
            data_source=dataset_val,
            batch_size=batching.batch_size,
            shuffle=False,
            drop_last=False,
            num_buckets=batching.num_buckets,
            dynamic_batch_size=batching.dynamic_batch_size,
            target_total_events=batching.target_total_events,
            seed=seed,
        )
        dataloader_train = DataLoader(dataset=dataset_train, batch_sampler=train_sampler, collate_fn=collate_fn_train)
        dataloader_val = DataLoader(dataset=dataset_val, batch_sampler=val_sampler, collate_fn=collate_fn_val)
    else:
        print("\nStandard random batching")
        dataloader_train = DataLoader(
            dataset=dataset_train,
            batch_size=batching.batch_size,
            shuffle=True,
            collate_fn=collate_fn_train,
        )
        dataloader_val = DataLoader(
            dataset=dataset_val,
            batch_size=batching.batch_size,
            shuffle=False,
            collate_fn=collate_fn_val,
        )

    return dataloader_train, dataloader_val


# ---------------------------------------------------------------------------
# Save path
# ---------------------------------------------------------------------------


def _default_save_path(root: Path, run_name: str, dataset_type: str) -> Path:
    """Default save path under ``$TORCHPPMODELS/<dataset_type>/<run>.pth``."""
    models_root = Path(os.environ.get("TORCHPPMODELS", root / "models"))
    return models_root / dataset_type / f"{run_name}.pth"


def resolve_save_path(
    save_cfg: SaveConfig,
    root: Path,
    run_name: str,
    dataset_type: str,
) -> Path:
    """Resolve the final model save path, honouring ``overwrite`` / ``auto_increment``.

    Raises :class:`FileExistsError` if the target exists and neither overwrite
    nor auto-increment is permitted.
    """
    if save_cfg.path is not None:
        save_path = Path(save_cfg.path)
        if not save_path.is_absolute():
            save_path = root / save_path
    else:
        save_path = _default_save_path(root, run_name, dataset_type)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        if save_cfg.overwrite:
            print(f"\nWarning: {save_path} exists and will be overwritten.")
        elif save_cfg.auto_increment:
            base, suffix, parent = save_path.stem, save_path.suffix, save_path.parent
            counter = 1
            while save_path.exists():
                save_path = parent / f"{base}_{counter}{suffix}"
                counter += 1
            print(f"\nFile exists; using: {save_path}")
        else:
            raise FileExistsError(
                f"Model file already exists: {save_path}\nUse save.overwrite=true or save.auto_increment=true."
            )

    return save_path


# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------


def init_wandb_run(wandb_cfg: WandbConfig, full_config: Any):
    """Create a Weights & Biases run if configured, otherwise return ``None``.

    *full_config* is serialised into the run's config payload (the entire
    :class:`FitConfig` is normally passed in).
    """
    if not wandb_cfg.enable:
        return None

    if wandb is None:
        print("wandb not installed; set wandb.enable=false or install wandb to enable logging.")
        return None

    return wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        name=wandb_cfg.run_name,
        tags=wandb_cfg.tags,
        mode=wandb_cfg.mode,  # type: ignore
        config=dataclasses.asdict(full_config),
    )
