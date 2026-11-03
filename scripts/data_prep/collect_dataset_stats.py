#!/usr/bin/env python
"""Collect dataset statistics for all datasets and write a CSV summary.

Datasets that are not present under $TORCHPPDATA are skipped.

Usage:
    python scripts/data_prep/collect_dataset_stats.py [--out tmp/dataset_statistics.csv]
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from torchpp.training.dataset_registry import load_split, get_data_root
from torchpp.utils.event_data import DatasetStats

DATA_ROOT = get_data_root()

# ── Dataset definitions ──────────────────────────────────────────────────────
# (name, dataset_type, cfg_dict)

DATASETS = [
    ("taxi", "benchmark", {"dataset_name": "taxi"}),
    ("stackoverflow", "benchmark", {"dataset_name": "stackoverflow"}),
    ("retweet", "benchmark", {"dataset_name": "retweet"}),
    ("amazon", "benchmark", {"dataset_name": "amazon"}),
    ("taobao", "benchmark", {"dataset_name": "taobao"}),
    ("earthquake", "benchmark", {"dataset_name": "earthquake"}),
    ("volcano", "benchmark", {"dataset_name": "volcano"}),
]

SPLITS = ["train", "val", "test"]

CSV_COLUMNS = [
    "dataset",
    "num_event_types",
    "terminating",
    "time_scale_factor",
    # Per-split counts
    "num_sequences_train",
    "num_sequences_val",
    "num_sequences_test",
    "num_events_train",
    "num_events_val",
    "num_events_test",
    # Stats (computed from train split)
    "mean_events_per_sequence",
    "mean_inter_event_time",
    "mean_sequence_time",
    "sequence_time_p0",
    "sequence_time_p50",
    "sequence_time_p75",
    "sequence_time_p90",
    "sequence_time_p95",
    "sequence_time_p100",
    "inter_event_time_p95",
]


def compute_split_stats(dataset, D: int, terminating: bool) -> DatasetStats:
    """Compute DatasetStats for a split."""
    return DatasetStats.from_sequences(dataset, D, T_obs=None, terminating=terminating)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "tmp" / "dataset_statistics.csv", help="output CSV path")
    out_path = parser.parse_args().out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for name, dataset_type, cfg in DATASETS:
        print(f"\n{'=' * 60}")
        print(f"Processing: {name} ({dataset_type})")
        print(f"{'=' * 60}")

        # Load all splits
        split_datasets = {}
        for split in SPLITS:
            try:
                ds = load_split(dataset_type, cfg, DATA_ROOT, split)
                split_datasets[split] = ds
                print(f"  {split}: {len(ds)} sequences")
            except Exception as e:
                print(f"  {split}: FAILED — {e}")
                split_datasets[split] = None

        train_ds = split_datasets["train"]
        if train_ds is None:
            print(f"  Skipping {name}: no train split.")
            continue

        D = train_ds.num_event_types
        terminating = train_ds.terminating
        tsf = train_ds.time_scale_factor

        # Compute stats per split (gives us event counts via mean_events_per_sequence)
        event_counts = {}
        train_stats = None
        for split in SPLITS:
            ds = split_datasets[split]
            if ds is not None:
                print(f"  Computing stats for {split} ({len(ds)} seqs)...")
                stats = compute_split_stats(ds, D, terminating)
                n_events = round(stats.mean_events_per_sequence * len(ds))
                event_counts[split] = n_events
                print(f"    events: {n_events}, mean_events/seq: {stats.mean_events_per_sequence:.2f}")
                if split == "train":
                    train_stats = stats
            else:
                event_counts[split] = ""

        # Convert time stats to natural units (multiply by time_scale_factor)
        s = train_stats
        row = {
            "dataset": name,
            "num_event_types": D,
            "terminating": terminating,
            "time_scale_factor": tsf,
            "num_sequences_train": len(train_ds) if split_datasets["train"] else "",
            "num_sequences_val": len(split_datasets["val"]) if split_datasets["val"] else "",
            "num_sequences_test": len(split_datasets["test"]) if split_datasets["test"] else "",
            "num_events_train": event_counts["train"],
            "num_events_val": event_counts["val"],
            "num_events_test": event_counts["test"],
            "mean_events_per_sequence": f"{s.mean_events_per_sequence:.4f}",
            "mean_inter_event_time": f"{s.mean_inter_event_time * tsf:.6f}",
            "mean_sequence_time": f"{s.mean_sequence_time * tsf:.6f}",
            "sequence_time_p0": f"{s.sequence_time_p0 * tsf:.6f}",
            "sequence_time_p50": f"{s.sequence_time_p50 * tsf:.6f}",
            "sequence_time_p75": f"{s.sequence_time_p75 * tsf:.6f}",
            "sequence_time_p90": f"{s.sequence_time_p90 * tsf:.6f}",
            "sequence_time_p95": f"{s.sequence_time_p95 * tsf:.6f}",
            "sequence_time_p100": f"{s.sequence_time_p100 * tsf:.6f}",
            "inter_event_time_p95": f"{s.inter_event_time_p95 * tsf:.6f}",
        }
        rows.append(row)

    # Write CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 60}")
    print(f"CSV written to: {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
