#!/usr/bin/env python3
"""Analyze preprocessed MIMIC-IV benchmark datasets and write summary reports.

The script loads the generated ``mimic_iv`` splits through the normal dataset
registry and writes CSV, JSON, and Markdown summaries under
``tmp/mimic_iv_reports`` by default.

Outputs
-------
For each dataset, the script writes:

* ``<dataset>_summary.json``
* ``<dataset>_type_frequencies.csv``
* ``<dataset>_topk.csv``

For ``icu_interventions`` it also writes:

* ``icu_interventions_category_frequencies.csv``

And one combined Markdown report:

* ``mimic_iv_report.md``
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torchpp.training.dataset_registry import get_data_root, load_split
from torchpp.utils.event_data import DatasetStats, MVEventData


DEFAULT_DATASET_NAMES = ("hospital_flow", "icu_interventions")
DEFAULT_K_VALUES = (8, 16, 32, 64, 128, 256, 512)
DEFAULT_RARE_THRESHOLDS = (5, 10, 25, 50, 100, 250, 500)
DEFAULT_COVERAGE_TARGETS = (0.80, 0.90, 0.95, 0.99)
SPLITS = ("train", "val", "test")


def parse_int_list(raw: str) -> List[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected a comma-separated list of integers")
    return sorted(set(values))


def load_dataset_splits(dataset_name: str, data_root: Path, mimic_root: str):
    cfg = {"dataset_name": dataset_name, "mimic_root": mimic_root}
    return {split: load_split("mimic_iv", cfg, data_root, split) for split in SPLITS}


def sequence_lengths(sequences: Sequence[MVEventData]) -> np.ndarray:
    return np.asarray([int(len(seq.time_points)) for seq in sequences], dtype=np.int64)


def observation_windows(sequences: Sequence[MVEventData], time_scale_factor: float) -> np.ndarray:
    values = []
    for seq in sequences:
        if seq.T_obs is None:
            values.append(float(seq.time_points[-1]) if len(seq.time_points) else 0.0)
        else:
            values.append(float(seq.T_obs))
    return np.asarray(values, dtype=np.float64) * time_scale_factor


def compute_type_counts(sequences: Sequence[MVEventData], num_event_types: int) -> tuple[np.ndarray, np.ndarray]:
    event_counts = np.zeros(num_event_types, dtype=np.int64)
    sequence_counts = np.zeros(num_event_types, dtype=np.int64)
    for seq in sequences:
        if len(seq.event_types) == 0:
            continue
        types = seq.event_types.detach().cpu().numpy().astype(np.int64, copy=False)
        valid = types[(types >= 0) & (types < num_event_types)]
        if valid.size == 0:
            continue
        event_counts += np.bincount(valid, minlength=num_event_types)
        sequence_counts[np.unique(valid)] += 1
    return event_counts, sequence_counts


def stats_to_summary(stats: DatasetStats, time_scale_factor: float) -> Dict[str, float]:
    return {
        "mean_events_per_sequence": float(stats.mean_events_per_sequence),
        "mean_inter_event_time_days": float(stats.mean_inter_event_time * time_scale_factor),
        "mean_sequence_time_days": float(stats.mean_sequence_time * time_scale_factor),
        "sequence_time_p0_days": float(stats.sequence_time_p0 * time_scale_factor),
        "sequence_time_p50_days": float(stats.sequence_time_p50 * time_scale_factor),
        "sequence_time_p75_days": float(stats.sequence_time_p75 * time_scale_factor),
        "sequence_time_p90_days": float(stats.sequence_time_p90 * time_scale_factor),
        "sequence_time_p95_days": float(stats.sequence_time_p95 * time_scale_factor),
        "sequence_time_p100_days": float(stats.sequence_time_p100 * time_scale_factor),
        "inter_event_time_p95_days": float(stats.inter_event_time_p95 * time_scale_factor),
    }


def percentiles(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def topk_rows(
    event_counts: np.ndarray,
    k_values: Sequence[int],
    coverage_targets: Sequence[float],
) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    total_events = int(event_counts.sum())
    if total_events == 0:
        zero_rows = [{"k": int(k), "n_types": 0, "covered_events": 0, "event_coverage": 0.0} for k in k_values]
        target_rows = {
            f"{int(target * 100)}pct": {"k": 0, "covered_events": 0, "event_coverage": 0.0}
            for target in coverage_targets
        }
        return zero_rows, target_rows

    order = np.argsort(-event_counts, kind="stable")
    sorted_counts = event_counts[order]
    cumulative = np.cumsum(sorted_counts)

    rows: List[Dict[str, Any]] = []
    for k in sorted(set(min(int(k), len(sorted_counts)) for k in k_values)):
        covered_events = int(cumulative[k - 1]) if k > 0 else 0
        rows.append(
            {
                "k": k,
                "n_types": k,
                "covered_events": covered_events,
                "event_coverage": float(covered_events / total_events),
            }
        )

    targets: Dict[str, Dict[str, float]] = {}
    fractions = cumulative / total_events
    for target in coverage_targets:
        idx = int(np.searchsorted(fractions, target, side="left"))
        idx = min(idx, len(sorted_counts) - 1)
        covered_events = int(cumulative[idx])
        targets[f"{int(target * 100)}pct"] = {
            "k": int(idx + 1),
            "covered_events": covered_events,
            "event_coverage": float(covered_events / total_events),
        }

    return rows, targets


def rare_type_rows(event_counts: np.ndarray, thresholds: Sequence[int]) -> Dict[str, Dict[str, float]]:
    total_events = int(event_counts.sum())
    rows: Dict[str, Dict[str, float]] = {}
    for threshold in thresholds:
        mask = event_counts <= int(threshold)
        rare_events = int(event_counts[mask].sum())
        rows[str(int(threshold))] = {
            "n_types": int(mask.sum()),
            "event_count": rare_events,
            "event_fraction": float(rare_events / total_events) if total_events else 0.0,
        }
    return rows


def resolve_type_metadata(metadata: Dict[str, Any], type_id: int) -> Dict[str, Any]:
    if "type_id_to_label" in metadata:
        label = metadata["type_id_to_label"][type_id]
        return {
            "label": label,
            "itemid": "",
            "abbreviation": "",
            "category": "",
            "linksto": "",
        }

    itemid = int(metadata["type_id_to_itemid"][type_id])
    item_meta = metadata.get("itemid_metadata", {}).get(str(itemid), {})
    return {
        "label": str(item_meta.get("label", "")),
        "itemid": itemid,
        "abbreviation": str(item_meta.get("abbreviation", "")),
        "category": str(item_meta.get("category", "")),
        "linksto": str(item_meta.get("linksto", "")),
    }


def build_type_rows(
    metadata: Dict[str, Any],
    event_counts: np.ndarray,
    sequence_counts: np.ndarray,
    n_train_sequences: int,
) -> List[Dict[str, Any]]:
    total_events = int(event_counts.sum())
    order = np.argsort(-event_counts, kind="stable")
    cumulative = np.cumsum(event_counts[order]) if total_events else np.zeros_like(event_counts)

    rows: List[Dict[str, Any]] = []
    for rank, type_id in enumerate(order, start=1):
        info = resolve_type_metadata(metadata, int(type_id))
        event_count = int(event_counts[type_id])
        rows.append(
            {
                "rank": rank,
                "type_id": int(type_id),
                "event_count": event_count,
                "event_fraction": float(event_count / total_events) if total_events else 0.0,
                "cumulative_event_fraction": float(cumulative[rank - 1] / total_events) if total_events else 0.0,
                "sequence_count": int(sequence_counts[type_id]),
                "sequence_fraction": float(sequence_counts[type_id] / n_train_sequences) if n_train_sequences else 0.0,
                **info,
            }
        )
    return rows


def build_category_rows(type_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    category_totals: Dict[str, Dict[str, Any]] = {}
    total_events = sum(int(row["event_count"]) for row in type_rows)

    for row in type_rows:
        category = str(row.get("category") or "(missing)")
        record = category_totals.setdefault(
            category,
            {"category": category, "num_types": 0, "event_count": 0, "top_label": "", "top_itemid": ""},
        )
        record["num_types"] += 1
        record["event_count"] += int(row["event_count"])
        if int(row["event_count"]) > 0 and not record["top_label"]:
            record["top_label"] = row.get("label", "")
            record["top_itemid"] = row.get("itemid", "")

    rows = sorted(category_totals.values(), key=lambda row: (-int(row["event_count"]), row["category"]))
    cumulative = 0
    for row in rows:
        cumulative += int(row["event_count"])
        row["event_fraction"] = float(int(row["event_count"]) / total_events) if total_events else 0.0
        row["cumulative_event_fraction"] = float(cumulative / total_events) if total_events else 0.0
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sequence_summary(dataset, time_scale_factor: float) -> Dict[str, Any]:
    lengths = sequence_lengths(dataset)
    windows = observation_windows(dataset, time_scale_factor)
    stats = DatasetStats.from_sequences(dataset, dataset.num_event_types, terminating=dataset.terminating)
    return {
        "n_sequences": int(len(dataset)),
        "n_events": int(lengths.sum()),
        "length_percentiles": percentiles(lengths.astype(np.float64)),
        "observation_window_days": percentiles(windows),
        "stats": stats_to_summary(stats, time_scale_factor),
    }


def build_dataset_summary(
    dataset_name: str,
    dataset_splits,
    *,
    k_values: Sequence[int],
    rare_thresholds: Sequence[int],
    coverage_targets: Sequence[float],
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_ds = dataset_splits["train"]
    metadata = dict(train_ds.metadata)
    time_scale_factor = float(train_ds.time_scale_factor)
    event_counts, sequence_counts = compute_type_counts(train_ds, train_ds.num_event_types)
    type_rows = build_type_rows(metadata, event_counts, sequence_counts, len(train_ds))
    topk_table, target_rows = topk_rows(event_counts, k_values, coverage_targets)
    rare_rows = rare_type_rows(event_counts, rare_thresholds)
    category_rows = build_category_rows(type_rows) if dataset_name == "icu_interventions" else []

    summary = {
        "dataset_name": dataset_name,
        "num_event_types": int(train_ds.num_event_types),
        "time_scale_factor_days": time_scale_factor,
        "terminating": bool(train_ds.terminating),
        "metadata": metadata,
        "splits": {split: sequence_summary(ds, time_scale_factor) for split, ds in dataset_splits.items()},
        "train_type_coverage": {
            "topk": topk_table,
            "coverage_targets": target_rows,
            "rare_type_thresholds": rare_rows,
        },
        "category_summary": category_rows,
    }
    return summary, type_rows, topk_table, category_rows


def render_recommendations(summary: Dict[str, Any]) -> List[str]:
    dataset_name = summary["dataset_name"]
    coverage = summary["train_type_coverage"]["coverage_targets"]
    rare = summary["train_type_coverage"]["rare_type_thresholds"]

    if dataset_name == "hospital_flow":
        return [
            f"Keep the raw vocabulary. {summary['num_event_types']} care-unit states is already compact for a benchmark.",
            (
                f"The smallest train-only cutoff that reaches 95% event coverage is top-"
                f"{coverage['95pct']['k']}; trimming below that would mostly throw away clinically relevant state changes."
            ),
        ]

    num_categories = len(summary.get("category_summary", []))
    return [
        (
            f"Use the raw {summary['num_event_types']}-type benchmark only as the full-fidelity setting; "
            f"the smallest train-only cutoffs reaching 90% and 95% event coverage are top-"
            f"{coverage['90pct']['k']} and top-{coverage['95pct']['k']} respectively."
        ),
        (
            f"If you want a compact benchmark without manual clinical clustering, top-{coverage['95pct']['k']} + OTHER "
            f"preserves at least 95% of train events while collapsing the extreme tail."
        ),
        (
            f"There are {rare['100']['n_types']} types with at most 100 train events; that is the long tail to trim or merge first."
        ),
        (
            f"For a coarse ablation, collapse itemids to the {num_categories} `d_items.category` groups before modelling."
        ),
    ]


def write_markdown_report(report_path: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# MIMIC-IV Benchmark Report",
        "",
        "The counts below are computed from the generated `mimic_iv` datasets already stored under the TorchPP data root.",
        "Top-K and rare-type cutoffs are train-split only so they can be used without leakage.",
        "",
    ]

    for summary in summaries:
        train_split = summary["splits"]["train"]
        lines.extend(
            [
                f"## {summary['dataset_name']}",
                "",
                f"- Event types: {summary['num_event_types']}",
                f"- Time scale factor: {summary['time_scale_factor_days']:.6f} days",
                f"- Train/dev/test sequences: {summary['splits']['train']['n_sequences']:,} / {summary['splits']['val']['n_sequences']:,} / {summary['splits']['test']['n_sequences']:,}",
                f"- Train events: {train_split['n_events']:,}",
                f"- Train median sequence length: {train_split['length_percentiles']['p50']:.1f}",
                f"- Train p95 sequence length: {train_split['length_percentiles']['p95']:.1f}",
                f"- Train median observation window: {train_split['observation_window_days']['p50']:.3f} days",
                f"- Train p95 observation window: {train_split['observation_window_days']['p95']:.3f} days",
                "",
                "Recommendations:",
            ]
        )
        for item in render_recommendations(summary):
            lines.append(f"- {item}")
        lines.append("")

    report_path.write_text("\n".join(lines))


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_dataset_reports(
    dataset_name: str,
    data_root: Path,
    mimic_root: str,
    output_dir: Path,
    *,
    k_values: Sequence[int],
    rare_thresholds: Sequence[int],
    coverage_targets: Sequence[float],
) -> Dict[str, Any]:
    dataset_splits = load_dataset_splits(dataset_name, data_root, mimic_root)
    summary, type_rows, topk_table, category_rows = build_dataset_summary(
        dataset_name,
        dataset_splits,
        k_values=k_values,
        rare_thresholds=rare_thresholds,
        coverage_targets=coverage_targets,
    )

    write_csv(output_dir / f"{dataset_name}_type_frequencies.csv", type_rows)
    write_csv(output_dir / f"{dataset_name}_topk.csv", topk_table)
    if category_rows:
        write_csv(output_dir / f"{dataset_name}_category_frequencies.csv", category_rows)

    with (output_dir / f"{dataset_name}_summary.json").open("w") as handle:
        json.dump(to_builtin(summary), handle, indent=2, sort_keys=True)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASET_NAMES),
        choices=list(DEFAULT_DATASET_NAMES),
        help="Which generated MIMIC-IV datasets to analyze.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=get_data_root(),
        help="TorchPP data root containing the generated mimic_iv directory.",
    )
    parser.add_argument(
        "--mimic-root",
        default="mimic_iv",
        help="Subdirectory below the data root where generated MIMIC-IV datasets live.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "tmp" / "mimic_iv_reports",
        help="Directory where analysis artifacts are written.",
    )
    parser.add_argument(
        "--k-values",
        type=parse_int_list,
        default=list(DEFAULT_K_VALUES),
        help="Comma-separated top-K cutoffs to evaluate on the train split.",
    )
    parser.add_argument(
        "--rare-thresholds",
        type=parse_int_list,
        default=list(DEFAULT_RARE_THRESHOLDS),
        help="Comma-separated rare-type event-count thresholds for train-split analysis.",
    )
    parser.add_argument(
        "--coverage-targets",
        type=float,
        nargs="+",
        default=list(DEFAULT_COVERAGE_TARGETS),
        help="Coverage targets in [0, 1] used to propose train-split top-K cutoffs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for dataset_name in args.datasets:
        print(f"Analyzing {dataset_name} …")
        summary = write_dataset_reports(
            dataset_name,
            args.data_root,
            args.mimic_root,
            args.output_dir,
            k_values=args.k_values,
            rare_thresholds=args.rare_thresholds,
            coverage_targets=args.coverage_targets,
        )
        summaries.append(summary)
        print(
            f"  train/dev/test sequences: {summary['splits']['train']['n_sequences']:,} / "
            f"{summary['splits']['val']['n_sequences']:,} / {summary['splits']['test']['n_sequences']:,}"
        )
        print(f"  D={summary['num_event_types']}, report={args.output_dir / f'{dataset_name}_summary.json'}")

    write_markdown_report(args.output_dir / "mimic_iv_report.md", summaries)
    print(f"Combined Markdown report: {args.output_dir / 'mimic_iv_report.md'}")


if __name__ == "__main__":
    main()
