#!/usr/bin/env python3
"""Recreate the MIMIC-IV TorchPP benchmarks and their summary reports.

This wrapper runs the end-to-end MIMIC-IV benchmark pipeline in one command:

1. Build the generated datasets under ``$TORCHPPDATA/mimic_iv/``.
2. Analyze the generated splits and write reports under ``tmp/mimic_iv_reports/``.

Existing dataset directories are skipped by default. Use ``--force-data`` to
rebuild them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_prep.analyze_mimic_iv_datasets import (  # noqa: E402
    DEFAULT_COVERAGE_TARGETS,
    DEFAULT_K_VALUES,
    DEFAULT_RARE_THRESHOLDS,
    parse_int_list,
    write_dataset_reports,
    write_markdown_report,
)
from scripts.data_prep.prepare_mimic_iv_datasets import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    DEFAULT_ZIP,
    create_dataset,
)


DEFAULT_DATASETS = ("hospital_flow", "icu_interventions")


def dataset_is_present(out_root: Path, dataset_name: str) -> bool:
    dataset_dir = out_root / dataset_name
    required = [dataset_dir / "train.pt", dataset_dir / "dev.pt", dataset_dir / "test.pt", dataset_dir / "metadata.pt"]
    return all(path.exists() for path in required)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=DEFAULT_ZIP,
        help=f"Path to the downloaded MIMIC-IV zip archive (default: {DEFAULT_ZIP}).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help=f"Output root for generated datasets (default: {DEFAULT_OUT_ROOT}).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "tmp" / "mimic_iv_reports",
        help="Directory where the analysis reports are written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=list(DEFAULT_DATASETS),
        help="Which MIMIC-IV benchmark datasets to build and analyze.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Subject split seed.")
    parser.add_argument("--min-events", type=int, default=2, help="Minimum events per retained sequence.")
    parser.add_argument("--chunk-size", type=int, default=500_000, help="Chunk size for streaming ICU event tables.")
    parser.add_argument("--force-data", action="store_true", help="Overwrite already generated dataset directories.")
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Store event times as float32 (the cohort of the MIMIC-IV knot sweep). Default float64.",
    )
    parser.add_argument("--skip-prepare", action="store_true", help="Skip dataset construction and only run analysis.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis and only build the datasets.")
    parser.add_argument(
        "--k-values",
        type=parse_int_list,
        default=list(DEFAULT_K_VALUES),
        help="Comma-separated top-K cutoffs to evaluate in the report.",
    )
    parser.add_argument(
        "--rare-thresholds",
        type=parse_int_list,
        default=list(DEFAULT_RARE_THRESHOLDS),
        help="Comma-separated rare-type event-count thresholds for the report.",
    )
    parser.add_argument(
        "--coverage-targets",
        type=float,
        nargs="+",
        default=list(DEFAULT_COVERAGE_TARGETS),
        help="Coverage targets in [0, 1] used to propose top-K cutoffs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_prepare and not args.zip_path.exists():
        raise FileNotFoundError(f"MIMIC-IV archive not found: {args.zip_path}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_prepare:
        print(f"Using archive: {args.zip_path}")
        print(f"Dataset root: {args.out_root}")
        for dataset_name in args.datasets:
            if dataset_is_present(args.out_root, dataset_name) and not args.force_data:
                print(f"[{dataset_name}] already present, skipping dataset build")
                continue
            create_dataset(
                dataset_name,
                args.zip_path,
                args.out_root,
                seed=args.seed,
                min_events=args.min_events,
                chunk_size=args.chunk_size,
                force=args.force_data,
                time_dtype=torch.float32 if args.float32 else torch.float64,
            )

    if args.skip_analysis:
        return

    args.report_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for dataset_name in args.datasets:
        if not dataset_is_present(args.out_root, dataset_name):
            raise FileNotFoundError(
                f"Generated dataset not found for {dataset_name}: {args.out_root / dataset_name}. "
                "Run without --skip-prepare or point --out-root at the existing data."
            )
        summaries.append(
            write_dataset_reports(
                dataset_name,
                args.out_root.parent,
                args.out_root.name,
                args.report_dir,
                k_values=args.k_values,
                rare_thresholds=args.rare_thresholds,
                coverage_targets=args.coverage_targets,
            )
        )

    write_markdown_report(args.report_dir / "mimic_iv_report.md", summaries)
    print(f"Analysis reports written to {args.report_dir}")


if __name__ == "__main__":
    main()
