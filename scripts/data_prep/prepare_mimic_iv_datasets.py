#!/usr/bin/env python3
"""Create two recurring MIMIC-IV benchmark datasets for TorchPP.

The script reads the credentialed MIMIC-IV archive directly from the downloaded
``mimic-iv-3.1.zip`` file and writes preprocessed ``{train,dev,test}.pt``
splits under ``$TORCHPPDATA/mimic_iv/``.

Datasets
--------
1. ``hospital_flow``
   One sequence per hospital admission (``hadm_id``). Events are care-unit
   transfers extracted from ``hosp/transfers.csv.gz``. Consecutive duplicate
   care units are collapsed.

2. ``icu_interventions``
   One sequence per ICU stay (``stay_id``). Events are drawn from
   ``icu/inputevents.csv.gz`` and ``icu/procedureevents.csv.gz`` using the raw
   ICU ``itemid`` vocabulary.

Times are converted to days relative to the observation-window start and then
scaled by the mean training-set observation window so that a typical sequence
spans roughly ``O(1)`` time units for model training.
"""

from __future__ import annotations

import argparse
import gzip
import math
import sys
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from torchpp.training.dataset_registry import get_data_root  # noqa: E402
from torchpp.utils.event_data import MVEventData  # noqa: E402


DEFAULT_ZIP = Path.home() / "data" / "mimic-iv-3.1.zip"
DEFAULT_OUT_ROOT = get_data_root() / "mimic_iv"

TRANSFERS_MEMBER = "mimic-iv-3.1/hosp/transfers.csv.gz"
ICUSTAYS_MEMBER = "mimic-iv-3.1/icu/icustays.csv.gz"
INPUTEVENTS_MEMBER = "mimic-iv-3.1/icu/inputevents.csv.gz"
PROCEDUREEVENTS_MEMBER = "mimic-iv-3.1/icu/procedureevents.csv.gz"
D_ITEMS_MEMBER = "mimic-iv-3.1/icu/d_items.csv.gz"

SECONDS_PER_DAY = 24.0 * 60.0 * 60.0
EPS_DAYS = 1e-6


@contextmanager
def open_member(zip_path: Path, member: str):
    """Yield a readable file object for a ``.csv.gz`` member inside the zip."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as raw:
            with gzip.GzipFile(fileobj=raw) as gz:
                yield gz


def read_csv_member(
    zip_path: Path,
    member: str,
    *,
    usecols: Optional[Sequence[str]] = None,
    parse_dates: Optional[Sequence[str]] = None,
    dtype: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    with open_member(zip_path, member) as stream:
        return pd.read_csv(stream, usecols=usecols, parse_dates=parse_dates, dtype=dtype)


def iter_csv_member(
    zip_path: Path,
    member: str,
    *,
    usecols: Optional[Sequence[str]] = None,
    parse_dates: Optional[Sequence[str]] = None,
    dtype: Optional[Dict[str, str]] = None,
    chunksize: int = 500_000,
) -> Iterator[pd.DataFrame]:
    with open_member(zip_path, member) as stream:
        yield from pd.read_csv(
            stream,
            usecols=usecols,
            parse_dates=parse_dates,
            dtype=dtype,
            chunksize=chunksize,
        )


def ensure_strictly_increasing(times: np.ndarray, eps_days: float = EPS_DAYS) -> np.ndarray:
    """Make timestamps strictly increasing with tiny forward jitter when needed."""
    fixed = np.asarray(times, dtype=np.float64).copy()
    for i in range(1, len(fixed)):
        if not np.isfinite(fixed[i]) or fixed[i] <= fixed[i - 1]:
            fixed[i] = fixed[i - 1] + eps_days
    return fixed


def make_subject_splits(subject_ids: Sequence[int], seed: int) -> Tuple[set[int], set[int], set[int]]:
    unique_subjects = np.array(sorted(set(int(s) for s in subject_ids)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique_subjects)

    n_total = len(perm)
    n_train = int(0.8 * n_total)
    n_dev = int(0.1 * n_total)

    train_subjects = set(int(x) for x in perm[:n_train])
    dev_subjects = set(int(x) for x in perm[n_train : n_train + n_dev])
    test_subjects = set(int(x) for x in perm[n_train + n_dev :])
    return train_subjects, dev_subjects, test_subjects


def scale_sequences(sequences: List[MVEventData], factor: float) -> List[MVEventData]:
    scaled: List[MVEventData] = []
    for seq in sequences:
        scaled.append(
            MVEventData(
                time_points=seq.time_points / factor,
                event_types=seq.event_types.clone(),
                covariates=seq.covariates,
                T_obs=(seq.T_obs / factor) if seq.T_obs is not None else None,
            )
        )
    return scaled


def compute_time_scale_factor(sequences: Sequence[MVEventData]) -> float:
    obs = [float(seq.T_obs) for seq in sequences if seq.T_obs is not None and seq.T_obs > 0]
    if not obs:
        return 1.0
    scale = float(np.mean(obs))
    return scale if math.isfinite(scale) and scale > 0 else 1.0


def summarise_lengths(sequences: Sequence[MVEventData]) -> Dict[str, float]:
    if not sequences:
        return {"n_sequences": 0, "mean_events": 0.0, "median_events": 0.0, "max_events": 0.0}
    lengths = np.array([len(seq.time_points) for seq in sequences], dtype=np.int64)
    return {
        "n_sequences": int(len(sequences)),
        "mean_events": float(lengths.mean()),
        "median_events": float(np.median(lengths)),
        "max_events": int(lengths.max()),
    }


def save_dataset_bundle(
    out_dir: Path,
    train: List[MVEventData],
    dev: List[MVEventData],
    test: List[MVEventData],
    metadata: Dict,
    *,
    force: bool,
) -> None:
    if out_dir.exists() and not force:
        raise FileExistsError(f"Output directory already exists: {out_dir}. Use --force to overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(train, out_dir / "train.pt")
    torch.save(dev, out_dir / "dev.pt")
    torch.save(test, out_dir / "test.pt")
    torch.save(metadata, out_dir / "metadata.pt")


def _timestamp_to_days(ts: pd.Timestamp, ref: pd.Timestamp) -> float:
    return float((ts - ref).total_seconds() / SECONDS_PER_DAY)


def build_hospital_flow_dataset(
    zip_path: Path, min_events: int, time_dtype: torch.dtype = torch.float64
) -> Tuple[List[Dict], Dict]:
    print("\n[hospital_flow] Reading transfers table …")
    transfer_events: Dict[int, Dict] = {}
    label_set: set[str] = set()
    for chunk_idx, chunk in enumerate(
        iter_csv_member(
            zip_path,
            TRANSFERS_MEMBER,
            usecols=["subject_id", "hadm_id", "transfer_id", "eventtype", "careunit", "intime", "outtime"],
            parse_dates=["intime", "outtime"],
            chunksize=500_000,
        ),
        start=1,
    ):
        chunk = chunk.dropna(subset=["subject_id", "hadm_id", "intime"]).copy()
        if chunk.empty:
            continue
        chunk["subject_id"] = chunk["subject_id"].astype("int64")
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk["outtime"] = chunk["outtime"].fillna(chunk["intime"])
        chunk["careunit_label"] = chunk["careunit"].fillna("EVENTTYPE::" + chunk["eventtype"].fillna("UNKNOWN"))

        for row in chunk.itertuples(index=False):
            hadm_id = int(row.hadm_id)
            subject_id = int(row.subject_id)
            label = str(row.careunit_label)
            label_set.add(label)
            record = transfer_events.get(hadm_id)
            if record is None:
                record = {
                    "subject_id": subject_id,
                    "events": [],
                    "end_time": row.outtime,
                }
                transfer_events[hadm_id] = record

            record["events"].append((row.intime, int(row.transfer_id), label))
            if row.outtime > record["end_time"]:
                record["end_time"] = row.outtime

        if chunk_idx % 5 == 0:
            print(f"    processed {chunk_idx} transfer chunks, admissions tracked={len(transfer_events):,}")

    labels = sorted(label_set)
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    sequences: List[Dict] = []
    skipped_short = 0
    for hadm_id, record in transfer_events.items():
        subject_id = int(record["subject_id"])
        events_sorted = sorted(record["events"], key=lambda x: (x[0], x[1]))
        start_time = events_sorted[0][0]
        end_time = record["end_time"]

        times_rel = [_timestamp_to_days(intime, start_time) for intime, _, _ in events_sorted]
        labels_rel = [label for _, _, label in events_sorted]

        collapsed_times: List[float] = []
        collapsed_labels: List[str] = []
        for time_value, label in zip(times_rel, labels_rel):
            if not collapsed_labels or label != collapsed_labels[-1]:
                collapsed_times.append(float(time_value))
                collapsed_labels.append(label)

        if len(collapsed_labels) < min_events:
            skipped_short += 1
            continue

        event_times = ensure_strictly_increasing(np.asarray(collapsed_times, dtype=np.float64))
        event_types = np.asarray([label_to_id[label] for label in collapsed_labels], dtype=np.int64)
        t_obs = max(float((end_time - start_time).total_seconds() / SECONDS_PER_DAY), float(event_times[-1]))

        sequences.append(
            {
                "subject_id": subject_id,
                "sequence": MVEventData(
                    # float64 by default: the strictly-increasing jitter (EPS_DAYS = 1e-6
                    # days) is below the float32 ULP for multi-day timestamps, so float32
                    # collapses simultaneous events back to Δt = 0.
                    time_points=torch.from_numpy(event_times).to(time_dtype),
                    event_types=torch.from_numpy(event_types).long(),
                    T_obs=t_obs,
                ),
            }
        )

    metadata = {
        "dataset_name": "hospital_flow",
        "sequence_unit": "hadm_id",
        "event_definition": "care-unit transfer events from hosp/transfers with consecutive duplicate care units collapsed",
        "time_unit": "days",
        "num_event_types": len(label_to_id),
        "type_id_to_label": labels,
        "skipped_short_sequences": skipped_short,
        "source_tables": [TRANSFERS_MEMBER],
    }
    return sequences, metadata


def build_icu_interventions_dataset(
    zip_path: Path,
    min_events: int,
    chunk_size: int,
    time_dtype: torch.dtype = torch.float64,
) -> Tuple[List[Dict], Dict]:
    print("\n[icu_interventions] Reading ICU stay windows …")
    stays = read_csv_member(
        zip_path,
        ICUSTAYS_MEMBER,
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
        parse_dates=["intime", "outtime"],
    )
    stays = stays.dropna(subset=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
    stays["subject_id"] = stays["subject_id"].astype("int64")
    stays["hadm_id"] = stays["hadm_id"].astype("int64")
    stays["stay_id"] = stays["stay_id"].astype("int64")
    stays_lookup = stays[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]]
    stay_window_lookup: Dict[int, Tuple[int, int, pd.Timestamp, pd.Timestamp]] = {
        int(row.stay_id): (int(row.subject_id), int(row.hadm_id), row.intime, row.outtime)
        for row in stays_lookup.itertuples(index=False)
    }

    print("[icu_interventions] Reading d_items lookup …")
    d_items = read_csv_member(
        zip_path,
        D_ITEMS_MEMBER,
        usecols=["itemid", "label", "abbreviation", "category", "linksto"],
    )
    d_items = d_items.dropna(subset=["itemid"])
    d_items["itemid"] = d_items["itemid"].astype("int64")
    item_metadata_lookup: Dict[int, Dict[str, str]] = {}
    for row in d_items.itertuples(index=False):
        item_metadata_lookup[int(row.itemid)] = {
            "label": "" if pd.isna(row.label) else str(row.label),
            "abbreviation": "" if pd.isna(row.abbreviation) else str(row.abbreviation),
            "category": "" if pd.isna(row.category) else str(row.category),
            "linksto": "" if pd.isna(row.linksto) else str(row.linksto),
        }

    stay_events: Dict[int, Dict] = defaultdict(lambda: {"events": []})
    source_counts: Counter[str] = Counter()
    unique_itemids: set[int] = set()
    source_members = [
        (INPUTEVENTS_MEMBER, "inputevents"),
        (PROCEDUREEVENTS_MEMBER, "procedureevents"),
    ]
    for member, source_name in source_members:
        print(f"[icu_interventions] Streaming {source_name} …")
        kept_rows = 0
        for chunk_idx, chunk in enumerate(
            iter_csv_member(
                zip_path,
                member,
                usecols=["subject_id", "hadm_id", "stay_id", "starttime", "itemid"],
                parse_dates=["starttime"],
                chunksize=chunk_size,
            ),
            start=1,
        ):
            chunk = chunk.dropna(subset=["stay_id", "starttime", "itemid"])
            if chunk.empty:
                continue
            chunk = chunk.copy()

            chunk["stay_id"] = chunk["stay_id"].astype("int64")
            chunk["itemid"] = chunk["itemid"].astype("int64")

            matched = chunk[chunk["stay_id"].isin(stay_window_lookup)]
            if matched.empty:
                continue

            for row in matched.itertuples(index=False):
                stay_id = int(row.stay_id)
                subject_id, hadm_id, intime, outtime = stay_window_lookup[stay_id]
                if row.starttime < intime or row.starttime > outtime:
                    continue

                record = stay_events[stay_id]
                if "subject_id" not in record:
                    record["subject_id"] = subject_id
                    record["hadm_id"] = hadm_id
                    record["intime"] = intime
                    record["outtime"] = outtime

                record["events"].append((row.starttime, int(row.itemid), source_name))
                unique_itemids.add(int(row.itemid))
                kept_rows += 1
                source_counts[source_name] += 1

            if chunk_idx % 10 == 0:
                print(f"    {source_name}: processed {chunk_idx} chunks, kept {kept_rows:,} rows")

        print(f"    {source_name}: kept {kept_rows:,} rows total")

    if not stay_events:
        raise RuntimeError("No ICU intervention events were extracted from the archive.")

    unique_itemids = sorted(unique_itemids)
    itemid_to_type_id = {itemid: idx for idx, itemid in enumerate(unique_itemids)}

    sequences: List[Dict] = []
    skipped_short = 0
    for stay_id, record in stay_events.items():
        subject_id = int(record["subject_id"])
        intime = record["intime"]
        outtime = record["outtime"]
        events_sorted = sorted(record["events"], key=lambda x: (x[0], x[1], x[2]))

        times_rel = np.asarray(
            [_timestamp_to_days(starttime, intime) for starttime, _, _ in events_sorted], dtype=np.float64
        )
        if len(times_rel) < min_events:
            skipped_short += 1
            continue

        event_times = ensure_strictly_increasing(times_rel)
        event_types = np.asarray([itemid_to_type_id[itemid] for _, itemid, _ in events_sorted], dtype=np.int64)
        t_obs = max(float((outtime - intime).total_seconds() / SECONDS_PER_DAY), float(event_times[-1]))

        sequences.append(
            {
                "subject_id": subject_id,
                "sequence": MVEventData(
                    # float64 by default: the strictly-increasing jitter (EPS_DAYS = 1e-6
                    # days) is below the float32 ULP for multi-day timestamps, so float32
                    # collapses simultaneous events back to Δt = 0.
                    time_points=torch.from_numpy(event_times).to(time_dtype),
                    event_types=torch.from_numpy(event_types).long(),
                    T_obs=t_obs,
                ),
            }
        )

    metadata = {
        "dataset_name": "icu_interventions",
        "sequence_unit": "stay_id",
        "event_definition": "ICU intervention stream from inputevents and procedureevents using raw itemid vocabulary",
        "time_unit": "days",
        "num_event_types": len(itemid_to_type_id),
        "type_id_to_itemid": unique_itemids,
        "itemid_metadata": {str(itemid): item_metadata_lookup.get(itemid, {}) for itemid in unique_itemids},
        "source_tables": [INPUTEVENTS_MEMBER, PROCEDUREEVENTS_MEMBER],
        "source_event_counts": dict(source_counts),
        "skipped_short_sequences": skipped_short,
    }
    return sequences, metadata


def split_and_scale_sequences(
    sequence_records: List[Dict],
    *,
    seed: int,
) -> Tuple[List[MVEventData], List[MVEventData], List[MVEventData], float, Dict[str, int]]:
    subject_ids = [int(record["subject_id"]) for record in sequence_records]
    train_subjects, dev_subjects, test_subjects = make_subject_splits(subject_ids, seed)

    train = [record["sequence"] for record in sequence_records if int(record["subject_id"]) in train_subjects]
    dev = [record["sequence"] for record in sequence_records if int(record["subject_id"]) in dev_subjects]
    test = [record["sequence"] for record in sequence_records if int(record["subject_id"]) in test_subjects]

    scale = compute_time_scale_factor(train)
    train = scale_sequences(train, scale)
    dev = scale_sequences(dev, scale)
    test = scale_sequences(test, scale)

    split_subject_counts = {
        "train": len(train_subjects),
        "dev": len(dev_subjects),
        "test": len(test_subjects),
    }
    return train, dev, test, scale, split_subject_counts


def create_dataset(
    dataset_name: str,
    zip_path: Path,
    out_root: Path,
    *,
    seed: int,
    min_events: int,
    chunk_size: int,
    force: bool,
    time_dtype: torch.dtype = torch.float64,
) -> None:
    if dataset_name == "hospital_flow":
        sequence_records, metadata = build_hospital_flow_dataset(
            zip_path, min_events=min_events, time_dtype=time_dtype
        )
    elif dataset_name == "icu_interventions":
        sequence_records, metadata = build_icu_interventions_dataset(
            zip_path,
            min_events=min_events,
            chunk_size=chunk_size,
            time_dtype=time_dtype,
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    train, dev, test, scale, split_subject_counts = split_and_scale_sequences(sequence_records, seed=seed)
    metadata["time_scale_factor"] = scale
    metadata["scaling"] = {
        "strategy": "divide_by_mean_train_observation_window",
        "time_scale_factor": scale,
    }
    metadata["split_subject_counts"] = split_subject_counts
    metadata["split_sequence_stats"] = {
        "train": summarise_lengths(train),
        "dev": summarise_lengths(dev),
        "test": summarise_lengths(test),
    }

    print(
        f"[{dataset_name}] sequences after split: "
        f"train={len(train):,}, dev={len(dev):,}, test={len(test):,}, D={metadata['num_event_types']:,}"
    )
    print(f"[{dataset_name}] time_scale_factor={scale:.6f} days")

    out_dir = out_root / dataset_name
    save_dataset_bundle(out_dir, train, dev, test, metadata, force=force)
    print(f"[{dataset_name}] saved to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=DEFAULT_ZIP,
        help=f"Path to the downloaded MIMIC-IV zip archive (default: {DEFAULT_ZIP}).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hospital_flow", "icu_interventions"],
        choices=["hospital_flow", "icu_interventions"],
        help="Which datasets to build.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help=f"Output root for generated datasets (default: {DEFAULT_OUT_ROOT}).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subject-level train/dev/test splitting.")
    parser.add_argument(
        "--min-events",
        type=int,
        default=2,
        help="Minimum number of events required for a sequence to be kept.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Chunk size for streaming the large ICU event tables.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dataset directories if they already exist.",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Store event times as float32 (the cohort of the MIMIC-IV knot sweep). Default float64.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.zip_path.exists():
        raise FileNotFoundError(f"MIMIC-IV archive not found: {args.zip_path}")

    print(f"Using archive: {args.zip_path}")
    print(f"Output root:  {args.out_root}")
    for dataset_name in args.datasets:
        create_dataset(
            dataset_name,
            args.zip_path,
            args.out_root,
            seed=args.seed,
            min_events=args.min_events,
            chunk_size=args.chunk_size,
            force=args.force,
            time_dtype=torch.float32 if args.float32 else torch.float64,
        )


if __name__ == "__main__":
    main()
