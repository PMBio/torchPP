"""Convert a UK Biobank field export into the TorchPP UKB `.bin` format.

Requires an approved UK Biobank application; no UKB data is distributed with this
repository. The token vocabulary (`labels.csv`) and the ICD-10 mapping
(`icd10_codes_mod.tsv`) follow the Delphi repository
(https://github.com/gerstung-lab/Delphi) and are read from there. Output is a
70/20/10 train/val/test split with the token IDs used by `UKBDataset`.

Both UKB export header styles (`f.31.0.0` and `31-0.0`) are accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


CANCER_SLOTS = 17
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.2
TEST_FRACTION = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, required=True, help="Path to the raw UKB CSV export")
    parser.add_argument("--labels-file", type=Path, required=True, help="Path to labels.csv (Delphi token vocabulary)")
    parser.add_argument(
        "--icd-map-file",
        type=Path,
        required=True,
        help="Path to icd10_codes_mod.tsv (Delphi ICD-10 mapping)",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for train/val/test .bin files")
    parser.add_argument("--sep", default=",", help="CSV delimiter used by the export")
    parser.add_argument("--chunksize", type=int, default=1000, help="Rows per pandas chunk")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Stop after this many chunks. Useful for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files in out-dir",
    )
    return parser.parse_args()


def translate_export_name(legacy_name: str) -> str:
    """Translate `f.X.Y.Z` field IDs to the newer `X-Y.Z` export format."""

    if legacy_name == "f.eid":
        return "eid"
    if not legacy_name.startswith("f."):
        return legacy_name

    body = legacy_name[2:]
    parts = body.split(".")
    if len(parts) != 3:
        return body
    return f"{parts[0]}-{parts[1]}.{parts[2]}"


def load_header(csv_path: Path, sep: str) -> list[str]:
    with csv_path.open(newline="") as handle:
        return next(csv.reader(handle, delimiter=sep))


def load_label_dict(labels_file: Path) -> dict[str, int]:
    label_dict: dict[str, int] = {}
    index = -1
    for line in labels_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        label_dict[stripped.split(" ")[0]] = index + 1
        index += 1
    return label_dict


def load_legacy_field_map(icd_map_file: Path) -> tuple[dict[str, str], list[str]]:
    legacy_to_canonical: dict[str, str] = {}
    disease_labels: list[str] = []
    with icd_map_file.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            legacy_field = parts[0]
            label = parts[5]
            legacy_to_canonical[legacy_field] = label
            disease_labels.append(label)

    legacy_to_canonical["f.31.0.0"] = "sex"
    legacy_to_canonical["f.34.0.0"] = "YEAR"
    legacy_to_canonical["f.52.0.0"] = "MONTH"
    legacy_to_canonical["f.40000.0.0"] = "Death"
    legacy_to_canonical["f.53.0.0"] = "assessment_date"
    legacy_to_canonical["f.21001.0.0"] = "BMI"
    legacy_to_canonical["f.1239.0.0"] = "smoking"
    legacy_to_canonical["f.1558.0.0"] = "alcohol"

    for slot in range(CANCER_SLOTS):
        legacy_to_canonical[f"f.40005.{slot}.0"] = f"cancer_date_{slot}"
        legacy_to_canonical[f"f.40006.{slot}.0"] = f"cancer_type_{slot}"

    return legacy_to_canonical, disease_labels


def resolve_columns(
    header: Iterable[str],
    legacy_to_canonical: dict[str, str],
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    available = set(header)
    actual_to_canonical: dict[str, str] = {}
    canonical_to_actual: dict[str, str] = {}
    missing_legacy: list[str] = []

    for legacy_name, canonical_name in legacy_to_canonical.items():
        translated_name = translate_export_name(legacy_name)
        actual_name = None
        if translated_name in available:
            actual_name = translated_name
        elif legacy_name in available:
            actual_name = legacy_name

        if actual_name is None:
            missing_legacy.append(legacy_name)
            continue

        actual_to_canonical[actual_name] = canonical_name
        canonical_to_actual[canonical_name] = actual_name

    return actual_to_canonical, canonical_to_actual, missing_legacy


def infer_sex_mapping(
    csv_path: Path,
    sex_column: str,
    sep: str,
    chunksize: int,
    max_chunks: int | None,
) -> dict[int, int]:
    observed: set[int] = set()
    iterator = pd.read_csv(
        csv_path,
        sep=sep,
        usecols=[sex_column],
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk_index, chunk in enumerate(iterator):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        values = pd.to_numeric(chunk[sex_column], errors="coerce").dropna().astype(int).unique().tolist()
        observed.update(values)
        if len(observed) >= 2:
            break

    if len(observed) < 2:
        raise ValueError(f"Could not infer sex coding from column '{sex_column}'. Observed values: {sorted(observed)}")

    female_value, male_value = sorted(observed)[:2]
    return {female_value: 2, male_value: 3}


def count_valid_sequences(
    csv_path: Path,
    sex_column: str,
    sep: str,
    chunksize: int,
    max_chunks: int | None,
) -> int:
    total = 0
    iterator = pd.read_csv(
        csv_path,
        sep=sep,
        usecols=[sex_column],
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk_index, chunk in enumerate(tqdm(iterator, desc="counting valid rows", unit="chunk")):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        total += int(chunk[sex_column].notna().sum())
    return total


def existing_outputs(out_dir: Path) -> list[Path]:
    return [out_dir / name for name in ("train.bin", "val.bin", "test.bin", "summary.json")]


def ensure_output_dir(out_dir: Path, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    present = [path for path in existing_outputs(out_dir) if path.exists()]
    if present and not overwrite:
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace them: " + ", ".join(str(path) for path in present)
        )
    if overwrite:
        for path in present:
            path.unlink()


def canonical_date_columns(disease_labels: list[str]) -> list[str]:
    cols = list(disease_labels)
    cols.append("Death")
    cols.append("assessment_date")
    for slot in range(CANCER_SLOTS):
        cols.append(f"cancer_date_{slot}")
    return cols


def convert_dates_in_place(chunk: pd.DataFrame, date_columns: Iterable[str]) -> None:
    for column in date_columns:
        if column not in chunk.columns:
            continue
        converted = pd.to_datetime(chunk[column], format="%Y-%m-%d", errors="coerce")
        chunk.loc[:, column] = (converted - chunk["dob"]).dt.days


def append_tokenized_events(
    parts: list[np.ndarray],
    frame: pd.DataFrame,
    time_column: str,
    token: int,
) -> None:
    if time_column not in frame.columns:
        return
    rel = frame.loc[frame[time_column].notna(), ["batch_id", time_column]]
    if rel.empty:
        return
    token_col = np.full((len(rel), 1), token, dtype=np.int64)
    parts.append(np.hstack((rel.to_numpy(dtype=np.int64), token_col)))


def build_chunk_events(
    chunk: pd.DataFrame,
    batch_start: int,
    disease_labels: list[str],
    label_tokens: dict[str, int],
    sex_mapping: dict[int, int],
) -> tuple[np.ndarray, int]:
    chunk = chunk.dropna(subset=["sex"]).copy()
    if chunk.empty:
        return np.empty((0, 3), dtype=np.uint32), 0

    batch_ids = np.arange(batch_start, batch_start + len(chunk), dtype=np.int64)
    chunk.insert(0, "batch_id", batch_ids)

    chunk.loc[:, "sex"] = pd.to_numeric(chunk["sex"], errors="coerce")
    chunk.loc[:, "YEAR"] = pd.to_numeric(chunk["YEAR"], errors="coerce")
    chunk.loc[:, "MONTH"] = pd.to_numeric(chunk["MONTH"], errors="coerce")
    chunk.loc[:, "dob"] = pd.to_datetime(
        {"year": chunk["YEAR"], "month": chunk["MONTH"], "day": 1},
        errors="coerce",
    )

    convert_dates_in_place(chunk, canonical_date_columns(disease_labels))

    parts: list[np.ndarray] = []
    for label in disease_labels:
        append_tokenized_events(parts, chunk, label, label_tokens[label])
    append_tokenized_events(parts, chunk, "Death", label_tokens["Death"])

    sex_values = chunk["sex"].astype(int).map(sex_mapping)
    if sex_values.isna().any():
        unknown = sorted(chunk.loc[sex_values.isna(), "sex"].astype(int).unique().tolist())
        raise ValueError(f"Encountered unexpected sex codes: {unknown}")
    sex_part = np.column_stack(
        (
            chunk["batch_id"].to_numpy(dtype=np.int64),
            np.zeros(len(chunk), dtype=np.int64),
            sex_values.to_numpy(dtype=np.int64),
        )
    )
    parts.append(sex_part)

    for slot in range(CANCER_SLOTS):
        date_col = f"cancer_date_{slot}"
        type_col = f"cancer_type_{slot}"
        if date_col not in chunk.columns or type_col not in chunk.columns:
            continue
        cancer = chunk.loc[chunk[date_col].notna() & chunk[type_col].notna(), ["batch_id", date_col, type_col]].copy()
        if cancer.empty:
            continue
        cancer.loc[:, "cancer_code"] = cancer[type_col].astype(str).str.slice(0, 3)
        cancer.loc[:, "cancer_token"] = cancer["cancer_code"].map(label_tokens)
        cancer = cancer.dropna(subset=["cancer_token"])
        if cancer.empty:
            continue
        parts.append(cancer[["batch_id", date_col, "cancer_token"]].to_numpy(dtype=np.int64))

    if "assessment_date" in chunk.columns and "BMI" in chunk.columns:
        bmi = chunk.loc[
            chunk["assessment_date"].notna() & chunk["BMI"].notna(), ["batch_id", "assessment_date", "BMI"]
        ].copy()
        if not bmi.empty:
            bmi.loc[:, "BMI"] = pd.to_numeric(bmi["BMI"], errors="coerce")
            bmi = bmi.dropna(subset=["BMI"])
            if not bmi.empty:
                bmi.loc[:, "token"] = np.where(
                    bmi["BMI"] > 28,
                    label_tokens["BMI_high"],
                    np.where(bmi["BMI"] > 22, label_tokens["BMI_mid"], label_tokens["BMI_low"]),
                )
                parts.append(bmi[["batch_id", "assessment_date", "token"]].to_numpy(dtype=np.int64))

    if "assessment_date" in chunk.columns and "smoking" in chunk.columns:
        smoking = chunk.loc[
            chunk["assessment_date"].notna() & chunk["smoking"].notna(),
            ["batch_id", "assessment_date", "smoking"],
        ].copy()
        if not smoking.empty:
            smoking.loc[:, "smoking"] = pd.to_numeric(smoking["smoking"], errors="coerce")
            smoking = smoking.dropna(subset=["smoking"])
            smoking = smoking.loc[smoking["smoking"] != -3].copy()
            if not smoking.empty:
                smoking.loc[:, "token"] = np.where(
                    smoking["smoking"] == 1,
                    label_tokens["Smoking_high"],
                    np.where(
                        smoking["smoking"] == 2,
                        label_tokens["Smoking_mid"],
                        label_tokens["Smoking_low"],
                    ),
                )
                parts.append(smoking[["batch_id", "assessment_date", "token"]].to_numpy(dtype=np.int64))

    if "assessment_date" in chunk.columns and "alcohol" in chunk.columns:
        alcohol = chunk.loc[
            chunk["assessment_date"].notna() & chunk["alcohol"].notna(),
            ["batch_id", "assessment_date", "alcohol"],
        ].copy()
        if not alcohol.empty:
            alcohol.loc[:, "alcohol"] = pd.to_numeric(alcohol["alcohol"], errors="coerce")
            alcohol = alcohol.dropna(subset=["alcohol"])
            alcohol = alcohol.loc[alcohol["alcohol"] != -3].copy()
            if not alcohol.empty:
                alcohol.loc[:, "token"] = np.where(
                    alcohol["alcohol"] == 1,
                    label_tokens["Alcohol_high"],
                    np.where(alcohol["alcohol"] < 4, label_tokens["Alcohol_mid"], label_tokens["Alcohol_low"]),
                )
                parts.append(alcohol[["batch_id", "assessment_date", "token"]].to_numpy(dtype=np.int64))

    if not parts:
        return np.empty((0, 3), dtype=np.uint32), len(chunk)

    data = np.vstack(parts)
    data = data[data[:, 1] >= 0]
    if data.size == 0:
        return np.empty((0, 3), dtype=np.uint32), len(chunk)

    death_token = label_tokens["Death"]
    sex_mask = np.isin(data[:, 2], [2, 3])
    death_mask = data[:, 2] == death_token
    priority = np.where(sex_mask, 0, np.where(death_mask, 2, 1))
    order = np.lexsort((data[:, 2], priority, data[:, 1], data[:, 0]))
    data = data[order]
    data = pd.DataFrame(data).drop_duplicates([0, 2], keep="first").to_numpy(dtype=np.int64)
    return data.astype(np.uint32, copy=False), len(chunk)


def write_split_records(
    data: np.ndarray,
    handles: dict[str, object],
    train_stop: int,
    val_stop: int,
) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    if data.size == 0:
        return counts

    train_mask = data[:, 0] < train_stop
    val_mask = (data[:, 0] >= train_stop) & (data[:, 0] < val_stop)
    test_mask = data[:, 0] >= val_stop

    for split, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        if not np.any(mask):
            continue
        rel = data[mask].reshape(-1)
        rel.tofile(handles[split])
        counts[split] = int(mask.sum())
    return counts


def main() -> None:
    args = parse_args()
    if not np.isclose(TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION, 1.0):
        raise AssertionError("Split fractions must sum to 1.0")

    header = load_header(args.csv_path, args.sep)
    label_tokens = load_label_dict(args.labels_file)
    legacy_to_canonical, disease_labels = load_legacy_field_map(args.icd_map_file)
    actual_to_canonical, canonical_to_actual, missing_legacy = resolve_columns(header, legacy_to_canonical)

    required = {"sex", "YEAR", "MONTH"}
    missing_required = sorted(required - set(canonical_to_actual))
    if missing_required:
        raise ValueError(f"Missing required columns after header translation: {missing_required}")

    ensure_output_dir(args.out_dir, overwrite=args.overwrite)

    sex_mapping = infer_sex_mapping(
        csv_path=args.csv_path,
        sex_column=canonical_to_actual["sex"],
        sep=args.sep,
        chunksize=args.chunksize,
        max_chunks=args.max_chunks,
    )
    total_sequences = count_valid_sequences(
        csv_path=args.csv_path,
        sex_column=canonical_to_actual["sex"],
        sep=args.sep,
        chunksize=args.chunksize,
        max_chunks=args.max_chunks,
    )
    train_stop = int(total_sequences * TRAIN_FRACTION)
    val_stop = train_stop + int(total_sequences * VAL_FRACTION)

    usecols = sorted(actual_to_canonical)
    event_counts = {"train": 0, "val": 0, "test": 0}
    batch_start = 0

    summary = {
        "csv_path": str(args.csv_path),
        "labels_file": str(args.labels_file),
        "icd_map_file": str(args.icd_map_file),
        "chunksize": args.chunksize,
        "max_chunks": args.max_chunks,
        "total_sequences": total_sequences,
        "train_sequences": train_stop,
        "val_sequences": val_stop - train_stop,
        "test_sequences": total_sequences - val_stop,
        "sex_mapping": sex_mapping,
        "missing_fields": [translate_export_name(name) for name in missing_legacy],
    }

    print("Resolved sex mapping:", sex_mapping)
    print(
        "Split sizes:",
        f"train={summary['train_sequences']}",
        f"val={summary['val_sequences']}",
        f"test={summary['test_sequences']}",
    )
    if missing_legacy:
        print(f"Warning: {len(missing_legacy)} archived fields are not present in this export")

    handles = {
        "train": (args.out_dir / "train.bin").open("wb"),
        "val": (args.out_dir / "val.bin").open("wb"),
        "test": (args.out_dir / "test.bin").open("wb"),
    }
    try:
        iterator = pd.read_csv(
            args.csv_path,
            sep=args.sep,
            usecols=usecols,
            chunksize=args.chunksize,
            low_memory=False,
        )
        for chunk_index, chunk in enumerate(tqdm(iterator, desc="converting", unit="chunk")):
            if args.max_chunks is not None and chunk_index >= args.max_chunks:
                break

            renamed = chunk.rename(columns=actual_to_canonical)
            data, valid_rows = build_chunk_events(
                chunk=renamed,
                batch_start=batch_start,
                disease_labels=disease_labels,
                label_tokens=label_tokens,
                sex_mapping=sex_mapping,
            )
            batch_start += valid_rows
            counts = write_split_records(data, handles, train_stop=train_stop, val_stop=val_stop)
            for split, count in counts.items():
                event_counts[split] += count
    finally:
        for handle in handles.values():
            handle.close()

    summary["events_written"] = event_counts
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote outputs to", args.out_dir)
    print("Event rows written:", event_counts)


if __name__ == "__main__":
    main()
