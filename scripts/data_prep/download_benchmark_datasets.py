"""
Download the standard TPP benchmark datasets and convert them to the TorchPP MVEventData format.

## Data Sources

All datasets are available on HuggingFace under the `easytpp` organisation:
  https://huggingface.co/easytpp

Datasets (HF identifier → num_event_types):
  easytpp/taxi          →  10
  easytpp/stackoverflow → 22
  easytpp/retweet       →   3
  easytpp/amazon        → 16
  easytpp/taobao        → 17  (json files fetched directly, times stored in float64; see below)
  easytpp/earthquake    →  7
  easytpp/volcano       →  1
  easytpp/synthetic     →  3  (generated from Hawkes process simulation)

For datasets not on HuggingFace use the Google Drive links published in the EasyTPP README:
  https://drive.google.com/drive/u/0/folders/1f8k82-NL6KFKuNMsUwozmbzDSFycYvz7

Taobao: use the HuggingFace release, not the Google Drive folder. The Drive folder holds several
Taobao versions, and in `tb_v8` the train/dev splits and the test split come from two different
preprocessings. The HuggingFace `datasets` loader only shows a preview of Taobao, so this script
downloads the three json files directly. Taobao is stored in float64, as in the NeurIPS 2026 paper.
The Drive versions keep absolute times of about 1.5e5, which float32 rounds to a 1/64 grid.

## Source data format (pkl files)

Each pkl file contains a dict:
  {
    "dim_process": int,         # number of event types
    "train" / "dev" / "test":   # list of sequences
      [
        {
          "time_since_start":     float,  # absolute time from sequence start
          "time_since_last_event": float,  # inter-arrival time
          "type_event":           int,    # 0-indexed event type
        },
        ...
      ]
  }

For HuggingFace JSON datasets the same fields appear at the top level as lists:
  {
    "dim_process":        int,
    "time_since_start":   list[float],
    "time_since_last_event": list[float],
    "type_event":         list[int],
  }

## TorchPP format

Datasets are saved to `$TORCHPPDATA/benchmarks/<dataset_name>/` (default `~/data/torchpp/`) as:
  train.pt  / dev.pt  / test.pt   –  list[MVEventData]

Each MVEventData holds:
  time_points:  float32 tensor of shape (L,)  – time_since_start values (float64 for taobao)
  event_types:  int64   tensor of shape (L,)  – 0-indexed event type ids

Usage
-----
  # All HuggingFace datasets
  python scripts/data_prep/download_benchmark_datasets.py

  # Specific datasets
  python scripts/data_prep/download_benchmark_datasets.py --datasets taxi stackoverflow retweet

  # Taobao (downloads train/dev/test.json from the HuggingFace hub)
  python scripts/data_prep/download_benchmark_datasets.py --datasets taobao

  # From local json files (train/dev/test.json, as served by the HuggingFace hub)
  python scripts/data_prep/download_benchmark_datasets.py --json_dir /path/to/taobao --name taobao --num_types 17

  # From a local pkl directory (e.g. after downloading from Google Drive)
  python scripts/data_prep/download_benchmark_datasets.py --pkl_dir /path/to/data --name <name> --num_types <D>
"""

import argparse
import hashlib
import json
import pickle
import urllib.request
from pathlib import Path

import torch

# ── project root so we can import torchpp ────────────────────────────────────
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torchpp.training.dataset_registry import get_data_root
from torchpp.utils.event_data import MVEventData

# ── Dataset registry ──────────────────────────────────────────────────────────
# Maps dataset short-name → (HuggingFace repo id, num_event_types)
DATASETS = {
    "taxi": ("easytpp/taxi", 10),
    "stackoverflow": ("easytpp/stackoverflow", 22),
    "retweet": ("easytpp/retweet", 3),
    "amazon": ("easytpp/amazon", 16),
    "earthquake": ("easytpp/earthquake", 7),
    "volcano": ("easytpp/volcano", 1),
    "taobao": ("easytpp/taobao", 17),
}

# Datasets whose json files are downloaded directly (the `datasets` loader only shows a preview),
# and the time dtype they are stored in.
JSON_DATASETS = {"taobao": torch.float64}

# md5 of the json files used for the NeurIPS 2026 paper (a mismatch only prints a warning)
JSON_MD5 = {
    ("taobao", "train"): "8b69ec879f22492995b6d36b32c8ebca",
    ("taobao", "dev"): "6716d9f225613541f951b43124af9638",
    ("taobao", "test"): "4e7a74b45d3a5afbd6d5902868411337",
}

DATA_DIR = get_data_root() / "benchmarks"


# ── Conversion helpers ────────────────────────────────────────────────────────


def seq_to_mv_event(seq: list[dict], dtype: torch.dtype = torch.float32) -> MVEventData:
    """Convert a single source sequence (list of event dicts) to MVEventData."""
    times = torch.tensor([e["time_since_start"] for e in seq], dtype=dtype)
    types = torch.tensor([e["type_event"] for e in seq], dtype=torch.long)
    return MVEventData(time_points=times, event_types=types)


def hf_row_to_mv_event(
    time_since_start: list[float], type_event: list[int], dtype: torch.dtype = torch.float32
) -> MVEventData:
    """Convert a single HuggingFace dataset row to MVEventData."""
    times = torch.tensor(time_since_start, dtype=dtype)
    types = torch.tensor(type_event, dtype=torch.long)
    return MVEventData(time_points=times, event_types=types)


# ── Download from HuggingFace ─────────────────────────────────────────────────


def load_from_hf(hf_id: str, num_types: int, out_dir: Path, dtype: torch.dtype = torch.float32):
    """Load dataset from HuggingFace and save as pt files."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The `datasets` package is required to download from HuggingFace.\nInstall it with:  pip install datasets"
        )

    print(f"  Downloading from HuggingFace: {hf_id} …")
    ds = load_dataset(hf_id)

    split_map = {"train": "train", "validation": "dev", "test": "test"}

    for hf_split, our_split in split_map.items():
        if hf_split not in ds:
            print(f"    Split '{hf_split}' not found, skipping.")
            continue

        split_data = ds[hf_split]
        sequences = []

        for row in split_data:
            mv = hf_row_to_mv_event(row["time_since_start"], row["type_event"], dtype=dtype)
            sequences.append(mv)

        out_path = out_dir / f"{our_split}.pt"
        torch.save(sequences, out_path)
        print(f"    {our_split}: {len(sequences)} sequences → {out_path}")


# ── Load from json files ──────────────────────────────────────────────────────


def download_json(hf_id: str, name: str, json_dir: Path):
    """Download train/dev/test.json of a HuggingFace dataset repo into json_dir."""
    json_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        url = f"https://huggingface.co/datasets/{hf_id}/resolve/main/{split}.json"
        path = json_dir / f"{split}.json"
        print(f"  Downloading {url} …")
        urllib.request.urlretrieve(url, path)
        expected = JSON_MD5.get((name, split))
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        if expected is not None and md5 != expected:
            print(f"    WARNING: md5 of {path.name} is {md5}, the paper used {expected}.")


def load_from_json(json_dir: Path, num_types: int, out_dir: Path, dtype: torch.dtype = torch.float32):
    """Convert train/dev/test.json files (EasyTPP HuggingFace layout) to pt files."""
    for split in ("train", "dev", "test"):
        path = json_dir / f"{split}.json"
        if not path.exists():
            print(f"    json file not found, skipping: {path}")
            continue
        rows = json.loads(path.read_text())
        for r in rows:
            assert r["dim_process"] == num_types, f"Expected dim_process={num_types} but got {r['dim_process']} in {path}"
            assert len(r["time_since_start"]) == len(r["type_event"]) == r["seq_len"], f"length mismatch in {path}"
        sequences = [hf_row_to_mv_event(r["time_since_start"], r["type_event"], dtype=dtype) for r in rows]
        out_path = out_dir / f"{split}.pt"
        torch.save(sequences, out_path)
        print(f"    {split}: {len(sequences)} sequences → {out_path}")


# ── Load from local pkl files ─────────────────────────────────────────────────


def load_from_pkl(pkl_dir: Path, num_types: int, out_dir: Path, dtype: torch.dtype = torch.float32):
    """Convert local pkl files (train/dev/test) to pt files."""
    for split in ("train", "dev", "test"):
        pkl_path = pkl_dir / f"{split}.pkl"
        if not pkl_path.exists():
            print(f"    pkl file not found, skipping: {pkl_path}")
            continue

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        assert data["dim_process"] == num_types, (
            f"Expected dim_process={num_types} but got {data['dim_process']} in {pkl_path}"
        )

        sequences = [seq_to_mv_event(seq, dtype=dtype) for seq in data[split]]
        out_path = out_dir / f"{split}.pt"
        torch.save(sequences, out_path)
        print(f"    {split}: {len(sequences)} sequences → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Download the benchmark datasets and convert them to TorchPP format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="Which datasets to download (default: all registered HF datasets).",
    )
    parser.add_argument(
        "--pkl_dir",
        type=Path,
        default=None,
        help=(
            "Path to a directory containing train.pkl / dev.pkl / test.pkl. "
            "Use together with --name and --num_types for datasets not on HuggingFace."
        ),
    )
    parser.add_argument(
        "--json_dir",
        type=Path,
        default=None,
        help=(
            "Path to a directory containing train.json / dev.json / test.json in the EasyTPP "
            "HuggingFace layout. Use together with --name and --num_types."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default=None,
        help="Time dtype of the stored splits (default: float64 for taobao, float32 otherwise).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Dataset name used as output subfolder when --pkl_dir is provided.",
    )
    parser.add_argument(
        "--num_types",
        type=int,
        default=None,
        help="Number of event types when --pkl_dir is provided.",
    )
    args = parser.parse_args()

    def dtype_for(name):
        if args.dtype is not None:
            return getattr(torch, args.dtype)
        return JSON_DATASETS.get(name, torch.float32)

    # ── Case 1: local pkl or json directory ──────────────────────────────────
    if args.pkl_dir is not None or args.json_dir is not None:
        assert args.name is not None and args.num_types is not None, (
            "--name and --num_types are required when --pkl_dir or --json_dir is set."
        )
        out_dir = DATA_DIR / args.name
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.json_dir is not None:
            print(f"\n[{args.name}] converting json files from {args.json_dir}")
            load_from_json(args.json_dir, args.num_types, out_dir, dtype=dtype_for(args.name))
        else:
            print(f"\n[{args.name}] converting pkl files from {args.pkl_dir}")
            load_from_pkl(args.pkl_dir, args.num_types, out_dir, dtype=dtype_for(args.name))
        return

    # ── Case 2: HuggingFace downloads ────────────────────────────────────────
    for name in args.datasets:
        hf_id, num_types = DATASETS[name]
        out_dir = DATA_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{name}] num_event_types={num_types}")
        if name in JSON_DATASETS:
            json_dir = out_dir / "json"
            download_json(hf_id, name, json_dir)
            load_from_json(json_dir, num_types, out_dir, dtype=dtype_for(name))
        else:
            load_from_hf(hf_id, num_types, out_dir, dtype=dtype_for(name))

    print("\nDone.")


if __name__ == "__main__":
    main()
