"""
Download EasyTPP benchmark datasets and convert them to the TorchPP MVEventData format.

## Data Sources

All datasets are available on HuggingFace under the `easytpp` organisation:
  https://huggingface.co/easytpp

Datasets (HF identifier → num_event_types):
  easytpp/taxi          →  10
  easytpp/stackoverflow → 22
  easytpp/retweet       →   3
  easytpp/amazon        → 16
  easytpp/taobao        → 17  (the `datasets` loader shows a preview only; convert the json files, see below)
  easytpp/earthquake    →  7
  easytpp/volcano       →  1
  easytpp/synthetic     →  3  (generated from Hawkes process simulation)

For datasets not on HuggingFace use the Google Drive links published in the EasyTPP README:
  https://drive.google.com/drive/u/0/folders/1f8k82-NL6KFKuNMsUwozmbzDSFycYvz7

## EasyTPP data format (pkl files)

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

Datasets are saved to `$TORCHPPDATA/easytpp/<dataset_name>/` (default `~/data/torchpp/`) as:
  train.pt  / dev.pt  / test.pt   –  list[MVEventData]

Each MVEventData holds:
  time_points:  float32 (default; --dtype float64 optional) tensor of shape (L,)  – time_since_start values
  event_types:  int64   tensor of shape (L,)  – 0-indexed event type ids

Usage
-----
  # All HuggingFace datasets
  python scripts/data_prep/download_easytpp_datasets.py

  # Specific datasets
  python scripts/data_prep/download_easytpp_datasets.py --datasets taxi stackoverflow retweet

  # From a local pkl directory (e.g. after downloading from Google Drive)
  python scripts/data_prep/download_easytpp_datasets.py --pkl_dir /path/to/taobao --name taobao --num_types 17

  # Taobao as used in the paper: the HuggingFace json files of easytpp/taobao, stored in float64
  # under the root easytpp_hf_f64 (which selects the matching time constants, see
  # torchpp/data/easytpp_data.py). Download the three files first:
  #   for s in train dev test; do
  #     curl -L -o tbhf/$s.json https://huggingface.co/datasets/easytpp/taobao/resolve/main/$s.json; done
  python scripts/data_prep/download_easytpp_datasets.py --json_dir tbhf --name taobao --num_types 17 \
      --dtype float64 --out_root $TORCHPPDATA/easytpp_hf_f64

  The default time dtype stays float32 (the stored EasyTPP data of the other datasets).
"""

import argparse
import pickle
from pathlib import Path

import torch

# ── project root so we can import torchpp ────────────────────────────────────
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torchpp.training.dataset_registry import get_data_root
from torchpp.utils.event_data import MVEventData

# ── Dataset registry ──────────────────────────────────────────────────────────
# Maps dataset short-name → (HuggingFace repo id or None, num_event_types)
# Set hf_id=None for datasets that must be obtained from Google Drive.
DATASETS = {
    "taxi": ("easytpp/taxi", 10),
    "stackoverflow": ("easytpp/stackoverflow", 22),
    "retweet": ("easytpp/retweet", 3),
    "amazon": ("easytpp/amazon", 16),
    "earthquake": ("easytpp/earthquake", 7),
    "volcano": ("easytpp/volcano", 1),
    # taobao: convert the HuggingFace json files with --json_dir (see the usage above)
    # "taobao":      (None,                    17),
}

DATA_DIR = get_data_root() / "easytpp"


# ── Conversion helpers ────────────────────────────────────────────────────────


TIME_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def seq_to_mv_event(seq: list[dict], dtype: torch.dtype = torch.float32) -> MVEventData:
    """Convert a single EasyTPP sequence (list of event dicts) to MVEventData."""
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


# ── Load from local pkl files ─────────────────────────────────────────────────


def load_from_pkl(pkl_dir: Path, num_types: int, out_dir: Path, dtype: torch.dtype = torch.float32):
    """Convert local EasyTPP pkl files (train/dev/test) to pt files."""
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


# ── Load from local HuggingFace json files ────────────────────────────────────


def load_from_json(json_dir: Path, num_types: int, out_dir: Path, dtype: torch.dtype = torch.float32):
    """Convert local HuggingFace EasyTPP json files (train/dev/test.json, as served at
    https://huggingface.co/datasets/<id>/resolve/main/<split>.json) to pt files. Times are kept as given."""
    import json

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


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Download EasyTPP datasets and convert to TorchPP format.",
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
            "Use together with --name and --num_types for datasets not on HuggingFace "
            "(e.g. taobao from Google Drive)."
        ),
    )
    parser.add_argument(
        "--json_dir",
        type=Path,
        default=None,
        help="Directory with HuggingFace train.json / dev.json / test.json (use with --name and --num_types).",
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
    parser.add_argument(
        "--dtype",
        choices=sorted(TIME_DTYPES),
        default="float32",
        help="dtype of the stored time_points (default float32 = the paper's EasyTPP data).",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=DATA_DIR,
        help="Root directory; files go to <out_root>/<name>/{train,dev,test}.pt (default: data/easytpp).",
    )
    args = parser.parse_args()
    dtype = TIME_DTYPES[args.dtype]

    # ── Case 1: local pkl directory ──────────────────────────────────────────
    if args.pkl_dir is not None:
        assert args.name is not None and args.num_types is not None, (
            "--name and --num_types are required when --pkl_dir is set."
        )
        out_dir = args.out_root / args.name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{args.name}] converting pkl files from {args.pkl_dir} (time dtype {args.dtype})")
        load_from_pkl(args.pkl_dir, args.num_types, out_dir, dtype=dtype)
        return

    # ── Case 1b: local HuggingFace json directory ────────────────────────────
    if args.json_dir is not None:
        assert args.name is not None and args.num_types is not None, (
            "--name and --num_types are required when --json_dir is set."
        )
        out_dir = args.out_root / args.name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{args.name}] converting HF json files from {args.json_dir} (time dtype {args.dtype})")
        load_from_json(args.json_dir, args.num_types, out_dir, dtype=dtype)
        return

    # ── Case 2: HuggingFace downloads ────────────────────────────────────────
    for name in args.datasets:
        hf_id, num_types = DATASETS[name]
        if hf_id is None:
            print(
                f"\n[{name}] No HuggingFace source registered. Download pkl files from Google Drive and use --pkl_dir."
            )
            continue

        out_dir = args.out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{name}] num_event_types={num_types} (time dtype {args.dtype})")
        load_from_hf(hf_id, num_types, out_dir, dtype=dtype)

    print("\nDone.")


if __name__ == "__main__":
    main()
