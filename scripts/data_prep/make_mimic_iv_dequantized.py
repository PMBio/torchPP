#!/usr/bin/env python3
"""Uniform dequantization of the processed MIMIC-IV ICU-interventions dataset.

Transforms an existing processed ``icu_interventions`` bundle (default
``$TORCHPPDATA/mimic_iv_f64``) into a new root (default ``mimic_iv_f64_dequant``);
the source is only read. Everything except the within-minute event times is
carried over unchanged (same stays, same order of stays, same splits, same raw
type ids -> identical top-K vocabulary at load time, same T_obs, same metadata
plus a ``dequantization`` record).

Transformation, per stay:

1. Undo the ``ensure_strictly_increasing`` tie jitter: consecutive events with a
   gap < 1e-5 d (the jitter is exactly k * 1e-6 d; real gaps are >= 1 min) form a
   tie group whose recorded starttime is the group's first time (stored untouched
   by the jitter). Checked per stay: residuals vs k * 1e-6, every group start on
   the stay's 1-minute grid, no group reaching the next minute.
2. Every event gets an independent offset u = U * w, U ~ Uniform[0, 1) (one fixed
   numpy draw, ``--seed``, in split order train, dev, test and stored stay order),
   w = min(1 min, T_obs - recorded time). The clip keeps events <= outtime (T_obs =
   outtime - intime for every stay in the source), so T_obs is unchanged; it only
   binds in the outtime minute.
3. Events are re-sorted by the jittered time within the stay (stable); types are
   permuted accordingly. Order across minutes is preserved, order within a minute
   becomes random; duplicates (same type, same minute) get independent offsets.

All arithmetic is in the stored (scaled) float64 time unit: new = t_rec + U * w with
t_rec the stored group-start value, so recorded times are reproduced bit-exactly.

    python scripts/data_prep/make_mimic_iv_dequantized.py \
        --src-root mimic_iv_f64 --out-root mimic_iv_f64_dequant --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from torchpp.training.dataset_registry import get_data_root  # noqa: E402
from torchpp.utils.event_data import MVEventData  # noqa: E402

MINUTE_D = 1.0 / 1440.0
EPS_D = 1e-6
TIE_THRESH_D = 1e-5
SPLITS = ("train", "dev", "test")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def recorded_times(t: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recorded (pre-tie-jitter) times in stored units, plus group id and position in group."""
    d = t * scale
    new_group = np.concatenate([[True], np.diff(d) > TIE_THRESH_D])
    gid = np.cumsum(new_group) - 1
    starts = np.flatnonzero(new_group)
    pos = np.arange(len(t)) - starts[gid]
    rec = t[starts][gid]
    # hard checks: exact jitter structure, group starts on the stay's minute grid
    resid = np.abs(d - rec * scale - pos * EPS_D).max()
    if resid > 1e-11:
        raise RuntimeError(f"tie-jitter residual {resid:.3g} d: recorded minutes not recoverable")
    x = (t[starts] - t[starts[0]]) * scale * 1440.0
    if np.abs(x - np.round(x)).max() > 1e-6:
        raise RuntimeError("group start not on the 1-minute grid")
    if (pos * EPS_D).max() >= MINUTE_D / 2:
        raise RuntimeError("tie group spans half a minute or more")
    return rec, gid, pos


def dequantize_split(
    seqs: List[MVEventData], scale: float, rng: np.random.Generator
) -> Tuple[List[MVEventData], Dict[str, int]]:
    minute_s = MINUTE_D / scale  # one minute in stored units
    out: List[MVEventData] = []
    c = dict(n_seq=0, n_events=0, n_clipped_events=0, n_stays_clipped=0, n_zero_width_events=0,
             n_exact_ties_after=0, n_reordered_stays=0, n_tobs_changed=0)
    for seq in seqs:
        t = seq.time_points.numpy().astype(np.float64)
        y = seq.event_types.numpy()
        T = float(seq.T_obs)
        rec, _, _ = recorded_times(t, scale)
        width = np.minimum(minute_s, T - rec)
        if (width < 0).any():
            raise RuntimeError("recorded time after T_obs")
        clipped = width < minute_s
        U = rng.random(len(t))
        new_t = rec + U * width
        order = np.argsort(new_t, kind="stable")
        new_t = new_t[order]
        new_y = y[order]
        n_ties = int((np.diff(new_t) <= 0).sum())
        if n_ties:
            raise RuntimeError(f"{n_ties} exact ties after dequantization (measure-zero; handle explicitly)")
        new_T = max(T, float(new_t[-1]))
        c["n_seq"] += 1
        c["n_events"] += len(t)
        c["n_clipped_events"] += int(clipped.sum())
        c["n_stays_clipped"] += int(clipped.any())
        c["n_zero_width_events"] += int((width == 0).sum())
        c["n_reordered_stays"] += int((order != np.arange(len(t))).any())
        c["n_tobs_changed"] += int(new_T != T)
        out.append(
            MVEventData(
                time_points=torch.from_numpy(new_t),  # float64
                event_types=torch.from_numpy(np.ascontiguousarray(new_y)).to(seq.event_types.dtype),
                covariates=seq.covariates,
                T_obs=new_T,
            )
        )
    return out, c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", default="mimic_iv_f64")
    ap.add_argument("--out-root", default="mimic_iv_f64_dequant")
    ap.add_argument("--dataset", default="icu_interventions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_root = get_data_root()
    src = data_root / args.src_root / args.dataset
    dst = data_root / args.out_root / args.dataset
    if dst.exists():
        raise FileExistsError(f"{dst} exists; refusing to overwrite")
    if src.resolve() == dst.resolve():
        raise ValueError("source and destination are identical")

    meta = torch.load(src / "metadata.pt", weights_only=False)
    scale = float(meta["time_scale_factor"])
    rng = np.random.default_rng(args.seed)
    results, counts = {}, {}
    for split in SPLITS:
        seqs = torch.load(src / f"{split}.pt", weights_only=False)
        results[split], counts[split] = dequantize_split(seqs, scale, rng)
        print(split, json.dumps(counts[split]), flush=True)

    new_meta = deepcopy(meta)
    new_meta["dequantization"] = {
        "method": "uniform_within_recorded_minute",
        "description": (
            "recorded starttime (tie jitter of ensure_strictly_increasing removed) + U*w, "
            "U~Uniform[0,1) iid per event, w = min(1 min, T_obs - recorded time); events re-sorted "
            "by jittered time within each stay; no tie jitter; T_obs unchanged"
        ),
        "seed": args.seed,
        "rng": "numpy.random.default_rng(seed).random(n) per stay, splits in order train, dev, test",
        "source_root": args.src_root,
        "source_sha256": {f"{s}.pt": sha256(src / f"{s}.pt") for s in (*SPLITS, "metadata")},
        "counts": counts,
        "script": "scripts/data_prep/make_mimic_iv_dequantized.py",
    }
    dst.mkdir(parents=True)
    for split in SPLITS:
        torch.save(results[split], dst / f"{split}.pt")
    torch.save(new_meta, dst / "metadata.pt")
    sums = {f"{s}.pt": sha256(dst / f"{s}.pt") for s in (*SPLITS, "metadata")}
    (dst / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in sums.items()))
    print("SHA256", json.dumps(sums))
    print(f"saved to {dst}")


if __name__ == "__main__":
    main()
