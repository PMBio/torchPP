#!/usr/bin/env python3
"""Verify a dequantized MIMIC-IV ICU bundle against its source (read-only).

    python scripts/data_prep/verify_mimic_iv_dequantized.py \
        --orig-root mimic_iv_f64 --new-root mimic_iv_f64_dequant

Prints per-check results and a final ``VERIFY_JSON`` line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from torchpp.training.dataset_registry import get_data_root  # noqa: E402
from torchpp.data.mimic_iv_data import MIMICIVDataset, _icu_topk_spec  # noqa: E402

MINUTE_D = 1.0 / 1440.0
TIE_THRESH_D = 1e-5
SEC_D = 1.0 / 86400.0
TRUNC = 316


def minute_index(t: np.ndarray, t0: float, scale: float) -> np.ndarray:
    """Recorded-minute index on the stay's grid (origin = first recorded event time)."""
    return np.floor((t - t0) * scale * 1440.0 + 1e-9).astype(np.int64)


def gap_stats(seqs, scale, limit=None, trunc=None):
    gaps = []
    for seq in seqs[:limit] if limit else seqs:
        t = seq.time_points.numpy()
        if trunc:
            t = t[:trunc]
        gaps.append(np.diff(t) * scale)
    g = np.concatenate(gaps)
    return {
        "n_targets": int(len(g)),
        "frac_exact_zero": float((g == 0).mean()),
        "frac_lt_1e-5d": float((g < TIE_THRESH_D).mean()),
        "frac_lt_1e-6d": float((g < 1e-6).mean()),
        "frac_lt_1s": float((g < SEC_D).mean()),
        "frac_lt_1min": float((g < MINUTE_D * (1 - 1e-9)).mean()),
        "min_gap_d": float(g.min()),
        "min_gap_s": float(g.min() * 86400),
        "q01_gap_s": float(np.quantile(g, 0.01) * 86400),
        "median_gap_s": float(np.median(g) * 86400),
        "n_nonpositive": int((g <= 0).sum()),
    }


def order_stats(seqs, t0s, scale):
    """Within-recorded-minute consecutive pairs: type ascending / equal / descending."""
    asc = eq = desc = 0
    for seq, t0 in zip(seqs, t0s):
        t = seq.time_points.numpy()
        y = seq.event_types.numpy()
        m = minute_index(t, t0, scale)
        same = m[1:] == m[:-1]
        dy = np.diff(y)[same]
        asc += int((dy > 0).sum())
        eq += int((dy == 0).sum())
        desc += int((dy < 0).sum())
    n = asc + eq + desc
    return dict(n_pairs=n, frac_ascending=asc / n, frac_equal=eq / n, frac_descending=desc / n,
                frac_ascending_among_distinct=asc / max(asc + desc, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-root", default="mimic_iv_f64")
    ap.add_argument("--new-root", default="mimic_iv_f64_dequant")
    ap.add_argument("--dataset", default="icu_interventions")
    args = ap.parse_args()
    root = get_data_root()
    od, nd = root / args.orig_root / args.dataset, root / args.new_root / args.dataset
    om = torch.load(od / "metadata.pt", weights_only=False)
    nm = torch.load(nd / "metadata.pt", weights_only=False)
    scale = float(om["time_scale_factor"])
    res = {"metadata_equal_except_dequant": {k: v for k, v in nm.items() if k != "dequantization"} == om,
           "time_scale_factor": scale}
    assert float(nm["time_scale_factor"]) == scale

    for split in ("train", "dev", "test"):
        O = torch.load(od / f"{split}.pt", weights_only=False)
        N = torch.load(nd / f"{split}.pt", weights_only=False)
        r = dict(n_seq_orig=len(O), n_seq_new=len(N))
        assert len(O) == len(N)
        bad = dict(len=0, tobs=0, types_multiset=0, minute_type_multiset=0, minute_multiset=0,
                   not_strict=0, outside_minute=0, after_tobs=0, before_first_rec=0, dtype=0)
        n_ev = 0
        t0s = []
        trunc_changed_types = trunc_changed_mt = trunc_straddle = n_long = 0
        for o, n in zip(O, N):
            to, tn = o.time_points.numpy(), n.time_points.numpy()
            yo, yn = o.event_types.numpy(), n.event_types.numpy()
            n_ev += len(to)
            bad["dtype"] += int(n.time_points.dtype != torch.float64)
            if len(to) != len(tn):
                bad["len"] += 1
                continue
            bad["tobs"] += int(float(o.T_obs) != float(n.T_obs))
            t0 = float(to[0])  # first event is never tie-jittered -> recorded time of first minute
            t0s.append(t0)
            mo, mn = minute_index(to, t0, scale), minute_index(tn, t0, scale)
            bad["types_multiset"] += int(not np.array_equal(np.sort(yo), np.sort(yn)))
            bad["minute_multiset"] += int(not np.array_equal(np.sort(mo), np.sort(mn)))
            ko, kn = np.sort(mo * 100000 + yo), np.sort(mn * 100000 + yn)
            bad["minute_type_multiset"] += int(not np.array_equal(ko, kn))
            bad["not_strict"] += int((np.diff(tn) <= 0).any())
            frac = (tn - t0) * scale * 1440.0 - mn
            bad["outside_minute"] += int(((frac < -1e-9) | (frac >= 1.0)).any())
            bad["after_tobs"] += int(tn[-1] > float(n.T_obs))
            bad["before_first_rec"] += int(tn[0] < t0)
            if len(to) > TRUNC:
                n_long += 1
                trunc_changed_types += int(not np.array_equal(np.sort(yo[:TRUNC]), np.sort(yn[:TRUNC])))
                trunc_changed_mt += int(not np.array_equal(np.sort(mo[:TRUNC] * 100000 + yo[:TRUNC]),
                                                           np.sort(mn[:TRUNC] * 100000 + yn[:TRUNC])))
                trunc_straddle += int(mo[TRUNC - 1] == mo[TRUNC])
        r.update(n_events=n_ev, failures=bad,
                 trunc316=dict(n_stays_longer=n_long, n_boundary_minute_straddles=trunc_straddle,
                               n_first316_type_multiset_changed=trunc_changed_types,
                               n_first316_minute_type_multiset_changed=trunc_changed_mt))
        r["gaps_full_orig"] = gap_stats(O, scale)
        r["gaps_full_new"] = gap_stats(N, scale)
        r["order_orig"] = order_stats(O, t0s, scale)
        r["order_new"] = order_stats(N, t0s, scale)
        if split == "test":
            r["gaps_eval_orig"] = gap_stats(O, scale, limit=1000, trunc=TRUNC)
            r["gaps_eval_new"] = gap_stats(N, scale, limit=1000, trunc=TRUNC)
            r["gaps_eval500_new"] = gap_stats(N, scale, limit=500, trunc=TRUNC)
        res[split] = r
        print(split, json.dumps(r), flush=True)

    # Top-116 vocabulary: computed at load time from training counts -> must be identical
    so = _icu_topk_spec(str(od), 116)
    sn = _icu_topk_spec(str(nd), 116)
    res["top116_old_to_new_identical"] = bool(torch.equal(so["old_to_new"], sn["old_to_new"]))
    res["top116_retained_identical"] = so["retained_raw_type_ids"] == sn["retained_raw_type_ids"]
    res["top116_coverage"] = [so["retained_event_coverage"], sn["retained_event_coverage"]]
    ds = MIMICIVDataset.load({"dataset_name": args.dataset, "mimic_root": args.new_root,
                              "top_k_event_types": 116}, root, "test")
    res["loader"] = dict(repr=repr(ds), D=ds.num_event_types, tsf=ds.time_scale_factor,
                         max_type=int(max(int(s.event_types.max()) for s in ds)))
    print("VERIFY_JSON " + json.dumps(res))


if __name__ == "__main__":
    main()
