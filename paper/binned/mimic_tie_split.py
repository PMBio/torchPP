#!/usr/bin/env python
"""MIMIC-IV metrics split by whether the target was recorded in the same minute as the previous event.

Joins, per main-run checkpoint, the per-item evaluation dump (evaluate.py with TORCHPP_EVAL_DUMP_ITEMS) and the
binned-likelihood arrays (binned_time_ll.py, 1-minute floor bins anchored at the recorded minute) on
(sequence, position); both are emitted by make_binned_commands.py --mimic-dumps.

    python paper/binned/mimic_tie_split.py --dir binned_out --out paper/results/mimic_same_minute_split.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

MIN = 1.0 / 1440
MODELS = ["poisson", "spline_poisson", "hawkes", "spline_exp_hawkes", "const_spline_kernel", "spline_spline_kernel",
          "inhibitive_const_exp_hawkes", "inhibitive_spline_spline_kernel", "neural_poisson", "neural_log_linear",
          "easytpp_sahp", "easytpp_thp", "neural_spline", "intensity_free", "nhp"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, required=True, help="--out-dir of make_binned_commands.py")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for dump in sorted((a.dir / "dump").glob("*_mimic_iv_icu*_rep*.npz")):
        job = dump.stem
        model = max((m for m in MODELS if job.startswith(m + "_mimic_iv_icu")), key=len)
        seed = int(re.search(r"_rep(\d+)$", job).group(1))
        z = np.load(dump)
        b = np.load(a.dir / f"binned_main_{job}.npz")
        j = int(np.where((np.abs(b["deltas"] - MIN) < 1e-15) & ~b["centered"] & ~b["prev_anchor"])[0][0])
        same = {(int(s), int(h)): bool(t) for s, h, t in zip(b["seq_idx"], b["history_length"], b["same_bin"][:, j])}
        tie = np.array([same[(int(s), int(h))] for s, h in zip(z["seq_idx"], z["history_length"])])
        lts = float(z["log_time_scale"])
        for subset, m in (("all", np.ones_like(tie)), ("other_minute", ~tie), ("same_minute", tie)):
            rows.append(dict(model_type=model, seed=seed, subset=subset, n=int(m.sum()),
                             joint_ll=float(z["joint_ll_scaled"][m].mean() - lts),
                             marginal_time_ll=float(z["time_ll_scaled"][m].mean() - lts),
                             cond_type_ll=float(z["cond_type_ll"][m].mean()),
                             cond_top1_accuracy=float(z["cond_top1_correct"][m].mean())))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {df.model_type.nunique()} models, {len(df) // 3} runs")


if __name__ == "__main__":
    main()
