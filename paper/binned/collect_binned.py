#!/usr/bin/env python
"""Collect the BINNED_METRICS_JSON lines of binned_time_ll.py runs into results/binned_time_ll.csv.

    python paper/binned/collect_binned.py logs/*.log --out paper/results/binned_time_ll.csv

One row per run and bin width. The run, dataset, model, K and seed come from the --meta of
make_binned_commands.py. Ties: exact zero gaps on the EasyTPP data, the same recorded minute on MIMIC-IV.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

COLS = ["run", "dataset", "model_type", "K", "seed", "ci", "n_targets", "n_tie", "tie_definition", "cont_all",
        "delta", "centered", "bin_anchor", "binned_all", "binned_tie", "binned_nontie"]


DATASET = {"icu_interventions": "mimic_iv"}  # dataset_name -> label


def rows(rec):
    ds = DATASET.get(rec["dataset"], rec["dataset"])
    rect = "tie_bin" in rec  # MIMIC-IV: same recorded minute as the previous event
    tie = "rectie" if rect else "tie"
    base = dict(run=rec["run"], dataset=ds, model_type=rec["model"],
                K=int(rec["K"]) if str(rec.get("K", "")).strip() else None, seed=int(rec["seed"]), ci=rec["ci"],
                n_targets=rec["n_items"], n_tie=rec["n_rectie"] if rect else rec["n_tie"],
                tie_definition="same_recorded_minute" if rect else "zero_gap", cont_all=rec["cont_all"])
    for j, d in enumerate(rec["deltas"]):
        anchor = ("previous_event" if rec["prev_anchor"][j] else
                  "recorded_minute" if ds == "mimic_iv" else "absolute")
        yield {**base, "delta": d, "centered": bool(rec["centered"][j]), "bin_anchor": anchor,
               "binned_all": rec["binned_all"][j],
               "binned_tie": rec[f"binned_{tie}"][j] if f"binned_{tie}" in rec else None,
               "binned_nontie": rec[f"binned_non{tie}"][j] if f"binned_non{tie}" in rec else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    recs = {}
    for f in a.logs:
        for m in re.findall(r"^BINNED_METRICS_JSON (.*)$", f.read_text(errors="ignore"), flags=re.M):
            r = json.loads(m)
            recs[(r["run"], r["dataset"], r["model"], str(r.get("K", "")), int(r["seed"]))] = r  # last one wins
    df = pd.DataFrame([row for r in recs.values() for row in rows(r)], columns=COLS)
    df["K"] = df["K"].astype("Int64")
    df = df.sort_values(["dataset", "run", "model_type", "K", "seed", "delta", "centered"])
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(recs)} runs, {len(df)} rows")


if __name__ == "__main__":
    main()
