"""Join the TorchPP and EasyTPP equivalence CSVs into a head-to-head table.

Compares, per (model, dataset), the matched-hyperparameter native-TorchPP run
against the fresh EasyTPP reference:

    LL  : torchpp joint_ll          vs  easytpp loglike      (per-event joint, day-scaled)
    Acc : torchpp cond_top1_accuracy vs  easytpp acc          (next-mark top-1)
    RMSE: torchpp time_rmse_mean_days vs easytpp rmse          (mean-based; for IFTPP the
          heavy-tailed mean is unstable — torchpp median RMSE is also shown)

Multiple seeds are averaged (mean ± std). Prints a Markdown table and writes it
to --out. Usage:
  python aggregate_equivalence.py --torchpp torchpp.csv --easytpp easytpp.csv --out table.md
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from statistics import mean, pstdev

NAME_MAP = {"intensity_free": "IntensityFree", "nhp": "NHP", "attnhp": "AttNHP"}


def _read(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def _f(row, key):
    v = row.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return (mean(vals), pstdev(vals) if len(vals) > 1 else 0.0)


def _fmt(stat):
    if stat is None:
        return "—"
    m, s = stat
    return f"{m:.3f}±{s:.3f}" if s else f"{m:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torchpp", required=True)
    ap.add_argument("--easytpp", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tp = [r for r in _read(args.torchpp) if r.get("status") == "SUCCESS"]
    ez = [r for r in _read(args.easytpp) if r.get("status") == "SUCCESS"]

    # collect per (model_easytpp_name, dataset)
    tp_by, ez_by = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for r in tp:
        key = (NAME_MAP.get(r["model"], r["model"]), r["dataset"])
        tp_by[key]["ll"].append(_f(r, "joint_ll"))
        tp_by[key]["acc"].append(_f(r, "cond_top1_accuracy"))
        tp_by[key]["rmse_mean"].append(_f(r, "time_rmse_mean_days"))
        tp_by[key]["rmse_med"].append(_f(r, "time_rmse_median_days"))
    for r in ez:
        key = (r["model"], r["dataset"])
        ez_by[key]["ll"].append(_f(r, "loglike"))
        ez_by[key]["acc"].append(_f(r, "acc"))
        ez_by[key]["rmse"].append(_f(r, "rmse"))

    keys = sorted(set(tp_by) | set(ez_by))
    lines = [
        "# Equivalence: native TorchPP vs EasyTPP (matched hyperparameters, fresh runs)",
        "",
        "LL = per-event joint log-likelihood (higher better). Acc = next-mark top-1.",
        "RMSE in days. IFTPP mean-RMSE is heavy-tail-unstable → TorchPP median also shown.",
        "",
        "| Model | Dataset | LL (TorchPP) | LL (EasyTPP) | ΔLL | Acc (TorchPP) | Acc (EasyTPP) | RMSE (TorchPP mean/med) | RMSE (EasyTPP) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model, ds in keys:
        t, e = tp_by[(model, ds)], ez_by[(model, ds)]
        t_ll, e_ll = _agg(t["ll"]), _agg(e["ll"])
        dll = "—" if (t_ll is None or e_ll is None) else f"{t_ll[0] - e_ll[0]:+.3f}"
        rmse_tp = f"{_fmt(_agg(t['rmse_mean']))} / {_fmt(_agg(t['rmse_med']))}"
        lines.append(
            f"| {model} | {ds} | {_fmt(t_ll)} | {_fmt(e_ll)} | {dll} | "
            f"{_fmt(_agg(t['acc']))} | {_fmt(_agg(e['acc']))} | {rmse_tp} | {_fmt(_agg(e['rmse']))} |"
        )
    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")
        print(f"\n[written to {args.out}]")


if __name__ == "__main__":
    main()
