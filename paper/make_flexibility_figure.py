#!/usr/bin/env python
"""Figure 2: change in log p(t) (continuous and binned) and log p(d|t) with the number of spline knots K (knot sweep).

Each curve is shifted by its value at K=20, so datasets with different log-likelihood ranges share one axis.
The binned log p(t) is evaluated at the recorded resolution of each dataset (Retweet: integer time stamps,
Amazon: the minimum gap 0.01, MIMIC-IV: 1 minute; bins starting at the grid points), from the same checkpoints.

    python paper/make_flexibility_figure.py [--stack] [--out paper/figures]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "knot_sweep.csv"
BINNED_CSV = HERE / "results" / "binned_time_ll.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--stack", action="store_true", help="3x1 vertical layout")
ap.add_argument("--out", type=Path, default=HERE / "figures")
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

# primary bin per dataset, in the data time unit (MIMIC-IV: days); floor bins (centered=False)
BIN = {"retweet": 1.0, "amazon": 0.01, "mimic_iv": 1 / 1440}
# K=10 underfits the time model, so the curves are anchored at K=20.
ANCHOR_K = 20
K_MAX = 400  # the knot grid of the reported runs
DATASETS = [("retweet", "Retweet"), ("amazon", "Amazon"), ("mimic_iv", "MIMIC-IV")]
# Okabe-Ito colourblind-safe
COLOR = {"retweet": "#0072B2", "amazon": "#D55E00", "mimic_iv": "#009E73"}
MODELS = [("neural_spline", "NeuralSplineTPP", "s", "-"),
          ("spline_spline_kernel", "Spline/Spline Hawkes", "x", ":")]
PANELS = [("marginal_time_ll", r"$\Delta\,\log p(t)$, continuous"),
          ("binned_time_ll", r"$\Delta\,\log p(t)$, binned"),
          ("cond_type_ll", r"$\Delta\,\log p(d\mid t)$")]

df = pd.read_csv(CSV)
_b = pd.read_csv(BINNED_CSV)
_b = _b[(_b.run == "knot_sweep") & ~_b.centered].astype({"K": int})
_rows = []
for ds in BIN:
    sub = _b[(_b.dataset == ds) & ((_b.delta - BIN[ds]).abs() < 1e-9 * BIN[ds] + 1e-12)]
    for (mt, K), g in sub.groupby(["model_type", "K"]):
        _rows.append({"dataset": ds, "model_type": mt, "K": K, "binned_time_ll_mean": g.binned_all.mean(),
                      "binned_time_ll_std": g.binned_all.std(ddof=1), "n_binned": len(g)})
_bm = pd.DataFrame(_rows)
assert (_bm.n_binned == 5).all(), _bm[_bm.n_binned != 5]
df = df.merge(_bm, on=["dataset", "model_type", "K"], how="left")
STACK = args.stack
plt.rcParams.update({"font.size": 8.5 if STACK else 12, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5})
if STACK:
    fig, axes = plt.subplots(3, 1, figsize=(3.1, 6.6), sharey=True, sharex=True)
else:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), sharey=True)

for i, (ax, (metric, title)) in enumerate(zip(axes, PANELS)):
    mean_col, std_col = f"{metric}_mean", f"{metric}_std"
    for ds, _ in DATASETS:
        for mt, _, marker, ls in MODELS:
            sub = df[(df.dataset == ds) & (df.model_type == mt)
                     & (df.K >= ANCHOR_K) & (df.K <= K_MAX)].sort_values("K")
            if sub.empty:
                continue
            base = sub[mean_col].iloc[0]           # value at K=ANCHOR_K for this curve
            ax.errorbar(sub["K"], sub[mean_col] - base, yerr=sub[std_col],
                        color=COLOR[ds], marker=marker, markersize=5, linestyle=ls,
                        linewidth=1.4, elinewidth=0.7, capsize=2, alpha=0.9)
    ax.axhline(0, color="0.6", linewidth=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([20, 50, 100, 200, 400])
    ax.set_xticklabels([20, 50, 100, 200, 400])
    ax.set_title(title)
    if (not STACK) or i == len(PANELS) - 1:
        ax.set_xlabel("spline knots $K$")
    if STACK:
        ax.set_ylabel(r"$\Delta$ LL [nats]")

if not STACK:
    axes[0].set_ylabel(r"$\Delta$ log-likelihood from $K{=}20$  [nats]")

ds_handles = [Line2D([], [], color=COLOR[d], lw=2.2, label=lbl) for d, lbl in DATASETS]
mdl_handles = [Line2D([], [], color="0.35", marker=mk, linestyle=ls, lw=1.4, label=lbl)
               for _, lbl, mk, ls in MODELS]
leg1 = axes[0].legend(handles=ds_handles, title="Dataset", loc="upper left",
                      fontsize=10.5, title_fontsize=10.5, framealpha=0.9)
axes[0].add_artist(leg1)
axes[-1].legend(handles=mdl_handles, title="Model", loc="upper right",
               fontsize=10.5, title_fontsize=10.5, framealpha=0.9)

fig.tight_layout()
stem = "flexibility_decomposition_stacked" if STACK else "flexibility_decomposition"
for ext in ("png", "pdf"):
    fig.savefig(args.out / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote {args.out / stem}.pdf")
