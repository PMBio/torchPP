"""Run one native-TorchPP equivalence replica (model, dataset, seed) via train.py
and append a CSV row of test metrics.

This is the TorchPP half of the EasyTPP equivalence check for the ported models
(intensity_free, nhp, attnhp).  Hyperparameters are *matched to EasyTPP's*
benchmark config (so it is a true head-to-head, not best-vs-best):

    NHP    : hidden 64, 2 layers
    AttNHP : hidden 32, 2 layers, 2 heads, time_emb 16
    IFTPP  : hidden 64, 2 GRU layers, 64 mixture components

Common (all): Adam-equivalent (AdamW + weight_decay=0 + scheduler 'none'),
lr 1e-3, batch 64, dropout 0, ~20 epochs (num_steps = 20 * ceil(n_train/64)),
mirroring EasyTPP.  Metrics are parsed from train.py's "Test Evaluation Results"
block.

Eval cost control: AttNHP recomputes attention at every CI grid point, so eval
is capped (limit_test_size / grid density) — set per-model defaults below,
overridable on the CLI.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TRAIN_SIZES = {"taxi": 1400, "retweet": 9000, "amazon": 6454, "stackoverflow": 1401, "taobao": 1300}

# Matched-to-EasyTPP architecture overrides (dotted train.py keys), per model.
MODEL_ARCH = {
    "intensity_free": ["arch.iftpp_encoder=gru", "arch.num_mix_components=64", "arch.n_layer=2", "arch.n_embd=64"],
    "nhp": ["arch.n_layer=2", "arch.n_embd=64"],
    "attnhp": ["arch.n_layer=2", "arch.n_embd=32", "arch.n_head=2", "arch.attnhp_time_emb_size=16"],
}

# Per-model eval caps (test-subset size, grid points/unit-time, eval batch).
# AttNHP is far slower per intensity call, so it gets tighter caps.
EVAL_DEFAULTS = {
    "intensity_free": dict(limit=None, grid=200.0, ebs=64),
    "nhp": dict(limit=None, grid=200.0, ebs=64),
    "attnhp": dict(limit=2000, grid=100.0, ebs=32),
}

DISPLAY_TO_KEY = {
    "Marginal time log-likelihood": "marginal_time_ll",
    "Conditional type log-likelihood p(m|t,H)": "cond_type_ll",
    "Marginal type log-likelihood p(m|H)": "marginal_type_ll",
    "Joint log-likelihood": "joint_ll",
    "Top-1 Accuracy (marginal)": "top1_accuracy",
    "Top-1 Accuracy (conditional)": "cond_top1_accuracy",
    "Top-5 Accuracy (conditional)": "cond_top5_accuracy",
    "Time MAE mean (days)": "time_mae_mean_days",
    "Time MAE median (days)": "time_mae_median_days",
    "Time RMSE mean (days)": "time_rmse_mean_days",
    "Time RMSE median (days)": "time_rmse_median_days",
    "Total predictions": "num_predictions",
}
METRIC_LINE = re.compile(r"^\s*(?P<key>[A-Za-z0-9_\-\(\)\| ,]+):\s+(?P<val>-?[0-9eE.+\-]+)\s*$")


def _parse_test_metrics(stdout: str) -> dict:
    metrics, in_block = {}, False
    for line in stdout.splitlines():
        if "Test Evaluation Results:" in line:
            in_block = True
            continue
        if not in_block:
            continue
        m = METRIC_LINE.match(line)
        if not m:
            continue
        display = m.group("key").strip()
        if display in DISPLAY_TO_KEY:
            try:
                metrics[DISPLAY_TO_KEY[display]] = float(m.group("val"))
            except ValueError:
                pass
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--model", required=True, choices=list(MODEL_ARCH))
    p.add_argument("--dataset", required=True, choices=list(TRAIN_SIZES))
    p.add_argument("--seed", type=int, default=2019)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-steps", type=int, default=None, help="Override; else epochs*ceil(n_train/batch).")
    p.add_argument("--limit-test-size", type=int, default=None, help="Override per-model eval cap (0 = full).")
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    n_train = TRAIN_SIZES[args.dataset]
    steps_per_epoch = math.ceil(n_train / args.batch)
    num_steps = args.num_steps or args.epochs * steps_per_epoch
    # Match EasyTPP's per-epoch validation + best-checkpoint selection (valid_freq=1).
    eval_freq = steps_per_epoch
    ev = EVAL_DEFAULTS[args.model]
    limit = args.limit_test_size if args.limit_test_size is not None else ev["limit"]
    args.log_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{args.model}_{args.dataset}_seed{args.seed}"

    cmd = [
        str(args.python), "train.py",
        "-c", "easytpp/default", "-c", "training/transformer", "-c", "wandb/disabled",
        "-c", f"data/easytpp_{args.dataset}", "-c", f"model/{args.model}",
        f"training.num_steps={num_steps}",
        f"training.eval_freq={eval_freq}",
        "training.scheduler_type=none",
        "training.scheduler_warmup_steps=0",
        f"training.learning_rate={args.lr}",
        "training.weight_decay=0.0",
        "training.early_stopping_patience=0",
        "training.grad_clip=0.0",  # EasyTPP uses plain Adam with no gradient clipping
        f"batching.batch_size={args.batch}",
        "arch.dropout=0.0",
        f"arch.seed={args.seed}",
        # Match EasyTPP loss_integral_num_sample_per_step=20 for the numerical-CI
        # models (per-interval survival quadrature).
        *(["arch.num_ci_integration_steps=20"] if args.model in ("nhp", "attnhp") else []),
        *MODEL_ARCH[args.model],
        f"eval.grid_points_per_unit_time={ev['grid']}",
        f"eval.eval_batch_size={ev['ebs']}",
        f"eval.limit_test_size={'null' if not limit else limit}",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[{time.strftime('%H:%M:%S')}] >>> {run_tag} steps={num_steps} limit={limit} (gpu={args.gpu})", flush=True)
    t0 = time.time()
    out_log = args.log_dir / f"{run_tag}.out"
    err_log = args.log_dir / f"{run_tag}.err"
    with out_log.open("w") as fo, err_log.open("w") as fe:
        proc = subprocess.run(cmd, cwd=str(args.repo), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        fo.write(proc.stdout)
        fe.write(proc.stderr)
    elapsed = time.time() - t0
    metrics = _parse_test_metrics(proc.stdout)
    status = "SUCCESS" if proc.returncode == 0 and metrics else f"FAILED(rc={proc.returncode})"

    row = {
        "framework": "torchpp", "run_tag": run_tag, "model": args.model, "dataset": args.dataset,
        "seed": args.seed, "num_steps": num_steps, "limit_test_size": limit or 0,
        "status": status, "elapsed_seconds": f"{elapsed:.1f}",
        **{k: metrics.get(k) for k in DISPLAY_TO_KEY.values()},
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output_csv.exists() or args.output_csv.stat().st_size == 0
    with args.output_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[{time.strftime('%H:%M:%S')}] <<< {run_tag} status={status} elapsed={elapsed:.1f}s "
          f"joint_ll={metrics.get('joint_ll')} cond_acc={metrics.get('cond_top1_accuracy')} "
          f"rmse_med={metrics.get('time_rmse_median_days')}", flush=True)
    sys.exit(0 if status == "SUCCESS" else 1)


if __name__ == "__main__":
    main()
