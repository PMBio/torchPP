"""Run EasyTPP reference experiments (NHP, AttNHP, IntensityFree) fresh on the
standard datasets and append one CSV row of test metrics per run.

This is the EasyTPP half of the equivalence check.  It is self-contained: it
builds a single-experiment EasyTPP config dict with the correct local data
paths (under $TORCHPPDATA) and the matched benchmark hyperparameters, writes a
temp YAML, runs the EasyTPP Runner, and parses test loglike/acc/rmse/mae from
the run's log.

Must be invoked with the EasyTPP venv python:
  EasyTemporalPointProcess/.venv/bin/python

taobao is omitted: TorchPP uses the 17-type Taobao release, so the EasyTPP side
needs the same pkl files to be comparable. Add them to the data root to include it.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
import time
import traceback
from pathlib import Path

import yaml

# dataset short name -> (subdir under easytpp_gdrive, num_event_types)
DATASETS = {
    "taxi": ("taxi", 10),
    "retweet": ("retweet", 3),
    "amazon": ("amazon", 16),
    "stackoverflow": ("so_v2", 22),
}

THINNING = dict(num_seq=2, num_sample=10, num_exp=50, look_ahead_time=2,
                patience_counter=5, over_sample_rate=5, num_samples_boundary=5,
                dtime_max=2, num_step_gen=1)


def model_config(model: str) -> dict:
    if model == "NHP":
        return {"hidden_size": 64, "loss_integral_num_sample_per_step": 20, "thinning": THINNING}
    if model == "AttNHP":
        return {"hidden_size": 32, "time_emb_size": 16, "num_layers": 2, "num_heads": 2,
                "loss_integral_num_sample_per_step": 20, "use_ln": False, "thinning": THINNING}
    if model == "IntensityFree":
        return {"hidden_size": 64, "num_layers": 2, "loss_integral_num_sample_per_step": 20,
                "model_specs": {"num_mix_components": 64}, "thinning": THINNING}
    raise ValueError(model)


def build_config(model: str, dataset: str, seed: int, base_dir: Path, epochs: int = 20) -> dict:
    subdir, n_types = DATASETS[dataset]
    root = os.environ["TORCHPPDATA"].rstrip("/") + "/easytpp_gdrive/" + subdir
    exp_id = f"{model}_{dataset}_train"
    return {
        "pipeline_config_id": "runner_config",
        "data": {
            dataset: {
                "data_format": "pkl",
                "train_dir": f"{root}/train.pkl",
                "valid_dir": f"{root}/dev.pkl",
                "test_dir": f"{root}/test.pkl",
                "data_specs": {"num_event_types": n_types, "pad_token_id": n_types,
                               "padding_side": "right", "truncation_side": "right"},
            }
        },
        exp_id: {
            "base_config": {"stage": "train", "backend": "torch", "dataset_id": dataset,
                            "runner_id": "std_tpp", "model_id": model, "base_dir": str(base_dir)},
            "trainer_config": {"batch_size": 64, "max_epoch": epochs, "shuffle": True, "optimizer": "adam",
                               "learning_rate": 1.0e-3, "valid_freq": 1, "use_tfb": False,
                               "metrics": ["acc", "rmse", "mae"], "seed": seed, "gpu": 0},
            "model_config": model_config(model),
        },
    }, exp_id


def parse_metrics_from_log(log_path: str) -> dict | None:
    if not log_path or not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        lines = f.readlines()
    best, last = None, None
    for line in lines:
        if "(test)" in line and "test " in line:
            last = line
        if "best updated at this epoch" in line and last is not None:
            best = last
    best = best or last
    if best is None:
        return None
    metrics = {}
    for m in re.finditer(r"(\w+) is ([0-9eE.+-]+)", best):
        k, v = m.group(1), m.group(2)
        if k != "num_events":
            try:
                metrics[k] = float(v)
            except ValueError:
                pass
    return metrics or None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["NHP", "AttNHP", "IntensityFree"])
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--seed", type=int, default=2019)
    p.add_argument("--epochs", type=int, default=20, help="max_epoch (per-epoch val + best-checkpoint selection)")
    p.add_argument("--base-dir", type=Path, required=True, help="EasyTPP checkpoints/logs base dir")
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from easy_tpp.config_factory import Config
    from easy_tpp.runner import Runner

    run_dir = args.base_dir / f"{args.model}_{args.dataset}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg, exp_id = build_config(args.model, args.dataset, args.seed, run_dir, epochs=args.epochs)
    tmp_yaml = run_dir / "config.yaml"
    tmp_yaml.write_text(yaml.safe_dump(cfg))

    run_tag = f"{args.model}_{args.dataset}_seed{args.seed}"
    print(f"[{time.strftime('%H:%M:%S')}] >>> EasyTPP {run_tag} (gpu={args.gpu})", flush=True)
    t0 = time.time()
    metrics, status, log_dir = None, "FAILED", None
    try:
        config = Config.build_from_yaml_file(str(tmp_yaml), experiment_id=exp_id)
        runner = Runner.build_from_config(config)
        log_dir = runner.runner_config.base_config.specs.get("saved_log_dir")
        runner.run()
        metrics = parse_metrics_from_log(log_dir)
        status = "SUCCESS" if metrics else "NO_METRICS"
    except Exception as e:
        traceback.print_exc()
        status = f"FAILED: {type(e).__name__}"
    elapsed = time.time() - t0

    row = {
        "framework": "easytpp", "run_tag": run_tag, "model": args.model, "dataset": args.dataset,
        "seed": args.seed, "status": status, "elapsed_seconds": f"{elapsed:.1f}",
        "loglike": (metrics or {}).get("loglike"),
        "acc": (metrics or {}).get("acc"),
        "rmse": (metrics or {}).get("rmse"),
        "mae": (metrics or {}).get("mae"),
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output_csv.exists() or args.output_csv.stat().st_size == 0
    with args.output_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[{time.strftime('%H:%M:%S')}] <<< EasyTPP {run_tag} status={status} elapsed={elapsed:.1f}s "
          f"loglike={row['loglike']} acc={row['acc']} rmse={row['rmse']}", flush=True)
    raise SystemExit(0 if status == "SUCCESS" else 1)


if __name__ == "__main__":
    main()
