#!/usr/bin/env python
"""Emit the binned-likelihood jobs of the paper as [JOB] blocks (for scripts/tuning/tune_local_generic.sh).

Main runs: the architecture of every checkpoint is taken from the main-run generator (ACTION=eval); knot sweep:
from the knot-sweep generator. Each job runs paper/binned/binned_time_ll.py on the first 1,000 test sequences
(MIMIC-IV truncated to 316 events), with the bins of the dataset (Table "time resolution" of the paper).
With --mimic-dumps, also the per-item evaluation dumps of the MIMIC-IV main runs used by mimic_tie_split.py.

    python paper/binned/make_binned_commands.py --out-dir binned_out --mimic-dumps     # print the jobs
    bash scripts/tuning/tune_local_generic.sh paper/binned/generate_binned_commands.sh --out-dir binned_out --mimic-dumps

Checkpoints: main runs under $TORCHPPMODELS/main_run/ (generate_main_run_commands.sh), knot-sweep runs under
--sweep-dir (default $TORCHPPMODELS/knot_sweep/, i.e. run the sweep with MODEL_SUBDIR=knot_sweep).
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "scripts" / "tuning"
DAY_S, MIN = 86400.0, 1.0 / 1440

# Bin widths in the native time unit; suffix c = centered bins, p = bins anchored at the previous event.
DELTAS = {
    "taxi": "0.000277777777777778,0.00277777777777778,0.0277777777777778,0.000277777777777778c,0.00277777777777778c",
    "stackoverflow": "0.0001220703125,0.001220703125,0.01220703125,0.0001220703125c,0.001220703125c",
    "retweet": "0.1,1,10,100,1c,10c",
    "amazon": "0.0001,0.001,0.01,0.1",
    "taobao": "0.0001p,0.0001pc,0.001p,0.001pc,0.01p,0.01pc",
    "mimic_iv": f"{10 / DAY_S!r},{MIN!r},{5 * MIN!r},{60 * MIN!r},{MIN!r}c",
}
# MIMIC-IV: same-recorded-minute subsets, bin grid anchored at the recorded minute of the first event.
MIMIC_EXTRA = f"--tie-bin {MIN!r} --anchor-mimic-root mimic_iv_f64"
EVAL = {"easytpp": "eval.compute_marginal_type=false eval.limit_test_size=1000 eval.eval_batch_size=64",
        "mimic_iv": "eval.compute_marginal_type=false eval.limit_test_size=1000 eval.eval_batch_size=16 "
                    "eval.max_sequence_length=316"}
STD = ["taxi", "stackoverflow", "retweet", "amazon", "taobao"]
GEN_DS = {"mimic_iv": "mimic_iv_icu"}                        # main-run generator label
SWEEP_DS = {"mimic_iv": "mimic_iv_icu_interventions_top116"}  # knot-sweep generator label
# evaluate.py accepts no training.*/batching.*/save.*/wandb.* keys
DROP = re.compile(r"^(training\.|batching\.|save\.|wandb\.|eval\.|model_path=)")


def jobs(script, env=None, args=()):
    out = subprocess.run(["bash", str(GEN / script), *args], cwd=REPO, env=dict(os.environ, **(env or {})),
                         check=True, capture_output=True, text=True).stdout
    return [(n, c.split()) for n, c in re.findall(r"\[JOB\] (\S+)\n\s+(.*?)\n\n", out, re.S)]


def eval_args(tokens):
    """data/model configs and arch/dataset overrides of a train.py or evaluate.py command."""
    out, it = [], iter(tokens[2:])
    for t in it:
        if t == "-c":
            c = next(it)
            if c.startswith(("data/", "model/")):
                out += ["-c", c]
        elif not DROP.match(t):
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True, help="per-item npz output directory")
    ap.add_argument("--datasets", default=",".join(STD + ["mimic_iv"]))
    ap.add_argument("--main-dir", type=Path, default=Path(os.environ.get("TORCHPPMODELS", "models")) / "main_run")
    ap.add_argument("--sweep-dir", type=Path, default=Path(os.environ.get("TORCHPPMODELS", "models")) / "knot_sweep")
    ap.add_argument("--no-main", action="store_true")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--mimic-dumps", action="store_true", help="also emit the per-item eval dumps (MIMIC-IV main runs)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    n = 0
    for ds in a.datasets.split(","):
        kind = "mimic_iv" if ds == "mimic_iv" else "easytpp"
        extra = MIMIC_EXTRA if ds == "mimic_iv" else ""
        ev = f"{EVAL[kind]} device={a.device}"
        todo = []
        if not a.no_main:
            for name, tok in jobs("generate_main_run_commands.sh", {"ACTION": "eval", "DATASETS": GEN_DS.get(ds, ds)}):
                job = name.removeprefix("eval_")
                if job.startswith("neural_log_linear"):
                    continue  # not reported in the paper
                model = next(t for t in tok if t.startswith("model/"))[len("model/"):]
                seed = int(job.rsplit("_rep", 1)[1])
                todo.append((f"binned_main_{job}", "main", model, "", seed, eval_args(tok), a.main_dir / f"{job}.pth"))
                if a.mimic_dumps and ds == "mimic_iv":
                    print(f"[JOB] dump_main_{job}\n  TORCHPP_EVAL_DUMP_ITEMS={a.out_dir / 'dump' / (job + '.npz')} "
                          f"python evaluate.py -c default_eval {' '.join(eval_args(tok))} "
                          f"model_path={a.main_dir / (job + '.pth')} {ev}\n")
                    n += 1
        if not a.no_sweep:
            args = [f"--datasets={SWEEP_DS.get(ds, ds)}"]
            env = {"PROFILE": "mimic"} if ds == "mimic_iv" else {}
            for job, tok in jobs("generate_ksweep_commands.sh", env, args):
                model = next(t for t in tok if t.startswith("model/"))[len("model/"):]
                if model == "spline_poisson" and ds != "mimic_iv":
                    continue
                K = 100 if model == "spline_poisson" else int(re.search(r"_K(\d+)_rep", job).group(1))
                seed = int(job.rsplit("_rep", 1)[1]) - 1
                todo.append((f"binned_sweep_{job}", "knot_sweep", model, K, seed, eval_args(tok),
                             a.sweep_dir / f"{job}.pth"))
        for name, run, model, K, seed, ea, ckpt in todo:
            meta = f"run={run},dataset={ds},model={model},K={K},seed={seed}"
            print(f"[JOB] {name}\n  python paper/binned/binned_time_ll.py --deltas {DELTAS[ds]} {extra} "
                  f"--out {a.out_dir / (name + '.npz')} --meta {meta} -- -c default_eval {' '.join(ea)} "
                  f"model_path={ckpt} {ev}\n")
            n += 1
    print(f"binned: emitted {n} jobs", file=sys.stderr)


if __name__ == "__main__":
    main()
