"""Calibration of trained models: time-rescaling goodness of fit and mark calibration.

For each (model, dataset) cell:
  * Time rescaling: u_i = 1 - exp(-Lambda*(t_{i-1}, t_i)) from ``survival_time``; u ~ U(0,1) if calibrated.
    Reports the Kolmogorov-Smirnov distance (events with a zero inter-event time excluded).
  * Mark calibration: p(d | t_i, H) at the true event time via ``prob_mark_given_time``.
    Reports top-label and classwise ECE (15 bins).

The architecture of each cell is taken from the main-run generator (ACTION=eval), so the checkpoint
loads into an identical model. Run from the repository root:

    python paper/calibration/run_calibration.py --checkpoint-dir <dir> [--reps 0] [--models ...] [--datasets ...]
    python paper/calibration/run_calibration.py --checkpoint-dir <dir> --datasets mimic_iv_icu --max-events 5000

The paper uses --max-events 10000 on the EasyTPP datasets and 5000 on MIMIC-IV.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

import torchpp.data  # noqa: F401  (registers datasets)
from torchpp.training.config import StandaloneEvalConfig
from torchpp.training.config_utils import load_config
from torchpp.training.dataset_registry import get_data_root
from torchpp.training.model_factory import create_model
from torchpp.training.training_helpers import load_test_dataset
from torchpp.utils.event_data import MVEventData, BatchedMVEventData
from torchpp.utils.evaluation import _build_prediction_items

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "scripts" / "tuning" / "generate_main_run_commands.sh"

MODELS = ["hawkes", "spline_spline_kernel", "neural_poisson", "neural_spline"]
DATASETS = ["taxi", "stackoverflow", "retweet", "amazon", "taobao"]

N_ECE_BINS = 15
EVAL_BATCH = 32
# The K=400 models build large (B, N, D, D) / (B, D, K) tensors at D=117, so they use small batches.
SMALL_BATCH = {"spline_spline_kernel": 2, "neural_spline": 4}


# ---------------------------------------------------------------------------
# 1. Per-cell evaluate.py command from the main-run generator.
# ---------------------------------------------------------------------------
def eval_commands(models, datasets):
    """Run the generator in eval mode; return {job_name: [argv tokens]}."""
    env = dict(os.environ, ACTION="eval", MODELS=" ".join(models), DATASETS=" ".join(datasets))
    out = subprocess.run(["bash", str(GEN)], cwd=str(REPO), env=env,
                         capture_output=True, text=True, check=True).stdout
    cmds = {}
    name = None
    for line in out.splitlines():
        line = line.rstrip()
        if line.startswith("[JOB] "):
            name = line[len("[JOB] "):].strip()
        elif name and line.strip().startswith("python evaluate.py"):
            cmds[name] = line.split()
            name = None
    return cmds


def build_argv(tokens, ckpt_dir):
    """Turn an emitted `python evaluate.py ...` token list into a load_config argv,
    pointing model_path at ``ckpt_dir``."""
    argv = []
    it = iter(tokens[2:])  # drop 'python', 'evaluate.py'
    for tok in it:
        if tok == "-c":
            argv += ["-c", next(it)]
        elif tok.startswith("model_path="):
            argv.append(f"model_path={ckpt_dir / Path(tok.split('=', 1)[1]).name}")
        else:
            argv.append(tok)
    return argv


# ---------------------------------------------------------------------------
# 2. Metric helpers.
# ---------------------------------------------------------------------------
def ks_uniform(u):
    """Two-sided KS statistic of samples u against Uniform(0,1)."""
    u = np.sort(np.clip(u, 0.0, 1.0))
    n = u.size
    i = np.arange(1, n + 1)
    d_plus = np.max(i / n - u)
    d_minus = np.max(u - (i - 1) / n)
    return float(max(d_plus, d_minus))


def top_label_ece(conf, correct, nbins=N_ECE_BINS):
    edges = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, nbins - 1)
    N = conf.size
    ece = 0.0
    bin_conf, bin_acc, bin_cnt = [], [], []
    for b in range(nbins):
        m = idx == b
        c = int(m.sum())
        if c == 0:
            bin_conf.append(np.nan); bin_acc.append(np.nan); bin_cnt.append(0)
            continue
        cf = float(conf[m].mean()); ac = float(correct[m].mean())
        ece += (c / N) * abs(ac - cf)
        bin_conf.append(cf); bin_acc.append(ac); bin_cnt.append(c)
    return float(ece), np.array(bin_conf), np.array(bin_acc), np.array(bin_cnt)


class ClasswiseAccumulator:
    """Pools (p_k, 1[true==k]) over all events and classes into equal-width bins."""

    def __init__(self, nbins=N_ECE_BINS):
        self.nbins = nbins
        self.edges = np.linspace(0.0, 1.0, nbins + 1)
        self.sum_p = np.zeros(nbins)
        self.sum_y = np.zeros(nbins)
        self.cnt = np.zeros(nbins, dtype=np.int64)

    def update(self, probs, onehot):
        p = probs.reshape(-1)
        y = onehot.reshape(-1)
        idx = np.clip(np.digitize(p, self.edges) - 1, 0, self.nbins - 1)
        np.add.at(self.sum_p, idx, p)
        np.add.at(self.sum_y, idx, y)
        np.add.at(self.cnt, idx, 1)

    def result(self):
        N = self.cnt.sum()
        conf = np.divide(self.sum_p, self.cnt, out=np.full(self.nbins, np.nan), where=self.cnt > 0)
        acc = np.divide(self.sum_y, self.cnt, out=np.full(self.nbins, np.nan), where=self.cnt > 0)
        ece = 0.0
        for b in range(self.nbins):
            if self.cnt[b] > 0:
                ece += (self.cnt[b] / N) * abs(acc[b] - conf[b])
        return float(ece), conf, acc, self.cnt.copy()


# ---------------------------------------------------------------------------
# 3. Per-cell evaluation.
# ---------------------------------------------------------------------------
def load_cell(argv):
    """Load config + dataset + model for one cell. Returns (config, dataset, model, load_report)."""
    config = load_config(StandaloneEvalConfig, argv)
    data_root = get_data_root()
    ds = load_test_dataset(
        config.dataset_type, dict(config.dataset), data_root,
        limit_test_size=config.eval.limit_test_size,
    )
    D = ds.num_event_types
    # Build the model WITHOUT loading, then load strict=False to capture the key
    # report explicitly (shape mismatch still raises even with strict=False).
    model, _ = create_model(
        model_type=config.model_type, D=D, arch=config.arch,
        terminating=ds.terminating, load_path=None,
    )
    sd = torch.load(str(config.model_path), map_location="cpu", weights_only=True)
    incompat = model.load_state_dict(sd, strict=False)
    load_report = {
        "missing_keys": list(incompat.missing_keys),
        "unexpected_keys": list(incompat.unexpected_keys),
        "ci_mode": model.ci_config.mode,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    model.eval()
    return config, ds, model, D, load_report


def trust_chain_check(model, ds, D):
    """On one test sequence, verify u=cdf_time equals 1-exp(-Lambda) with Lambda from
    cumulative_intensity over [t_{i-1}, t_i] (summed over types). Returns max abs diff."""
    # first sequence with >= 3 events
    seq = None
    for s in ds:
        if len(s) >= 3:
            seq = s
            break
    if seq is None:
        return None
    max_diff_u = 0.0
    max_diff_lam = 0.0
    n = min(len(seq), 12)
    for idx in range(1, n):
        hist = MVEventData(seq.time_points[:idx], seq.event_types[:idx],
                           covariates=getattr(seq, "covariates", None))
        batch = BatchedMVEventData(mv_events=[hist])
        t_target = torch.tensor([float(seq.time_points[idx])], dtype=torch.float64)
        t_prev = torch.tensor([float(seq.time_points[idx - 1])], dtype=torch.float64)
        prepare = getattr(model, "prepare_batch", None)
        clear = getattr(model, "clear_batch", None)
        with torch.no_grad():
            if prepare is not None:
                prepare(batch)
            try:
                u_cdf = float(model.cdf_time(t_target, batch, log=False)[0])
                logS = float(model.survival_time(t_target, batch, log=True)[0])
                # independent Lambda over [t_prev, t_target], summed over types
                lam_ci = float(model.cumulative_intensity(t_target, batch, T_low=t_prev).sum(dim=1)[0])
            finally:
                if clear is not None:
                    clear()
        u_recon = 1.0 - np.exp(-lam_ci)
        lam_from_S = -logS
        max_diff_u = max(max_diff_u, abs(u_cdf - u_recon))
        max_diff_lam = max(max_diff_lam, abs(lam_ci - lam_from_S))
    return {"max_abs_diff_u": max_diff_u, "max_abs_diff_lambda": max_diff_lam, "n_checked": n - 1}


def eval_cell(model, ds, D, ignore_indices, time_scale_factor, batch_size=EVAL_BATCH, max_events=None):
    items = _build_prediction_items(list(ds), ignore_indices)  # idx>=1: excludes first + censored intervals
    n_total = len(items)
    if max_events is not None and len(items) > max_events:
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(len(items), size=max_events, replace=False))
        items = [items[i] for i in sel]
    n_capped = len(items)
    keep = None
    if ignore_indices:
        keep = torch.ones(D, dtype=torch.bool)
        keep[list(ignore_indices)] = False

    # Reported LL convention: p_days = p_scaled / time_scale_factor, i.e.
    # log p_days = log p_scaled - log(time_scale_factor) (evaluation.py:526-533).
    log_time_scale = float(np.log(time_scale_factor))

    u_all, lam_all, tie_all = [], [], []
    conf_all, correct_all = [], []
    joint_ll_all, time_ll_all, cond_ll_all = [], [], []
    internal_diffs = []
    cw = ClasswiseAccumulator()

    prepare = getattr(model, "prepare_batch", None)
    clear = getattr(model, "clear_batch", None)

    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        histories = [it["history"] for it in chunk]
        batch = BatchedMVEventData(mv_events=histories)
        t_target = torch.tensor([float(it["target_time"]) for it in chunk], dtype=torch.float64)
        tgt_type = torch.tensor([int(it["target_type"]) for it in chunk], dtype=torch.long)
        # inter-event time τ = t_target − (history's last event time); τ ≤ 0 == exact tie
        # (Δt=0 simultaneity). Ties pin u=F*(0)=0, so they are dropped from the PIT below.
        last_time = torch.tensor([float(it["history"].time_points[-1]) for it in chunk], dtype=torch.float64)
        is_tie = (t_target - last_time) <= 0.0

        with torch.no_grad():
            if prepare is not None:
                prepare(batch)
            try:
                logS = model.survival_time(t_target, batch, log=True)          # (B,)
                pmark = model.prob_mark_given_time(t_target, batch, log=False)  # (B, D)
                # One extra contract call: the joint gives the full LL triple —
                # joint at true type directly, the time-marginal via the mark-axis
                # logsumexp identity (== density_time, base_tpp.py:312), cond by subtraction.
                log_joint = model.density_joint(t_target, batch, log=True)      # (B, D)
            finally:
                if clear is not None:
                    clear()

        logS = logS.double()
        lam = (-logS).cpu().numpy()
        u = (-torch.expm1(logS)).cpu().numpy()
        u_all.append(u.astype(np.float64))
        lam_all.append(lam.astype(np.float64))
        tie_all.append(is_tie.cpu().numpy())

        # --- density-contract LLs on ALL items (ties included, as in the reported LL) ---
        log_joint = log_joint.double()
        bidx = torch.arange(len(chunk))
        log_time = torch.logsumexp(log_joint, dim=-1)  # (B,) = density_time(log=True)
        jll = log_joint[bidx, tgt_type]                # (B,) log p(t, d_true | H), scaled time
        tll = log_time                                 # (B,) log p(t | H), scaled time
        cll = jll - tll                                # (B,) log p(d_true | t, H), unit-free
        joint_ll_all.append(jll.cpu().numpy())
        time_ll_all.append(tll.cpu().numpy())
        cond_ll_all.append(cll.cpu().numpy())

        # Internal check: density_joint[true] == density_time + log p(d_true | t, H),
        # using the independently-called prob_mark_given_time (catches indexing/masking bugs).
        p_raw = pmark.double()
        tiny = torch.finfo(p_raw.dtype).tiny
        log_pm_true = torch.log(p_raw[bidx, tgt_type].clamp(min=tiny))
        internal_diffs.append((jll - (tll + log_pm_true)).abs().cpu().numpy())

        # --- mark calibration on ALL items (unchanged) ---
        if keep is not None:
            pmark = pmark.masked_fill(~keep.unsqueeze(0), 0.0)
            pmark = pmark / pmark.sum(dim=1, keepdim=True).clamp_min(1e-300)
        p = pmark.cpu().numpy()
        conf = p.max(axis=1)
        pred = p.argmax(axis=1)
        tt = tgt_type.numpy()
        conf_all.append(conf)
        correct_all.append((pred == tt).astype(np.float64))

        onehot = np.zeros_like(p)
        onehot[np.arange(len(tt)), tt] = 1.0
        cw.update(p, onehot)

    u_all = np.concatenate(u_all)
    lam_all = np.concatenate(lam_all)
    tie_all = np.concatenate(tie_all)
    conf_all = np.concatenate(conf_all)
    correct_all = np.concatenate(correct_all)
    joint_ll_all = np.concatenate(joint_ll_all)
    time_ll_all = np.concatenate(time_ll_all)
    cond_ll_all = np.concatenate(cond_ll_all)
    internal_diffs = np.concatenate(internal_diffs)

    n_ties = int(tie_all.sum())
    tie_fraction = (n_ties / n_capped) if n_capped else 0.0

    # tie-free PIT / KS / moments (ties give u=0, pinning the empirical CDF at 0)
    tf = ~tie_all
    u_tf = u_all[tf]
    lam_tf = lam_all[tf]

    ks = ks_uniform(u_tf)
    ks_with_ties = ks_uniform(u_all)  # pre-exclusion (preserved for comparison)
    top_ece, tconf, tacc, tcnt = top_label_ece(conf_all, correct_all)
    cw_ece, cwconf, cwacc, cwcnt = cw.result()

    # QQ arrays (tie-free): 200 probability grid points.
    pg = (np.arange(1, 201) - 0.5) / 200.0
    emp_q = np.quantile(u_tf, pg)

    joint_ll = float(joint_ll_all.mean()) - log_time_scale
    marginal_time_ll = float(time_ll_all.mean()) - log_time_scale
    cond_type_ll = float(cond_ll_all.mean())  # unit-free (the Jacobian cancels)

    return {
        "n_events": int(u_tf.size),                # tie-free items used for the PIT
        "n_events_with_ties": int(n_capped),       # capped item set (tie_fraction denom / LL n)
        "n_total_events": int(n_total),
        "n_ties": n_ties,
        "tie_fraction": float(tie_fraction),
        "ks_D": ks,                                # tie-free (primary)
        "ks_D_with_ties": ks_with_ties,            # pre-exclusion (preserved)
        "mean_lambda": float(lam_tf.mean()),
        "var_lambda": float(lam_tf.var()),
        "mean_lambda_with_ties": float(lam_all.mean()),
        "var_lambda_with_ties": float(lam_all.var()),
        "top_label_ece": top_ece,
        "classwise_ece": cw_ece,
        "acc_at_t": float(correct_all.mean()),
        # LLs for comparison with the reported test metrics (all items, days units)
        "joint_ll": joint_ll,
        "marginal_time_ll": marginal_time_ll,
        "cond_type_ll": cond_type_ll,
        "time_scale_factor": float(time_scale_factor),
        "internal_max_abs_diff": float(internal_diffs.max()),
    }, {
        "u": u_tf.astype(np.float32),
        "lambda": lam_tf.astype(np.float32),
        "u_with_ties": u_all.astype(np.float32),
        "lambda_with_ties": lam_all.astype(np.float32),
        "tie_mask": tie_all.astype(bool),
        "qq_prob_grid": pg,
        "qq_empirical": emp_q,
        "top_bin_conf": tconf, "top_bin_acc": tacc, "top_bin_cnt": tcnt,
        "cw_bin_conf": cwconf, "cw_bin_acc": cwacc, "cw_bin_cnt": cwcnt,
    }


# ---------------------------------------------------------------------------
def cell_name(model_type, ds):
    return f"{model_type}_{ds}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint-dir", type=Path, required=True, help="directory with the trained .pth files")
    ap.add_argument("--out", type=Path, default=Path("calibration_out"), help="output directory")
    ap.add_argument("--reps", type=int, nargs="+", default=[0])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--max-events", type=int, default=10_000, help="random subsample of prediction items per cell")
    args = ap.parse_args()

    arr_dir = args.out / "arrays"
    arr_dir.mkdir(parents=True, exist_ok=True)
    cmds = eval_commands(args.models, args.datasets)

    results = []          # per (model, ds, rep)
    trust = []            # trust-chain per (model, ds, rep0)
    failures = []

    for model_type in args.models:
        for ds_name in args.datasets:
            for rep in args.reps:
                # find the matching job (job name embeds rep and optional _K<k>)
                cand = [n for n in cmds
                        if n.startswith(f"eval_{model_type}_{ds_name}")
                        and n.endswith(f"_rep{rep}")]
                if not cand:
                    failures.append((model_type, ds_name, rep, "no matching generator job"))
                    continue
                job = cand[0]
                argv = build_argv(cmds[job], args.checkpoint_dir)
                ckpt = [a for a in argv if a.startswith("model_path=")][0].split("=", 1)[1]
                tag = f"{cell_name(model_type, ds_name)}_rep{rep}"
                print(f"\n=== {tag}  (job {job}) ===", flush=True)
                if not Path(ckpt).exists():
                    failures.append((model_type, ds_name, rep, f"checkpoint missing: {ckpt}"))
                    print(f"  SKIP: missing {ckpt}")
                    continue
                try:
                    config, dsobj, model, D, lr = load_cell(argv)
                except Exception as e:  # noqa: BLE001
                    failures.append((model_type, ds_name, rep, f"load error: {type(e).__name__}: {e}"))
                    print(f"  LOAD FAILED: {e}")
                    continue
                if lr["missing_keys"] or lr["unexpected_keys"]:
                    failures.append((model_type, ds_name, rep,
                                     f"key mismatch missing={lr['missing_keys'][:3]} unexpected={lr['unexpected_keys'][:3]}"))
                print(f"  D={D} terminating={dsobj.terminating} ci_mode={lr['ci_mode']} "
                      f"params={lr['n_params']:,} clean_load={not (lr['missing_keys'] or lr['unexpected_keys'])}")

                ig = list(dsobj.ignore_indices) if dsobj.ignore_indices else None
                tc = trust_chain_check(model, dsobj, D)
                if tc:
                    print(f"  trust-chain: max|u_cdf - (1-e^-Lambda)|={tc['max_abs_diff_u']:.2e} "
                          f"max|Lambda_ci - (-logS)|={tc['max_abs_diff_lambda']:.2e}")
                    trust.append(dict(model=model_type, dataset=ds_name, rep=rep, **tc))

                cell_bs = SMALL_BATCH.get(model_type, EVAL_BATCH)
                summary, arrays = eval_cell(model, dsobj, D, ig, dsobj.time_scale_factor, batch_size=cell_bs,
                                            max_events=args.max_events)
                summary.update(model=model_type, dataset=ds_name, rep=rep, D=D,
                               ci_mode=lr["ci_mode"], clean_load=not (lr["missing_keys"] or lr["unexpected_keys"]),
                               spline_kernel_K=getattr(config.arch, "spline_kernel_K", None))
                results.append(summary)
                np.savez_compressed(arr_dir / f"{tag}.npz", **arrays)
                print(f"  n_events={summary['n_events']} (tie_frac={summary['tie_fraction']:.3f})  "
                      f"KS_D={summary['ks_D']:.4f} (w/ties {summary['ks_D_with_ties']:.4f})  "
                      f"meanL={summary['mean_lambda']:.3f} varL={summary['var_lambda']:.3f}  "
                      f"topECE={summary['top_label_ece']:.4f} cwECE={summary['classwise_ece']:.4f} "
                      f"Acc|t={summary['acc_at_t']:.4f}")
                print(f"    LL(days): joint={summary['joint_ll']:.4f} time={summary['marginal_time_ll']:.4f} "
                      f"cond={summary['cond_type_ll']:.4f}  tsf={summary['time_scale_factor']:.4g}  "
                      f"internal_max|Δ|={summary['internal_max_abs_diff']:.2e}")

    with open(args.out / "calibration_raw.json", "w") as f:
        json.dump({"results": results, "trust_chain": trust, "failures": failures}, f, indent=2)
    print(f"\nWrote {args.out / 'calibration_raw.json'}  ({len(results)} cells, {len(failures)} failures)")


if __name__ == "__main__":
    main()
