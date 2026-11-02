"""Fit EasyTPP's multivariate Hawkes (MHP) baseline with `tick` and report next-type accuracy Acc|t.

Recipe (EasyTPP, Appendix B): exponential kernel with fixed decay 1.0, base intensity 0.2.
Acc|t = argmax_d lambda_d(t_i) at the true next event time, with causal history (t_j < t_i);
the first event of each sequence is conditioned on.

Fits:
  tick LS        tick HawkesExpKern(decays=1.0), default least-squares; learns mu and A.
  tick ML        the same with gofit="likelihood" (fails in tick 0.8.0.2; reported as nan).
  MLE            positivity-constrained exponential-Hawkes MLE (scipy L-BFGS-B), learns mu and A.
  tick LS, mu=0.2  tick LS adjacency evaluated with the base intensity set to 0.2.
  MLE, mu=0.2    base intensity fixed at 0.2, learns A (the literal recipe).

Intensity (tick convention): lambda_k(t) = mu_k + sum_{t_m<t} A[k, d_m] * beta * exp(-beta (t - t_m)).
Times are EasyTPP's native `time_since_start` with each sequence shifted to start at 0.

Data: taxi / stackoverflow / retweet / amazon from the EasyTPP Google Drive pickles (--pkl-root, containing
taxi/, so_v2/, retweet/, amazon/); taobao (17 types) from the torchpp EasyTPP data root (--pt-root, as
written by scripts/data_prep/download_easytpp_datasets.py).

    python paper/tick/tick_hawkes_repro.py --pkl-root <gdrive> --pt-root <easytpp> [--out tick_repro.csv] [datasets ...]
"""
import argparse
import csv
import pickle
import time
import warnings

import numpy as np
from scipy.optimize import minimize
from tick.hawkes import HawkesExpKern

warnings.filterwarnings("ignore")

BETA = 1.0

# dataset -> (source, sub-directory, number of event types)
DATASETS = {
    "taxi": ("pkl", "taxi", 10),
    "stackoverflow": ("pkl", "so_v2", 22),
    "retweet": ("pkl", "retweet", 3),
    "amazon": ("pkl", "amazon", 16),
    "taobao": ("pt", "taobao", 17),
}

# Acc|t (%) reported for MHP by EasyTPP (Xue et al., 2024).
EASYTPP_REPORTED = {"taxi": 90.47, "stackoverflow": 35.00, "retweet": 44.30, "amazon": 24.10, "taobao": 31.90}


# --------------------------------------------------------------------------- #
# Data loading (native units, per-sequence offset removed)
# --------------------------------------------------------------------------- #
def load_pkl(path, split):
    obj = pickle.load(open(f"{path}/{split}.pkl", "rb"))
    key = [k for k in obj if k != "dim_process"][0]
    out = []
    for s in obj[key]:
        t = np.array([float(e["time_since_start"]) for e in s], dtype=float)
        y = np.array([int(e["type_event"]) for e in s], dtype=int)
        t = t - t[0]                      # remove external-reference offset
        out.append((t, y))
    return out

def load_pt(path, split):
    from torchpp.data.easytpp_data import load_easytpp_sequences
    out = []
    for s in load_easytpp_sequences(path, split):
        t = s["time_points"].numpy().astype(float)
        y = s["event_types"].numpy().astype(int)
        t = t - t[0]
        out.append((t, y))
    return out

def load(name, split, roots):
    kind, sub, _ = DATASETS[name]
    return (load_pkl if kind == "pkl" else load_pt)(f"{roots[kind]}/{sub}", split)


# --------------------------------------------------------------------------- #
# tick realizations
# --------------------------------------------------------------------------- #
def to_realizations(seqs, K):
    reals = []
    for t, y in seqs:
        per = [t[y == k] for k in range(K)]
        # tick requires strictly-increasing timestamps per component
        fixed = []
        for arr in per:
            a = np.sort(arr.astype(float))
            if a.size > 1:
                d = np.diff(a)
                bad = d <= 0
                if bad.any():                        # break ties with tiny jitter
                    eps = 1e-9
                    for i in range(1, a.size):
                        if a[i] <= a[i-1]:
                            a[i] = a[i-1] + eps
            fixed.append(a)
        reals.append(fixed)
    return reals


# --------------------------------------------------------------------------- #
# Acc|t : argmax lambda at true event times, causal history
# --------------------------------------------------------------------------- #
def acc_at_t(seqs, mu, A, beta=BETA):
    K = len(mu); correct = 0; total = 0
    for t, y in seqs:
        n = len(t)
        cur = np.zeros(K)
        for i in range(1, n):
            d = np.exp(-beta * (t[i] - t[i-1]))
            cur = d * cur
            cur[y[i-1]] += d                          # event i-1 decayed to t_i
            lam = mu + beta * (A @ cur)
            correct += int(np.argmax(lam) == y[i])
            total += 1
    return correct / total, total

def majority_acc(train, test, K):
    counts = np.zeros(K)
    for _, y in train:
        counts += np.bincount(y, minlength=K)
    maj = int(np.argmax(counts))
    correct = total = 0
    for t, y in test:
        correct += int((y[1:] == maj).sum()); total += len(y) - 1
    return correct / total


# --------------------------------------------------------------------------- #
# Custom convex fixed-decay MLE (identical intensity model to tick)
#   optimize mu (optional) + A>=0 ; used for V2 (fixed mu) and V1 cross-check
# --------------------------------------------------------------------------- #
def _precompute(train, K, beta=BETA):
    states = []; Ttot = 0.0; S = np.zeros(K)
    for t, y in train:
        n = len(t); R = np.zeros((n, K)); cur = np.zeros(K)
        for i in range(1, n):
            d = np.exp(-beta * (t[i] - t[i-1]))
            cur = d * cur; cur[y[i-1]] += d; R[i] = cur
        states.append((y, R))
        T = t[-1]; Ttot += T
        S += np.bincount(y, weights=(1 - np.exp(-beta * (T - t))), minlength=K)
    return states, Ttot, S

def fit_mle(train, K, beta=BETA, fix_mu=None):
    states, Ttot, S = _precompute(train, K, beta)
    def obj(x):
        if fix_mu is None:
            mu = x[:K]; A = x[K:].reshape(K, K)
        else:
            mu = fix_mu; A = x.reshape(K, K)
        nll = 0.0; gmu = np.zeros(K); gA = np.zeros((K, K))
        for y, R in states:
            lam = mu[y] + beta * np.einsum("ij,ij->i", R, A[y])
            lam = np.clip(lam, 1e-12, None)
            nll -= np.log(lam).sum()
            inv = 1.0 / lam
            np.add.at(gA, y, -beta * R * inv[:, None])
            if fix_mu is None:
                np.add.at(gmu, y, -inv)
        nll += mu.sum() * Ttot + (A * S[None, :]).sum()
        gA += S[None, :]
        if fix_mu is None:
            gmu += Ttot
            return nll, np.concatenate([gmu, gA.ravel()])
        return nll, gA.ravel()

    if fix_mu is None:
        x0 = np.concatenate([np.full(K, 0.2), np.full(K*K, 0.01)])
        bounds = [(1e-8, None)]*K + [(0.0, None)]*(K*K)
    else:
        x0 = np.full(K*K, 0.01)
        bounds = [(0.0, None)]*(K*K)
    res = minimize(obj, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 1000, "maxfun": 20000})
    if fix_mu is None:
        return res.x[:K], res.x[K:].reshape(K, K), res
    return fix_mu.copy(), res.x.reshape(K, K), res


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run_one(name, roots):
    K = DATASETS[name][2]
    train = load(name, "train", roots)
    test = load(name, "test", roots)
    print(f"\n===== {name}  K={K}  train_events={sum(len(y) for _, y in train)} "
          f"test_events={sum(len(y) for _, y in test)} =====", flush=True)

    maj = majority_acc(train, test, K)

    reals = to_realizations(train, K)
    m = HawkesExpKern(decays=BETA, gofit="least-squares", penalty="l2", C=1e3,
                      solver="agd", max_iter=1000, tol=1e-6, verbose=False)
    m.fit(reals)
    mu_t = np.asarray(m.baseline, float)
    A_t = np.asarray(m.adjacency, float)
    tick_ls, n = acc_at_t(test, mu_t, A_t)
    tick_ls_mu02, _ = acc_at_t(test, np.full(K, 0.2), A_t)

    try:
        m2 = HawkesExpKern(decays=BETA, gofit="likelihood", penalty="l2", C=1e3,
                           solver="agd", max_iter=1000, tol=1e-6, verbose=False)
        m2.fit(reals)
        tick_ml, _ = acc_at_t(test, np.asarray(m2.baseline, float), np.asarray(m2.adjacency, float))
    except Exception as e:  # tick 0.8.0.2: "The sum of the influence on someone cannot be negative"
        tick_ml = float("nan")
        print(f"  tick ML failed: {type(e).__name__}: {e}", flush=True)

    t0 = time.time()
    mu_c, A_c, _ = fit_mle(train, K, fix_mu=None)
    mle, _ = acc_at_t(test, mu_c, A_c)
    mu_f, A_f, _ = fit_mle(train, K, fix_mu=np.full(K, 0.2))
    mle_mu02, _ = acc_at_t(test, mu_f, A_f)
    print(f"  MLE fits: {time.time() - t0:.1f}s", flush=True)

    row = dict(dataset=name, n_types=K, n_eval=n, majority=maj * 100, tick_ls=tick_ls * 100,
               tick_ml=tick_ml * 100, mle=mle * 100, tick_ls_mu02=tick_ls_mu02 * 100,
               mle_mu02=mle_mu02 * 100, easytpp_reported=EASYTPP_REPORTED[name])
    print("  " + "  ".join(f"{k}={v:.2f}" for k, v in row.items() if isinstance(v, float)), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--pkl-root", required=True, help="EasyTPP Google Drive folder (taxi/, so_v2/, retweet/, amazon/)")
    ap.add_argument("--pt-root", required=True, help="torchpp EasyTPP data root (taobao/)")
    ap.add_argument("--out", default="tick_repro.csv")
    args = ap.parse_args()
    rows = [run_one(n, {"pkl": args.pkl_root, "pt": args.pt_root}) for n in args.datasets]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
