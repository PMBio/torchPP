"""Binned (discretized) time log-likelihood of a trained model, alongside the continuous log p(t).

A continuous log-density can grow without bound by sharpening a spike on (near-)deterministic gaps, e.g. on
data recorded on a time grid. The binned likelihood

    log P(t_i in [a, b) | H) = log( exp(-Λ*(t_{i-1}, a)) − exp(-Λ*(t_{i-1}, b)) )
                             = −Λ*(t_{i-1}, a) + log(1 − exp(−Λ*(a, b)))          (≤ 0)

is bounded and insensitive to how sharp a spike inside one bin is. Bins live on the native time axis: bin
width Δ in the dataset's native units, edges at anchor + kΔ, a = max(bin start, t_{i-1}), b = bin end.

Bin anchors and Δ tokens (``--deltas``, comma-separated):
  * ``<Δ>``    anchor 0 for EasyTPP data (raw absolute timestamps) and the first event of the stay for MIMIC-IV
               (times are relative to the ICU intime, so the first event carries the minute-grid phase);
               with ``--anchor-mimic-root`` the first recorded minute is read from that (non-dequantized) root.
  * ``<Δ>p``   bins anchored at the previous event, [t_{i-1} + kΔ, t_{i-1} + (k+1)Δ), for data whose gaps lie
               on a grid while the absolute times do not (Taobao).
  * ``<Δ>c``   grid shifted by −Δ/2 ("centered" or rounding bins), so that a spike placed exactly at an
               observed grid point is not split between two bins.
Same target set and convention as the continuous log p(t) of the paper (every event after the first is a
target, no final survival term; same limit/truncation flags). Native timestamps are read in float64 from the
raw split files and checked against the model-facing sequences. Λ is the model's own: closed form where the
model has one (Λ(0, T) differences), else the evaluation quadrature (trapezoidal, the model's n_points, global
placement); density models use their log-survival.

Subsets: "tie" targets have a native gap <= ``--tie-eps`` (default: exact 0 on EasyTPP, 1e-5 days on
MIMIC-IV); ``--tie-bin <token>`` adds "rectie"/"nonrectie": the previous event lies in the same bin of that Δ
(MIMIC-IV: the same recorded minute).

Usage (the evaluate.py configuration follows ``--``):

    python paper/binned/binned_time_ll.py --deltas 1,10,1c --out res.npz --meta run=main,seed=0 -- \
        -c default_eval -c data/easytpp_retweet -c model/neural_spline ... model_path=...

Prints ``BINNED_METRICS_JSON {...}`` (aggregates; collected by paper/binned/collect_binned.py) and writes
per-item arrays to ``--out``. paper/binned/make_binned_commands.py emits the commands of the paper.
"""

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

import torchpp.data  # noqa: E402,F401
from torchpp.data.easytpp_data import easytpp_time_constants  # noqa: E402
from torchpp.tpp.base_tpp import DensityBasedTPP  # noqa: E402
from torchpp.training.config import StandaloneEvalConfig  # noqa: E402
from torchpp.training.config_utils import load_config  # noqa: E402
from torchpp.training.dataset_registry import get_data_root  # noqa: E402
from torchpp.training.model_factory import create_model  # noqa: E402
from torchpp.training.training_helpers import load_test_dataset, truncate_sequences_for_eval  # noqa: E402
from torchpp.utils.evaluation import _build_prediction_items, evaluate_at_time  # noqa: E402
from torchpp.utils.event_data import BatchedMVEventData  # noqa: E402

BIN_TOL = 1e-6  # in bins: absorbs float error of t/Δ at exact grid points


def log1mexp(x):
    """log(1 − exp(−x)) for x > 0, stable."""
    return torch.where(x < math.log(2.0), torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))


def native_times(config, test_dataset, eval_sequences, args):
    """Per evaluated sequence: float64 native times, the scaled→native map, and the bin anchor."""
    root = get_data_root()
    n = len(test_dataset)
    tsf = float(test_dataset.time_scale_factor)
    if config.dataset_type == "easytpp":
        name = config.dataset["dataset_name"]
        raw = torch.load(root / config.dataset.get("easytpp_root", "easytpp") / name / "test.pt", weights_only=False)
        nat = [s.time_points.double() for s in raw[:n]]
        miet = easytpp_time_constants(name, config.dataset.get("easytpp_root", "easytpp"))[0]
        offs = [float(t[0]) - miet for t in nat]  # native = scaled * tsf + off
        anchors = [0.0] * n  # raw EasyTPP timestamps are absolute
    elif config.dataset_type == "mimic_iv":
        name = config.dataset["dataset_name"]
        # Native root: default = the evaluated root itself (stored in float64).
        nat_root = args.native_mimic_root or config.dataset.get("mimic_root") or "mimic_iv_f64"
        raw = torch.load(root / nat_root / name / "test.pt", weights_only=False)
        nat = [s.time_points.double() * tsf for s in raw[:n]]  # days
        offs = [0.0] * n
        if args.anchor_mimic_root:
            # Dequantized roots: the first event is no longer on the minute grid. Take the grid
            # phase from the source root (same stays, same order, same scale), whose first stored
            # event is the first recorded minute itself.
            araw = torch.load(root / args.anchor_mimic_root / name / "test.pt", weights_only=False)
            assert len(araw) >= n
            anchors = [float(s.time_points[0].double()) * tsf for s in araw[:n]]
            off = np.array([float(t[0]) - a for t, a in zip(nat, anchors)])
            assert (off >= -1e-9).all() and (off < 1.0 / 1440 + 1e-9).all(), (
                f"anchor root {args.anchor_mimic_root} not aligned with {nat_root}: first-event offset "
                f"range [{off.min():.3g}, {off.max():.3g}] d (expected [0, 1 min))")
            print(f"anchors from {args.anchor_mimic_root}: first-event offset in [{off.min() * 1440:.4f}, "
                  f"{off.max() * 1440:.4f}] min")
        else:
            anchors = [float(t[0]) for t in nat]  # minute-grid phase (starttimes are whole minutes)
    else:
        raise ValueError(f"unsupported dataset_type {config.dataset_type}")
    if config.eval.max_sequence_length is not None:
        assert config.eval.truncation_strategy == "right", "only right (prefix) truncation is aligned"
        nat = [t[: config.eval.max_sequence_length] for t in nat]
    # Alignment check against the model-facing sequences (float32 storage → loose tolerance).
    worst = 0.0
    for s, (seq, t) in enumerate(zip(eval_sequences, nat)):
        assert len(seq) == len(t), f"seq {s}: length mismatch {len(seq)} vs {len(t)}"
        model_native = seq.time_points.double() * tsf + offs[s]
        worst = max(worst, float((model_native - t).abs().max() / max(1.0, float(t.abs().max()))))
    print(f"native-time alignment: max rel. deviation {worst:.3g}")
    assert worst < 1e-5, "native timestamps do not match the evaluated sequences"
    return nat, offs, anchors, tsf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", required=True,
                    help="comma-separated bin widths in NATIVE time units; suffix c = grid shifted by -Δ/2")
    ap.add_argument("--out", required=True, help="per-item npz")
    ap.add_argument("--tie-eps", type=float, default=None,
                    help="native gap below which a target counts as tie (default: MIMIC 1e-5 d, else exact 0)")
    ap.add_argument("--native-mimic-root", default=None,
                    help="root to read float64 native times from (default: the evaluated config's mimic_root "
                         "if it sets one, else mimic_iv_f64)")
    ap.add_argument("--anchor-mimic-root", default=None,
                    help="MIMIC: take the bin-grid anchor (first recorded minute) from this source root "
                         "instead of the first event (needed for dequantized roots)")
    ap.add_argument("--tie-bin", default=None,
                    help="Δ token (as in --deltas) whose same-bin indicator defines extra 'rectie'/'nonrectie' "
                         "subsets (previous event in the same recorded bin, e.g. the 1-min floor bin)")
    ap.add_argument("--meta", default="", help="k=v,k=v (or JSON) echoed into the result line (dataset/model/K/seed)")
    args, rest = ap.parse_known_args()
    rest = [r for r in rest if r != "--"]
    config: StandaloneEvalConfig = load_config(StandaloneEvalConfig, argv=rest)
    tokens = args.deltas.split(",")
    deltas = [float(x.rstrip("cp")) for x in tokens]
    centered = [x.endswith("c") for x in tokens]
    prev_anchor = ["p" in x for x in tokens]

    test_dataset = load_test_dataset(config.dataset_type, dict(config.dataset), get_data_root(),
                                     limit_test_size=config.eval.limit_test_size)
    D = test_dataset.num_event_types
    model, _ = create_model(model_type=config.model_type, D=D, arch=config.arch,
                            terminating=test_dataset.terminating, load_path=Path(config.model_path))
    model = model.to(config.device).eval()
    eval_sequences = list(test_dataset)
    if config.eval.max_sequence_length is not None:
        eval_sequences = truncate_sequences_for_eval(eval_sequences, max_length=config.eval.max_sequence_length,
                                                     strategy=config.eval.truncation_strategy, seed=config.arch.seed)
    nat, offs, anchors, tsf = native_times(config, test_dataset, eval_sequences, args)
    tie_eps = args.tie_eps if args.tie_eps is not None else (1e-5 if config.dataset_type == "mimic_iv" else 0.0)

    ignore = test_dataset.ignore_indices
    assert not ignore, "ignore_indices not supported here (none in the benchmark datasets)"
    items = [it for it in _build_prediction_items(eval_sequences, ignore) if it["valid"]]
    dev = torch.device(config.device)
    is_density = isinstance(model, DensityBasedTPP)
    eval_ci = None if is_density else replace(model.ci_config, rule="trapezoidal", placement="global")
    closed_form = (not is_density) and eval_ci.mode == "closed_form"
    print(f"model family: {'density' if is_density else 'intensity'}; CI: "
          f"{'n/a' if is_density else (eval_ci.mode + ('' if closed_form else f' n_points={eval_ci.n_points}'))}")

    bs = config.eval.eval_batch_size or 32
    rec = {k: [] for k in ["seq_idx", "history_length", "gap_native", "time_ll_scaled", "binned_ll", "same_bin"]}
    n_clamp = np.zeros(len(deltas), dtype=np.int64)
    n_nonpos = np.zeros(len(deltas), dtype=np.int64)
    n_bad_b = np.zeros(len(deltas), dtype=np.int64)
    for b0 in tqdm(range(0, len(items), bs), desc="binned eval"):
        chunk = items[b0:b0 + bs]
        B = len(chunk)
        batch = BatchedMVEventData(mv_events=[it["history"] for it in chunk]).to(dev)
        t = torch.tensor([float(it["target_time"]) for it in chunk], dtype=torch.float64, device=dev)
        types = torch.tensor([int(it["target_type"]) for it in chunk], dtype=torch.long, device=dev)
        t_last = torch.tensor([float(it["history"].time_points[-1]) for it in chunk], dtype=torch.float64, device=dev)
        si = [it["seq_idx"] for it in chunk]
        hi = [it["history_length"] for it in chunk]
        tn = np.array([float(nat[s][h]) for s, h in zip(si, hi)])
        tp = np.array([float(nat[s][h - 1]) for s, h in zip(si, hi)])
        anc = np.array([anchors[s] for s in si])
        off = np.array([offs[s] for s in si])

        # bin edges per Δ, native → scaled (float64), a := t_{i-1} (model value) if prev in the bin
        A, Bend, same = [], [], []
        for j, d in enumerate(deltas):
            base = tp if prev_anchor[j] else anc
            g0 = base - d / 2 if centered[j] else base
            k = np.floor((tn - g0) / d + BIN_TOL)
            lo = g0 + k * d
            # tolerance: on-grid data put t_{i-1} exactly at the bin start; float noise may place lo
            # a few ulp above it.
            sb = lo <= tp + BIN_TOL * d
            a_s = torch.tensor((lo - off) / tsf, dtype=torch.float64, device=dev)
            b_s = torch.tensor((lo + d - off) / tsf, dtype=torch.float64, device=dev)
            a_s = torch.where(torch.tensor(sb, device=dev), t_last, a_s)
            # float32 model times vs float64 native edges: a may land a few ulp past t (bin starting
            # exactly at the event, e.g. integer Retweet times) -> clamp, benign; b <= t would not be.
            n_clamp[j] += int(((a_s < t_last) | (a_s > t)).sum())
            n_bad_b[j] += int((b_s <= t).sum())
            a_s = torch.minimum(torch.maximum(a_s, t_last), t)
            b_s = torch.maximum(b_s, t)
            A.append(a_s); Bend.append(b_s); same.append(sb)

        with torch.no_grad():
            if is_density:
                prep = getattr(model, "prepare_batch", None)
                if prep is not None:
                    prep(batch)
                try:
                    log_time = torch.logsumexp(model.density_joint(t, batch, log=True), dim=1)  # as evaluate_tpp
                    binned = []
                    for a_s, b_s in zip(A, Bend):
                        lsa = model.survival_time(a_s, batch, log=True)
                        lsb = model.survival_time(b_s, batch, log=True)
                        binned.append((lsa, lsa - lsb))
                finally:
                    clr = getattr(model, "clear_batch", None)
                    if clr is not None:
                        clr()
            else:
                with model.use_ci(eval_ci):
                    # continuous: the exact evaluate_tpp intensity-family path
                    r = evaluate_at_time(model, t, batch, compute_marginals=False)
                    tiny = torch.finfo(r["intensity"].dtype).tiny
                    log_joint = torch.log(r["intensity"].clamp(min=tiny)) - r["ci"].sum(1, keepdim=True)
                    log_time = torch.logsumexp(log_joint, dim=1)
                    prep = getattr(model, "prepare_batch", None)
                    if prep is not None:
                        prep(batch)
                    try:
                        binned = []
                        if closed_form:
                            G = lambda T: model.cumulative_intensity(T, batch).sum(1)  # noqa: E731  Λ(0,T)
                            g_last = G(t_last)
                            for a_s, b_s in zip(A, Bend):
                                ga = G(a_s)
                                binned.append((-(ga - g_last), G(b_s) - ga))
                        else:
                            for a_s, b_s in zip(A, Bend):
                                la = model.cumulative_intensity(a_s, batch, T_low=t_last, ci_config=eval_ci).sum(1)
                                dl = model.cumulative_intensity(b_s, batch, T_low=a_s, ci_config=eval_ci).sum(1)
                                binned.append((-la, dl))
                    finally:
                        clr = getattr(model, "clear_batch", None)
                        if clr is not None:
                            clr()
            out = []
            for j, (lsa, dl) in enumerate(binned):
                n_nonpos[j] += int((dl <= 0).sum())
                dl = dl.clamp(min=1e-300)
                out.append((lsa.double() + log1mexp(dl.double())).cpu())
        rec["seq_idx"].append(np.array(si)); rec["history_length"].append(np.array(hi))
        rec["gap_native"].append(tn - tp)
        rec["time_ll_scaled"].append(log_time.double().cpu().numpy())
        rec["binned_ll"].append(torch.stack(out, 1).numpy())
        rec["same_bin"].append(np.stack(same, 1))
        _ = types  # (target types unused: time-only metric)

    R = {k: np.concatenate(v) for k, v in rec.items()}
    lts = math.log(tsf)
    tie = R["gap_native"] <= tie_eps
    meta = json.loads(args.meta) if args.meta.startswith("{") else dict(kv.split("=", 1) for kv in args.meta.split(",") if kv)
    res = {**meta, "model_type": config.model_type, "model_path": config.model_path,
           "dataset": config.dataset.get("dataset_name"), "limit_test_size": config.eval.limit_test_size,
           "max_sequence_length": config.eval.max_sequence_length, "n_items": int(len(tie)),
           "n_tie": int(tie.sum()), "tie_eps": tie_eps, "time_scale_factor": tsf, "deltas": deltas,
           "centered": centered, "prev_anchor": prev_anchor, "delta_tokens": tokens,
           "ci": "density" if is_density else ("closed_form" if closed_form else f"trapezoid{eval_ci.n_points}"),
           "n_clamped_a": n_clamp.tolist(), "n_bad_b": n_bad_b.tolist(), "n_nonpos_dlambda": n_nonpos.tolist()}
    subsets = [("all", np.ones_like(tie)), ("tie", tie), ("nontie", ~tie)]
    if args.tie_bin is not None:
        rectie = R["same_bin"][:, tokens.index(args.tie_bin)]
        res["tie_bin"] = args.tie_bin
        res["n_rectie"] = int(rectie.sum())
        subsets += [("rectie", rectie), ("nonrectie", ~rectie)]
    for sub, m in subsets:
        if m.sum() == 0:
            continue
        res[f"cont_{sub}"] = float(R["time_ll_scaled"][m].mean() - lts)
        res[f"binned_{sub}"] = [float(R["binned_ll"][m, j].mean()) for j in range(len(deltas))]
        res[f"samebin_frac_{sub}"] = [float(R["same_bin"][m, j].mean()) for j in range(len(deltas))]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, deltas=np.array(deltas), centered=np.array(centered), prev_anchor=np.array(prev_anchor), log_time_scale=lts, tie=tie, **R)
    print("BINNED_METRICS_JSON " + json.dumps(res))


if __name__ == "__main__":
    main()
