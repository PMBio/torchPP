#!/usr/bin/env python
"""Render the result tables of the paper from paper/results/.

    python paper/make_tables.py [--out paper/tables]

Tables that include UK Biobank results are rendered without the UKB part (UKB results are not released).
"""
import argparse
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parent / "tables"

STD = [("taxi", "Taxi"), ("stackoverflow", "StackOverflow"), ("retweet", "Retweet"),
       ("amazon", "Amazon"), ("taobao", "Taobao")]
MIMIC = "mimic_iv"

GROUPS = [
    ("Poisson processes", [("poisson", "Homogeneous Poisson"), ("spline_poisson", "Spline Poisson")]),
    ("Hawkes processes", [("hawkes", "Const./Exp Hawkes"), ("spline_exp_hawkes", "Spline/Exp Hawkes"),
                          ("const_spline_kernel", "Const./Spline Hawkes"),
                          ("spline_spline_kernel", "Spline/Spline Hawkes")]),
    ("Inhibitive Hawkes processes", [("inhibitive_const_exp_hawkes", "Inhib. Const./Exp Hawkes"),
                                     ("inhibitive_spline_spline_kernel", "Inhib. Spline/Spline Hawkes")]),
    ("Neural TPPs", [("neural_poisson", "NeuralConstTPP"), ("easytpp_thp", "THP"),
                     ("easytpp_sahp", "SAHP"), ("nhp", "NHP")]),
    ("Flexible neural TPPs", [("intensity_free", "IntensityFree"), ("neural_spline", "NeuralSplineTPP")]),
]
SHORT = {"poisson": "H. Poisson", "spline_poisson": "S. Poisson", "hawkes": "C/E Hawkes",
         "spline_exp_hawkes": "S/E Hawkes", "const_spline_kernel": "C/S Hawkes",
         "spline_spline_kernel": "S/S Hawkes", "inhibitive_const_exp_hawkes": "I. C/E Hawkes",
         "inhibitive_spline_spline_kernel": "I. S/S Hawkes", "neural_poisson": "N. Const",
         "easytpp_thp": "THP", "easytpp_sahp": "SAHP", "nhp": "NHP", "intensity_free": "IntensityFree",
         "neural_spline": "N. Spline"}
CLASSICAL = {m for g, rows in GROUPS[:3] for m, _ in rows}
ROSTER = [m for _, rows in GROUPS for m, _ in rows]


# ----------------------------------------------------------------------------- data
def _agg(df, spec, group):
    out = {}
    for idx, g in df.groupby(group):
        for key, (col, scale) in spec.items():
            v = g[col]
            v = v[np.isfinite(v)] * scale
            if len(v):
                out[(key,) + idx] = (v.mean(), v.std(ddof=1) if len(v) > 1 else np.nan, len(v))
    return out


def load_main():
    df = pd.read_csv(RES / "main_runs_perseed.csv")
    spec = {"joint": ("test_joint_ll", 1.0), "time": ("test_marginal_time_ll", 1.0),
            "type": ("test_cond_type_ll", 1.0), "acc_t": ("test_cond_top1_accuracy", 100.0)}
    return _agg(df, spec, ["model_type", "dataset"])


def load_grid():
    df = pd.read_csv(RES / "grid_eval_perseed.csv")
    out = _agg(df, {"acc": ("top1_accuracy", 100.0), "ptype": ("marginal_type_ll", 1.0)},
               ["model_type", "dataset"])
    # RMSE: seeds whose grid-integrated mean diverged are excluded.
    out.update(_agg(df[~df.rmse_divergent], {"rmse": ("time_rmse_mean_days", 1.0)}, ["model_type", "dataset"]))
    # IntensityFree: the log-normal-mixture mean is heavy-tailed -> report the median predictor.
    iff = df[df.model_type == "intensity_free"]
    out.update(_agg(iff, {"rmse": ("time_rmse_median_days", 1.0)}, ["model_type", "dataset"]))
    return out


MAIN = load_main()
GRID_EVAL = load_grid()


def val(key, model, ds):
    src = MAIN if key in {"joint", "time", "type", "acc_t"} else GRID_EVAL
    v = src.get((key, model, ds))
    if v is None or not np.isfinite(v[0]):
        return None
    return v


def load_ksweep():
    return pd.read_csv(RES / "knot_sweep.csv")


# ----------------------------------------------------------------------------- format
def rnd(v, dec):
    """Round half-up (not Python's round-half-even) to `dec` decimals, as a string."""
    return str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP))


def f_std(s, dec):
    if not np.isfinite(s):
        return ""
    t = rnd(s, dec)
    return t[1:] if t.startswith("0.") else t


def f_num(v, dec):
    t = rnd(v, dec)
    if t.startswith("-") and float(t) == 0:
        t = t[1:]
    return t


FEW = "\\(^{\\ast}\\)"


def cell(v, dec=2, bold=False, under=False, mark=""):
    if v is None:
        return "--"
    m, s, n = v
    if n < 5:
        mark = FEW + mark
    num = f_num(m, dec)
    if bold:
        num = f"\\mathbf{{{num}}}"
    body = f"{num}_{{\\pm {f_std(s, dec)}}}" if np.isfinite(s) else num
    out = f"\\({body}\\){mark}"
    return f"\\underline{{{out}}}" if under else out


def best(values, lower, dec=2, exclude=()):
    """Sets of best and best-classical models; ties at the displayed precision are all marked."""
    ok = {m: float(f_num(v[0], dec)) for m, v in values.items() if v is not None and m not in exclude}
    if not ok:
        return set(), set()
    pick = min if lower else max
    top = pick(ok.values())
    cl = {m: x for m, x in ok.items() if m in CLASSICAL}
    ctop = pick(cl.values()) if cl else None
    return {m for m, x in ok.items() if x == top}, {m for m, x in cl.items() if x == ctop}


def roster_table(label, caption, key, datasets, lower=False, dec=2, marks=None, colspec=None, exclude=()):
    marks = marks or {}
    ncol = len(datasets) + 1
    lines = ["\\begin{table}[t]", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
             "  \\centering", "  \\small", "  \\setlength{\\tabcolsep}{4pt}",
             "  \\begin{tabular}{" + (colspec or "l" + "c" * len(datasets)) + "}", "    \\toprule",
             "    Model & " + " & ".join(n for _, n in datasets) + " \\\\", "    \\midrule"]
    bests = {ds: best({m: val(key, m, ds) for m in ROSTER}, lower, dec, exclude) for ds, _ in datasets}
    for gi, (gname, rows) in enumerate(GROUPS):
        lines.append(f"    \\multicolumn{{{ncol}}}{{l}}{{\\textit{{{gname}}}}} \\\\")
        for m, disp in rows:
            cells = []
            for ds, _ in datasets:
                b, bc = bests[ds]
                cells.append(cell(val(key, m, ds), dec, bold=(m in b), under=(m in bc),
                                  mark=marks.get((m, ds), "")))
            lines.append(f"    {disp} & " + " & ".join(cells) + " \\\\")
        if gi < len(GROUPS) - 1:
            lines.append("    \\midrule")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


COMMON = ("Mean \\(\\pm\\) std over 5 seeds. \\textbf{Bold}: best per dataset; "
          "\\underline{underline}: best classical (Poisson or Hawkes) model; ties at the displayed precision are all marked.")
GRID = (" Grid-based metrics are evaluated on the first 1{,}000 test sequences. \\(^{\\ast}\\): mean over "
        "3--4 seeds, as the evaluation of a seed is missing or its RMSE diverged. Spline/Spline Hawkes on Amazon "
        "was not evaluated with the grid-based metrics.")


def write(name, text):
    (OUT / name).write_text("% generated by paper/make_tables.py -- do not edit by hand\n" + text)
    print("wrote", name)


# ----------------------------------------------------------------------------- standard benchmarks
def standard_tables():
    write("app_std_joint.tex", roster_table(
        "tab:joint_ll_results",
        "Joint test log-likelihood per event \\(\\log p(t, d \\mid \\mathcal{H})\\) (nats, higher is "
        "better) on the standard benchmarks. It is the sum of Tables~\\ref{tab:time_ll_results} and "
        "\\ref{tab:cond_type_ll_results}. " + COMMON, "joint", STD))
    write("app_std_time.tex", roster_table(
        "tab:time_ll_results",
        "Time log-likelihood per event \\(\\log p(t \\mid \\mathcal{H})\\) (nats, higher is better) on "
        "the standard benchmarks (Equation~\\ref{eq:time_marginal}). " + COMMON, "time", STD))
    write("app_std_type.tex", roster_table(
        "tab:cond_type_ll_results",
        "Time-conditional type log-likelihood per event \\(\\log p(d \\mid t, \\mathcal{H})\\) at the true "
        "next-event time (nats, higher is better; Equation~\\ref{eq:type_conditional}). " + COMMON,
        "type", STD))
    write("app_std_acct.tex", roster_table(
        "tab:cond_acc_results",
        "Time-conditional top-1 type accuracy \\(\\mathrm{Acc}|t\\) (\\%, higher is better): "
        "\\(\\arg\\max_d p(d \\mid t, \\mathcal{H})\\) at the true next-event time, the next-type metric "
        "reported by EasyTPP~\\citep{xue2024easytpp}. " + COMMON, "acc_t", STD))
    write("app_std_acc.tex", roster_table(
        "tab:acc_results",
        "Marginal top-1 type accuracy \\(\\mathrm{Acc}\\) (\\%, higher is better): "
        "\\(\\arg\\max_d p(d \\mid \\mathcal{H})\\) from the marginal type distribution "
        "(Equation~\\ref{eq:type_marginal}), computed by numerical integration." + GRID + " " + COMMON,
        "acc", STD))
    write("app_std_ptype.tex", roster_table(
        "tab:marg_type_ll_results",
        "Marginal type log-likelihood per event \\(\\log p(d \\mid \\mathcal{H})\\) (nats, higher is "
        "better; Equation~\\ref{eq:type_marginal}), computed by numerical integration." + GRID + " " + COMMON,
        "ptype", STD))
    marks = {("intensity_free", ds): "\\(^\\ddagger\\)" for ds, _ in STD}
    write("app_std_rmse.tex", roster_table(
        "tab:rmse_results",
        "Next-event time RMSE (lower is better) using the predictive mean "
        "\\(\\mathbb{E}[t \\mid \\mathcal{H}]\\), in each dataset's native EasyTPP time unit (not comparable "
        "across datasets). \\(^\\ddagger\\)IntensityFree: the closed-form mean of its log-normal mixture is dominated "
        "by components with large variance and overflows for individual seeds, so we report the median predictor "
        "instead and exclude it from the bold marking. " + GRID + " " + COMMON,
        "rmse", STD, lower=True, marks=marks, exclude=("intensity_free",)))


# ----------------------------------------------------------------------------- MIMIC full
def mimic_table():
    cols = [("joint", "\\(\\log p(t,d)\\)", False, 2, 1), ("time", "\\(\\log p(t)\\)", False, 2, 1),
            ("type", "\\(\\log p(d|t)\\)", False, 2, 1), ("ptype", "\\(\\log p(d)\\)", False, 2, 1),
            ("acc_t", "Acc\\(|t\\) (\\%)", False, 1, 1), ("acc", "Acc (\\%)", False, 1, 1),
            ("rmse", "RMSE\\(\\times 100\\)", True, 2, 100)]
    lines = ["\\begin{table}[t]",
             "  \\caption{Full MIMIC-IV results (\\(D = 117\\), including the \\texttt{OTHER} class). "
             "\\(\\log p(t,d) = \\log p(t) + \\log p(d|t)\\); \\(\\log p(d)\\), Acc and RMSE are computed by "
             "numerical integration on the first 500 test stays, all other metrics on the first 1{,}000. "
             "RMSE in days, multiplied by 100. \\(^\\ddagger\\)Median predictor (see "
             "Table~\\ref{tab:rmse_results}). \\(^\\S\\)NeuralConstTPP uses a Softplus head on MIMIC-IV, as its "
             "exponential head diverged for individual seeds. " + COMMON.replace("per dataset", "per column") + "}",
             "  \\label{tab:full_mimic_results}", "  \\centering", "  \\small",
             "  \\resizebox{\\textwidth}{!}{%", "  \\begin{tabular}{l" + "c" * len(cols) + "}",
             "    \\toprule", "    Model & " + " & ".join(c[1] for c in cols) + " \\\\", "    \\midrule"]

    def v(key, m, scale):
        x = val(key, m, MIMIC)
        return None if x is None else (x[0] * scale, x[1] * scale, x[2])

    bests = {c[0]: best({m: v(c[0], m, c[4]) for m in ROSTER}, c[2], c[3],
                        ("intensity_free",) if c[0] == "rmse" else ()) for c in cols}
    for gi, (gname, rows) in enumerate(GROUPS):
        lines.append(f"    \\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textit{{{gname}}}}} \\\\")
        for m, disp in rows:
            cs = []
            for key, _, _, dec, scale in cols:
                b, bc = bests[key]
                mark = "\\(^\\ddagger\\)" if (key == "rmse" and m == "intensity_free") else ""
                cs.append(cell(v(key, m, scale), dec, bold=(m in b), under=(m in bc), mark=mark))
            name = disp + ("\\(^\\S\\)" if m == "neural_poisson" else "")
            lines.append(f"    {name} & " + " & ".join(cs) + " \\\\")
        if gi < len(GROUPS) - 1:
            lines.append("    \\midrule")
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table}"]
    write("app_mimic_full.tex", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- IntensityFree encoder ablation
def iftpp_table():
    tr = pd.read_csv(RES / "main_runs_perseed.csv")
    tr = tr[tr.model_type == "intensity_free"]
    gru = pd.read_csv(RES / "intensity_free_gru_perseed.csv")
    cols = ["test_joint_ll", "test_marginal_time_ll", "test_cond_type_ll"]

    def tex(g, col):
        return f"\\({f_num(g[col].mean(), 2)}_{{\\pm {f_std(g[col].std(ddof=1), 2)}}}\\)"

    lines = ["\\begin{table}[t]",
             "  \\caption{IntensityFree with a GRU encoder (as in \\citet{IntensityFree}) vs.\\ the "
             "Transformer encoder used in the main tables. Per-event test log-likelihood, mean \\(\\pm\\) "
             "std over 5 seeds (higher is better). The Transformer is clearly better on Amazon, the GRU on MIMIC-IV and slightly on Taobao, and the two are close "
             "on Taxi, StackOverflow, and Retweet.}",
             "  \\label{tab:iftpp_encoder}", "  \\centering", "  \\small",
             "  \\begin{tabular}{llccc}", "    \\toprule",
             "    Dataset & Encoder & \\(\\log p(t,d)\\) & \\(\\log p(t)\\) & \\(\\log p(d|t)\\) \\\\", "    \\midrule"]
    for i, (ds, disp) in enumerate(STD + [(MIMIC, "MIMIC-IV")]):
        if i:
            lines.append("    \\midrule")
        for enc, df in [("Transformer", tr), ("GRU", gru)]:
            g = df[df.dataset == ds]
            lines.append(f"    {disp if enc == 'Transformer' else ''} & {enc} & "
                         + " & ".join(tex(g, c) for c in cols) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    write("app_iftpp_encoder.tex", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- knot sweep (MIMIC) + params
# Parameter counts of create_model(...) at D=117 (sum of p.numel() over the model parameters);
# NeuralSplineTPP additionally has a 208,832-parameter Transformer encoder.
PARAMS_SS = {10: 148_591, 20: 285_481, 50: 696_151, 100: 1_380_601, 200: 2_749_501, 400: 5_487_301}
PARAMS_NS = {10: 77_220, 20: 154_440, 50: 386_100, 100: 772_200, 200: 1_544_400, 400: 3_088_800}


def ksweep_table():
    k = load_ksweep()
    k = k[k.dataset == MIMIC].set_index(["model_type", "K"])

    def c(mt, K):
        r = k.loc[(mt, K)]
        t = f"\\({r.marginal_time_ll_mean:.2f}_{{\\pm {f_std(r.marginal_time_ll_std, 2)}}}\\)"
        y = f"\\({r.cond_type_ll_mean:.2f}_{{\\pm {f_std(r.cond_type_ll_std, 2)}}}\\)"
        a = f"\\({100 * r.cond_top1_acc_mean:.1f}_{{\\pm {f_std(100 * r.cond_top1_acc_std, 1)}}}\\)"
        return t, y, a

    lines = ["\\begin{table}[t]",
             "  \\caption{Knot-count sweep on MIMIC-IV: Spline/Spline Hawkes (kernel knots \\(K\\); baseline "
             "fixed at 100 knots) vs.\\ NeuralSplineTPP (head knots \\(K\\)). Mean \\(\\pm\\) std over 5 "
             "seeds. \\(\\log p(t)\\) is flat in \\(K\\) for both families, and \\(\\log p(d|t)\\) does not improve. "
             "\\#params: trainable parameters of the Hawkes model and of the "
             "NeuralSplineTPP head (its Transformer encoder adds 208{,}832).}",
             "  \\label{tab:ksweep_mimic}", "  \\centering", "  \\small", "  \\setlength{\\tabcolsep}{3pt}",
             "  \\resizebox{\\textwidth}{!}{%", "  \\begin{tabular}{r rccc rccc}", "    \\toprule",
             "    & \\multicolumn{4}{c}{Spline/Spline Hawkes} & \\multicolumn{4}{c}{NeuralSplineTPP} \\\\",
             "    \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
             "    \\(K\\) & \\#params & \\(\\log p(t)\\) & \\(\\log p(d|t)\\) & Acc\\(|t\\) (\\%) & "
             "\\#params & \\(\\log p(t)\\) & \\(\\log p(d|t)\\) & Acc\\(|t\\) (\\%) \\\\", "    \\midrule"]
    for K in [10, 20, 50, 100, 200, 400]:
        ps, pn = (f"{x:,}".replace(",", "{,}") for x in (PARAMS_SS[K], PARAMS_NS[K]))
        lines.append(f"    {K} & {ps} & " + " & ".join(c("spline_spline_kernel", K))
                     + f" & {pn} & " + " & ".join(c("neural_spline", K)) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table}"]
    write("app_ksweep_mimic.tex", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- non-stationary control
def ns_table():
    ns = pd.read_csv(RES / "ns_control.csv")
    k = load_ksweep()
    k = k[k.dataset == MIMIC].set_index(["model_type", "K"])

    def trip(t, ts, y, ys, a, as_):
        return (f"\\({t:.2f}_{{\\pm {f_std(ts, 2)}}}\\) & \\({y:.2f}_{{\\pm {f_std(ys, 2)}}}\\) & "
                f"\\({100 * a:.1f}_{{\\pm {f_std(100 * as_, 1)}}}\\)")

    def nsrow(model, kdt, kns):
        r = ns[(ns.model_type == model) & (ns.K_ns == kns) & ((ns.K_dt == kdt) if kdt else True)].iloc[0]
        return trip(r.time_ll_mean, r.time_ll_std, r.type_ll_mean, r.type_ll_std, r.acc1_mean, r.acc1_std)

    nst, nsy = val("time", "neural_spline", MIMIC)[0], val("type", "neural_spline", MIMIC)[0]
    lines = ["\\begin{table}[t]",
             "  \\caption{Non-stationary Hawkes control on MIMIC-IV. The kernel heights of the NS variants are "
             "themselves linear splines over absolute stay time with \\(K_{ns}\\) knots (\\(K_{ns}=1\\): the "
             "stationary model); \\(K_{\\Delta t}\\): knots of the kernel over elapsed time. Mean \\(\\pm\\) std over "
             "5 seeds. The NS runs use the configuration of Appendix~\\ref{app:hyperparams_final} with learning rate 0.05 for "
             "the NS Spline/Spline kernels. The stationary Spline/Spline rows are taken from the knot sweep "
             "(Table~\\ref{tab:ksweep_mimic}). At \\(K_{\\Delta t} = 12\\), their nearest counterparts are \\(K = 10\\) and \\(20\\) there. "
             "No non-stationary cell improves the joint log-likelihood of its nearest stationary counterpart by more than 0.002 nats, and all remain "
             f"below NeuralSplineTPP (\\(\\log p(t) = {nst:.2f}\\), \\(\\log p(d|t) = {nsy:.2f}\\)).}}",
             "  \\label{tab:ns_control}", "  \\centering", "  \\small",
             "  \\begin{tabular}{lrrccc}", "    \\toprule",
             "    Model & \\(K_{\\Delta t}\\) & \\(K_{ns}\\) & \\(\\log p(t)\\) & \\(\\log p(d|t)\\) & Acc\\(|t\\) (\\%) \\\\",
             "    \\midrule"]
    t, y, a = (val(x, "spline_exp_hawkes", MIMIC) for x in ("time", "type", "acc_t"))
    lines.append("    Spline/Exp Hawkes & -- & 1 & " + trip(t[0], t[1], y[0], y[1], a[0] / 100, a[1] / 100) + " \\\\")
    lines.append("    \\quad + NS kernel & -- & 5 & " + nsrow("spline_ns_exp_hawkes", None, 5) + " \\\\")
    lines.append("    \\midrule")
    for Kns in [3, 5]:
        lines.append(f"    Spline/Spline Hawkes + NS kernel & 12 & {Kns} & " + nsrow("spline_ns_spline_kernel", 12, Kns) + " \\\\")
    for K in [20, 100]:
        st = k.loc[("spline_spline_kernel", K)]
        lines.append(f"    Spline/Spline Hawkes & {K} & 1 & " + trip(st.marginal_time_ll_mean, st.marginal_time_ll_std,
                                                                   st.cond_type_ll_mean, st.cond_type_ll_std,
                                                                   st.cond_top1_acc_mean, st.cond_top1_acc_std) + " \\\\")
        for Kns in [3, 5]:
            lines.append(f"    \\quad + NS kernel & {K} & {Kns} & " + nsrow("spline_ns_spline_kernel", K, Kns) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    write("app_ns_control.tex", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- tick reproduction
def tick_table():
    tick = pd.read_csv(RES / "tick_repro.csv").set_index("dataset")
    ours_best = {ds: max(val("acc_t", m, ds)[0] for m in CLASSICAL) for ds, _ in STD}
    lines = ["\\begin{table}[t]",
             "  \\caption{Reproducing EasyTPP's Hawkes (MHP) baseline. Time-conditional next-type accuracy "
             "\\(\\mathrm{Acc}|t\\) (\\%). The \\texttt{tick} fits, the majority rate and EasyTPP use the full test sets, "
             "our models the first 1{,}000 test sequences (Section~\\ref{sec:datasets-experiments/setup}). \\emph{Majority}: always predicting the most frequent "
             "type. \\emph{tick LS}: \\texttt{tick} \\texttt{HawkesExpKern} with fixed decay 1.0 and its default "
             "least-squares fit; \\emph{MLE}: the same exponential Hawkes fitted by positivity-constrained maximum "
             "likelihood; \\emph{MLE, \\(\\mu{=}0.2\\)}: additionally fixing the base intensity to 0.2, the "
             "literal recipe of EasyTPP's Appendix~B. Ours: our learned Const./Exp Hawkes and the best of our "
             "(inhibitive) Hawkes variants, from Table~\\ref{tab:cond_acc_results}.}",
             "  \\label{tab:tick_repro}", "  \\centering", "  \\small",
             "  \\resizebox{\\textwidth}{!}{%", "  \\begin{tabular}{lcccccc}", "    \\toprule",
             "    & & \\multicolumn{3}{c}{\\texttt{tick} MHP (our fit)} & & \\\\",
             "    \\cmidrule(lr){3-5}",
             "    Dataset & Majority & tick LS & MLE & MLE, \\(\\mu{=}0.2\\) & EasyTPP (reported) & "
             "Ours: C/E \\;/\\; best Hawkes \\\\", "    \\midrule"]
    for ds, disp in STD:
        v = tick.loc[ds]
        ce = val("acc_t", "hawkes", ds)[0]
        lines.append(f"    {disp} & {v.majority:.1f} & {v.tick_ls:.1f} & {v.mle:.1f} & "
                     f"{v.mle_mu02:.1f} & {v.easytpp_reported:.1f} & {ce:.1f} \\;/\\; {ours_best[ds]:.1f} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table}"]
    write("app_tick_repro.tex", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- calibration
def calibration_table():
    cal = pd.read_csv(RES / "calibration.csv").set_index(["model_type", "dataset"])
    models = [("hawkes", "Const./Exp Hawkes"), ("spline_spline_kernel", "Spline/Spline Hawkes"),
              ("neural_poisson", "NeuralConstTPP"), ("neural_spline", "NeuralSplineTPP")]
    dss = STD + [(MIMIC, "MIMIC-IV")]
    lines = ["\\begin{table}[t]",
             "  \\caption{Calibration of four representative models with a closed-form cumulative intensity (seed 0). Each cell "
             "is the time-rescaling Kolmogorov--Smirnov distance \\(\\sup|F_{\\mathrm{emp}}(u) - u|\\) of the "
             "probability integral transform \\(u_i = 1 - \\exp(-\\Lambda^*(t_{i-1}, t_i))\\) (events with a zero "
             "inter-event time excluded) and the top-label expected calibration error of \\(p(d \\mid t, "
             "\\mathcal{H})\\) (15 bins); lower is better for both. Up to 10{,}000 test events per cell "
             "(MIMIC-IV: 5{,}000). For Spline/Spline Hawkes on StackOverflow, the evaluated checkpoint uses learning rate 0.1 "
             "instead of the adjusted rate of Table~\\ref{tab:joint_ll_results}.}",
             "  \\label{tab:calibration}", "  \\centering", "  \\small", "  \\setlength{\\tabcolsep}{3pt}",
             "  \\resizebox{\\textwidth}{!}{%",
             "  \\begin{tabular}{l" + "c" * len(dss) + "}", "    \\toprule",
             "    Model (KS / ECE) & " + " & ".join(d for _, d in dss) + " \\\\", "    \\midrule"]
    for key, disp in models:
        cs = [f"{cal.loc[(key, ds), 'ks_D']:.3f} / {cal.loc[(key, ds), 'top_label_ece']:.3f}".replace("0.", ".")
              for ds, _ in dss]
        lines.append(f"    {disp} & " + " & ".join(cs) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table}"]
    write("app_calibration.tex", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- main-text packed tables
def packed_rows(getter, columns):
    """Rows of `time/type` cells. Bold: best per column and component; underline: best joint (time + type)."""
    best_t, best_y, best_j = {}, {}, {}
    for c in columns:
        vals = {m: getter(m, c) for m in ROSTER}
        vals = {m: v for m, v in vals.items() if v is not None}
        rt = {m: float(rnd(x[0], 2)) for m, x in vals.items()}
        ry = {m: float(rnd(x[1], 2)) for m, x in vals.items()}
        rj = {m: float(rnd(x[0] + x[1], 2)) for m, x in vals.items()}
        best_t[c] = {m for m, x in rt.items() if x == max(rt.values())}
        best_y[c] = {m for m, x in ry.items() if x == max(ry.values())}
        best_j[c] = {m for m, x in rj.items() if x == max(rj.values())}

    def tight(v, bold):
        s = rnd(v, 2)
        s = "{-}" + s[1:] if s.startswith("-") else s
        return f"\\mathbf{{{s}}}" if bold else s

    lines = []
    ncol = len(columns) + 1
    for gi, (gname, rows) in enumerate(GROUPS):
        lines.append(f"    \\multicolumn{{{ncol}}}{{l}}{{\\textit{{{gname}}}}} \\\\")
        for m, _ in rows:
            cells = []
            for c in columns:
                v = getter(m, c)
                if v is None:
                    cells.append("--")
                    continue
                cellv = f"\\({tight(v[0], m in best_t[c])}/{tight(v[1], m in best_y[c])}\\)"
                cells.append(f"\\underline{{{cellv}}}" if m in best_j[c] else cellv)
            lines.append(f"    {SHORT[m]} & " + " & ".join(cells) + " \\\\")
        if gi < len(GROUPS) - 1:
            lines.append("    \\midrule")
    return lines


def main_tables():
    def g_std(m, ds):
        t, y = val("time", m, ds), val("type", m, ds)
        return None if t is None or y is None else (t[0], y[0])

    lines = ["\\begin{table}[t]",
             "  \\caption{Main results: per-event test log-likelihood on the standard benchmark datasets (nats, higher is "
             "better; mean over 5 seeds; EasyTPP convention, i.e.\\ conditioned on the first event and without the final "
             "survival term, Section~\\ref{sec:datasets-experiments/setup}). Each cell is \\(\\log p(t)\\,/\\,\\log p(d\\mid t)\\) "
             "(time\\,/\\,type), and the joint \\(\\log p(t,d)=\\log p(t)+\\log p(d\\mid t)\\) is their sum "
             "(Table~\\ref{tab:joint_ll_results}; \\(\\pm\\)std in Appendix~\\ref{app:full_results}). Models differ chiefly in the "
             "time term \\(\\log p(t)\\), the axis that flexibility improves, while the type term \\(\\log p(d\\mid t)\\) is "
             "comparatively tight across history-aware models (near-flat on StackOverflow and Retweet). "
             "Row labels: baseline/kernel (C: constant, S: spline, E: exponential), I.: inhibitive, N.: neural. "
             "\\textbf{Bold}: best per dataset and component; \\underline{underline}: best joint log-likelihood; "
             "ties at the displayed precision are all marked.}",
             "  \\label{tab:main_results}", "  \\centering", "  \\small", "  \\setlength{\\tabcolsep}{3pt}",
             "  \\begin{tabular}{lccccc}", "    \\toprule",
             "    Model & Taxi & StackOv. & Retweet & Amazon & Taobao \\\\", "    \\midrule"]
    lines += packed_rows(g_std, [d for d, _ in STD])
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    write("main_std.tex", "\n".join(lines) + "\n")

    lines = ["\\begin{wraptable}{r}{0.50\\textwidth}", "\\vspace{-\\baselineskip}",
             "  \\caption{Large-scale medical results (nats, same convention as Table~\\ref{tab:main_results}): cell = "
             "\\(\\log p(t)/\\log p(d\\mid t)\\) (time/type). UKB results are not released and omitted here. Joint, \\(\\pm\\)std and full metrics in Appendix~\\ref{app:full_results}. \\textbf{Bold}: best per "
             "column and component; \\underline{underline}: best joint log-likelihood (ties all marked).}",
             "  \\label{tab:medical_results}", "  \\centering", "  \\footnotesize", "  \\setlength{\\tabcolsep}{3pt}",
             "  \\begin{tabular}{lc}", "    \\toprule", "    Model & MIMIC-IV \\\\", "    \\midrule"]
    lines += packed_rows(lambda m, c: g_std(m, MIMIC), ["mimic"])
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{wraptable}"]
    write("main_medical.tex", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- knot sweep, all datasets
def ksweep_all_table():
    k = load_ksweep()
    dss = [("taxi", "Taxi"), ("stackoverflow", "StackOverflow"), ("retweet", "Retweet"), ("amazon", "Amazon"),
           ("taobao", "Taobao"), (MIMIC, "MIMIC-IV")]
    Ks = [10, 20, 50, 100, 200, 400]

    def tight(v):
        s = f"{v:.2f}"
        return "{-}" + s[1:] if s.startswith("-") else s

    lines = ["\\begin{table}[ht]",
             "  \\caption{Knot-count sweep on all datasets: \\(\\log p(t)\\,/\\,\\log p(d\\mid t)\\) (nats, mean over 5 "
             "seeds) for NeuralSplineTPP (head knots \\(K\\)) and Spline/Spline Hawkes (kernel knots \\(K\\), baseline fixed at "
             "100 knots). The sweep runs are separate from the reported runs (20{,}000 steps on the standard benchmarks, "
             "learning rates without per-model adjustment), so values differ from the main tables.}",
             "  \\label{tab:ksweep_all}", "  \\centering", "  \\footnotesize", "  \\setlength{\\tabcolsep}{3pt}",
             "  \\resizebox{\\textwidth}{!}{%", "  \\begin{tabular}{r" + "c" * len(dss) + "}", "    \\toprule",
             "    \\(K\\) & " + " & ".join(n for _, n in dss) + " \\\\"]
    for mt, name in [("neural_spline", "NeuralSplineTPP"), ("spline_spline_kernel", "Spline/Spline Hawkes")]:
        lines += ["    \\midrule", f"    \\multicolumn{{{len(dss) + 1}}}{{l}}{{\\textit{{{name}}}}} \\\\"]
        for K in Ks:
            cells = []
            for ds, _ in dss:
                r = k[(k.dataset == ds) & (k.model_type == mt) & (k.K == K)]
                cells.append("--" if r.empty else
                             f"\\({tight(r.marginal_time_ll_mean.iloc[0])}/{tight(r.cond_type_ll_mean.iloc[0])}\\)")
            lines.append(f"    {K} & " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table}"]
    write("app_ksweep_all.tex", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- discretized (binned) time likelihood
# log P(t_i in [a, b)) with a = max(bin start, t_{i-1}), b = bin end, from the same checkpoints
# (paper/binned/binned_time_ll.py). Δ in the data time unit.
BIN_DS = [("taxi", "Taxi", 1 / 3600, 10 / 3600), ("stackoverflow", "StackOverflow", 2 ** -13, 10 * 2 ** -13),
          ("retweet", "Retweet", 1.0, 10.0), ("amazon", "Amazon", 0.01, 0.1),
          ("taobao", "Taobao", 1e-4, 1e-3), (MIMIC, "MIMIC-IV", 1 / 1440, 5 / 1440)]


def load_binned():
    return pd.read_csv(RES / "binned_time_ll.csv")


def at_bin(b, ds, delta, centered=False):
    return b[(b.dataset == ds) & (b.centered == centered) & ((b.delta - delta).abs() <= 1e-9 * delta)]


def resolution_table():
    r = pd.read_csv(RES / "time_resolution.csv").set_index("dataset")
    lines = ["\\begin{table}[ht]",
             "  \\caption{Time resolution of the evaluated event times (first 1{,}000 test sequences, MIMIC-IV "
             "truncated to 316 events) and the bin widths used for the binned time likelihood, in the time unit "
             "of each dataset. Tied: fraction of evaluated events with a zero gap to the previous event.}",
             "  \\label{tab:time_resolution}", "  \\centering", "  \\small",
             "  \\begin{tabular}{lllc}", "    \\toprule",
             "    Dataset & Recorded resolution & Bins & Tied (\\%) \\\\", "    \\midrule"]
    res = {"taxi": "1 s", "stackoverflow": "float32 (\\(2^{-13}\\))", "retweet": "integer time stamps",
           "amazon": "none (minimum gap 0.01)", "taobao": "gaps of \\(10^{-4}\\)",
           MIMIC: "1 min (dequantized)"}
    bins = {"taxi": "1 s, 10 s", "stackoverflow": "\\(2^{-13}\\), \\(10 \\cdot 2^{-13}\\)", "retweet": "1, 10",
            "amazon": "0.01, 0.1", "taobao": "\\(10^{-4}\\), \\(10^{-3}\\)", MIMIC: "1 min, 5 min"}
    for ds, name in [("taxi", "Taxi"), ("stackoverflow", "StackOverflow"), ("retweet", "Retweet"),
                     ("amazon", "Amazon"), ("taobao", "Taobao"), (MIMIC, "MIMIC-IV")]:
        lines.append(f"    {name} & {res[ds]} & {bins[ds]} & {rnd(100 * r.loc[ds, 'tied_frac'], 1)} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    write("app_time_resolution.tex", "\n".join(lines) + "\n")


def binned_ksweep_table():
    b = load_binned()
    b = b[b.run == "knot_sweep"].astype({"K": int})
    k = load_ksweep()

    def signed(v):
        s = f_num(v, 2)
        return s if s.startswith("-") else "+" + s

    lines = ["\\begin{table}[ht]",
             "  \\caption{Change of \\(\\log p(t)\\) from \\(K = 20\\) to \\(K = 400\\) knots (nats, mean over 5 seeds), "
             "continuous and binned at the recorded resolution, at a ten-fold (MIMIC-IV: five-fold) coarser bin, and "
             "with bins centered on the grid points instead of starting at them (Table~\\ref{tab:time_resolution}). "
             "Same checkpoints as Table~\\ref{tab:ksweep_all}.}",
             "  \\label{tab:binned_ksweep}", "  \\centering", "  \\small", "  \\setlength{\\tabcolsep}{4pt}",
             "  \\begin{tabular}{llccccc}", "    \\toprule",
             "    & & \\multicolumn{2}{c}{\\(\\log p(t)\\), continuous} & \\multicolumn{3}{c}{\\(\\Delta\\) binned} \\\\",
             "    \\cmidrule(lr){3-4}\\cmidrule(lr){5-7}",
             "    Dataset & Model & \\(K = 20\\) & \\(\\Delta\\) to \\(K = 400\\) & recorded & coarser & centered \\\\",
             "    \\midrule"]
    for i, (ds, name, d1, d2) in enumerate(BIN_DS):
        for j, (mt, mname) in enumerate([("neural_spline", "NeuralSplineTPP"), ("spline_spline_kernel", "Spline/Spline Hawkes")]):
            def dlt(sub):
                g = sub[sub.model_type == mt].groupby("K").binned_all.mean()
                return g.loc[400] - g.loc[20] if {20, 400} <= set(g.index) else None
            kk = k[(k.dataset == ds) & (k.model_type == mt)].set_index("K").marginal_time_ll_mean
            if not {20, 400} <= set(kk.index):
                continue
            cells = [f_num(kk.loc[20], 2), signed(kk.loc[400] - kk.loc[20]),
                     *("--" if x is None else signed(x) for x in
                       (dlt(at_bin(b, ds, d1)), dlt(at_bin(b, ds, d2)), dlt(at_bin(b, ds, d1, True))))]
            lines.append(f"    {name if j == 0 else ''} & {mname} & " + " & ".join(cells) + " \\\\")
        if i < len(BIN_DS) - 1:
            lines.append("    \\midrule")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    write("app_binned_ksweep.tex", "\n".join(lines) + "\n")


def binned_main_table():
    b = load_binned()
    b = b[b.run == "main"]
    vals = {}
    for ds, _, d1, _ in BIN_DS:
        for m, g in at_bin(b, ds, d1).groupby("model_type"):
            vals[(m, ds)] = (g.binned_all.mean(), g.binned_all.std(ddof=1), len(g))
    dss = [(ds, name) for ds, name, _, _ in BIN_DS]
    lines = ["\\begin{table}[t]",
             "  \\caption{Binned time log-likelihood \\(\\log \\Pr(t \\in \\text{bin})\\) at the recorded resolution "
             "(nats, higher is better; bins in Table~\\ref{tab:time_resolution}) for the models of the main tables. "
             "Only differences within a column are meaningful, as the level depends on the bin width. On Retweet "
             "the history-aware models are within a few tenths of a nat, while the continuous \\(\\log p(t)\\) "
             "(Table~\\ref{tab:time_ll_results}) spreads over more than two nats. " + COMMON + "}",
             "  \\label{tab:binned_main}", "  \\centering", "  \\small", "  \\setlength{\\tabcolsep}{3pt}",
             "  \\resizebox{\\textwidth}{!}{%", "  \\begin{tabular}{l" + "c" * len(dss) + "}", "    \\toprule",
             "    Model & " + " & ".join(n for _, n in dss) + " \\\\", "    \\midrule"]
    bests = {ds: best({m: vals.get((m, ds)) for m in ROSTER}, False, 2) for ds, _ in dss}
    for gi, (gname, rows) in enumerate(GROUPS):
        lines.append(f"    \\multicolumn{{{len(dss) + 1}}}{{l}}{{\\textit{{{gname}}}}} \\\\")
        for m, disp in rows:
            cs = [cell(vals.get((m, ds)), 2, bold=(m in bests[ds][0]), under=(m in bests[ds][1])) for ds, _ in dss]
            lines.append(f"    {disp} & " + " & ".join(cs) + " \\\\")
        if gi < len(GROUPS) - 1:
            lines.append("    \\midrule")
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table}"]
    write("app_binned_main.tex", "\n".join(lines) + "\n")


def mimic_ties_table():
    t = pd.read_csv(RES / "mimic_same_minute_split.csv")
    order = [m for m in ROSTER if m in set(t.model_type)]

    b = load_binned()
    b = at_bin(b[b.run == "main"], MIMIC, 1 / 1440)

    def v(m, kind, s, col, scale=1.0):
        if kind == "binned":
            g = b[b.model_type == m][col]
        else:
            g = t[(t.model_type == m) & (t.subset == s)][col] * scale
        return (g.mean(), g.std(ddof=1), len(g))

    cols = [("cont", "all", "marginal_time_ll", 1.0, 2), ("cont", "other_minute", "marginal_time_ll", 1.0, 2),
            ("binned", "all", "binned_all", 1.0, 2), ("binned", "other_minute", "binned_nontie", 1.0, 2),
            ("binned", "same_minute", "binned_tie", 1.0, 2),
            ("cont", "all", "cond_type_ll", 1.0, 2), ("cont", "other_minute", "cond_type_ll", 1.0, 2),
            ("cont", "same_minute", "cond_type_ll", 1.0, 2),
            ("cont", "all", "cond_top1_accuracy", 100.0, 1), ("cont", "other_minute", "cond_top1_accuracy", 100.0, 1),
            ("cont", "same_minute", "cond_top1_accuracy", 100.0, 1)]
    lines = ["\\begin{table}[t]",
             "  \\caption{MIMIC-IV results split by whether the evaluated event was recorded in the same minute as the "
             "previous event (same, \\(41.6\\%\\) of the evaluated events) or not (other). "
             "Same checkpoints and test sequences as Table~\\ref{tab:full_mimic_results}. Binned: "
             "\\(\\log \\Pr(t \\in \\text{bin})\\) at one minute (Appendix~\\ref{app:binned}). " + COMMON.replace("per dataset", "per column") + "}",
             "  \\label{tab:mimic_ties}", "  \\centering", "  \\small", "  \\setlength{\\tabcolsep}{3pt}",
             "  \\resizebox{\\textwidth}{!}{%", "  \\begin{tabular}{l" + "c" * len(cols) + "}", "    \\toprule",
             "    & \\multicolumn{2}{c}{\\(\\log p(t)\\)} & \\multicolumn{3}{c}{Binned \\(\\log p(t)\\)} & "
             "\\multicolumn{3}{c}{\\(\\log p(d|t)\\)} & \\multicolumn{3}{c}{Acc\\(|t\\) (\\%)} \\\\",
             "    \\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}",
             "    Model & all & other & all & other & same & all & other & same & all & other & same \\\\", "    \\midrule"]
    bests = [best({m: v(m, *c[:4]) for m in order}, False, c[4]) for c in cols]
    prev = None
    for m in order:
        g = next(gn for gn, rows in GROUPS if m in {x for x, _ in rows})
        if prev is not None and g != prev:
            lines.append("    \\midrule")
        prev = g
        disp = dict(r for _, rows in GROUPS for r in rows)[m]
        cs = [cell(v(m, *c[:4]), c[4], bold=(m in bs[0]), under=(m in bs[1])) for c, bs in zip(cols, bests)]
        lines.append(f"    {disp} & " + " & ".join(cs) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table}"]
    write("app_mimic_ties.tex", "\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT, help="output directory for the .tex tables")
    OUT = ap.parse_args().out
    OUT.mkdir(parents=True, exist_ok=True)
    main_tables()
    standard_tables()
    mimic_table()
    iftpp_table()
    ksweep_table()
    ksweep_all_table()
    ns_table()
    tick_table()
    calibration_table()
    resolution_table()
    binned_ksweep_table()
    binned_main_table()
    mimic_ties_table()
