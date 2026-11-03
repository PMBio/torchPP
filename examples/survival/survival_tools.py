"""Shared helpers for the survival-analysis notebook series.

Survival analysis is a temporal point process with one terminal event per
subject: the outcome fires once (uncensored) or not before the end of the
observation window (right-censored).  TorchPP models that directly — a
terminating TPP whose intensity is the *hazard* and whose survival function
S(t) is exp of the negative cumulative intensity.

This module is model-agnostic.  The TPP model classes live in the notebooks;
every function here takes a fitted (or to-be-fitted) ``model`` and works with
any :class:`IntensityBasedTPP`.  What it provides:

* synthetic survival data with known ground truth (exponential / Weibull /
  spline baseline hazards, a nonlinear-covariate case, and competing risks);
* prediction helpers that turn a model into survival / hazard / cumulative-
  incidence curves;
* hand-rolled non-parametric baselines (Kaplan-Meier, Aalen-Johansen,
  Harrell's C) — no external survival library;
* a full-batch maximum-likelihood ``fit_full_batch`` loop (``torchpp.fit`` is the mini-batch trainer) and a set of plotting helpers.

Dependencies: ``torch``, ``numpy``, ``matplotlib`` (headless ``Agg``) only.

Conventions.  Static covariates for subject ``i`` are stored as a 1-D tensor
under the key ``"x"`` on each :class:`MVEventData`; batching stacks them to
``(N, n_features)``.  Times are kept O(1) (default ``time_scale=1.0``).  A
subject with no observed event is an *empty* sequence carrying only its
covariates and the window ``T_obs``.
"""

from typing import Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchpp import BatchedMVEventData, MVEventData
from torchpp.utils import LinearSpline

Meta = Dict[str, object]


# ---------------------------------------------------------------------------
# Data generators — each returns (list[MVEventData], meta_dict)
# ---------------------------------------------------------------------------
# All follow proportional hazards (PH): h(t | x) = h0(t) * exp(x . beta).  The
# meta dict always carries the covariates ``x``, the true (uncensored) event
# time, and an ``observed`` flag; generators add their ground-truth params.


def _build_survival_seqs(
    x: torch.Tensor, event_time: torch.Tensor, T_obs: float
) -> List[MVEventData]:
    """Single-terminal-event sequences, right-censored at ``T_obs``.

    An event past ``T_obs`` becomes an empty sequence (only covariates + window);
    otherwise one event of type 0 at its time.  Shared by the single-event
    generators.
    """
    seqs: List[MVEventData] = []
    for i in range(x.shape[0]):
        cov = {"x": x[i]}
        if float(event_time[i]) > T_obs:  # right-censored
            seqs.append(
                MVEventData(torch.zeros(0), torch.zeros(0, dtype=torch.long), covariates=cov, T_obs=float(T_obs))
            )
        else:  # observed terminal event
            seqs.append(
                MVEventData(
                    event_time[i : i + 1].to(torch.float32),
                    torch.zeros(1, dtype=torch.long),
                    covariates=cov,
                    T_obs=float(T_obs),
                )
            )
    return seqs


def make_exponential_ph(
    n: int, beta: torch.Tensor, h0: float, T_obs: float, seed: int = 0
) -> Tuple[List[MVEventData], Meta]:
    """Exponential proportional-hazards data: constant baseline hazard ``h0``.

    Draws ``x ~ N(0, I)`` and an event time ``T ~ Exp(h0 * exp(x . beta))``,
    right-censored at ``T_obs``.
    """
    g = torch.Generator().manual_seed(seed)
    beta = torch.as_tensor(beta, dtype=torch.float32)
    x = torch.randn(n, beta.shape[0], generator=g)
    rate = h0 * torch.exp(x @ beta)  # (n,)
    event_time = -torch.log(torch.rand(n, generator=g)) / rate
    seqs = _build_survival_seqs(x, event_time, T_obs)
    meta: Meta = {
        "x": x,
        "event_time": event_time,
        "observed": event_time <= T_obs,
        "beta": beta,
        "h0": float(h0),
    }
    return seqs, meta


def make_competing_risks(
    n: int, betas: torch.Tensor, h0: torch.Tensor, T_obs: float, seed: int = 0
) -> Tuple[List[MVEventData], Meta]:
    """Competing-risks data with ``D`` mutually-exclusive causes.

    For each cause ``d`` draw a latent time ``T_d ~ Exp(h0[d] * exp(x . betas[d]))``;
    the observed outcome is the earliest, ``event_time = min_d T_d`` with
    ``cause = argmin_d T_d``.  An observed subject's window is set to its own
    event time (``T_obs = event_time``) so that every cause's survival integral
    runs only up to the first event — the standard competing-risks likelihood.
    A subject whose earliest cause falls past ``T_obs`` is right-censored (empty
    sequence, window ``T_obs``).

    Args:
        betas: Per-cause coefficients. Shape (D, n_features).
        h0: Per-cause baseline hazards. Shape (D,).
    """
    g = torch.Generator().manual_seed(seed)
    betas = torch.as_tensor(betas, dtype=torch.float32)  # (D, nf)
    h0 = torch.as_tensor(h0, dtype=torch.float32)  # (D,)
    D, nf = betas.shape
    x = torch.randn(n, nf, generator=g)
    rates = h0.unsqueeze(0) * torch.exp(x @ betas.T)  # (n, D)
    T_all = -torch.log(torch.rand(n, D, generator=g)) / rates  # (n, D)
    event_time, cause = T_all.min(dim=1)  # (n,), (n,)
    observed = event_time <= T_obs

    seqs: List[MVEventData] = []
    for i in range(n):
        cov = {"x": x[i]}
        if bool(observed[i]):  # window = event time: all causes integrate to the first event
            seqs.append(
                MVEventData(
                    event_time[i : i + 1].to(torch.float32),
                    cause[i : i + 1].to(torch.long),
                    covariates=cov,
                    T_obs=float(event_time[i]),
                )
            )
        else:  # right-censored: no cause fired within [0, T_obs]
            seqs.append(
                MVEventData(torch.zeros(0), torch.zeros(0, dtype=torch.long), covariates=cov, T_obs=float(T_obs))
            )
    meta: Meta = {
        "x": x,
        "event_time": event_time,
        "cause": cause,
        "observed": observed,
        "betas": betas,
        "h0": h0,
    }
    return seqs, meta


def make_weibull_ph(
    n: int,
    beta: torch.Tensor,
    shape_k: float,
    scale_lambda: float,
    T_obs: float,
    seed: int = 0,
) -> Tuple[List[MVEventData], Meta]:
    """Weibull proportional-hazards data (time-varying baseline hazard).

    Baseline hazard ``h0(t) = (k / lambda) * (t / lambda)**(k - 1)`` with
    ``k = shape_k``, ``lambda = scale_lambda``; ``k > 1`` rises with time,
    ``k < 1`` falls.  Sampled by inverse transform,

        T = lambda * ((-log U) / exp(x . beta))**(1 / k),

    right-censored at ``T_obs``.  ``meta["h0_true"]`` is the closed-form baseline
    hazard as a callable.
    """
    g = torch.Generator().manual_seed(seed)
    beta = torch.as_tensor(beta, dtype=torch.float32)
    x = torch.randn(n, beta.shape[0], generator=g)
    lin = torch.exp(x @ beta)  # (n,)
    u = -torch.log(torch.rand(n, generator=g))
    event_time = scale_lambda * (u / lin) ** (1.0 / shape_k)
    seqs = _build_survival_seqs(x, event_time, T_obs)

    def h0_true(t) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=torch.float64)
        return (shape_k / scale_lambda) * (t / scale_lambda) ** (shape_k - 1.0)

    meta: Meta = {
        "x": x,
        "event_time": event_time,
        "observed": event_time <= T_obs,
        "beta": beta,
        "shape_k": float(shape_k),
        "scale_lambda": float(scale_lambda),
        "h0_true": h0_true,
    }
    return seqs, meta


def make_spline_hazard(
    n: int,
    beta: torch.Tensor,
    knot_locs: torch.Tensor,
    true_heights: torch.Tensor,
    T_obs: float,
    seed: int = 0,
) -> Tuple[List[MVEventData], Meta]:
    """Linear-spline baseline-hazard PH data (flexible, positive baseline).

    The baseline hazard is a linear spline through ``(knot_locs, true_heights)``;
    heights must be strictly positive and ``knot_locs`` must start at 0.  Sampled
    by inverting the cumulative baseline hazard:

        event_time = H0^{-1}( (-log U) / exp(x . beta) ),

    where ``H0^{-1}`` is :meth:`LinearSpline.inverse_integral`.  Right-censored at
    ``T_obs``.  ``meta["h0_true"]`` is the spline evaluated as a callable.
    """
    g = torch.Generator().manual_seed(seed)
    beta = torch.as_tensor(beta, dtype=torch.float32)
    knot_locs = torch.as_tensor(knot_locs, dtype=torch.float64)  # (K,)
    true_heights = torch.as_tensor(true_heights, dtype=torch.float64)  # (K,)
    assert (true_heights > 0).all(), "spline baseline-hazard heights must be strictly positive"
    assert float(knot_locs[0]) == 0.0, "knot_locs must start at 0 (spline undefined before its first knot)"

    x = torch.randn(n, beta.shape[0], generator=g)
    lin = torch.exp(x @ beta).to(torch.float64)  # (n,)
    # Target cumulative baseline hazard H0(T) = (-log U) / exp(x.beta), inverted.
    u = (-torch.log(torch.rand(n, generator=g)).to(torch.float64) / lin).unsqueeze(1)  # (n, 1)
    y_knots = true_heights.unsqueeze(0)  # (D=1, K)
    event_time = LinearSpline.inverse_integral(knot_locs, y_knots, u).squeeze(1).to(torch.float32)  # (n,)
    seqs = _build_survival_seqs(x, event_time, T_obs)

    def h0_true(t) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=torch.float64).reshape(-1)
        return LinearSpline.interpolate(knot_locs, y_knots, t).squeeze(1)

    meta: Meta = {
        "x": x,
        "event_time": event_time,
        "observed": event_time <= T_obs,
        "beta": beta,
        "knot_locs": knot_locs,
        "heights": true_heights,
        "h0_true": h0_true,
    }
    return seqs, meta


def make_nonlinear_covariate(
    n: int,
    n_features: int,
    effect_fn: Callable[[torch.Tensor], torch.Tensor],
    h0: float = 1.0,
    T_obs: float = 3.0,
    seed: int = 0,
) -> Tuple[List[MVEventData], Meta]:
    """Exponential-hazard data with a nonlinear covariate effect.

    The log-risk is ``g = effect_fn(x)`` (an arbitrary nonlinear function of the
    features, e.g. an interaction), giving hazard ``h = h0 * exp(g(x))`` and event
    time ``T ~ Exp(h)``, right-censored at ``T_obs``.  A linear Cox model cannot
    recover ``g``; a neural hazard can.  ``meta["g"]`` stores the true log-risk.

    Args:
        effect_fn: Maps covariates ``x`` of shape (n, n_features) to a log-risk
            vector of shape (n,).
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, n_features, generator=g)
    grisk = torch.as_tensor(effect_fn(x), dtype=torch.float32).reshape(-1)  # (n,)
    rate = h0 * torch.exp(grisk)
    event_time = -torch.log(torch.rand(n, generator=g)) / rate
    seqs = _build_survival_seqs(x, event_time, T_obs)
    meta: Meta = {"x": x, "event_time": event_time, "observed": event_time <= T_obs, "g": grisk, "h0": float(h0)}
    return seqs, meta


# ---------------------------------------------------------------------------
# Batch / prediction helpers
# ---------------------------------------------------------------------------


def _as_2d(x) -> torch.Tensor:
    """Covariates as a float ``(N, n_features)`` tensor (a single feature is a column)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    return x.unsqueeze(1) if x.dim() == 1 else x


def survival_batch(sequences: List[MVEventData]) -> BatchedMVEventData:
    """Collate survival sequences into one padded batch."""
    return BatchedMVEventData.from_sequences(sequences)


def covariate_batch(x, T_obs: Optional[float] = None) -> BatchedMVEventData:
    """Empty-history batch carrying only covariates — the predictive-query batch.

    Each subject becomes an empty sequence holding just its covariates ``x[i]``.
    Without ``T_obs`` the forecast starts at time 0, so the predictive methods
    (:meth:`survival_time`, :meth:`intensity`) give the survival / hazard curve from 0
    at any ``t >= 0``.  A ``T_obs`` makes them condition on no event up to ``T_obs``.

    Args:
        x: Covariates. Shape (N, n_features) (a 1-D input is treated as one feature).
        T_obs: Optional end of an event-free observation window.
    """
    x = _as_2d(x)
    seqs = [
        MVEventData(
            torch.zeros(0),
            torch.zeros(0, dtype=torch.long),
            covariates={"x": x[i]},
            T_obs=None if T_obs is None else float(T_obs),
        )
        for i in range(x.shape[0])
    ]
    return BatchedMVEventData.from_sequences(seqs)


def _query_grid(model, x, t_grid) -> Tuple[BatchedMVEventData, torch.Tensor, torch.Tensor]:
    """Build the covariate batch and the ``(N, G)`` query-time tensor for a shared grid."""
    x = _as_2d(x)
    t_grid = torch.as_tensor(t_grid, dtype=torch.float64)
    batch = covariate_batch(x)
    t = t_grid.unsqueeze(0).expand(x.shape[0], -1)  # (N, G)
    return batch, t, t_grid


def model_survival_curve(model, x, t_grid) -> torch.Tensor:
    """Predicted survival S(t) for each subject over a shared grid.  Shape (N, G).

    ``survival_time`` returns the LOG survival by default (``S(t) = exp(-Lambda)``);
    we exponentiate to probabilities.
    """
    batch, t, _ = _query_grid(model, x, t_grid)
    with torch.no_grad():
        log_S = model.survival_time(t, batch, log=True)  # (N, G)
    return log_S.exp()


def model_hazard_curve(model, x, t_grid) -> torch.Tensor:
    """Predicted hazard (intensity) over a shared grid.

    Shape (N, G) for a single event type, else (N, G, D).
    """
    batch, t, _ = _query_grid(model, x, t_grid)
    with torch.no_grad():
        lam = model.intensity(t, batch)  # (N, G, D)
    return lam.squeeze(-1) if lam.shape[-1] == 1 else lam


def cumulative_incidence(model, x, t_grid) -> torch.Tensor:
    """Competing-risks cumulative incidence CIF_d(t) = integral_0^t S(u) h_d(u) du.

    ``S`` is the all-cause survival (over every event type); ``h_d`` is the cause-``d``
    hazard.  Integrated by the trapezoid rule over ``t_grid``.  Shape (N, D, G).
    """
    batch, t, t_grid = _query_grid(model, x, t_grid)
    with torch.no_grad():
        S = model.survival_time(t, batch, log=True).exp()  # (N, G)
        lam = model.intensity(t, batch)  # (N, G, D)
    integrand = S.unsqueeze(-1) * lam  # (N, G, D)
    cif = _cumtrapz_time(integrand, t_grid)  # (N, G, D)
    return cif.permute(0, 2, 1).contiguous()  # (N, D, G)


def _cumtrapz_time(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Cumulative trapezoid integral of ``y`` (N, G, D) along the grid ``t`` (G,).

    Returns (N, G, D) with a leading 0 (the integral over ``[t_0, t_0]``).
    """
    dt = (t[1:] - t[:-1]).to(y.dtype)  # (G-1,)
    incr = 0.5 * (y[:, 1:, :] + y[:, :-1, :]) * dt.view(1, -1, 1)  # (N, G-1, D)
    cum = torch.cumsum(incr, dim=1)  # (N, G-1, D)
    zero = torch.zeros(y.shape[0], 1, y.shape[2], dtype=y.dtype, device=y.device)
    return torch.cat([zero, cum], dim=1)  # (N, G, D)


# ---------------------------------------------------------------------------
# Non-parametric estimators (hand-rolled, numpy)
# ---------------------------------------------------------------------------


def kaplan_meier(event_times, observed) -> Tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier product-limit survival estimate.

    Args:
        event_times: Observed time per subject, min(true time, censoring time). (N,)
        observed: 1 if the event was observed, 0 if right-censored. (N,)

    Returns:
        ``(t_steps, S_hat)`` — step times (0 prepended) and survival at each step.
    """
    et = np.asarray(event_times, dtype=float)
    ev = np.asarray(observed).astype(bool)
    death_times = np.unique(et[ev])  # distinct times where an event occurs
    t_steps, S_hat, S = [0.0], [1.0], 1.0
    for tj in death_times:
        d = int(np.sum((et == tj) & ev))  # events at tj
        at_risk = int(np.sum(et >= tj))  # subjects still at risk just before tj
        S *= 1.0 - d / at_risk
        t_steps.append(float(tj))
        S_hat.append(S)
    return np.array(t_steps), np.array(S_hat)


def aalen_johansen(event_times, causes, observed, n_causes) -> Tuple[np.ndarray, np.ndarray]:
    """Aalen-Johansen non-parametric cumulative incidence, per cause.

    CIF_d(t) = sum_{t_j <= t} S(t_{j-1}) * (d_{d,j} / n_j), where ``S`` is the
    all-cause Kaplan-Meier survival, ``d_{d,j}`` the cause-``d`` events at ``t_j``,
    and ``n_j`` the number at risk.

    Args:
        event_times: Observed time per subject. (N,)
        causes: Cause index per subject (ignored where censored). (N,)
        observed: 1 if any event observed, 0 if censored. (N,)
        n_causes: Number of competing causes D.

    Returns:
        ``(t_steps, CIF)`` — step times (0 prepended) and ``CIF`` of shape (D, T_steps).
    """
    et = np.asarray(event_times, dtype=float)
    ev = np.asarray(observed).astype(bool)
    cz = np.asarray(causes)
    death_times = np.unique(et[ev])
    t_steps = [0.0]
    cif_rows = [np.zeros(n_causes)]
    cur = np.zeros(n_causes)
    S_prev = 1.0  # all-cause survival just before the current time
    for tj in death_times:
        at_risk = int(np.sum(et >= tj))
        d_total = int(np.sum((et == tj) & ev))
        for d in range(n_causes):
            d_dj = int(np.sum((et == tj) & ev & (cz == d)))
            cur[d] += S_prev * d_dj / at_risk
        t_steps.append(float(tj))
        cif_rows.append(cur.copy())
        S_prev *= 1.0 - d_total / at_risk
    return np.array(t_steps), np.array(cif_rows).T  # (D, T_steps)


def concordance_index(risk_scores, event_times, observed) -> float:
    """Harrell's concordance index (C-index) over comparable pairs.

    A pair ``(i, j)`` is comparable when subject ``i`` has an observed event before
    ``event_times[j]``.  It is concordant when the higher risk score goes with the
    shorter time (``risk_scores[i] > risk_scores[j]``); tied risks count 0.5.
    ``C = 0.5`` is random, ``C = 1.0`` perfect.

    Args:
        risk_scores: Higher = higher predicted risk (shorter expected time). (N,)
        event_times: Observed time per subject. (N,)
        observed: 1 if the event was observed, 0 if censored. (N,)
    """
    r = np.asarray(risk_scores, dtype=float)
    et = np.asarray(event_times, dtype=float)
    ev = np.asarray(observed).astype(bool)
    concordant, comparable = 0.0, 0.0
    for i in range(len(r)):
        if not ev[i]:
            continue
        later = et > et[i]  # subjects who outlived i's event -> comparable with i
        comparable += int(np.sum(later))
        concordant += float(np.sum(r[i] > r[later])) + 0.5 * float(np.sum(r[i] == r[later]))
    return concordant / comparable if comparable > 0 else float("nan")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def fit_full_batch(
    model,
    batch: BatchedMVEventData,
    steps: int = 800,
    lr: float = 0.05,
    normalize_by_events: bool = True,
    log_every: int = 200,
    val_batch: Optional[BatchedMVEventData] = None,
) -> Dict[str, object]:
    """Full-batch Adam maximum-likelihood fit.

    Minimises ``model.loss(batch).sum() / n_events + model.regularization(batch)``,
    where ``n_events`` is the number of real (uncensored) events; censored subjects
    still contribute their survival term through ``loss``.

    Args:
        model: Any :class:`IntensityBasedTPP` (or TPP with ``loss`` / ``likelihood``).
        batch: Training sequences, already collated.
        steps: Number of Adam steps.
        lr: Learning rate.
        normalize_by_events: Divide the summed loss by the event count (per-event NLL).
        log_every: Print every this many steps (0 disables printing).
        val_batch: Optional held-out batch; its per-event log-likelihood is returned.

    Returns:
        ``{"losses": [...], "val_loglik": float | None}``.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_events = batch.real_counts.sum()
    denom = n_events if (normalize_by_events and n_events > 0) else 1.0

    losses: List[float] = []
    for step in range(steps):
        opt.zero_grad()
        loss = model.loss(batch).sum() / denom + model.regularization(batch)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if log_every and step % log_every == 0:
            print(f"step {step:4d}  loss = {losses[-1]:.4f}")

    val_loglik: Optional[float] = None
    if val_batch is not None:
        with torch.no_grad():
            n_val = val_batch.real_counts.sum()
            total = model.likelihood(val_batch).sum()
            val_loglik = float(total / n_val) if n_val > 0 else float(total)
    return {"losses": losses, "val_loglik": val_loglik}


# ---------------------------------------------------------------------------
# Plotting (matplotlib, Agg backend)
# ---------------------------------------------------------------------------


def _get_ax(ax: Optional[plt.Axes]) -> Tuple[plt.Figure, plt.Axes]:
    """Return ``(fig, ax)``, creating a figure when ``ax`` is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    return fig, ax


def plot_survival_km_vs_model(
    t_km, S_km, t_grid, S_model, ax: Optional[plt.Axes] = None, labels: Optional[List[str]] = None
) -> plt.Figure:
    """Kaplan-Meier step curve(s) against model survival curve(s).

    Single stratum: ``S_model`` is 1-D (G,), ``t_km`` / ``S_km`` are the KM arrays.
    Multiple strata: ``S_model`` is 2-D (n_strata, G), ``t_km`` / ``S_km`` are lists
    of KM arrays (one per stratum).  Matched colours; KM dashed, model solid.
    """
    fig, ax = _get_ax(ax)
    t_grid = np.asarray(t_grid, dtype=float)
    S_model = np.asarray(S_model, dtype=float)
    if S_model.ndim == 1:
        S_model = S_model[None, :]
        km_list = [(np.asarray(t_km, float), np.asarray(S_km, float))]
    else:
        km_list = [(np.asarray(tk, float), np.asarray(sk, float)) for tk, sk in zip(t_km, S_km)]
    n_strata = S_model.shape[0]
    labels = labels or [f"stratum {i}" for i in range(n_strata)]
    cmap = plt.get_cmap("tab10")
    for i in range(n_strata):
        c = cmap(i % 10)
        tk, sk = km_list[i] if i < len(km_list) else km_list[0]
        ax.step(tk, sk, where="post", color=c, linestyle="--", alpha=0.8, label=f"{labels[i]} (KM)")
        ax.plot(t_grid, S_model[i], color=c, label=f"{labels[i]} (model)")
    ax.set_xlabel("time")
    ax.set_ylabel("survival S(t)")
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=8)
    return fig


def plot_hazard(t_grid, h_model, h_true=None, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Learned baseline hazard vs the true hazard.

    ``h_true`` may be an array over ``t_grid`` or a callable ``h0(t)``.
    """
    fig, ax = _get_ax(ax)
    t = np.asarray(t_grid, dtype=float)
    ax.plot(t, np.asarray(h_model, dtype=float).reshape(-1), label="model h0(t)")
    if h_true is not None:
        ht = h_true(t_grid) if callable(h_true) else h_true
        ax.plot(t, np.asarray(ht, dtype=float).reshape(-1), "k--", label="true h0(t)")
    ax.set_xlabel("time")
    ax.set_ylabel("baseline hazard")
    ax.legend(fontsize=8)
    return fig


def plot_cumulative_incidence(
    t_grid, cif_model, t_aj=None, cif_aj=None, ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Per-cause cumulative incidence: model curves vs Aalen-Johansen steps.

    ``cif_model`` is (D, G) for one representative subject / stratum; ``cif_aj`` is
    (D, T_steps) at times ``t_aj`` (both optional).
    """
    fig, ax = _get_ax(ax)
    t = np.asarray(t_grid, dtype=float)
    cif_model = np.asarray(cif_model, dtype=float)
    if cif_model.ndim == 1:
        cif_model = cif_model[None, :]
    cmap = plt.get_cmap("tab10")
    for d in range(cif_model.shape[0]):
        c = cmap(d % 10)
        ax.plot(t, cif_model[d], color=c, label=f"cause {d} (model)")
        if cif_aj is not None and t_aj is not None:
            ax.step(
                np.asarray(t_aj, float),
                np.asarray(cif_aj, float)[d],
                where="post",
                color=c,
                linestyle="--",
                alpha=0.8,
                label=f"cause {d} (A-J)",
            )
    ax.set_xlabel("time")
    ax.set_ylabel("cumulative incidence")
    ax.legend(fontsize=8)
    return fig


def plot_calibration(pred_S_at_t, observed, t, n_bins: int = 10, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Reliability diagram: predicted vs observed survival at a horizon ``t``.

    Subjects are binned by their predicted ``S(t)``; each bin plots mean predicted
    against the observed survival fraction.  ``observed[i] = 1`` if subject ``i`` was
    still event-free at ``t`` (survived past it), else 0.  A well-calibrated model
    sits on the diagonal.
    """
    fig, ax = _get_ax(ax)
    p = np.asarray(pred_S_at_t, dtype=float).reshape(-1)
    obs = np.asarray(observed, dtype=float).reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = which == b
        if m.sum() == 0:
            continue
        xs.append(p[m].mean())
        ys.append(obs[m].mean())
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="ideal")
    ax.plot(xs, ys, "o-", label=f"binned (t={t})")
    ax.set_xlabel("predicted S(t)")
    ax.set_ylabel("observed survival")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8)
    return fig


def plot_coefficients(beta_true, beta_hat, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Recovered vs true coefficients as grouped bars (hazard ratio = exp(beta))."""
    fig, ax = _get_ax(ax)
    bt = np.asarray(beta_true, dtype=float).reshape(-1)
    bh = np.asarray(beta_hat, dtype=float).reshape(-1)
    idx = np.arange(len(bt))
    w = 0.38
    ax.bar(idx - w / 2, bt, w, label="true")
    ax.bar(idx + w / 2, bh, w, label="recovered")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"beta{i}" for i in idx])
    ax.set_ylabel("coefficient")
    ax.legend(fontsize=8)
    return fig
