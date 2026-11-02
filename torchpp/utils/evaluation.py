"""
Evaluation utilities for temporal point process models.

This module provides functions for evaluating TPP models on test sequences,
computing various metrics like likelihoods, accuracy, and time prediction errors.
"""

import warnings
from dataclasses import replace
from typing import Dict, List, Union, Any, Optional
import numpy as np
import torch
from tqdm import tqdm

from ..tpp.base_tpp import TemporalPointProcess, IntensityBasedTPP, DensityBasedTPP
from .event_data import MVEventData, BatchedMVEventData
from ..data.base_dataset import TPPDataset
from .math import IntensityGrid


# Numerical constants (value-preserving names for previously-inlined literals).
_GRID_POINTS_PER_UNIT_TIME = 200.0  # marginal / mean-time integration grid density
_EVAL_SNAP_ATOL = 1e-8  # snap a query time within this of t_last onto t_last
_COND_TYPE_INTENSITY_FLOOR = 1e-12  # floor on the conditional-type-distribution divisor


def evaluate_at_time(
    model: IntensityBasedTPP,
    T: torch.Tensor,
    ts: BatchedMVEventData,
    ignore_indices: Optional[List[int]] = None,
    compute_marginals: bool = True,
    integrate_marginal_via_grid: bool = True,
    grid_points_per_unit_time: float = _GRID_POINTS_PER_UNIT_TIME,
) -> dict:
    """
    Evaluate a TPP model at specific query times ``T``.

    Computes intensity, cumulative intensity, PDF, survival, CDF, and
    conditional type distribution at ``T`` given history ``ts``.
    When ``compute_marginals=True``, also computes grid-based quantities:
    mean and median next-event time and marginal type distribution
    :math:`p(m | H)` (integrated over future time).

    Neural models that implement ``prepare_batch`` / ``clear_batch`` for
    lazy parameter caching are handled transparently via duck-typing:
    the cache is populated before each computation phase and cleared
    afterwards.

    Args:
        model: The TPP model to query.
        T: Query times. Shape ``(B,)``; must satisfy ``T >= ts.max_time``.
        ts: Batched, padded event history. Shape ``(B, N)``.
        ignore_indices: Event-type indices to treat as absent. Their
            intensities and cumulative intensities are zeroed before all
            derived quantities are computed, so survival, PDF,
            ``cond_type_dist``, and all grid-based quantities reflect only
            the remaining types. ``None`` or empty list → no masking.
        compute_marginals: If ``True``, also builds an integration grid and
            returns ``mean_time``, ``median_time``, and ``marginal_type_dist``.
            This is more expensive (requires a full grid build), so set
            ``False`` when only point-wise quantities are needed.
        integrate_marginal_via_grid: If ``True``, the integration grid uses
            trapezoidal prefix sums for cumulative intensity instead of
            calling the analytical CI function at every grid point.  This
            roughly halves grid-building cost for analytical-CI models
            (e.g. Hawkes) with negligible accuracy loss at ≥200 grid points.
            Only affects the grid; point-wise CI at ``T`` still uses the
            model’s native (analytical or numerical) path.

    Returns:
        Dictionary with keys:

        - ``"intensity"``: :math:`\\lambda(T)`, zeroed at ignored types. Shape ``(B, D)``.
        - ``"ci"``: :math:`\\Lambda(T) - \\Lambda(t_{\\text{last}})`, zeroed at ignored types. Shape ``(B, D)``.
        - ``"pdf"``: :math:`p(T, m | H)` for valid types. Shape ``(B, D)``.
        - ``"survival"``: :math:`S(T)` based on valid types only. Shape ``(B, 1)``.
        - ``"cdf"``: :math:`P(t < T | H)` based on valid types only. Shape ``(B,)``.
        - ``"cond_type_dist"``: :math:`p(m | T, H)` over valid types. Shape ``(B, D)``.

        If ``compute_marginals=True``, additionally:

        - ``"mean_time"``: :math:`\\mathbb{E}[t_{\\text{next}} | H]` from the grid. Shape ``(B,)``.
        - ``"median_time"``: Median of :math:`p(t_{\\text{next}} | H)` from the grid. Shape ``(B,)``.
        - ``"marginal_type_dist"``: :math:`p(m | H)` marginalised over time. Shape ``(B, D)``.
    """
    T = T.double()
    close_mask = torch.abs(T - ts.max_time) < _EVAL_SNAP_ATOL
    T = torch.where(close_mask, ts.max_time, T)

    assert (T >= ts.max_time).all(), "evaluate_at_time is only defined for T >= last event time."

    # --- Point evaluation (intensity + CI) ---
    # Neural models cache transformer outputs per batch; handle via duck-typing.
    prepare = getattr(model, "prepare_batch", None)
    clear = getattr(model, "clear_batch", None)
    if prepare is not None:
        prepare(ts)
    try:
        intensity = model.intensity(T, ts)  # (B, D)
        # Window CI over [t_last, T] with GLOBAL placement, via the CIConfig dispatcher.
        # The [t_last, T] window has no interior events, so this is the single-interval
        # whole-window quadrature the metrics have always used; forcing placement="global"
        # keeps interval-placement models (NHP/AttNHP/…) on that exact scheme (closed-form
        # models ignore placement).  Replaces the old ``use_analytical_ci`` mode-branch and
        # the direct ``_numerical_cumulative_intensity`` call.
        ci = model.cumulative_intensity(
            T, ts, T_low=ts.max_time, ci_config=replace(model.ci_config, placement="global")
        )  # (B, D)
    finally:
        if clear is not None:
            clear()

    # --- Apply ignore_indices mask ---
    # Setting λ_k = 0 and Λ_k = 0 is equivalent to removing those types
    # from the process, so all derived quantities reflect the restricted sub-process.
    if ignore_indices:
        valid_mask = torch.ones(1, intensity.shape[1], dtype=intensity.dtype, device=intensity.device)
        valid_mask[0, ignore_indices] = 0.0
        intensity = intensity * valid_mask
        ci = ci * valid_mask

    # --- Derived point quantities ---
    total_ci = ci.sum(dim=1, keepdim=True)  # (B, 1)
    survival = torch.exp(-total_ci)  # (B, 1)
    pdf = intensity * survival  # (B, D)
    cdf = 1 - survival.squeeze(1)  # (B,)
    total_intensity = intensity.sum(dim=1, keepdim=True).clamp(min=_COND_TYPE_INTENSITY_FLOOR)  # (B, 1)
    cond_type_dist = intensity / total_intensity  # (B, D)

    result = {
        "intensity": intensity,
        "ci": ci,
        "pdf": pdf,
        "survival": survival,
        "cdf": cdf,
        "cond_type_dist": cond_type_dist,
    }

    if not compute_marginals:
        return result

    # --- Grid-based marginal quantities ---
    # model.build_integration_grid handles its own prepare_batch via the
    # neural override, so no extra wrapping is needed here.
    grid: IntensityGrid = model.build_integration_grid(
        ts, integrate_marginal_via_grid=integrate_marginal_via_grid, points_per_unit_time=grid_points_per_unit_time
    )

    if ignore_indices:
        D = intensity.shape[1]
        grid_mask = torch.ones(1, 1, D, dtype=grid.intensities.dtype, device=grid.intensities.device)
        grid_mask[0, 0, ignore_indices] = 0.0
        grid.intensities = grid.intensities * grid_mask  # (B, num_points, D)
        grid.ci_prefix = grid.ci_prefix * grid_mask  # (B, num_points, D)

    B = T.shape[0]
    result["mean_time"] = model._mean_next_event_time_from_grid(grid)  # (B,)
    result["median_time"] = model._inverse_cdf_from_grid(
        grid, torch.tensor(0.5, device=T.device, dtype=T.dtype).expand(B)
    )  # (B,)
    result["marginal_type_dist"] = model._marginal_class_distribution_from_grid(grid)  # (B, D)

    return result


_ITEM_KEYS = (
    "seq_idx",
    "history_length",
    "target_type",
    "gap",
    "time_ll_scaled",
    "joint_ll_scaled",
    "cond_type_ll",
    "cond_pred_top1",
    "cond_top1_correct",
)


def _build_prediction_items(
    sequences: List[MVEventData],
    ignore_indices: Optional[List[int]] = None,
) -> List[dict]:
    """
    Flatten all sequences into a list of individual prediction items.

    Each item contains:
      - history: MVEventData of events before the prediction target
      - target_time: scalar tensor
      - target_type: scalar tensor
      - valid: whether the target type should be included in evaluation

    Args:
        sequences: List of event sequences to flatten.
        ignore_indices: Event-type indices to exclude from evaluation
            (marked ``valid=False``).  ``None`` or empty list → all types
            are included.

    Returns:
        List of prediction item dicts, sorted by descending history length
        for efficient batching (longest first → minimizes padding).
    """
    _ignore = set(ignore_indices) if ignore_indices else set()
    items = []
    for seq_idx, ts_all in enumerate(sequences):
        if len(ts_all) < 2:
            continue
        for idx in range(1, len(ts_all)):
            target_type = ts_all.event_types[idx]
            items.append(
                {
                    "history": MVEventData(
                        ts_all.time_points[:idx],
                        ts_all.event_types[:idx],
                        covariates=getattr(ts_all, "covariates", None),
                    ),
                    "target_time": ts_all.time_points[idx],
                    "target_type": target_type,
                    # int() is load-bearing: target_type is a 0-dim tensor, and
                    # tensors hash by identity, so `tensor in set_of_ints` is
                    # always False — without the cast nothing is ever dropped.
                    "valid": bool(int(target_type) not in _ignore),
                    "history_length": idx,
                    "seq_idx": seq_idx,
                }
            )

    # Sort by descending history length to minimize padding waste in batches.
    items.sort(key=lambda x: -x["history_length"])
    return items


def evaluate_tpp(
    model: TemporalPointProcess,
    sequences: Union[TPPDataset, List[MVEventData]],
    num_event_types: int,
    device: str = "cpu",
    ignore_indices: Optional[List[int]] = None,
    eval_batch_size: Optional[int] = None,
    time_scale_factor: Optional[float] = None,
    compute_marginals: bool = True,
    integrate_marginal_via_grid: bool = True,
    grid_points_per_unit_time: float = _GRID_POINTS_PER_UNIT_TIME,
    return_items: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a temporal point process model on sequences.

    This function computes comprehensive metrics for evaluating TPP models:
    - Joint log-likelihood p(t, m | H)
    - Marginal time log-likelihood p(t | H)
    - Conditional type log-likelihood p(m | t, H)
    - Marginal type log-likelihood p(m | H) (optional, expensive)
    - Top-1 and Top-5 accuracy (marginal, from p(m|H))
    - Conditional Top-1 and Top-5 accuracy (from p(m|t,H))
    - Time prediction MAE and RMSE (mean and median of time distribution vs actual)

    Performance:
        Prediction items from all sequences are pooled, sorted by history length,
        and evaluated in batches. This dramatically improves throughput compared to
        processing one sequence at a time (especially for numerical CI models),
        while also minimizing padding waste via length-sorted batching.

    Note on time units:
        TPP models can operate in a scaled time where 1.0 corresponds to
        ``time_scale_factor`` time units.  Time-related metrics are converted back to
        time units using this factor.  When a :class:`~torchpp.data.base_dataset.TPPDataset`
        is passed as ``sequences`` its ``time_scale_factor`` attribute is used
        automatically; otherwise supply ``time_scale_factor`` explicitly.

    Args:
        model: The TPP model to evaluate.
        sequences: Either a :class:`~torchpp.data.base_dataset.TPPDataset` or a
            plain :class:`list` of :class:`~torchpp.utils.event_data.MVEventData`
            sequences.  When a ``TPPDataset`` is provided its
            ``time_scale_factor`` attribute is used unless overridden.
        num_event_types: Number of event types (D).
        device: Device to run evaluation on.
        ignore_indices: Event-type indices to exclude from evaluation metrics.
            ``None`` or empty list includes all event types.
        eval_batch_size: Number of prediction items per evaluation batch.
            Larger = faster but more memory. None = auto-select based on
            median history length and num_event_types.
        time_scale_factor: Conversion factor from the model's scaled time unit
            to days (``t_days = t_scaled * time_scale_factor``).  If ``None``,
            the value is taken from the dataset's ``time_scale_factor`` attribute
            when a :class:`~torchpp.data.base_dataset.TPPDataset` is passed.

    Returns:
        Dictionary with evaluation metrics (all values are floats):
            - marginal_time_ll: Marginal time log-likelihood (in days)
            - cond_type_ll: Conditional type log-likelihood p(m|t,H)
            - marginal_type_ll: Marginal type log-likelihood p(m|H) (if computed)
            - joint_ll: Joint log-likelihood (in days)
            - top1_accuracy: Top-1 type prediction accuracy (marginal p(m|H))
            - top5_accuracy: Top-5 type prediction accuracy (marginal p(m|H))
            - cond_top1_accuracy: Top-1 type prediction accuracy (conditional p(m|t,H))
            - cond_top5_accuracy: Top-5 type prediction accuracy (conditional p(m|t,H))
            - time_mae_mean_days: Time prediction MAE in days (mean of distribution vs actual)
            - time_mae_median_days: Time prediction MAE in days (median of distribution vs actual)
            - time_rmse_mean_days: Time prediction RMSE in days (mean of distribution vs actual)
            - time_rmse_median_days: Time prediction RMSE in days (median of distribution vs actual)
            - num_predictions: Total number of predictions made
        If ``return_items=True``, ``metrics["items"]`` additionally holds per-item numpy
        arrays (scaled-time log-likelihoods, conditional top-1 hit, inter-event gap, ...).
    """
    if isinstance(sequences, TPPDataset):
        if time_scale_factor is None:
            time_scale_factor = sequences.time_scale_factor
        sequences = list(sequences)
    else:
        if time_scale_factor is None:
            raise ValueError("time_scale_factor must be provided when sequences is not a TPPDataset")
    log_time_scale = np.log(time_scale_factor)

    model.eval()
    DEVICE = torch.device(device)
    model = model.to(DEVICE)

    # =========================================================================
    # Step 1: Flatten all sequences into individual prediction items
    # =========================================================================
    all_items = _build_prediction_items(sequences, ignore_indices)

    if len(all_items) == 0:
        return {"error": "No valid predictions made"}

    # Drop items whose target type is in ignore_indices to avoid computing on
    # tokens that are excluded from evaluation (e.g. lifestyle tokens).
    valid_items = [item for item in all_items if item["valid"]]
    items_to_eval = valid_items

    # In the standard benchmarks ignored types are stripped from the sequences
    # by the data loaders, so nothing is dropped here. If ignored-type events
    # DO appear interleaved with valid ones, the current scoring has a known
    # incoherence: the survival mass of windows ending in an ignored event is
    # never charged (the dropped item carried it), so the time-axis LL is not
    # the likelihood of a well-defined process. Warn loudly rather than score
    # silently.
    n_dropped = len(all_items) - len(valid_items)
    if n_dropped > 0:
        warnings.warn(
            f"evaluate_tpp dropped {n_dropped}/{len(all_items)} prediction items whose target type is in "
            f"ignore_indices. Ignored-type events are present in the sequences, so the survival factor of "
            f"windows preceding them is unscored and the reported time/joint LL is biased. Strip ignored "
            f"types from the sequences (as the standard loaders do) or re-target to the next valid event.",
            stacklevel=2,
        )

    if len(items_to_eval) == 0:
        return {"error": "No valid predictions made"}

    total_predictions = len(items_to_eval)

    # =========================================================================
    # Step 2: Determine batch size
    # =========================================================================
    if eval_batch_size is None:
        median_hist_len = np.median([item["history_length"] for item in items_to_eval])
        # Conservative heuristic targeting ~10K total events per batch.
        target_total_events = 10_000
        eval_batch_size = max(4, int(target_total_events / max(median_hist_len, 1)))
        eval_batch_size = min(eval_batch_size, 512)

    # =========================================================================
    # Step 3: Evaluate in batches
    # =========================================================================
    # Log-likelihoods are accumulated in log space (log λ − Λ) rather than as
    # densities (λ·exp(−Λ)): the survival factor exp(−Λ) underflows to 0 for
    # Λ ≳ 88 (float32) / 745 (float64), which would silently floor the metric.
    # In log space any representable Λ yields the exact log-density; only a
    # literally-zero intensity needs a floor (finfo.tiny, counted below).
    joint_log_likelihood = []
    time_log_likelihood = []
    cond_type_log_likelihood = []
    marginal_type_likelihood = []
    num_floored_intensity = 0  # events where the model assigned λ_target < tiny
    top1_correct = []
    top5_correct = []
    cond_top1_correct = []
    cond_top5_correct = []
    time_mean_abs_errors = []
    time_median_abs_errors = []
    time_mean_sq_errors = []
    time_median_sq_errors = []

    num_batches = (len(items_to_eval) + eval_batch_size - 1) // eval_batch_size
    item_records: Dict[str, list] = {k: [] for k in _ITEM_KEYS} if return_items else {}

    # Density models (DensityBasedTPP) realize the family metrics natively from the
    # inter-event density + mark head (no intensity / CI); the intensity-family path
    # below (evaluate_at_time + eval CIConfig) does not apply to them.
    is_density = isinstance(model, DensityBasedTPP)

    # Deterministic CI for evaluation (intensity family only): force trapezoidal
    # quadrature so metrics do not depend on the mc_trapezoidal draw (the stochastic
    # training default that guards against node-gaming), and global placement so the
    # [t_last, T] window is integrated as one whole-window quadrature (the scheme the
    # metrics have always used — and what the contract methods' CI now routes through).
    # Closed-form models keep mode="closed_form" (rule/placement ignored) and stay exact.
    eval_ci_config = None if is_density else replace(model.ci_config, rule="trapezoidal", placement="global")

    for batch_start in tqdm(range(0, len(items_to_eval), eval_batch_size), total=num_batches, desc="Evaluating"):
        batch_items = items_to_eval[batch_start : batch_start + eval_batch_size]
        B = len(batch_items)

        histories = [item["history"] for item in batch_items]
        batch = BatchedMVEventData(mv_events=histories)
        target_time = torch.tensor([item["target_time"] for item in batch_items], dtype=torch.float64, device=DEVICE)
        target_type = torch.tensor([item["target_type"] for item in batch_items], dtype=torch.long, device=DEVICE)
        batch = batch.to(DEVICE)

        batch_indices = torch.arange(B, device=DEVICE)

        if is_density:
            # --- Density family: everything via the predictive contract (no intensity / CI) ---
            with torch.no_grad():
                prepare = getattr(model, "prepare_batch", None)
                if prepare is not None:
                    prepare(batch)  # cache the encoder pass for the whole batch's queries
                try:
                    log_joint = model.density_joint(target_time, batch, log=True)  # (B, D)
                    log_pm = model.prob_mark(batch, log=True)  # (B, D) = p(m | t) too (mark ⟂ time)
                    if compute_marginals:
                        mean_time_pred = model.time_mean(batch)  # (B,)
                        median_time_pred = model.time_quantile(0.5, batch)  # (B,)
                finally:
                    clear = getattr(model, "clear_batch", None)
                    if clear is not None:
                        clear()

                if ignore_indices:
                    # Restrict to retained types: mask their mark columns and renormalise.
                    neg_inf = torch.finfo(log_joint.dtype).min
                    keep = torch.ones(1, log_joint.shape[1], dtype=torch.bool, device=DEVICE)
                    keep[0, ignore_indices] = False
                    log_joint = log_joint.masked_fill(~keep, neg_inf)
                    log_pm = log_pm.masked_fill(~keep, neg_inf)
                    log_pm = log_pm - torch.logsumexp(log_pm, dim=1, keepdim=True)

                log_time = torch.logsumexp(log_joint, dim=1)  # (B,) = density_time(log=True)
                batch_joint_ll = log_joint[batch_indices, target_type]
                batch_time_ll = log_time
                batch_cond_ll = batch_joint_ll - log_time  # = prob_mark_given_time(log=True)[target]

                # mark ⟂ time -> the conditional p(m|t) equals the marginal p(m|H).
                type_distribution_at_t = log_pm.exp()  # (B, D)
                marginal_type_probs = type_distribution_at_t
        else:
            with torch.no_grad(), model.use_ci(eval_ci_config):
                # Single call returns point-wise quantities and grid-based marginals.
                eval_result = evaluate_at_time(
                    model,
                    target_time,
                    batch,
                    ignore_indices=ignore_indices,
                    compute_marginals=compute_marginals,
                    integrate_marginal_via_grid=integrate_marginal_via_grid,
                    grid_points_per_unit_time=grid_points_per_unit_time,
                )
                type_intensity_at_t = eval_result["intensity"]  # (B, D) — ignored cols = 0
                type_distribution_at_t = eval_result["cond_type_dist"]  # (B, D) — for conditional accuracy

                tiny = torch.finfo(type_intensity_at_t.dtype).tiny

                # Diagnostic: events the model assigned below-tiny intensity (intensity-specific).
                num_floored_intensity += int((type_intensity_at_t[batch_indices, target_type] < tiny).sum())

                # --- Universal LLs via the predictive contract (family-agnostic) ---
                # density_joint(log=True) = log λ_d(T) − Λ_total(T); the time-marginal and
                # conditional-mark LLs follow by the exact mark-axis identities (logsumexp /
                # subtraction).  ``ignore_indices`` restricts the process to the retained types
                # — an intensity-specific operation — so that path reduces the ignore-masked
                # intensity/CI from ``evaluate_at_time`` instead of the full-support contract.
                if ignore_indices:
                    total_ci = eval_result["ci"].sum(dim=1)  # (B,) over retained types
                    log_target = torch.log(type_intensity_at_t[batch_indices, target_type].clamp(min=tiny))
                    log_total = torch.log(type_intensity_at_t.sum(dim=1).clamp(min=tiny))
                    batch_joint_ll = log_target - total_ci
                    batch_time_ll = log_total - total_ci
                    batch_cond_ll = log_target - log_total
                else:
                    # density_joint(log=True) = log λ_d(T) − Λ_total(T): reuse the intensity + CI
                    # ``evaluate_at_time`` already computed (identical clamp/global-CI semantics to
                    # IntensityBasedTPP.density_joint), avoiding a second forward pass.
                    total_ci = eval_result["ci"].sum(dim=1)  # (B,)
                    log_joint = torch.log(type_intensity_at_t.clamp(min=tiny)) - total_ci.unsqueeze(1)  # (B, D)
                    log_time = torch.logsumexp(log_joint, dim=1)  # (B,) = density_time(log=True)
                    batch_joint_ll = log_joint[batch_indices, target_type]
                    batch_time_ll = log_time
                    batch_cond_ll = batch_joint_ll - log_time  # = prob_mark_given_time(log=True)[target]

                if compute_marginals:
                    mean_time_pred = eval_result["mean_time"]  # (B,)
                    median_time_pred = eval_result["median_time"]  # (B,)
                    marginal_type_probs = eval_result["marginal_type_dist"]  # (B, D)

        with torch.no_grad():
            joint_log_likelihood.append(batch_joint_ll.cpu())
            time_log_likelihood.append(batch_time_ll.cpu())
            cond_type_log_likelihood.append(batch_cond_ll.cpu())

            # Conditional Top-1 and Top-5 accuracy based on p(m | t, H)
            D = type_distribution_at_t.shape[1]
            k5 = min(5, D)

            cond_pred_top1 = type_distribution_at_t.argmax(dim=1)
            cond_top1_correct.append((cond_pred_top1 == target_type).cpu())

            _, cond_pred_top5 = type_distribution_at_t.topk(k5, dim=1)
            cond_top5_hits = (cond_pred_top5 == target_type.unsqueeze(1)).any(dim=1)
            cond_top5_correct.append(cond_top5_hits.cpu())

            if return_items:
                last_time = torch.tensor(
                    [float(item["history"].time_points[-1]) for item in batch_items], dtype=torch.float64
                )
                rec = item_records
                rec["seq_idx"].append(torch.tensor([item["seq_idx"] for item in batch_items]))
                rec["history_length"].append(torch.tensor([item["history_length"] for item in batch_items]))
                rec["target_type"].append(target_type.cpu())
                rec["gap"].append(target_time.cpu() - last_time)
                rec["time_ll_scaled"].append(batch_time_ll.double().cpu())
                rec["joint_ll_scaled"].append(batch_joint_ll.double().cpu())
                rec["cond_type_ll"].append(batch_cond_ll.double().cpu())
                rec["cond_pred_top1"].append(cond_pred_top1.cpu())
                rec["cond_top1_correct"].append((cond_pred_top1 == target_type).cpu())

            if compute_marginals:
                # Marginal type distribution p(m | H) — grid-integrated (intensity) or
                # the mark head (density).
                marginal_type_likelihood_at_target = marginal_type_probs[batch_indices, target_type]
                marginal_type_likelihood.append(marginal_type_likelihood_at_target.cpu())

                # Top-1 and Top-5 accuracy based on marginal type probs p(m | H)
                pred_top1 = marginal_type_probs.argmax(dim=1)
                top1_correct.append((pred_top1 == target_type).cpu())

                _, pred_top5 = marginal_type_probs.topk(k5, dim=1)
                top5_hits = (pred_top5 == target_type.unsqueeze(1)).any(dim=1)
                top5_correct.append(top5_hits.cpu())

                # Time prediction errors (in scaled time; converted to days at aggregation).
                # Drop non-finite predictions (e.g. a defective-survival quantile → +inf) so a
                # single unbounded item does not corrupt the aggregate MAE/RMSE.
                mean_err = mean_time_pred - target_time
                median_err = median_time_pred - target_time
                mean_fin = torch.isfinite(mean_err)
                median_fin = torch.isfinite(median_err)
                time_mean_abs_errors.append(mean_err[mean_fin].abs().cpu())
                time_median_abs_errors.append(median_err[median_fin].abs().cpu())
                time_mean_sq_errors.append((mean_err[mean_fin] ** 2).cpu())
                time_median_sq_errors.append((median_err[median_fin] ** 2).cpu())

    # =========================================================================
    # Step 4: Aggregate metrics
    # =========================================================================

    # Time likelihoods need a Jacobian correction for the linear time rescaling
    # f(x) = x / time_scale_factor:
    #   log p_days(x) = log p_scaled(x) - log(time_scale_factor)
    marginal_time_ll = torch.mean(torch.cat(time_log_likelihood)) - log_time_scale
    joint_ll = torch.mean(torch.cat(joint_log_likelihood)) - log_time_scale

    # Type likelihoods are unit-free and need no conversion.
    cond_type_ll = torch.mean(torch.cat(cond_type_log_likelihood))

    metrics = {
        "marginal_time_ll": float(marginal_time_ll),
        "cond_type_ll": float(cond_type_ll),
        "joint_ll": float(joint_ll),
        "cond_top1_accuracy": float(torch.cat(cond_top1_correct).float().mean()),
        "cond_top5_accuracy": float(torch.cat(cond_top5_correct).float().mean()),
        "num_predictions": total_predictions,
        # Events whose target-type intensity was below finfo.tiny and got
        # floored inside log(). 0 means the LLs above carry no floor at all.
        "num_floored_intensity": num_floored_intensity,
    }

    # Marginal (unconditional) type accuracy is only populated when
    # compute_marginals=True; guard the cat exactly like the metrics below.
    if len(top1_correct) > 0:
        metrics["top1_accuracy"] = float(torch.cat(top1_correct).float().mean())
    if len(top5_correct) > 0:
        metrics["top5_accuracy"] = float(torch.cat(top5_correct).float().mean())

    if len(marginal_type_likelihood) > 0:
        # Grid-integrated probability; tiny-floor only guards an exact zero.
        marginal_type_cat = torch.cat(marginal_type_likelihood)
        metrics["marginal_type_ll"] = float(
            torch.mean(torch.log(marginal_type_cat.clamp(min=torch.finfo(marginal_type_cat.dtype).tiny)))
        )

    # Time errors are converted from scaled time back to days.
    if len(time_mean_abs_errors) > 0:
        metrics["time_mae_mean_days"] = float(torch.mean(torch.cat(time_mean_abs_errors)) * time_scale_factor)
    if len(time_median_abs_errors) > 0:
        metrics["time_mae_median_days"] = float(torch.mean(torch.cat(time_median_abs_errors)) * time_scale_factor)

    # RMSE = sqrt(mean(err²)) * scale
    if len(time_mean_sq_errors) > 0:
        metrics["time_rmse_mean_days"] = float(
            torch.sqrt(torch.mean(torch.cat(time_mean_sq_errors))) * time_scale_factor
        )
    if len(time_median_sq_errors) > 0:
        metrics["time_rmse_median_days"] = float(
            torch.sqrt(torch.mean(torch.cat(time_median_sq_errors))) * time_scale_factor
        )

    if return_items:
        metrics["items"] = {k: torch.cat(v).numpy() for k, v in item_records.items()}
        metrics["items"]["log_time_scale"] = np.array(log_time_scale)

    return metrics


def format_metrics(metrics: Dict[str, Union[float, torch.Tensor]], prefix: str = "") -> str:
    """Format evaluation metrics as a readable string.

    Args:
        metrics: Dictionary of metrics from evaluate_tpp
        prefix: Optional prefix for each line

    Returns:
        Formatted string with metrics
    """
    lines = []
    metric_names = {
        "marginal_time_ll": "Marginal time log-likelihood",
        "cond_type_ll": "Conditional type log-likelihood p(m|t,H)",
        "marginal_type_ll": "Marginal type log-likelihood p(m|H)",
        "joint_ll": "Joint log-likelihood",
        "top1_accuracy": "Top-1 Accuracy (marginal)",
        "top5_accuracy": "Top-5 Accuracy (marginal)",
        "cond_top1_accuracy": "Top-1 Accuracy (conditional)",
        "cond_top5_accuracy": "Top-5 Accuracy (conditional)",
        "time_mae_mean_days": "Time MAE mean (days)",
        "time_mae_median_days": "Time MAE median (days)",
        "time_rmse_mean_days": "Time RMSE mean (days)",
        "time_rmse_median_days": "Time RMSE median (days)",
        "num_predictions": "Total predictions",
    }

    for key, display_name in metric_names.items():
        if key in metrics:
            value = metrics[key]
            if isinstance(value, (int, np.integer)):
                lines.append(f"{prefix}{display_name}: {value}")
            else:
                lines.append(f"{prefix}{display_name}: {value:.6f}")

    return "\n".join(lines)
