"""
Evaluation utilities for temporal point process models.

This module provides functions for evaluating TPP models on test sequences,
computing various metrics like likelihoods, accuracy, and time prediction errors.
"""

import warnings
from typing import Dict, List, Union, Any, Optional
import numpy as np
import torch
from tqdm import tqdm

from ..tpp.base_tpp import TemporalPointProcess, IntensityBasedTPP, DensityBasedTPP
from .event_data import MVEventData, BatchedMVEventData, TokenRoles, scored_event_mask
from ..data.base_dataset import TPPDataset
from .math import IntensityGrid


# Numerical-stability constants (guards, not tuning knobs).  The integration-grid
# density is a CIConfig knob now: pass ``grid_points_per_unit_time=None`` to let the
# model resolve it from ``model.ci_config.grid_points_per_unit_time``.
_EVAL_SNAP_ATOL = 1e-8  # snap a query time within this of t_last onto t_last
_COND_TYPE_INTENSITY_FLOOR = 1e-12  # floor on the conditional-type-distribution divisor


def evaluate_at_time(
    model: IntensityBasedTPP,
    T: torch.Tensor,
    ts: BatchedMVEventData,
    ignored_types: Optional[List[int]] = None,
    compute_marginals: bool = True,
    integrate_marginal_via_grid: bool = True,
    grid_points_per_unit_time: Optional[float] = None,
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
        T: Query times. Shape ``(B,)``; must satisfy ``T >= ts.last_event_time``.
        ts: Batched, padded event history. Shape ``(B, N)``.
        ignored_types: Event-type indices to treat as absent. Their
            intensities and cumulative intensities are zeroed before all
            derived quantities are computed, so survival, PDF,
            ``cond_type_dist``, and all grid-based quantities reflect only
            the remaining types. Unioned with the model's context set
            (``~model.predictable_mask``), which is always ignored — context
            types are never scored or integrated. ``None`` or empty list, with
            an all-True mask → no masking.
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
    origin = model._forecast_origin(ts).double()  # T_obs, at least the last event
    close_mask = torch.abs(T - origin) < _EVAL_SNAP_ATOL
    T = torch.where(close_mask, origin, T)

    if not bool((T >= origin).all()):
        raise ValueError("evaluate_at_time is only defined for T >= T_obs (the last event by default).")

    # Context types are never scored or integrated, so they are always removed together
    # with the caller's ignored types.
    context = (~model.predictable_mask).nonzero(as_tuple=True)[0].tolist()
    ignore = sorted(set(ignored_types or []) | set(context))

    # --- Point evaluation (intensity + CI) ---
    # Neural models cache transformer outputs per batch; handle via duck-typing.
    prepare = getattr(model, "prepare_batch", None)
    clear = getattr(model, "clear_batch", None)
    if prepare is not None:
        prepare(ts)
    try:
        intensity = model.intensity(T, ts)  # (B, D)
        # Window CI over [origin, T] with the model's deterministic predictive scheme.
        ci = model.cumulative_intensity(T, ts, T_low=origin, ci_config=model._predictive_ci_config())  # (B, D)
    finally:
        if clear is not None:
            clear()

    # --- Apply the (context ∪ ignore) mask ---
    # Setting λ_k = 0 and Λ_k = 0 is equivalent to removing those types
    # from the process, so all derived quantities reflect the restricted sub-process.
    if ignore:
        valid_mask = torch.ones(1, intensity.shape[1], dtype=intensity.dtype, device=intensity.device)
        valid_mask[0, ignore] = 0.0
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
    # model._build_integration_grid handles its own prepare_batch via the
    # neural override, so no extra wrapping is needed here.
    # The integration grid runs in the model's internal SCALED units:
    # ``_build_integration_grid`` uses the scaled cores, so it must be fed the scaled
    # history and grows a scale-covariant grid.  Grid-derived TIMES are converted back
    # to native units (× s); the marginal mark distribution is dimensionless.
    # The grid is restricted to the retained types while it grows, so it extends until the
    # restricted process reaches the target CDF.
    s = model.time_scale
    grid: IntensityGrid = model._build_integration_grid(
        model._scaled_ts(ts),
        integrate_marginal_via_grid=integrate_marginal_via_grid,
        points_per_unit_time=grid_points_per_unit_time,
        ignore_indices=ignore,
    )

    B = T.shape[0]
    target_cdf = model.ci_config.grid_target_cdf
    result["mean_time"] = model._mean_next_event_time_from_grid(grid, target_cdf=target_cdf) * s  # (B,) native
    result["median_time"] = model._inverse_cdf_from_grid(
        grid, torch.tensor(0.5, device=T.device, dtype=T.dtype).expand(B)
    ) * s  # (B,) scaled → native
    result["marginal_type_dist"] = model._marginal_class_distribution_from_grid(grid)  # (B, D) dimensionless

    return result


def _build_prediction_items(
    sequences: List[MVEventData],
    ignore_indices: Optional[List[int]] = None,
    has_origin: bool = False,
    predictable_mask: Optional[torch.Tensor] = None,
) -> List[dict]:
    """Flatten sequences into next-event prediction items, one per scored event.

    The targets are exactly the events the likelihood scores
    (:func:`~torchpp.utils.event_data.scored_event_mask`): real events of predictable
    types; with ``has_origin=True`` this includes the first event (its history is
    empty and the wait is measured from the origin at 0), without an origin the first
    event and events tied with it are history only. Context types stay in the history
    but are never targets.

    Each item holds ``history`` (the events before the target), ``target_time``,
    ``target_type``, ``valid`` (False for a type in ``ignore_indices``) and
    ``history_length``. Items are sorted by descending history length so batches pad
    little.
    """
    _ignore = set(ignore_indices) if ignore_indices else set()
    items = []
    for ts_all in sequences:
        if len(ts_all) == 0:
            continue
        predictable = predictable_mask
        if predictable is None:
            predictable = torch.ones(int(ts_all.event_types.max()) + 1, dtype=torch.bool)
        device = ts_all.event_types.device
        scored = scored_event_mask(ts_all.to_batch(), predictable.to(device), has_origin)[0]
        for idx in torch.nonzero(scored, as_tuple=False).flatten().tolist():
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
                    # int(): a 0-d tensor never compares equal to the ints in the set.
                    "valid": bool(int(target_type) not in _ignore),
                    "history_length": idx,
                }
            )

    items.sort(key=lambda x: -x["history_length"])
    return items


def _sequence_log_likelihood(model, sequences: List[MVEventData], device, ci_config) -> tuple[float, int]:
    """Summed full-sequence log-likelihood (events + censored tail up to ``T_obs``) and scored-event count."""
    order = sorted(range(len(sequences)), key=lambda i: -len(sequences[i]))
    median_len = max(int(np.median([len(s) for s in sequences])), 1)
    batch_size = max(1, min(64, 10_000 // median_len))
    total_ll, total_scored = 0.0, 0
    for start in range(0, len(order), batch_size):
        batch = BatchedMVEventData(mv_events=[sequences[i] for i in order[start : start + batch_size]]).to(device)
        with torch.no_grad():
            if ci_config is not None:
                with model.use_ci(ci_config):
                    ll = model.likelihood(batch)
            else:
                ll = model.likelihood(batch)
        total_ll += float(ll.sum())
        total_scored += int(model.num_scored_events(batch).sum())
    return total_ll, total_scored


def evaluate_tpp(
    model: TemporalPointProcess,
    sequences: Union[TPPDataset, List[MVEventData]],
    num_event_types: Optional[int] = None,
    device: str = "cpu",
    roles: Optional[TokenRoles] = None,
    eval_batch_size: Optional[int] = None,
    compute_marginals: bool = True,
    integrate_marginal_via_grid: bool = True,
    grid_points_per_unit_time: Optional[float] = None,
    verbose: bool = False,
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
        Metrics are reported in the dataset's **native time units**.  The model owns
        time normalisation: it computes internally in scaled units but
        de-normalises its outputs, so ``evaluate_tpp`` consumes native-unit quantities
        directly — there is no dataset-level de-normalisation here.

    Args:
        model: The TPP model to evaluate.
        sequences: Either a :class:`~torchpp.data.base_dataset.TPPDataset` or a
            plain :class:`list` of :class:`~torchpp.utils.event_data.MVEventData`
            sequences.
        num_event_types: Number of event types; defaults to ``model.D``.
        device: Device to run evaluation on.
        roles: :class:`~torchpp.utils.event_data.TokenRoles` for the evaluation; ``None``
            uses ``model.token_roles``. Only ``ignored_types`` (types excluded from the
            targets) may differ from the model's roles; the origin regime and the context
            types are properties of the model.
        eval_batch_size: Number of prediction items per evaluation batch.
            Larger = faster but more memory. None = auto-select based on
            median history length and num_event_types.
        verbose: Show a progress bar.

    Targets:
        The next-event metrics use one prediction per scored event (the events the
        likelihood scores): with ``model.has_origin`` the first event is a target, its
        wait measured from the origin; without an origin the first event and events tied
        with it are history only (common for benchmarks with an arbitrary time origin). Context types are never
        targets. Sequences with zero or one event are allowed.

    Returns:
        Dictionary with evaluation metrics (all values are floats), in the dataset's
        native time units:
            - ll_per_event: Full-sequence log-likelihood (all scored events plus the
              censored tail up to ``T_obs``) divided by the number of scored events; the
              quantity training and validation report
            - ll_per_sequence: The same log-likelihood averaged over sequences
            - num_scored_events, num_sequences: The counts behind the two averages
            - marginal_time_ll: Marginal time log-likelihood p(t|H)
            - cond_type_ll: Conditional type log-likelihood p(m|t,H)
            - marginal_type_ll: Marginal type log-likelihood p(m|H) (if computed)
            - joint_ll: Joint log-likelihood p(t,m|H)
            - top1_accuracy: Top-1 type prediction accuracy (marginal p(m|H))
            - top5_accuracy: Top-5 type prediction accuracy (marginal p(m|H))
            - cond_top1_accuracy: Top-1 type prediction accuracy (conditional p(m|t,H))
            - cond_top5_accuracy: Top-5 type prediction accuracy (conditional p(m|t,H))
            - time_mae_mean: Time prediction MAE (mean of distribution vs actual)
            - time_mae_median: Time prediction MAE (median of distribution vs actual)
            - time_rmse_mean: Time prediction RMSE (mean of distribution vs actual)
            - time_rmse_median: Time prediction RMSE (median of distribution vs actual)
            - num_predictions: Total number of predictions made
    """
    if isinstance(sequences, TPPDataset):
        sequences = list(sequences)
    if num_event_types is None:
        num_event_types = model.D
    model_roles = model.token_roles
    roles = model_roles if roles is None else roles
    if roles.has_origin != model_roles.has_origin or roles.context_types != model_roles.context_types:
        raise ValueError(
            f"roles {roles} disagree with the model's {model_roles}: only ignored_types may differ."
        )
    ignore_indices = list(roles.ignored_types)

    # Evaluate in eval mode on ``device``; the caller's mode and device are restored afterwards.
    was_training = model.training
    first = next(iter(model.buffers()), None)
    first = first if first is not None else next(iter(model.parameters()), None)
    original_device = first.device if first is not None else None
    model.eval()
    DEVICE = torch.device(device)
    model.to(DEVICE)
    try:
        return _evaluate_items(
            model,
            sequences,
            num_event_types,
            DEVICE,
            ignore_indices,
            eval_batch_size,
            compute_marginals,
            integrate_marginal_via_grid,
            grid_points_per_unit_time,
            verbose,
        )
    finally:
        model.train(was_training)
        if original_device is not None:
            model.to(original_device)


def _evaluate_items(
    model,
    sequences,
    num_event_types,
    DEVICE,
    ignore_indices,
    eval_batch_size,
    compute_marginals,
    integrate_marginal_via_grid,
    grid_points_per_unit_time,
    verbose=False,
) -> Dict[str, Any]:
    """Body of :func:`evaluate_tpp` (model already in eval mode on ``DEVICE``)."""

    # =========================================================================
    # Step 1: Flatten all sequences into individual prediction items
    # =========================================================================
    if len(sequences) == 0:
        raise ValueError("evaluate_tpp: no sequences to evaluate.")
    is_density = isinstance(model, DensityBasedTPP)

    # Full-sequence log-likelihood per scored event: the training objective, with a
    # deterministic quadrature rule.
    seq_ci = None if is_density else model._predictive_ci_config()
    total_ll, total_scored = _sequence_log_likelihood(model, sequences, DEVICE, seq_ci)
    sequence_metrics = {
        "ll_per_event": total_ll / total_scored if total_scored > 0 else float("nan"),
        "ll_per_sequence": total_ll / len(sequences),
        "num_scored_events": total_scored,
        "num_sequences": len(sequences),
    }

    all_items = _build_prediction_items(
        sequences, ignore_indices, has_origin=model.has_origin, predictable_mask=model.predictable_mask
    )
    if len(all_items) == 0:
        return {**sequence_metrics, "num_predictions": 0}

    # Drop items whose target type is ignored (excluded from the metrics).
    valid_items = [item for item in all_items if item["valid"]]
    items_to_eval = valid_items

    # In the bundled datasets ignored types are stripped from the sequences
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
            f"roles.ignored_types. Ignored-type events are present in the sequences, so the survival factor of "
            f"windows preceding them is unscored and the reported time/joint LL is biased. Strip ignored "
            f"types from the sequences (as the standard loaders do) or re-target to the next valid event.",
            stacklevel=2,
        )

    if len(items_to_eval) == 0:
        return {**sequence_metrics, "num_predictions": 0}

    total_predictions = len(items_to_eval)
    context = (~model.predictable_mask).nonzero(as_tuple=True)[0].tolist()
    density_ignore = sorted(set(ignore_indices or []) | set(context))

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


    # Deterministic CI for evaluation (intensity family): the model's own scheme with a
    # trapezoid rule in place of a Monte-Carlo training rule, as in the predictive methods.
    eval_ci_config = None if is_density else model._predictive_ci_config()

    for batch_start in tqdm(range(0, len(items_to_eval), eval_batch_size), total=num_batches, desc="Evaluating", disable=not verbose):
        batch_items = items_to_eval[batch_start : batch_start + eval_batch_size]
        B = len(batch_items)

        histories = [item["history"] for item in batch_items]
        batch = BatchedMVEventData(mv_events=histories)
        target_time = torch.tensor([item["target_time"] for item in batch_items], dtype=torch.float64, device=DEVICE)
        target_type = torch.tensor([item["target_type"] for item in batch_items], dtype=torch.long, device=DEVICE)
        batch = batch.to(DEVICE)

        batch_indices = torch.arange(B, device=DEVICE)

        if is_density:
            # --- Density family: everything via the contract (no intensity / CI) ---
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

                if density_ignore:
                    # Restrict to retained types (ignored + context): mask their mark columns and renormalise.
                    keep = torch.ones(1, log_joint.shape[1], dtype=torch.bool, device=DEVICE)
                    keep[0, density_ignore] = False
                    log_joint = log_joint.masked_fill(~keep, torch.finfo(log_joint.dtype).min)
                    log_pm = log_pm.masked_fill(~keep, torch.finfo(log_pm.dtype).min)
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
                    ignored_types=ignore_indices,
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

                # Time prediction errors (native units; the model's predictions and the
                # targets are both native). Drop non-finite predictions
                # (e.g. a defective-survival quantile → +inf) so a single unbounded item
                # does not corrupt the aggregate MAE/RMSE.
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

    # The model reports native-unit densities directly, so the LLs need
    # no dataset-level Jacobian here.
    marginal_time_ll = torch.mean(torch.cat(time_log_likelihood))
    joint_ll = torch.mean(torch.cat(joint_log_likelihood))
    cond_type_ll = torch.mean(torch.cat(cond_type_log_likelihood))

    metrics = {
        **sequence_metrics,
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

    # Time errors are already in native units (model predictions + targets both native).
    if len(time_mean_abs_errors) > 0:
        metrics["time_mae_mean"] = float(torch.mean(torch.cat(time_mean_abs_errors)))
    if len(time_median_abs_errors) > 0:
        metrics["time_mae_median"] = float(torch.mean(torch.cat(time_median_abs_errors)))

    # RMSE = sqrt(mean(err²))
    if len(time_mean_sq_errors) > 0:
        metrics["time_rmse_mean"] = float(torch.sqrt(torch.mean(torch.cat(time_mean_sq_errors))))
    if len(time_median_sq_errors) > 0:
        metrics["time_rmse_median"] = float(torch.sqrt(torch.mean(torch.cat(time_median_sq_errors))))

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
        "ll_per_event": "Log-likelihood per scored event (full sequence)",
        "num_scored_events": "Scored events",
        "num_sequences": "Sequences",
        "marginal_time_ll": "Marginal time log-likelihood",
        "cond_type_ll": "Conditional type log-likelihood p(m|t,H)",
        "marginal_type_ll": "Marginal type log-likelihood p(m|H)",
        "joint_ll": "Joint next-event log-likelihood",
        "top1_accuracy": "Top-1 Accuracy (marginal)",
        "top5_accuracy": "Top-5 Accuracy (marginal)",
        "cond_top1_accuracy": "Top-1 Accuracy (conditional)",
        "cond_top5_accuracy": "Top-5 Accuracy (conditional)",
        "time_mae_mean": "Time MAE mean (native units)",
        "time_mae_median": "Time MAE median (native units)",
        "time_rmse_mean": "Time RMSE mean (native units)",
        "time_rmse_median": "Time RMSE median (native units)",
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
