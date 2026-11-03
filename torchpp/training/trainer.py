"""Training loop: optimiser, LR schedule, gradient accumulation, early stopping, resume.

:func:`fit` trains any :class:`~torchpp.tpp.base_tpp.TemporalPointProcess` on a training
set and returns the model with the best validation weights.  ``train.py`` is a thin
command-line wrapper around it.

.. code-block:: python

    from torchpp.training import create_model, default_fit_config, fit

    model = create_model("const_exp_hawkes", D=dataset.num_event_types)
    config = default_fit_config(num_steps=2000, learning_rate=0.05)
    result = fit(model, train_sequences, val_sequences, config)
    result.model, result.best_val_ll
"""

from __future__ import annotations

import copy
import math
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchpp.tpp.base_tpp import TemporalPointProcess
from omegaconf import MISSING

from torchpp.training.arch import ArchConfig
from torchpp.training.config import BatchingConfig, FitConfig, TrainingConfig
from torchpp.training.training_helpers import make_collate_fns, make_train_val_dataloaders

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


# Number of recent steps over which the training LL moving average is computed.
_LL_MA_WINDOW = 50


@dataclass
class FitResult:
    """Outcome of :func:`fit`."""

    model: TemporalPointProcess
    """The trained model, with the best validation weights restored (if validation ran)."""

    best_val_ll: Optional[float]
    """Best per-event validation log-likelihood (``None`` without validation)."""

    train_ll: list[float] = field(default_factory=list)
    """Per-step per-event training log-likelihood."""

    val_ll: list[float] = field(default_factory=list)
    """Per-event validation log-likelihood at every validation step."""

    steps: int = 0
    """Number of optimiser steps run."""


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module, transparently through torch.compile."""
    return getattr(model, "_orig_mod", model)


# ---------------------------------------------------------------------------
# Resumable checkpointing
# ---------------------------------------------------------------------------
# A periodic checkpoint bundles everything needed to continue a run (model, optimiser,
# scheduler, step/epoch/early-stop counters, RNG). It is written atomically (tmp file +
# os.replace), so a kill mid-write cannot corrupt it.


def _save_checkpoint(path: Path, **state: Any) -> None:
    """Atomically write a resume checkpoint to ``path``.

    A write failure (full disk, quota) is warned and swallowed rather than crashing an
    otherwise healthy run.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001 — checkpointing must never kill training
        print(f"  WARNING: checkpoint write to {path} failed ({e}); continuing without it.")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _load_checkpoint(path: Path, model, optim, scheduler, device) -> Optional[dict]:
    """Restore model/optimiser/scheduler in place from ``path``; return the bookkeeping
    state (global_step, epoch, best_val_ll, …), or ``None`` if no checkpoint exists."""
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device)
    _unwrap(model).load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if ckpt.get("rng_cpu") is not None:
        torch.set_rng_state(ckpt["rng_cpu"].cpu())
    if device.type == "cuda" and ckpt.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng_cuda"]])
    return ckpt


# ---------------------------------------------------------------------------
# Validation aggregation + early stopping
# ---------------------------------------------------------------------------


def _num_scored_events(model, batch) -> torch.Tensor:
    """Per-sequence count of scored events ``(B,)``; ``real_counts`` for models without the method."""
    counter = getattr(model, "num_scored_events", None)
    return counter(batch) if counter is not None else batch.real_counts


def _aggregate_val_ll(model, test_events, device) -> tuple[float, int]:
    """Per-event mean validation reward ``-loss()`` (higher is better), NaN-safe.

    Runs in eval mode (dropout off) and restores the previous mode. Early stopping and
    best-model selection use this quantity. It equals the per-event
    log-likelihood for intensity and density models (whose ``loss`` is ``-likelihood``)
    and excludes the regulariser.

    Non-finite per-sequence values are excluded from both the summed reward and the event
    count, so a single bad batch cannot poison the validation number. Returns
    ``(current_val_ll, n_nonfinite)``; ``current_val_ll`` is ``nan`` only when no finite
    events remain.
    """
    total_val_ll = 0.0
    total_val_events = 0
    n_nonfinite = 0
    is_module = isinstance(model, torch.nn.Module)
    was_training = model.training if is_module else False
    if is_module:
        model.eval()
    with torch.no_grad():
        for val_batch in test_events:
            val_batch = val_batch.to(device)
            ll = -model.loss(val_batch)  # (B,)
            finite = torch.isfinite(ll)
            n_nonfinite += int((~finite).sum().item())
            total_val_ll += ll[finite].sum().item()
            total_val_events += int(_num_scored_events(model, val_batch)[finite].sum().item())
    if is_module:
        model.train(was_training)
    if total_val_events == 0 and n_nonfinite > 0:
        return float("nan"), n_nonfinite
    return total_val_ll / max(total_val_events, 1), n_nonfinite


def _early_stopping_step(
    current_val_ll: float,
    best_val_ll: float,
    early_stop_counter: int,
    min_delta: float,
    patience: int,
) -> tuple[float, bool, int, bool]:
    """Early-stopping state transition for one validation step.

    Returns ``(best_val_ll, improved, early_stop_counter, triggered)``. A non-finite
    ``current_val_ll`` leaves the best value and the patience counter unchanged.
    """
    if not math.isfinite(current_val_ll):
        return best_val_ll, False, early_stop_counter, False
    threshold = best_val_ll + min_delta * abs(best_val_ll) if best_val_ll != float("-inf") else float("-inf")
    if current_val_ll > threshold:
        return current_val_ll, True, 0, False
    if patience > 0:
        early_stop_counter += 1
        return best_val_ll, False, early_stop_counter, early_stop_counter >= patience
    return best_val_ll, False, early_stop_counter, False


def make_scheduler(optim: torch.optim.Optimizer, training) -> torch.optim.lr_scheduler.LRScheduler:
    """Learning-rate schedule from a :class:`~torchpp.training.config.TrainingConfig`.

    ``scheduler_type``: ``step``, ``cosine``, ``linear_warmup_cosine`` or ``none``.
    """
    if training.scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer=optim, step_size=training.scheduler_step_size, gamma=training.scheduler_gamma
        )
    if training.scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optim, T_max=training.num_steps, eta_min=training.scheduler_eta_min
        )
    if training.scheduler_type == "linear_warmup_cosine":
        warmup_steps = training.scheduler_warmup_steps
        total_steps = training.num_steps
        eta_min_ratio = training.scheduler_eta_min / training.learning_rate

        def _warmup_cosine_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=_warmup_cosine_lambda)
    if training.scheduler_type == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda _: 1.0)
    raise ValueError(f"Unknown scheduler type: {training.scheduler_type}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _train_loop(
    model: TemporalPointProcess,
    events_batch: DataLoader,
    config: FitConfig,
    test_events: Optional[DataLoader] = None,
    wandb_run=None,
    profiler=None,
    checkpoint_path: Optional[Path] = None,
    resume: bool = False,
    checkpoint_freq: int = 0,
    verbose: bool = True,
) -> FitResult:
    """Optimisation loop behind :func:`fit` (data already in dataloaders)."""
    log = print if verbose else (lambda *args, **kwargs: None)
    device = config.device
    model = model.to(device=device)
    model.train()
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = make_scheduler(optim, config.training)

    likelihoods: list[float] = []
    test_likelihoods: list[float] = []
    last_ll = 0.0
    ll_window: deque[float] = deque(maxlen=_LL_MA_WINDOW)
    global_step = 0
    epoch = 0

    best_val_ll: float = float("-inf")
    best_model_state = None
    early_stop_counter: int = 0
    early_stopping_triggered: bool = False

    if resume and checkpoint_path is not None:
        ckpt = _load_checkpoint(checkpoint_path, model, optim, scheduler, torch.device(device))
        if ckpt is not None:
            global_step = ckpt["global_step"]
            epoch = ckpt["epoch"]
            best_val_ll = ckpt["best_val_ll"]
            best_model_state = ckpt["best_model_state"]
            early_stop_counter = ckpt["early_stop_counter"]
            likelihoods = ckpt.get("likelihoods", [])
            test_likelihoods = ckpt.get("test_likelihoods", [])
            log(f"\n  Resumed from {checkpoint_path} at step {global_step} "
                  f"(best val LL = {best_val_ll:.6f}).")
        else:
            log(f"\n  save.resume=true but no checkpoint at {checkpoint_path}; starting fresh.")

    accumulate_gradients = config.batching.accumulate_gradients
    log(f"\n  Fixed gradient accumulation: {accumulate_gradients} steps")

    def _write_checkpoint() -> None:
        _save_checkpoint(
            checkpoint_path,
            model=_unwrap(model).state_dict(),
            optim=optim.state_dict(),
            scheduler=scheduler.state_dict(),
            global_step=global_step,
            epoch=epoch,
            best_val_ll=best_val_ll,
            best_model_state=best_model_state,
            early_stop_counter=early_stop_counter,
            likelihoods=likelihoods,
            test_likelihoods=test_likelihoods,
            rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if torch.device(device).type == "cuda" else None,
        )

    progress_bar = tqdm(total=config.training.num_steps, initial=global_step, disable=not verbose)
    # The dataloader is not fast-forwarded on resume: it restarts from a fresh shuffle.
    data_iter = iter(events_batch)

    while global_step < config.training.num_steps:
        optim.zero_grad()
        accumulated_ll = 0.0
        accumulated_events = 0

        for _ in range(accumulate_gradients):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                log(f"Starting epoch {epoch}...")
                data_iter = iter(events_batch)
                batch = next(data_iter)

            batch = batch.to(device)
            N = _num_scored_events(model, batch).float()

            # The model owns its objective (default: negative log-likelihood) and regulariser.
            per_seq_loss = model.loss(batch)  # (B,)
            # Per scored event, like the validation and test LL; keeps lr and grad_clip
            # independent of sequence length.
            nll = per_seq_loss.sum() / (N.sum().clamp(min=1) * accumulate_gradients)
            nll = nll + model.regularization(batch) / accumulate_gradients

            nll.backward()

            accumulated_events += N.sum().item()
            accumulated_ll += (-per_seq_loss).sum().item()

        if config.training.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

        grad_ok = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if not grad_ok:
            print(f"  WARNING: NaN/Inf gradient at step {global_step}, skipping optimizer step.")
            optim.zero_grad()
        else:
            optim.step()
        scheduler.step()

        last_ll = accumulated_ll / max(accumulated_events, 1)
        likelihoods.append(last_ll)
        ll_window.append(last_ll)
        ll_ma = sum(ll_window) / len(ll_window)

        if wandb_run is not None and wandb is not None:
            wandb.log({"train/log_likelihood": last_ll}, step=global_step)

        if test_events is not None and (
            global_step % config.training.eval_freq == 0 or global_step == config.training.num_steps - 1
        ):
            current_val_ll, n_nonfinite_val = _aggregate_val_ll(model, test_events, device)
            if n_nonfinite_val > 0:
                print(
                    f"  WARNING: {n_nonfinite_val} non-finite val contribution(s) at step "
                    f"{global_step}; excluded from the val LL."
                )
            log(f" Step {global_step}, Val LL (per event): {current_val_ll}")
            test_likelihoods.append(current_val_ll)

            if wandb_run is not None and wandb is not None:
                wandb.log({"val/log_likelihood": current_val_ll}, step=global_step)

            best_val_ll, improved, early_stop_counter, triggered = _early_stopping_step(
                current_val_ll,
                best_val_ll,
                early_stop_counter,
                config.training.early_stopping_min_delta,
                config.training.early_stopping_patience,
            )
            if improved:
                best_model_state = {k: v.cpu().clone() for k, v in _unwrap(model).state_dict().items()}
                log(f"   → New best val LL: {best_val_ll:.6f}  (best weights kept in memory)")
            elif not math.isfinite(current_val_ll):
                log("   → Non-finite val LL; uninformative eval (best-val and patience unchanged).")
            elif config.training.early_stopping_patience > 0:
                log(f"   → No improvement ({early_stop_counter}/{config.training.early_stopping_patience})")
            if triggered:
                log(
                    f"\nEarly stopping triggered at step {global_step} "
                    f"(patience={config.training.early_stopping_patience})."
                )
                early_stopping_triggered = True

        global_step += 1
        progress_bar.set_postfix(epoch=epoch, LL_ma50=f"{ll_ma:.4f}", refresh=True)
        progress_bar.update(1)

        if profiler is not None:
            profiler.step()

        if checkpoint_freq > 0 and checkpoint_path is not None and global_step % checkpoint_freq == 0:
            _write_checkpoint()

        if early_stopping_triggered:
            break

    progress_bar.close()

    # A clean exit (finished or early-stopped) drops the resume checkpoint, so launching the
    # finished job again does not continue it. A wall-clock kill leaves it for resume.
    if checkpoint_freq > 0 and checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_path.unlink()

    if best_model_state is not None:
        _unwrap(model).load_state_dict(best_model_state)
        log(f"\nRestored best model (val LL = {best_val_ll:.6f}).")

    return FitResult(
        model=model,
        best_val_ll=best_val_ll if math.isfinite(best_val_ll) else None,
        train_ll=likelihoods,
        val_ll=test_likelihoods,
        steps=global_step,
    )


def default_fit_config(
    num_steps: int = 1000,
    learning_rate: float = 0.01,
    eval_freq: int = 100,
    batch_size: int = 32,
    **training: Any,
) -> FitConfig:
    """A :class:`FitConfig` for Python use: AdamW without a schedule, early stopping off.

    Extra keyword arguments set further :class:`~torchpp.training.config.TrainingConfig`
    fields (e.g. ``scheduler_type="cosine"``, ``early_stopping_patience=5``).
    """
    return FitConfig(
        model_type="",
        training=TrainingConfig(
            num_steps=num_steps,
            learning_rate=learning_rate,
            eval_freq=eval_freq,
            scheduler_type=training.pop("scheduler_type", "none"),
            scheduler_step_size=training.pop("scheduler_step_size", num_steps),
            scheduler_gamma=training.pop("scheduler_gamma", 1.0),
            **training,
        ),
        batching=BatchingConfig(batch_size=batch_size, analyze_lengths=False),
        arch=ArchConfig(),
    )


def _check_required(config: FitConfig) -> None:
    unset = [
        f"{section}.{name}"
        for section in ("training", "batching")
        for name, value in vars(getattr(config, section)).items()
        if value == MISSING
    ]
    if unset:
        raise ValueError(f"Unset required config fields: {unset}. Use default_fit_config() or set them.")


def _as_loaders(
    train, val, config: FitConfig, verbose: bool = False, has_origin: bool = False
) -> tuple[DataLoader, Optional[DataLoader]]:
    """Dataloaders from datasets/sequence lists (honouring ``config.batching``), or pass through."""
    if isinstance(train, DataLoader):
        if val is not None and not isinstance(val, DataLoader):
            raise TypeError("Pass either dataloaders for both train and val, or datasets/sequence lists for both.")
        return train, val
    collate_train, collate_val = make_collate_fns(config.batching, config.arch.seed, has_origin=has_origin)
    loader_train, loader_val = make_train_val_dataloaders(
        train,
        val if val is not None else train,
        config.batching,
        config.arch.seed,
        collate_train,
        collate_val,
        verbose=verbose,
    )
    return loader_train, (loader_val if val is not None else None)


def fit(
    model: TemporalPointProcess,
    train: Union[DataLoader, Sequence],
    val: Optional[Union[DataLoader, Sequence]] = None,
    config: Optional[FitConfig] = None,
    *,
    device: Optional[str] = None,
    wandb_run=None,
    profiler=None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    resume: bool = False,
    checkpoint_freq: int = 0,
    verbose: bool = False,
) -> FitResult:
    """Train ``model`` and return it with the best validation weights.

    Uses AdamW, the learning-rate schedule, gradient accumulation, gradient clipping and
    early stopping of ``config`` (``config.training`` and ``config.batching``).  Validation
    runs every ``training.eval_freq`` steps and at the last step; the weights with the best
    per-event validation log-likelihood are restored at the end.

    Args:
        model:  Any TemporalPointProcess (e.g. from :func:`~torchpp.training.create_model`).
        train:  Training data: a ``DataLoader`` yielding ``BatchedMVEventData``, or a
            dataset / list of ``MVEventData`` (batched with ``config.batching``).
        val:    Validation data, same kinds as ``train``.  ``None`` trains without
            validation or early stopping.
        config: A :class:`~torchpp.training.config.FitConfig`; ``None`` uses
            :func:`default_fit_config`.
        device: Device to train on. Defaults to ``config.device``; when that is ``"auto"``
            (the default), the model trains where it already is, so move it with
            ``model.to("cuda")`` or pass ``device="cuda"`` to train on a GPU. The returned
            model stays on the training device.
        wandb_run: An active W&B run to log to, or ``None``.
        profiler: A ``torch.profiler.profile`` to step once per optimiser step.
        checkpoint_path: File for resumable checkpoints (with ``checkpoint_freq > 0``).
        resume: Continue from ``checkpoint_path`` if it exists.
        checkpoint_freq: Steps between resumable checkpoints; 0 disables them.
        verbose: Print progress (a progress bar, validation results, epochs).

    Returns:
        :class:`FitResult`.  ``config`` is not modified.
    """
    config = copy.deepcopy(config) if config is not None else default_fit_config()
    _check_required(config)
    if device is not None:
        config.device = device
    if config.device == "auto":
        first = next(iter(model.parameters()), None)
        config.device = str(first.device) if first is not None else "cpu"
    if isinstance(config.arch, dict):  # a FitConfig built by hand; the trainer reads only the seed
        config.arch = ArchConfig(seed=config.arch.get("seed"))
    if config.arch.seed is None:
        config.arch.seed = 0
    loader_train, loader_val = _as_loaders(
        train, val, config, verbose=verbose, has_origin=getattr(model, "has_origin", False)
    )
    return _train_loop(
        model,
        loader_train,
        config,
        test_events=loader_val,
        wandb_run=wandb_run,
        profiler=profiler,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path is not None else None,
        resume=resume,
        checkpoint_freq=checkpoint_freq,
        verbose=verbose,
    )
