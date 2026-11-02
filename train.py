"""Unified training entry point for TorchPP.

Dataset selection is controlled by the ``dataset_type`` field in the config
(``"ukb"`` | ``"easytpp"`` | ``"mimic_iv"``); the same script covers every
registered dataset.

Usage examples
--------------
.. code-block:: bash

    # UKB — spline-Hawkes on the real cohort
    python train.py -c default_fit -c model/spline_exp_hawkes -c data/ukb_real_full

    # EasyTPP — neural SAHP on retweet
    python train.py -c easytpp/default -c model/neural_sahp \\
        -c training/transformer dataset.dataset_name=retweet

    # CPU smoke run with a short schedule
    python train.py -c easytpp/default -c model/hawkes \\
        training.num_steps=500 device=cpu

Datasets must be available locally. For EasyTPP datasets run::

    python scripts/data_prep/download_easytpp_datasets.py
"""

import math
import os
from collections import deque
from pathlib import Path
from typing import Any, Optional, Union

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity

# Importing the data package triggers @register_dataset decorators for all
# built-in dataset classes.
import torchpp.data  # noqa: F401
from torchpp.tpp.base_tpp import TemporalPointProcess
from torchpp.training.config import FitConfig
from torchpp.training.config_utils import load_config, print_config
from torchpp.training.dataset_registry import get_data_root
from torchpp.training.model_factory import create_model
from torchpp.training.training_helpers import (
    init_wandb_run,
    load_dataset_stats,
    load_test_dataset,
    load_train_val_datasets,
    make_collate_fns,
    make_train_val_dataloaders,
    print_dataset_stats,
    resolve_save_path,
    truncate_sequences_for_eval,
)
from torchpp.utils.batching import print_length_analysis
from torchpp.utils.evaluation import evaluate_tpp, format_metrics
from torchpp.utils.event_data import MVEventData

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


# Number of recent steps over which the training LL moving average is computed.
_LL_MA_WINDOW = 50


# ---------------------------------------------------------------------------
# Resumable checkpointing
# ---------------------------------------------------------------------------
# A periodic checkpoint bundles everything needed to continue a run bit-for-bit
# (model + optimiser + scheduler + step/epoch/early-stop counters + RNG). It is
# written atomically (tmp file + os.replace) so a wall-clock kill mid-write can
# never corrupt it. A job stopped by a wall-clock limit can be restarted with
# ``save.resume=true`` and continues where it stopped.


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module, transparently through torch.compile."""
    return getattr(model, "_orig_mod", model)


def _save_checkpoint(path: Path, **state: Any) -> None:
    """Atomically write a resume checkpoint to ``path``.

    Best-effort: a write failure (full disk / quota — ``torch.save`` raises
    ``RuntimeError: basic_ios`` or ``OSError``) is warned and swallowed rather
    than crashing an otherwise-healthy run. The stale ``.tmp`` is cleaned up.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)  # atomic on POSIX — either the old or the new file, never a partial one
    except Exception as e:  # noqa: BLE001 — checkpointing must never kill training
        print(f"  WARNING: checkpoint write to {path} failed ({e}); continuing without it.")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _load_checkpoint(path: Path, model, optim, scheduler, device) -> Optional[dict]:
    """Restore model/optim/scheduler in-place from ``path``; return the bookkeeping
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


def _aggregate_val_ll(model, test_events, device) -> tuple[float, int]:
    """Per-event mean validation log-likelihood, nan-safe.

    Non-finite per-sequence contributions (NaN/inf LL from a pathological batch)
    are excluded from both the summed LL and the event count, so a single bad
    batch cannot poison the whole validation number.  Returns
    ``(current_val_ll, n_nonfinite)``.  ``current_val_ll`` is ``nan`` only when
    non-finite contributions left no finite events at all (fully uninformative
    eval); with all-finite input this reproduces the plain per-event mean.
    """
    total_val_ll = 0.0
    total_val_events = 0
    n_nonfinite = 0
    with torch.no_grad():
        for val_batch in test_events:
            val_batch = val_batch.to(device)
            ll = model.likelihood(val_batch)  # (B,)
            finite = torch.isfinite(ll)
            n_nonfinite += int((~finite).sum().item())
            total_val_ll += ll[finite].sum().item()
            total_val_events += int(val_batch.real_counts[finite].sum().item())
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
    """Early-stopping state transition for one validation eval.

    Returns ``(best_val_ll, improved, early_stop_counter, triggered)``.

    A non-finite ``current_val_ll`` is treated as an uninformative eval: the best
    is left untouched and patience is **not** incremented, so a transient NaN val
    batch cannot spuriously trigger early stopping.
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def batched_train_loop(
    model: Union[TemporalPointProcess, Any],
    events_batch: DataLoader,
    config: FitConfig,
    test_events: Optional[DataLoader] = None,
    wandb_run=None,
    profiler=None,
    checkpoint_path: Optional[Path] = None,
    resume: bool = False,
    checkpoint_freq: int = 0,
):
    """Run the main training loop, returning the trained model and LL histories.

    Implements gradient accumulation, optional early stopping, per-step W&B
    logging, and best-model checkpointing based on validation log-likelihood.

    When ``checkpoint_freq > 0`` a resumable checkpoint is written to
    ``checkpoint_path`` every ``checkpoint_freq`` steps; when ``resume`` is set
    and that file exists, training continues from it (step, optimiser, scheduler,
    best model and RNG all restored).

    Returns ``(model, train_likelihoods, val_likelihoods)`` — the best model
    (by validation LL), and per-step LL lists.
    """
    device = config.device
    model = model.to(device=device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    if config.training.scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optim,
            step_size=config.training.scheduler_step_size,
            gamma=config.training.scheduler_gamma,
        )
    elif config.training.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optim,
            T_max=config.training.num_steps,
            eta_min=config.training.scheduler_eta_min,
        )
    elif config.training.scheduler_type == "linear_warmup_cosine":
        warmup_steps = config.training.scheduler_warmup_steps
        total_steps = config.training.num_steps
        eta_min_ratio = config.training.scheduler_eta_min / config.training.learning_rate

        def _warmup_cosine_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=_warmup_cosine_lambda)
    elif config.training.scheduler_type == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda _: 1.0)
    else:
        raise ValueError(f"Unknown scheduler type: {config.training.scheduler_type}")

    torch.autograd.set_detect_anomaly(False)

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

    # ── Resume from a previous checkpoint (wall-clock-safe requeue) ────────────
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
            print(f"\n  Resumed from {checkpoint_path} at step {global_step} "
                  f"(best val LL = {best_val_ll:.6f}).")
        else:
            print(f"\n  save.resume=true but no checkpoint at {checkpoint_path}; starting fresh.")

    accumulate_gradients = config.batching.accumulate_gradients
    print(f"\n  Fixed gradient accumulation: {accumulate_gradients} steps")

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

    progress_bar = tqdm(total=config.training.num_steps, initial=global_step)
    # NOTE: the dataloader is not fast-forwarded on resume — it restarts from a
    # fresh shuffle. Step/optimiser/scheduler continuity is preserved, which is
    # what matters for convergence; a partial epoch's exact batch order is not.
    data_iter = iter(events_batch)

    while global_step < config.training.num_steps:
        optim.zero_grad()
        accumulated_ll = 0.0
        accumulated_events = 0

        for _ in range(accumulate_gradients):
            try:
                batch = next(data_iter)
            except StopIteration:
                print(f"Starting epoch {epoch}...")
                epoch += 1
                data_iter = iter(events_batch)
                batch = next(data_iter)

            N = batch.real_counts.float()
            batch = batch.to(device)

            ll = model.likelihood(batch)
            # Normalise by total real events so the loss scale matches the
            # per-event val LL metric, making lr/grad_clip dataset-agnostic.
            nll = -ll.sum() / (N.to(device).sum() * accumulate_gradients)

            # Optional spline smoothness regulariser (NeuralSplineProcess).
            # Computed on the cached full params populated by the likelihood
            # forward, so no extra encoder pass is required.
            sm_lambda = getattr(config.arch, "spline_smoothness_reg", 0.0)
            sm_fn = getattr(model, "spline_smoothness_penalty", None)
            if sm_lambda > 0.0 and callable(sm_fn):
                sm_pen = sm_fn(batch) / accumulate_gradients
                nll = nll + sm_lambda * sm_pen

            nll.backward()

            accumulated_events += N.sum().item()
            accumulated_ll += ll.sum().item()

        if config.training.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

        # Skip optimizer step if gradients contain NaN/Inf
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
            print(f" Step {global_step}, Val LL (per event): {current_val_ll}")
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
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"   → New best val LL: {best_val_ll:.6f}  (checkpoint saved)")
            elif not math.isfinite(current_val_ll):
                print("   → Non-finite val LL; uninformative eval (best-val and patience unchanged).")
            elif config.training.early_stopping_patience > 0:
                print(f"   → No improvement ({early_stop_counter}/{config.training.early_stopping_patience})")
            if triggered:
                print(
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

    # Clean exit (finished num_steps or early-stopped): drop the resume checkpoint
    # so launching this finished job again does not continue it. A
    # wall-clock kill bypasses this line, leaving the last checkpoint for resume.
    if checkpoint_freq > 0 and checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_path.unlink()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model (val LL = {best_val_ll:.6f}).")

    return model, likelihoods, test_likelihoods


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Unified training entry point. Dataset selection is driven by config."""
    config: FitConfig = load_config(FitConfig)
    print_config(config, title="Training Configuration")

    ROOT = Path(__file__).resolve().parent
    DATA_ROOT = get_data_root()

    # ── Load data ─────────────────────────────────────────────────────────────
    dataset_train, dataset_val, D = load_train_val_datasets(
        config.dataset_type, dict(config.dataset), DATA_ROOT
    )

    # ── Dataset statistics ────────────────────────────────────────────────────
    dataset_stats = load_dataset_stats(
        config.stats, ROOT, dataset_train, D, config.dataset_type, dict(config.dataset)
    )
    if dataset_stats is not None:
        print_dataset_stats(dataset_stats, dataset_train)

    # ── Length analysis (optional) ────────────────────────────────────────────
    if config.batching.analyze_lengths:
        print_length_analysis(list(dataset_train))
        if len(dataset_val) > 0:
            print("\nValidation set:")
            print_length_analysis(list(dataset_val))

    # ── Collate functions ─────────────────────────────────────────────────────
    collate_fn_train, collate_fn_val = make_collate_fns(config.batching, config.arch.seed)

    if config.batching.max_sequence_length is not None:
        print(
            f"\nSequence truncation: max_length={config.batching.max_sequence_length}, "
            f"strategy='{config.batching.truncation_strategy}'"
        )

    # ── Dataloaders ───────────────────────────────────────────────────────────
    dataloader_train, dataloader_val = make_train_val_dataloaders(
        dataset_train, dataset_val, config.batching, config.arch.seed, collate_fn_train, collate_fn_val
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model, run_name = create_model(
        model_type=config.model_type,
        D=D,
        arch=config.arch,
        terminating=dataset_train.terminating,
        seed=config.arch.seed,
        dataset_stats=dataset_stats,
    )

    if config.compile.enable:
        print(f"\ntorch.compile(mode='{config.compile.mode}', backend='{config.compile.backend}')…")
        model = torch.compile(model, mode=config.compile.mode, backend=config.compile.backend)  # type: ignore

    # ── Save path ─────────────────────────────────────────────────────────────
    save_path = resolve_save_path(config.save, ROOT, run_name, config.dataset_type)
    print(f"Model will be saved to: {save_path}")
    # Resume checkpoint lives beside the model file. Stable only when the save
    # path is stable across launches (save.overwrite=true, as the run generators
    # set) — auto_increment would move it and defeat resume.
    checkpoint_path = save_path.with_suffix(".ckpt")
    if config.save.checkpoint_freq > 0:
        print(f"Resumable checkpoint every {config.save.checkpoint_freq} steps → {checkpoint_path}"
              f"{' (resume enabled)' if config.save.resume else ''}")

    # ── W&B ───────────────────────────────────────────────────────────────────
    if config.wandb.run_name is None:
        config.wandb.run_name = run_name
    wandb_run = init_wandb_run(config.wandb, config)

    # ── Training ──────────────────────────────────────────────────────────────
    num_epochs = (config.training.num_steps * config.batching.batch_size) / max(len(dataset_train), 1)
    print(f"\nApprox. epochs: {num_epochs:.2f}")
    print("Starting training…\n")

    if config.profiling.enable:
        print(f"\nProfiling enabled: {config.profiling.steps} steps after 5 warmup steps")
        print(f"Profile output: {config.profiling.output}")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=5, active=config.profiling.steps, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            fit_model, train_lls, val_lls = batched_train_loop(
                model,
                dataloader_train,
                config,
                test_events=dataloader_val,
                wandb_run=wandb_run,
                profiler=prof,
                checkpoint_path=checkpoint_path,
                resume=config.save.resume,
                checkpoint_freq=config.save.checkpoint_freq,
            )
        prof.export_chrome_trace(config.profiling.output)
        print(f"\nProfile trace saved to: {config.profiling.output}")
        print("\n" + "=" * 60)
        print("Profiler Summary (sorted by CUDA time):")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        print("\n" + "=" * 60)
        print("Profiler Summary (sorted by CPU time):")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    else:
        fit_model, train_lls, val_lls = batched_train_loop(
            model,
            dataloader_train,
            config,
            test_events=dataloader_val,
            wandb_run=wandb_run,
            checkpoint_path=checkpoint_path,
            resume=config.save.resume,
            checkpoint_freq=config.save.checkpoint_freq,
        )

    best_val_ll = max(val_lls) if val_lls else float("-inf")

    torch.save(fit_model.state_dict(), str(save_path))
    print(f"\nModel saved to {save_path}")

    print("\nTraining complete!")
    if train_lls:
        print(f"  Final train LL: {train_lls[-1]:.4f}")
    if val_lls:
        print(f"  Final val LL  : {val_lls[-1]:.4f}")
        print(f"  Best  val LL  : {best_val_ll:.4f}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_metrics = None
    if config.eval.run_test_eval:
        print("\n" + "=" * 60)
        print("Running Test Evaluation")
        print("=" * 60)

        test_dataset = load_test_dataset(
            config.dataset_type, dict(config.dataset), DATA_ROOT, config.eval.limit_test_size
        )
        eval_sequences: Union[Any, list[MVEventData]] = test_dataset
        eval_kwargs: dict[str, Any] = {}

        if config.batching.max_sequence_length is not None:
            print(
                "Applying training truncation policy to test split: "
                f"max_length={config.batching.max_sequence_length}, "
                f"strategy='{config.batching.truncation_strategy}'"
            )
            eval_sequences = truncate_sequences_for_eval(
                list(test_dataset),
                max_length=config.batching.max_sequence_length,
                strategy=config.batching.truncation_strategy,
                seed=config.arch.seed,
            )
            eval_kwargs["time_scale_factor"] = test_dataset.time_scale_factor

        print("\nEvaluating model on test set…")
        test_metrics = evaluate_tpp(
            model=fit_model,
            sequences=eval_sequences,
            num_event_types=D,
            device=config.device,
            eval_batch_size=config.eval.eval_batch_size,
            ignore_indices=dataset_train.ignore_indices,
            compute_marginals=config.eval.compute_marginal_type,
            integrate_marginal_via_grid=config.eval.integrate_marginal_via_grid,
            grid_points_per_unit_time=config.eval.grid_points_per_unit_time,
            **eval_kwargs,
        )

        print("\n" + "=" * 60)
        print("Test Evaluation Results:")
        print("=" * 60)
        print(format_metrics(test_metrics, prefix="  "))

    # ── Log final metrics to W&B ──────────────────────────────────────────────
    if wandb_run is not None and wandb is not None:
        final_train_ll = float(train_lls[-1]) if train_lls else None
        final_val_ll = float(val_lls[-1]) if val_lls else None

        wandb.summary["best_val_ll"] = best_val_ll
        wandb.summary["final_val_ll"] = final_val_ll
        wandb.summary["final_train_ll"] = final_train_ll

        if test_metrics is not None:
            for key, value in test_metrics.items():
                wandb.summary[f"test_{key}"] = value

        log_payload = {
            "final/train_log_likelihood": final_train_ll,
            "final/val_log_likelihood": final_val_ll,
            "final/best_val_log_likelihood": best_val_ll,
            "artifact/save_path": str(save_path),
        }
        if test_metrics is not None:
            for key, value in test_metrics.items():
                log_payload[f"test/{key}"] = value

        wandb.log(log_payload)
        wandb.finish()


if __name__ == "__main__":
    main()
