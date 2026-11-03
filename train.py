"""Unified training entry point for TorchPP.

Dataset selection is controlled by the ``dataset_type`` field in the config
(built in: ``"benchmark"``); the same script covers every registered dataset.

Usage examples
--------------
.. code-block:: bash

    # spline-baseline Hawkes on Taxi
    python train.py -c default_fit -c model/spline_exp_hawkes -c data/taxi

    # SAHP on Retweet
    python train.py -c default_fit -c model/sahp -c data/retweet

    # CPU run with a short schedule
    python train.py -c default_fit -c model/const_exp_hawkes -c data/taxi \\
        training.num_steps=500 device=cpu

Datasets must be available locally. For the benchmark datasets run::

    python scripts/data_prep/download_benchmark_datasets.py
"""

import json
from pathlib import Path
import sys
from typing import Any, Union

import torch
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
from torch.profiler import profile, ProfilerActivity

# Importing the data package triggers @register_dataset decorators for all
# built-in dataset classes.
import torchpp.data  # noqa: F401
from torchpp.training.checkpoint import save_model
from torchpp.training.config import FitConfig
from torchpp.training.config_utils import load_config, print_config
from torchpp.training.dataset_registry import get_data_root
from torchpp.training.model_factory import create_model
from torchpp.training.registry import resolve_model_type
from torchpp.training.trainer import fit
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Unified training entry point. Dataset selection is driven by config."""
    try:
        run()
    except MissingMandatoryValue as err:
        key = getattr(err, "full_key", "") or str(err)
        hint = " Add -c model/<type>." if "model_type" in key else ""
        sys.exit(f"error: missing required value '{key}' (set {key}=...).{hint}")
    except (ValueError, FileNotFoundError) as err:
        sys.exit(f"error: {err}")


def run() -> None:
    """Load the config, train, save and evaluate (see :func:`main` for error handling)."""
    config: FitConfig = load_config(
        FitConfig,
        configs_dir=Path(__file__).resolve().parent / "configs",
        description="Train a TPP model from YAML presets and key=value overrides, save it and evaluate it.",
    )
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
    collate_fn_train, collate_fn_val = make_collate_fns(
        config.batching, config.arch.seed, has_origin=dataset_train.token_roles.has_origin
    )

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
    model = create_model(
        model_type=config.model_type,
        D=D,
        arch=config.arch,
        terminating=dataset_train.terminating,
        seed=config.arch.seed,
        dataset_stats=dataset_stats,
        token_roles=dataset_train.token_roles,
        verbose=True,
    )
    run_name = model.run_name
    # Global RNG draws during training (dropout, Monte-Carlo integration) follow the same seed.
    torch.manual_seed(config.arch.seed)

    if config.compile.enable:
        print(f"\ntorch.compile(mode='{config.compile.mode}', backend='{config.compile.backend}')…")
        model = torch.compile(model, mode=config.compile.mode, backend=config.compile.backend)  # type: ignore

    # ── Save path ─────────────────────────────────────────────────────────────
    dataset_name = config.dataset.get("dataset_name")
    save_subdir = f"{config.dataset_type}/{dataset_name}" if dataset_name else config.dataset_type
    save_path = resolve_save_path(config.save, ROOT, run_name, save_subdir)
    print(f"Model will be saved to: {save_path}")
    # Resume checkpoint lives beside the model file. Stable only when the save
    # path is stable across launches (save.overwrite=true) — auto_increment would
    # move it and defeat resume.
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
    print("Starting training…")
    print(
        "('Val LL (per event)' is the validation-set log-likelihood per scored event: full sequences, "
        "survival term included. The test reports the same quantity as ll_per_event.)\n"
    )

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
            result = fit(
                model,
                dataloader_train,
                dataloader_val,
                config,
                wandb_run=wandb_run,
                profiler=prof,
                checkpoint_path=checkpoint_path,
                resume=config.save.resume,
                checkpoint_freq=config.save.checkpoint_freq,
                verbose=True,
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
        result = fit(
            model,
            dataloader_train,
            dataloader_val,
            config,
            wandb_run=wandb_run,
            checkpoint_path=checkpoint_path,
            resume=config.save.resume,
            checkpoint_freq=config.save.checkpoint_freq,
            verbose=True,
        )

    fit_model, train_lls, val_lls = result.model, result.train_ll, result.val_ll
    best_val_ll = result.best_val_ll

    save_model(
        save_path,
        fit_model,
        model_type=resolve_model_type(config.model_type),
        arch=config.arch,
        D=D,
        seed=config.arch.seed,
        dataset_type=config.dataset_type,
        dataset=dict(config.dataset),
    )
    save_path.with_suffix(".yaml").write_text(OmegaConf.to_yaml(OmegaConf.structured(config)))
    print(f"\nModel saved to {save_path} (config: {save_path.with_suffix('.yaml')})")

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
                has_origin=dataset_train.token_roles.has_origin,
            )

        print("\nEvaluating model on test set…")
        test_metrics = evaluate_tpp(
            model=fit_model,
            sequences=eval_sequences,
            num_event_types=D,
            device=config.device,
            eval_batch_size=config.eval.eval_batch_size,
            compute_marginals=config.eval.compute_marginal_type,
            integrate_marginal_via_grid=config.eval.integrate_marginal_via_grid,
            grid_points_per_unit_time=config.eval.grid_points_per_unit_time,
            verbose=True,
        )

        print("\n" + "=" * 60)
        print("Test Evaluation Results:")
        print("=" * 60)
        print(format_metrics(test_metrics, prefix="  "))

    metrics_path = save_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(
        {"best_val_ll": best_val_ll, "steps": result.steps, "test": test_metrics}, indent=2, default=float
    ))
    print(f"Metrics written to {metrics_path}")

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
