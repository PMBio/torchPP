# UKB Real Hyperparameter Search

This document records the staged search strategy implemented in
`scripts/tuning/tune_ukb_real_combined_local.sh`.

## Scope

Dataset:
- `ukb_real`

The search is staged so that reusable architectural choices are selected once
and then reused by the more expressive families.

## Global Training Settings

Shared settings for all tuning runs:
- config stack: `-c default_fit -c training/transformer -c wandb/$WANDB_MODE -c data/ukb_real_full` (`WANDB_MODE` defaults to `offline`)
- `training.num_steps=50000`
- `training.eval_freq=500`
- `training.scheduler_warmup_steps=1000`
- `training.early_stopping_patience=6`
- `training.scheduler_eta_min = learning_rate / 100`
- `arch.spline_T_factor=1.0`
- `arch.spline_kernel_knot_power=4.5`
- `stats.cache=true` with cache files stored under `tmp/dataset_stats`
- `stats.max_sequences=20000` in `data/ukb_real_full` so initialisation stats are estimated from a deterministic subset instead of rescanning the full UKB training set every run
- checkpoints are written by the runner to `$TORCHPPMODELS/tuning/<job_name>.pth` (default `models/tuning/`)

Family-specific defaults:

| Family | Learning rate | `scheduler_eta_min` | Training steps |
| --- | ---: | ---: | ---: |
| Hawkes / spline / inhibitive | `0.01` | `lr / 100` | `50000` |
| Transformer-backed neural | `0.001` | `lr / 100` | `50000` |
| Numerical neural families | `0.001` | `lr / 100` | `50000` |

Operational knobs in the launcher:
- `--no-inhibitive` skips the 3 inhibitive stage-2 jobs and runs the tractable remainder.
- `--inhibitive-only` runs only the 3 inhibitive stage-2 jobs.
- `CI_NUMERICAL_STEPS` controls `arch.num_ci_integration_steps` for `neural_hawkes`, `neural_mlp`, and `neural_spline_hawkes`.
- `NUM_STEPS_INHIBITIVE` and `CI_INHIBITIVE_STEPS` let inhibitive models use a shorter schedule or cheaper numerical CI than the main stage-2 sweep.

Fixed neural defaults:
- `arch.dropout=0.1`
- transformer-backed models use fixed `training.weight_decay=1e-4` in the first pass
- `neural_hawkes` fixes `arch.neural_kernel_emb_dim=256`

## Stage 1: Reusable Component Tuning

Purpose:
- tune spline-baseline complexity,
- tune Hawkes-side spline-kernel complexity,
- tune the transformer backbone once,
- collect the best reusable values for later stages.

Models and grids:

| Model | Grid |
| --- | --- |
| `poisson` | single config |
| `inhomogeneous_poisson` | single config |
| `spline_poisson` | `arch.spline_K ∈ {15, 20}` |
| `hawkes` | single config |
| `linear_exp_hawkes` | single config |
| `const_spline_kernel` | `arch.spline_kernel_K ∈ {5, 8, 10}`, `arch.spline_kernel_T_factor ∈ {15, 20}` |
| `neural_poisson` | `arch.n_layer ∈ {8, 12}`, `arch.n_head ∈ {8, 16}`, `arch.n_embd ∈ {64, 128, 256}` |

Outputs reused later:
- `BEST_BASELINE_K_MAP[ukb_real]` from `spline_poisson`
- `BEST_KERNEL_K_MAP[ukb_real]` from `const_spline_kernel`
- `BEST_KERNEL_TF_MAP[ukb_real]` from `const_spline_kernel`
- `BEST_NEURAL_NL_MAP[ukb_real]` from `neural_poisson`
- `BEST_NEURAL_NH_MAP[ukb_real]` from `neural_poisson`
- `BEST_NEURAL_NE_MAP[ukb_real]` from `neural_poisson`

These values are stored in `scripts/tuning/tuning_ukb_real_best_params.sh`.

## Stage 2: Composite Family Tuning

Purpose:
- reuse the stage-1 winners instead of re-searching the same architectural axes,
- tune only the parameters that remain model-specific.

Models and grids:

| Model | Reused values | Grid |
| --- | --- | --- |
| `spline_exp_hawkes` | best baseline `K` | single config |
| `inhibitive_spline_exp_hawkes` | best baseline `K` | single config with `arch.inhibitive_clamping_fn=softplus` |
| `spline_spline_kernel` | best baseline `K`, best kernel `K`, best kernel `T_factor` | single config |
| `inhibitive_spline_spline_kernel` | best baseline `K`, best kernel `K`, best kernel `T_factor` | single config with `arch.inhibitive_clamping_fn=softplus` |
| `inhibitive_const_exp_hawkes` | none | single config with `arch.inhibitive_clamping_fn=softplus` |
| `neural_log_linear` | best transformer backbone | single config |
| `neural_spline` | best transformer backbone | `arch.spline_kernel_K ∈ {5, 8, 10}`, `arch.spline_kernel_T_factor ∈ {15, 20}` |
| `neural_hawkes` | best baseline `K` | `arch.neural_kernel_n_hidden ∈ {64, 128, 256}`, `arch.neural_kernel_n_mlp_layers ∈ {1, 2, 3}`, with `arch.neural_kernel_emb_dim=256` and `arch.neural_kernel_time_embd_dim=32` fixed |
| `neural_mlp` | best transformer backbone | `arch.mlp_n_hidden ∈ {64, 128, 256}`, `arch.mlp_n_layers ∈ {1, 2, 3}`, with `arch.mlp_time_embd_dim=32` fixed |
| `neural_spline_hawkes` | best baseline `K`, best kernel `K`, best kernel `T_factor` | `arch.neural_kernel_n_hidden ∈ {64, 128, 256}`, `arch.neural_kernel_n_mlp_layers ∈ {1, 2, 3}`, with `arch.neural_kernel_emb_dim=256` fixed |

## Stage 3: Backward-Compatible Alias

`--stage3` remains available as a convenience alias for running only the
`neural_spline_hawkes` subset. The actual search design is now a two-stage
process, with `neural_spline_hawkes` folded into Stage 2.

## Parameter Relationships

- `arch.spline_K` is the staged baseline-complexity parameter. It is selected in Stage 1 and reused by every later model that contains a spline baseline.
- `arch.spline_kernel_K` and `arch.spline_kernel_T_factor` are the staged Hawkes-side spline-kernel parameters. They are selected in Stage 1 and reused by the Hawkes composite families.
- The Hawkes-side kernel winner should not be forced onto `neural_spline`. That model reuses the transformer backbone, but still tunes its own spline head in Stage 2.
- `arch.n_embd`, `arch.n_layer`, and `arch.n_head` form one transformer-backbone block. They are tuned together on `neural_poisson` and then reused by `neural_log_linear`, `neural_spline`, and `neural_mlp`.
- `neural_hawkes` is not transformer-backed in this repository. Its tuned capacity lives in `arch.neural_kernel_n_hidden` and `arch.neural_kernel_n_mlp_layers`, while `arch.neural_kernel_emb_dim` is fixed to `256` in the first pass.

## Operational Workflow

Recommended order:

1. Run Stage 1 on `ukb_real`.
2. Inspect the tuning results and update `scripts/tuning/tuning_ukb_real_best_params.sh`.
3. Run Stage 2.
4. Optionally use `--stage3` only to run just the
  `neural_spline_hawkes` subset.
5. If inhibitive models are too slow, run Stage 2 with `--no-inhibitive`
  and handle them separately with `--inhibitive-only`.

Examples:

```bash
# Stage 1 dry-run
bash scripts/tuning/tune_ukb_real_combined_local.sh --dry-run --stage1

# Stage 1 actual tuning
GPUS=0,1 bash scripts/tuning/tune_ukb_real_combined_local.sh --stage1

# After updating scripts/tuning/tuning_ukb_real_best_params.sh
GPUS=0,1 bash scripts/tuning/tune_ukb_real_combined_local.sh --stage2

# Skip the inhibitive jobs in the main stage-2 sweep
GPUS=0,1 bash scripts/tuning/tune_ukb_real_combined_local.sh --stage2 --no-inhibitive

# Use a cheaper CI grid for the numerical neural models
CI_NUMERICAL_STEPS=50 GPUS=0,1 bash scripts/tuning/tune_ukb_real_combined_local.sh --stage2 --no-inhibitive

# Run only the inhibitive jobs later with a cheaper schedule
NUM_STEPS_INHIBITIVE=5000 CI_INHIBITIVE_STEPS=64 GPUS=0 bash scripts/tuning/tune_ukb_real_combined_local.sh --stage2 --inhibitive-only

# Optional: neural_spline_hawkes subset only
GPUS=0,1 bash scripts/tuning/tune_ukb_real_combined_local.sh --stage3

# Train the selected configs with 5 seeds
GPUS=0,1 bash scripts/tuning/tune_local_generic.sh scripts/tuning/generate_replica_commands_ukb.sh
```

## Notes

This is intentionally a staged search rather than a flat Cartesian product.
The core architectural decisions are made once, stored, and then reused. That
keeps the UKB real benchmark tractable while still allowing model-specific head
parameters to adapt in later stages. The stats cache removes most repeated
startup overhead across tuning jobs; set `stats.max_sequences=null` when you
need exact full-dataset statistics again.
