# EasyTPP Hyperparameter Search

This document records the exact staged search strategy used by
`scripts/tuning/tune_easytpp_combined_local.sh`.

## Scope

Datasets:
- `taxi`
- `stackoverflow`
- `retweet`
- `amazon`
- `taobao`
- `earthquake`
- `volcano`

The search is split into three stages so that simple models identify reusable
per-dataset hyperparameters before more expressive composite models are tuned.

## Global training settings

Shared settings for all tuning runs:
- config stack: `-c easytpp/default -c training/transformer -c wandb/$WANDB_MODE` (`WANDB_MODE` defaults to `offline`)
- `training.eval_freq=250`
- `training.scheduler_warmup_steps=500`
- `training.early_stopping_patience=4`
- `arch.spline_T_factor=1.25`
- `arch.spline_kernel_knot_power=4.0`

Family-specific defaults:

| Family | Learning rate | `scheduler_eta_min` | Training steps |
| --- | ---: | ---: | ---: |
| Hawkes / spline / inhibitive | `0.01` | `lr / 100` | `10000` |
| Neural models | `0.01` | `lr / 100` | `20000` |

## Stage 1: Independent component tuning

Purpose:
- tune baseline complexity,
- tune spline-kernel complexity,
- tune transformer backbone size,
- store the best per-dataset winners for later stages.

Models and grids:

| Model | Grid |
| --- | --- |
| `poisson` | single config |
| `spline_poisson` | `arch.spline_K ∈ {5, 10, 15}` |
| `hawkes` | single config |
| `const_spline_kernel` | `arch.spline_kernel_K ∈ {2, 4, 8}`, `arch.spline_kernel_T_factor ∈ {5, 10, 15, 20}` |
| `neural_poisson` | `arch.n_layer ∈ {2, 4}`, `arch.n_head ∈ {2, 4}`, `arch.n_embd ∈ {32, 64}`, with `training.weight_decay=0` and `arch.dropout=0.1` |

Outputs reused later:
- `BEST_BASELINE_K_MAP[dataset]` from `spline_poisson`
- `BEST_KERNEL_K_MAP[dataset]` from `const_spline_kernel`
- `BEST_KERNEL_TF_MAP[dataset]` from `const_spline_kernel`
- `BEST_NEURAL_NL_MAP[dataset]` from `neural_poisson`
- `BEST_NEURAL_NH_MAP[dataset]` from `neural_poisson`
- `BEST_NEURAL_NE_MAP[dataset]` from `neural_poisson`

These values are stored in `scripts/tuning/tuning_easytpp_best_params.sh`.

## Stage 2: Composite EasyTPP model tuning

Purpose:
- reuse the stage-1 winners to avoid re-searching baseline and backbone sizes,
- only search the remaining composite-model dimensions.

Models and grids:

| Model | Reused values | Grid |
| --- | --- | --- |
| `spline_exp_hawkes` | best baseline `K` | single config |
| `spline_spline_kernel` | best baseline `K`, best kernel `K`, best kernel `T_factor` | single config |
| `inhibitive_const_exp_hawkes` | none | single config |
| `inhibitive_spline_spline_kernel` | best baseline `K`, best kernel `K`, best kernel `T_factor` | single config |
| `neural_log_linear` | best neural backbone `n_layer`, `n_head`, `n_embd` | single config |
| `neural_spline` | best neural backbone `n_layer`, `n_head`, `n_embd` | `arch.spline_kernel_K ∈ {2, 4, 8}`, `arch.spline_kernel_T_factor ∈ {5, 10, 15, 20}` |

## Stage 3: Neural Hawkes family tuning

Purpose:
- tune the neural Hawkes models using the same staged handoff,
- reuse the already selected per-dataset backbone for `neural_mlp` to keep the
  search space manageable.

Models and grids:

| Model | Reused values | Grid |
| --- | --- | --- |
| `neural_hawkes` | best baseline `K`, best neural backbone | `arch.neural_kernel_n_mlp_layers ∈ {1, 2}` with fixed `arch.neural_kernel_emb_dim=32`, `arch.neural_kernel_time_embd_dim=32`, `arch.neural_kernel_n_hidden=32`, `training.weight_decay=0`, `arch.dropout=0.1` |
| `neural_spline_hawkes` | best baseline `K`, best kernel `K`, best kernel `T_factor`, best neural backbone | `arch.neural_kernel_n_mlp_layers ∈ {1, 2}` with the same fixed neural-kernel dimensions and `training.weight_decay=0` |
| `neural_mlp` | best neural backbone | `arch.mlp_n_hidden ∈ {16, 32}`, `arch.mlp_n_layers ∈ {1, 2, 3}`, `arch.mlp_time_embd_dim=32`, `training.weight_decay ∈ {0, 1e-4, 1e-3}` |

## Operational workflow

Recommended order:

1. Run stage 1 on the target datasets.
2. Inspect the tuning results and update `scripts/tuning/tuning_easytpp_best_params.sh`.
3. Run stage 2 on the same datasets.
4. Run stage 3 on the same datasets.

Examples:

```bash
# Stage 1 for the missing datasets
bash scripts/tuning/tune_easytpp_combined_local.sh --stage1 --datasets=taobao,earthquake,volcano

# After updating scripts/tuning/tuning_easytpp_best_params.sh
bash scripts/tuning/tune_easytpp_combined_local.sh --stage2 --datasets=taobao,earthquake,volcano
bash scripts/tuning/tune_easytpp_combined_local.sh --stage3 --datasets=taobao,earthquake,volcano
```

## Notes for the paper

The key methodological point is that the search is not a flat Cartesian product
over all model hyperparameters. It is a staged search:

- simple baseline and backbone hyperparameters are tuned first,
- the best values are reused per dataset,
- later model families only search the parameters that remain model-specific.

This reduces the number of runs while preserving dataset-specific adaptation.