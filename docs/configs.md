# Configs

`torchpp` builds a single typed config object (`FitConfig` for training,
`StandaloneEvalConfig` for evaluation) by merging three layers, **later
wins**:

1. **Dataclass defaults** in
   [`torchpp/training/config.py`](../torchpp/training/config.py). Fields
   annotated `MISSING` must be filled by a later layer or training fails
   immediately.
2. **YAML presets** stacked via `-c <path>` (or `--config <path>`), repeatable.
3. **Dotted CLI overrides**, OmegaConf-style: `key=value`, with dots into
   nested fields.

Schema validation, YAML composition, and CLI parsing live in
[`torchpp/training/config_utils.py`](../torchpp/training/config_utils.py).
OmegaConf runs in *struct mode*, so unknown keys fail loudly instead of
being silently ignored.

## Resolving `-c <path>`

Each `-c <path>` is resolved in this order:

1. raw path as given,
2. `configs/<path>`,
3. `configs/<path>.yaml`.

So `-c model/spline_exp_hawkes` and
`-c configs/model/spline_exp_hawkes.yaml` are equivalent.

A YAML file can recursively compose other presets via a `defaults:` list.
For example, [`configs/default_fit.yaml`](../configs/default_fit.yaml)
pulls in `batching/bucket_default_small`, `training/very_long`,
`wandb/disabled`, and `eval/normal`. Entries in `defaults:` are resolved
with the same rules and merged left-to-right; the file's own keys win on
top.

## Available presets

```
configs/
├── default_fit.yaml         # Base training stack (batching + training schedule + eval)
├── default_eval.yaml        # Base evaluation stack
├── model/                   # One YAML per model_type (21 entries)
├── data/                    # ukb_real_full, ukb_simulated_*, easytpp_<name>, mimic_iv_*
├── training/                # transformer | very_long | easytpp_default | easytpp_neural
├── batching/                # bucket_default_small | bucket_numerical_small
├── eval/                    # fast | normal | full | skip
└── wandb/                   # online | offline | disabled
```

`model/<name>.yaml` always sets `model_type: <name>` and tweaks any arch
parameters specific to that model. Most are 5-10 lines. Example:

```yaml
# configs/model/spline_exp_hawkes.yaml
model_type: spline_exp_hawkes

arch:
  spline_K: 5
  spline_T_factor: 1.25
```

`data/<name>.yaml` sets `dataset_type` plus the `dataset:` dict that the
loader for that type expects (see
[`torchpp/training/dataset_registry.py`](../torchpp/training/dataset_registry.py)
for the registered loaders and their schemas).

## Stacking presets

Anything stacks. The whole `FitConfig` tree is exposed:

```bash
python train.py \
    -c default_fit \                  # base: batching + schedule + eval
    -c model/spline_spline_kernel \   # pick a model
    -c training/transformer \         # swap in cosine + warmup schedule
    -c wandb/online \                 # log to wandb
    arch.spline_K=10 \                # override any nested field
    arch.spline_kernel_K=8 \
    training.num_steps=20000 \
    training.learning_rate=0.01 \
    wandb.project=my-tpp-runs \
    device=cpu
```

Pass `--print` (or `-p`) to render the fully-merged config and exit
without training — the fastest way to confirm a complex `-c` stack:

```bash
python train.py -c default_fit -c model/spline_exp_hawkes \
    arch.spline_K=10 training.num_steps=500 --print
```

## Top-level `FitConfig`

Defined in [`torchpp/training/config.py`](../torchpp/training/config.py).
Each entry below is a nested dataclass; only the most-edited fields are
listed. See the source for the complete schema and per-field docstrings.

| Field | Notable knobs |
|---|---|
| `model_type` | one of [`VALID_MODEL_TYPES`](./models.md). **Required.** |
| `device` | `auto` (default), `cuda`, `cuda:1`, `cpu`. |
| `dataset_type` | `ukb`, `easytpp`, `mimic_iv`, … (registered in `dataset_registry.py`). |
| `dataset` | Dict consumed by the dataset loader. Validated at runtime. |
| `batching` | `batch_size`, `use_bucket_batching`, `dynamic_batch_size`, `target_total_events`, `max_sequence_length`, `truncation_strategy`. |
| `training` | `num_steps`, `learning_rate`, `weight_decay`, `eval_freq`, `scheduler_type`, `grad_clip`, `early_stopping_patience`. |
| `arch` | `spline_K`, `spline_kernel_K`, `spline_T_factor`, `spline_kernel_T_factor`, `inhibitive_clamping_fn`, `n_layer`, `n_embd`, `n_head`, `dropout`, MLP / neural-kernel sizes. |
| `eval` | `limit_test_size`, `compute_marginal_type`, `eval_batch_size`, `integrate_marginal_via_grid`, `grid_points_per_unit_time`. |
| `wandb` | `enable`, `project`, `entity`, `run_name`, `mode`. |
| `save` | `path` (defaults to `$TORCHPPMODELS/<dataset_type>/<run_name>.pth`), `overwrite`, `auto_increment`. |
| `stats` | `enable`, `cache`, `cache_dir`, `max_sequences`, `subsample_seed` — controls the empirical `DatasetStats` used for data-driven init. |
| `profiling` | `enable`, `steps`, `output` — PyTorch profiler. |
| `compile` | `enable`, `mode`, `backend` — `torch.compile()`. |

## Standalone evaluation: `StandaloneEvalConfig`

Same dataset / batching / arch / eval semantics as `FitConfig`, but the
required fields are `model_path` (path to the `.pth` checkpoint) and
`model_type` (must match the checkpoint).

```bash
python evaluate.py \
    -c default_eval \
    -c data/ukb_real_full \
    -c eval/full \
    model_path=models/<...>.pth \
    model_type=spline_exp_hawkes
```

## Where saved checkpoints live

`save.path` defaults to `None`, which expands to
`$TORCHPPMODELS/<dataset_type>/<run_name>.pth`, where `<run_name>` is
generated by
[`_build_run_name`](../torchpp/training/model_factory.py) from
`model_type` plus the relevant `arch` knobs (e.g.
`spline_K5_exp_hawkes`, `neural_4l_128d_spline_kK4`). With
`save.auto_increment=true`, a numerical suffix is appended on collision
instead of overwriting.
