# Configs

`torchpp` builds a single typed config object (`FitConfig` for training,
`StandaloneEvalConfig` for evaluation) by merging three layers. Later layers win over earlier
ones.

1. Dataclass defaults in
   [`torchpp/training/config.py`](../torchpp/training/config.py). Fields annotated `MISSING`
   must be filled by a later layer, otherwise training fails immediately.
2. YAML presets, stacked with `-c <path>` (or `--config <path>`). The flag can be repeated.
3. Dotted CLI overrides in the OmegaConf style (`key=value`), where dots select nested fields.

Schema validation, YAML composition and CLI parsing live in
[`torchpp/training/config_utils.py`](../torchpp/training/config_utils.py).
Mistakes stop the run with a one-line `error:` message. This happens for an unknown key (with a
"did you mean" suggestion), a value of the wrong type (`training.num_steps=abc`), an argument
without `=` (`training.num_steps 50`), an unknown flag, or a `dataset.*` key that the dataset
does not accept. `--help` lists the options and every preset.

## Resolving `-c <path>`

Each `-c <path>` is resolved in the following order:

1. The raw path as given.
2. `configs/<path>`.
3. `configs/<path>.yaml`.

This makes `-c model/spline_exp_hawkes` and `-c configs/model/spline_exp_hawkes.yaml`
equivalent.

A YAML file can recursively compose other presets through a `defaults:` list. For example,
[`configs/default_fit.yaml`](../configs/default_fit.yaml) pulls in `wandb/disabled`. Entries in
`defaults:` are resolved with the same rules and merged from left to right, and the file's own
keys are applied on top.

## Available presets

```
configs/
├── default_fit.yaml   # base training stack (batching, eval on the first 1000 test sequences)
├── default_eval.yaml  # base evaluation stack
├── demo.yaml          # every field with its default and a one-line comment
├── model/             # one YAML per model_type (26 files); each names its training family
├── data/              # the seven benchmark datasets: taxi, stackoverflow, retweet, amazon,
│                      #   taobao, earthquake, volcano
├── training/          # classical | neural (picked by the model preset) | smoke (10 steps)
├── eval/              # fast (200 test sequences) | full (all)
└── wandb/             # disabled (default) | offline | online
```

## Training presets

Each model preset names the training defaults of its family with `training_preset:`.

| Preset | Models | Learning rate | Schedule |
|---|---|---|---|
| `training/classical` | Poisson and Hawkes models (additive and inhibitive) | 0.1 | 40k steps, 500-step warmup, cosine to lr/100, early stopping (patience 4, eval every 250 steps) |
| `training/neural` | neural TPPs and the MLP-kernel Hawkes models | 0.01 (weight decay 1e-4) | same |

So `-c default_fit -c model/thp -c data/taxi` trains THP with the neural defaults, and no third
preset is needed. `training/smoke` changes only the length and schedule and keeps the learning
rate of the family. Any other schedule takes a few `training.*=` overrides (see
`configs/demo.yaml`).

The precedence, from lowest to highest, is:

1. The `training_preset` of the model preset, which is merged before every `-c` file.
2. The `-c` files in command-line order. Any `-c training/...` therefore overrides the family
   defaults, whether it comes before or after the model preset.
3. The `key=value` overrides (`training.learning_rate=0.003`). The override
   `training_preset=classical` switches the family.

`model/<name>.yaml` always sets `model_type: <name>` and the architecture fields of that model.
Most of these files are 5-10 lines long, as in this example.

```yaml
# configs/model/spline_exp_hawkes.yaml
model_type: spline_exp_hawkes
training_preset: classical

arch:
  baseline:
    num_knots: 5
    T_factor: 1.25
```

`data/<name>.yaml` sets `dataset_type` and the `dataset:` dict that the loader for that type
expects. The registered loaders and their schemas are in
[`torchpp/training/dataset_registry.py`](../torchpp/training/dataset_registry.py).

## Stacking presets

Presets and overrides can be stacked freely, and the whole `FitConfig` tree is exposed.

```bash
# base stack, a model, a dataset, W&B logging, then single-field overrides
python train.py \
    -c default_fit \
    -c model/spline_spline_hawkes \
    -c data/taxi \
    -c wandb/online \
    arch.baseline.num_knots=10 \
    arch.kernel.num_knots=8 \
    training.num_steps=20000 \
    training.learning_rate=0.01 \
    wandb.project=my-tpp-runs \
    device=cpu
```

Pass `--print` (or `-p`) to render the fully merged config and exit without training. This is
the fastest way to confirm a complex `-c` stack.

```bash
python train.py -c default_fit -c model/spline_exp_hawkes -c data/taxi \
    arch.baseline.num_knots=10 training.num_steps=500 --print
```

## Top-level `FitConfig`

`FitConfig` is defined in [`torchpp/training/config.py`](../torchpp/training/config.py).
Each entry below is a nested dataclass, and only the most-edited fields are listed. See the
source for the complete schema and the docstring of each field.

| Field | Notable knobs |
|---|---|
| `model_type` | one of [`VALID_MODEL_TYPES`](./models.md) (or the alias `hawkes`). **Required.** |
| `device` | `auto` (default), `cuda`, `cuda:1`, `cpu`. |
| `dataset_type` | `benchmark`, or any dataset registered with `@register_dataset`. |
| `dataset` | Dict consumed by the dataset loader. Validated at runtime. |
| `dataset_module`, `model_module` | Python modules imported before the config is validated, so datasets registered with `@register_dataset` and models registered with `register_model` there can be used. |
| `dataset.context_types` | Optional list of type indices read as history but never predicted. Overrides the `context_types` of the dataset's `default_token_roles`. |
| `dataset.ignored_types` | Optional list of type indices excluded from the evaluation targets. Overrides the `ignored_types` of the dataset's `default_token_roles`. |
| `batching` | `batch_size`, `use_bucket_batching`, `dynamic_batch_size`, `target_total_events`, `max_sequence_length`, `truncation_strategy`. |
| `training` | `num_steps`, `learning_rate`, `weight_decay`, `eval_freq`, `scheduler_type`, `grad_clip`, `early_stopping_patience`. |
| `arch` | The architecture of `model_type`. Its fields depend on the model (see [Architecture fields](#architecture-fields)). |
| `eval` | `limit_test_size`, `compute_marginal_type`, `eval_batch_size`, `integrate_marginal_via_grid`, `grid_points_per_unit_time`. |
| `wandb` | `enable`, `project`, `entity`, `run_name`, `mode`. |
| `save` | `path` (defaults to `$TORCHPPMODELS/<dataset_type>/<dataset_name>/<run_name>.pth`), `overwrite`, `auto_increment`. |
| `stats` | `enable`, `cache`, `cache_dir`, `max_sequences`, `subsample_seed`. These control the empirical `DatasetStats` used for data-driven initialization. |
| `profiling` | `enable`, `steps`, `output` for the PyTorch profiler. |
| `compile` | `enable`, `mode`, `backend` for `torch.compile()`. |

## Architecture fields

Each model type has its own architecture config
([`torchpp/training/arch.py`](../torchpp/training/arch.py)), so `arch:` accepts exactly the
fields of the selected model. Any other field fails with the list of valid fields.

```text
error: model_type 'spline_spline_hawkes' has no architecture field 'arch.encoder'. Valid fields:
arch.seed, arch.sampling, arch.ci, arch.baseline.num_knots, arch.baseline.T_factor, ...
```

The following fields are shared by every model:

| Field | Meaning |
|---|---|
| `arch.seed` | Seed for parameter initialization and training. `null` gives a random seed, which is recorded in the checkpoint. |
| `arch.ci` | For intensity-based models, overrides of the model's `CIConfig`, e.g. `arch.ci.n_points=200 arch.ci.rule=trapezoidal`. Numerically integrated models use `n_points: 100` unless set. |
| `arch.sampling` | Overrides of the model's `SamplingConfig`, e.g. `arch.sampling.mode=thinning`. |

The following sections are present only in the models built from the corresponding component:

| Section | Models | Fields (defaults) |
|---|---|---|
| `arch.baseline` | spline / piecewise baselines (`spline_*`, `piecewise_*`, `inhibitive_spline_*`) | `num_knots` (5), `T_factor` (1.0): extent = `T_factor × sequence_time_p95` |
| `arch.kernel` | spline / piecewise kernels (`*_spline_hawkes`, `const_piecewise_hawkes`) | `num_knots` (4), `T_factor` (5.0): extent = `T_factor × inter_event_time_p95`, `knot_schedule` (`geometric`), `knot_power` (2.0) |
| `arch.kernel` | `inhibitive_spline_mlp_hawkes` | `emb_dim` (32), `time_emb_dim` (32), `n_hidden` (64), `n_layers` (2) |
| `arch.kernel` | `inhibitive_spline_mlpspline_hawkes` | the spline-kernel fields plus `emb_dim`, `n_hidden`, `n_layers` |
| `arch.clamping_fn` | `inhibitive_*` | `softplus` (default), `relu`, `exp` |
| `arch.encoder` | `neural_*`, `thp`, `sahp` | `n_layers` (4), `n_heads` (4), `n_embd` (64), `dropout` (0.1) |
| `arch.encoder` | `intensity_free` | the transformer fields plus `type` (`gru` or `transformer`) |
| `arch.encoder` | `attnhp` | the transformer fields plus `time_emb_dim` (16), `use_layer_norm` (false) |
| `arch.encoder` | `nhp` | `hidden_dim` (64) |
| `arch.head` | `neural_spline`, `neural_piecewise_constant` | `num_bins` (4), `T_factor` (5.0), `knot_schedule`, `knot_power`, and for `neural_spline` also `smoothness_reg` (0.0) |
| `arch.head` | `neural_mlp` | `n_hidden` (128), `n_layers` (2), `time_emb_dim` (null → the encoder's `n_embd`) |
| `arch.head` | `intensity_free` | `num_mix_components` (8) |

`configs/demo.yaml` lists the same sections with comments. In Python, `create_model` takes the
same structure as a dict (`arch={"kernel": {"num_knots": 8}}`) or as the model's config
dataclass (`torchpp.build_arch("spline_spline_hawkes", {...})`).

`has_origin` is not a config option. It is part of the dataset's `default_token_roles`, which
is a fixed property of the data, so it is set on the dataset class and not overridden per run.
See [Notation → Token roles](./notation.md#token-roles).

## Standalone evaluation: `StandaloneEvalConfig`

`StandaloneEvalConfig` has the same dataset, arch and eval semantics as `FitConfig`. The only
required field is `model_path`. A checkpoint written by `train.py` stores its `model_type`,
architecture and dataset, and `evaluate.py` uses them unless the command line says otherwise.
A data preset or a `dataset.*` key selects another dataset, and explicit `arch.*=...` arguments
change single architecture fields (presets such as `default_eval` do not). A file holding only
a `state_dict` needs `model_type` and the training architecture (e.g. `-c model/<type>`).
Metrics are written to `output_path`, which defaults to the checkpoint path with `.eval.json` in
place of `.pth`.

```bash
python evaluate.py -c default_eval -c eval/full model_path=models/<...>.pth
python evaluate.py -c default_eval model_path=models/<...>.pth arch.ci.n_points=500
```

## Where saved checkpoints live

A relative `save.path` (and `model_path` in `evaluate.py`) is resolved against the current
working directory. `save.path` defaults to `None`, which expands to
`$TORCHPPMODELS/<dataset_type>/<dataset_name>/<run_name>.pth`. Here `<run_name>` is generated by
the model's run-name rule in
[`torchpp/training/registry.py`](../torchpp/training/registry.py) from `model_type` and the
relevant `arch` options (e.g. `spline_K5_exp_hawkes`, `neural_4l_128d_spline_kK4`). With
`save.auto_increment=true`, a numerical suffix is appended on collision instead of overwriting
the existing file. Next to the checkpoint, `train.py` writes the resolved configuration
(`<run_name>.yaml`) and the final metrics (`<run_name>.metrics.json`).
