# API reference

This page lists the public symbols of each subpackage with their signatures. Each section links
to the source, where every class and function has its full docstring. The concepts are explained
in [Notation](./notation.md), [Architecture](./architecture.md) and [Models](./models.md).

The symbols in the block below are re-exported from the package root. All other symbols are
imported from their subpackage, for example `from torchpp.training import default_fit_config`.

```python
from torchpp import (
    # data containers and batching
    MVEventData, BatchedMVEventData, DatasetStats, TokenRoles,
    BucketBatchSampler, TruncationStrategy, collate_batch_with_truncation,
    analyze_length_distribution, print_length_analysis,
    # base classes and configs
    TemporalPointProcess, AutoregressiveTPP, IntensityBasedTPP, DensityBasedTPP,
    BaselineTPP, HawkesProcess, InhibitiveHawkesProcess,
    NeuralParametrizedTPP, TransformerParametrizedTPP,
    CIConfig, SamplingConfig, Sample,
    # Poisson processes
    PoissonProcess, LogLinearPoissonProcess, SplinePoissonProcess, PiecewisePoissonProcess,
    # Hawkes processes
    ConstBaselineExpKernelHawkesProcess, LogLinearBaselineExpKernelHawkesProcess,
    SplineBaselineExpKernelHawkesProcess, PiecewiseBaselineExpKernelHawkesProcess,
    ConstBaselineSplineKernelHawkesProcess, ConstBaselinePiecewiseKernelHawkesProcess,
    SplineBaselineSplineKernelHawkesProcess,
    InhibitiveConstBaselineExpKernelHawkesProcess, InhibitiveSplineBaselineExpKernelHawkesProcess,
    InhibitiveSplineBaselineSplineKernelHawkesProcess,
    InhibitiveSplineBaselineMLPKernelHawkesProcess, InhibitiveSplineBaselineMLPSplineKernelHawkesProcess,
    # neural TPPs
    NeuralPoissonProcess, NeuralLogLinearProcess, NeuralSplineProcess,
    NeuralPiecewiseConstantProcess, NeuralMLPProcess,
    THPProcess, SAHPProcess, IntensityFreeProcess, NHPProcess, AttNHPProcess,
    # model registry, training, checkpoints
    create_model, register_model, MODEL_REGISTRY, MODEL_ALIASES, VALID_MODEL_TYPES,
    fit, FitResult, save_model, load_model,
    # datasets
    TPPDataset, BenchmarkDataset,
)
```

## `torchpp.tpp`: models

### Base classes

[`torchpp/tpp/base_tpp.py`](../torchpp/tpp/base_tpp.py)

- **`TemporalPointProcess(D, *, seed=None, terminating=False, time_scale=1.0, token_roles=None, sampling_config=None)`**:
  the abstract root class with the contract that every model implements. Times are passed and
  returned in native units. Time arguments `t` have shape `(B,)` or `(B, Q)`. Distribution
  methods take the keyword-only `log=True` (`log=False` returns probabilities and densities).
  Each method also accepts a single `MVEventData` instead of a batch.
  - `likelihood(ts, T=None, *, log=True) -> (B,)`: per-sequence log-likelihood.
  - `num_scored_events(ts) -> (B,)`: number of scored events per sequence. Divide a summed
    log-likelihood by it to get the per-event value (see
    [Notation → Scored events](./notation.md#scored-events)).
  - `loss(ts) -> (B,)`: training objective (`-likelihood` for autoregressive models).
  - `sample(ts=None, num_steps=1, rng=None, sampling_config=None) -> Sample`: a NamedTuple
    `(sequence, times, mark_probs)` with the history extended by up to `num_steps` sampled
    events, their absolute times and the mark distribution at each sampled time.
  - `density_joint(t, ts, *, log=True) -> (B, [Q,] D)`, `density_time(t, ts, *, log=True) -> (B, [Q])`.
  - `cdf_time(t, ts, *, log=True)`, `survival_time(t, ts, *, log=True) -> (B, [Q])`.
  - `prob_mark(ts, *, log=True) -> (B, D)`, `prob_mark_given_time(t, ts, *, log=True) -> (B, [Q,] D)`,
    `density_time_given_mark(t, ts, *, log=True) -> (B, [Q,] D)`.
  - `time_mean(ts) -> (B,)`, `time_quantile(u, ts) -> (B,)`. Both return `inf` when the next
    event may never occur.

  Predictions start at `T_obs` when it lies after the last event
  ([Notation → Forecasting](./notation.md#forecasting-from-t_obs)). `token_roles`
  (`model.token_roles`, a `TokenRoles`) is explained under
  [Notation → Token roles](./notation.md#token-roles). `model.has_origin` is
  `model.token_roles.has_origin`.
- **`AutoregressiveTPP`**: the base class for models whose likelihood factorizes over events.
  It provides `likelihood` and `loss` from a per-event hook.
- **`IntensityBasedTPP(D, *, seed=None, terminating=False, time_scale=1.0, token_roles=None, ci_config=None, sampling_config=None)`**:
  models defined by an intensity.
  - `intensity(t, ts) -> (B, [Q,] D)`, with termination masking.
  - `cumulative_intensity(T, ts, T_low=None, ci_config=None) -> (B, D)`: Λ over `[T_low or 0, T]`.
  - `ci_config` / `use_ci(cfg)` and `sampling_config` / `use_sampling(cfg)`: the active
    configuration, and context managers that change it for a block.
  - The grid builders and inverse-Λ helpers are internal and work in the model's internal time
    units. Use the `time_*`, `prob_*`, `density_*` and `intensity` methods above instead, which
    take and return native time.

  Subclasses must implement the hook `_intensities_at_queries(query_times, ts) -> (B, Q, D)`
  ([Extending](./extending.md)). The optional hooks are
  `_closed_form_ci(ts, T_low, T) -> (B, D)` or `_interval_ci(ts, t_lo, t_hi) -> (B, L+1, D)`, `_default_ci_config()`,
  `_event_intensities(ts) -> (B, L)`, `_sequence_intensities(ts) -> (B, L, D)`,
  and `_interval_intensities(query_times, ts)`.
- **`HawkesProcess` / `InhibitiveHawkesProcess`**: `baseline_values(t) -> (Q, D)`, `kernel_values(dt) -> (Q, D, D)`
  and `branching_matrix(t_max=None) -> (D, D)`, in native time units, indexed `[target, source]`
  ([Models](./models.md#hawkes-processes)). **`BaselineTPP`** (Poisson models): `baseline_values(t)`.
  `transform_params()` returns the raw parameters in internal time units.
- **`DensityBasedTPP(D, *, seed=None, terminating=False, time_scale=1.0, token_roles=None, sampling_config=None)`**:
  models defined by an inter-event-time density and a mark distribution, without an
  intensity (`IntensityFreeProcess`). Its hooks are `_log_density_time_given_mark`,
  `_log_survival_time_given_mark`, `_log_prob_mark`, `_event_log_density_sum`, `_draw_next`.
- **`CIConfig(mode='numerical', placement='interval', rule='mc_trapezoidal', n_points=100, chunk_size=None, ...)`**:
  how Λ is computed. `mode` is `closed_form` or `numerical`. Numerical integration is set by
  `placement`, `rule` (`trapezoidal`, `mc`, `mc_trapezoidal`) and `n_points`. With
  `placement='interval'` we integrate per inter-event interval (second order), and with
  `placement='global'` we use one grid over the window. A call argument takes precedence over
  `use_ci`, which takes precedence over the model default. Predictive methods always use a
  deterministic rule.
- **`SamplingConfig(mode='inversion', inversion_method='auto', newton_tol=1e-6, newton_max_iters=50, bracket_width=0.1, grid_density=None, ...)`**:
  how `sample` draws events. The mode is either `inversion` (inverse CDF, with the method
  `auto`, `closed_form`, `newton` or `grid`) or `thinning` (Ogata).


### Model classes

The concrete models, their `model_type` strings and parameters are listed in
[Models](./models.md). Every argument after `D` (or `encoder`) is keyword-only. Every
model accepts `seed`, `terminating`, `time_scale`, `token_roles` and `sampling_config`. Models
in the intensity family also accept `ci_config`. The constructor signatures are listed below.

[`torchpp/tpp/baseline.py`](../torchpp/tpp/baseline.py)

- `PoissonProcess(D=None, *, params=None, seed=None, terminating=False, ...)`
- `LogLinearPoissonProcess(D=None, *, params=None, seed=None, terminating=False, ...)`
- `SplinePoissonProcess(D, *, baseline_num_knots, baseline_delta_t, params=None, seed=None, terminating=False, ...)`
- `PiecewisePoissonProcess(D, *, baseline_num_knots, baseline_delta_t, params=None, seed=None, terminating=False, ...)`

[`torchpp/tpp/hawkes.py`](../torchpp/tpp/hawkes.py)

- `HawkesProcess(baseline, kernel, D, *, seed=None, terminating=False, ...)` adds a baseline
  and a kernel. `InhibitiveHawkesProcess(positive_constraint, baseline, kernel, D, *, ...)` takes
  the sum and then applies a positive clamp. The clamp is given by `resolve_clamping_fn(name)`,
  with the names in `VALID_CLAMPING_FNS` (`softplus`, `relu`, `exp`).
- `ConstBaselineExpKernelHawkesProcess(D, *, baseline_params=None, kernel_params=None, seed=None, ...)`,
  likewise `LogLinearBaselineExpKernelHawkesProcess` and `InhibitiveConstBaselineExpKernelHawkesProcess`.
- `SplineBaselineExpKernelHawkesProcess(D, *, baseline_num_knots, baseline_delta_t, ...)`, likewise
  `PiecewiseBaselineExpKernelHawkesProcess`, `InhibitiveSplineBaselineExpKernelHawkesProcess`
  and `InhibitiveSplineBaselineMLPKernelHawkesProcess`.
- `ConstBaselineSplineKernelHawkesProcess(D, *, kernel_num_knots, kernel_delta_t, ...)`, likewise
  `ConstBaselinePiecewiseKernelHawkesProcess`.
- `SplineBaselineSplineKernelHawkesProcess(D, *, baseline_num_knots, kernel_num_knots, baseline_delta_t, kernel_delta_t, ...)`,
  likewise `InhibitiveSplineBaselineSplineKernelHawkesProcess`.
- `InhibitiveSplineBaselineMLPSplineKernelHawkesProcess(D, *, baseline_num_knots, baseline_delta_t, kernel_num_knots, kernel_delta_t, ...)`.

[`torchpp/tpp/neural.py`](../torchpp/tpp/neural.py)

- `NeuralParametrizedTPP(encoder, *, terminating=False, seed=None, ...)` is the base class for
  encoder-conditioned intensity models. `TransformerParametrizedTPP(config, *, ...)` builds
  the transformer encoder from a `TransformerConfig`.
- The transformer-encoder models take `D` or `config=TransformerConfig(...)`, for example
  `NeuralSplineProcess(D=3, num_bins=5)`. These models are
  `NeuralPoissonProcess`, `NeuralLogLinearProcess`, `THPProcess`, `SAHPProcess`,
  `NeuralMLPProcess(..., n_hidden=128, n_layers=..., time_emb_dim=..., ...)`,
  `NeuralSplineProcess(D=None, *, config=None, num_bins=5, knot_gaps=None, t_max=1.0, ...)`,
  `NeuralPiecewiseConstantProcess(D=None, *, config=None, num_bins=5, knot_gaps=None, t_max=1.0, ...)`.
- `NHPProcess(encoder: ContTimeLSTMEncoder, *, ...)`, `AttNHPProcess(encoder: AttNHPEncoder, *, ..., quadrature_chunk_size=10)`,
  `IntensityFreeProcess(encoder: SequenceEncoder, *, num_mix_components=8, ...)`.

## `torchpp.training`: building, training, checkpoints

- **`create_model(model_type, D, *, arch=None, terminating=False, seed=None, load_path=None, dataset_stats=None, token_roles=None, verbose=False)`**
  ([`model_factory.py`](../torchpp/training/model_factory.py)): builds a model from its
  `model_type` and that model's architecture config. The config is a nested dict such as
  `{"kernel": {"num_knots": 8}}` or the dataclass from `build_arch`, and its `ci` / `sampling`
  entries override fields of the model's `CIConfig` / `SamplingConfig`. The initialization is
  data-driven, and the time unit comes from `dataset_stats`. `seed` seeds the initialization
  without touching the global RNG. `load_path` takes a `save_model` checkpoint or a bare `state_dict` that stores
  `time_scale`. The descriptive run name is `model.run_name`, and `model.model_type` and
  `model.arch_config` record how the model was built.
- **`MODEL_REGISTRY`**, **`MODEL_ALIASES`**, **`VALID_MODEL_TYPES`**, `resolve_model_type(name)`
  ([`registry.py`](../torchpp/training/registry.py)): one `ModelSpec` per `model_type`, with
  the class, description, builder, run-name rule and experimental flag. `hawkes` is an alias of
  `const_exp_hawkes`.
- **`register_model(name, cls=None, *, build=None, description='', run_name=None, experimental=False, aliases=())`**:
  adds your own model to the registry and can also be used as a class decorator. The model then
  works with `create_model`, checkpoints and the command line (`model_module=<module>`).
- **`fit(model, train, val=None, config=None, *, device=None, wandb_run=None, profiler=None, checkpoint_path=None, resume=False, checkpoint_freq=0, verbose=False) -> FitResult`**
  ([`trainer.py`](../torchpp/training/trainer.py)): AdamW with the configured schedule,
  gradient accumulation and clipping, periodic validation, early stopping, and the best
  validation weights restored at the end. `train`/`val` are `DataLoader`s or lists of
  `MVEventData`. Training runs on the model's device unless `device=` (or `config.device`) says
  otherwise. To train on a GPU, move the model with `model.to("cuda")`. `config` is not
  modified. `verbose=True` prints progress.
- **`FitResult(model, best_val_ll, train_ll, val_ll, steps)`**: `best_val_ll` is `None`
  without validation.
- **`default_fit_config(num_steps=1000, learning_rate=0.01, eval_freq=100, batch_size=32, **training) -> FitConfig`**:
  a complete config for Python use.
- **`save_model(path, model, *, model_type=None, D=None, arch=None, seed=None, dataset_type=None, dataset=None) -> Path`**
  (`model_type`, `D` and `arch` default to what `create_model` recorded and are checked against
  the model if given explicitly),
  **`load_model(path, device='cpu', **arch_overrides)`**, **`read_checkpoint(path) -> dict`**
  ([`checkpoint.py`](../torchpp/training/checkpoint.py)): checkpoints store the weights
  with the model class and its constructor arguments, the active `CIConfig` /
  `SamplingConfig`, and, for `create_model` models, the model type and architecture. They also
  store `D`, the terminating and origin flags, `time_scale`, the seed, the dataset and the
  TorchPP version. Any model is rebuilt exactly, however it was constructed. With
  `arch_overrides`, the model is rebuilt through `create_model`.
- Config dataclasses ([`config.py`](../torchpp/training/config.py)): `FitConfig`,
  `StandaloneEvalConfig` and their groups `BatchingConfig`, `TrainingConfig`,
  `EvalConfig`, `WandbConfig`, `SaveConfig`, `DatasetStatsConfig`, `ProfilingConfig`,
  `CompileConfig`. See [Configs](./configs.md).
- Architecture configs ([`arch.py`](../torchpp/training/arch.py)): one dataclass per model type,
  declared on its registry entry (`MODEL_REGISTRY[name].config_cls`). The base classes are
  `ArchConfig`, with the shared fields `seed` and `sampling`, and `IntensityArchConfig`, which
  adds `ci`. The component sections are `BaselineKnotsConfig`, `KernelKnotsConfig`, `MLPKernelConfig`, `MLPSplineKernelConfig`,
  `TransformerEncoderConfig`, `IntensityFreeEncoderConfig`, `CTLSTMEncoderConfig`,
  `AttNHPEncoderConfig`, `BinsHeadConfig`, `SplineHeadConfig`, `MLPHeadConfig`,
  `MixtureHeadConfig`. `build_arch(model_type, values=None)` validates a nested dict. Fields that
  the model does not have raise an error that lists the valid ones.
  `merge_arch(model_type, arch, overrides)` applies overrides.
- Dataset registry ([`dataset_registry.py`](../torchpp/training/dataset_registry.py)):
  `register_dataset(name)` (class decorator), `load_split(dataset_type, cfg, root, split)`,
  `get_data_root()` (`$TORCHPPDATA`, default `~/data/torchpp`).

## `torchpp.data`: datasets

- **`TPPDataset`** ([`base_dataset.py`](../torchpp/data/base_dataset.py)): abstract
  `Dataset` of `MVEventData` with class attributes `terminating`, `default_token_roles`
  and `cfg_keys`. `dataset.token_roles` applies per-run overrides. See
  [Data → Your own dataset](./data.md#your-own-dataset).
- **`BenchmarkDataset(data_dir, split, dataset_name, limit_size=None)`**,
  `load_benchmark_sequences(data_dir, split, limit_size=None) -> list[MVEventData]`,
  `num_event_types_for(dataset_name, data_dir=None) -> int`, `BENCHMARK_NUM_TYPES`
  ([`benchmark_data.py`](../torchpp/data/benchmark_data.py)).

## `torchpp.utils`: data containers, batching, evaluation

[`event_data.py`](../torchpp/utils/event_data.py)

- **`MVEventData(time_points, event_types, covariates=None, sort=False, T_obs=None)`**: one
  sequence. Times must be non-decreasing (or pass `sort=True`). Types are non-negative
  integers. Lists and NumPy arrays are converted.
- **`BatchedMVEventData(time_points=None, event_types=None, covariates=None, mv_events=None, sort=False, T_obs=None)`**:
  a right-padded batch. `BatchedMVEventData.from_sequences(seqs)` builds one and
  `batch.to_sequences()` splits it again. `last_event_time` is the last real event per
  sequence, and `T_obs` is the end of observation (never earlier). See [Data](./data.md) for the
  padding convention.
- **`TokenRoles(has_origin=False, context_types=(), ignored_types=())`**: how time 0 and each
  event type are treated. The object is shared by datasets (`default_token_roles`,
  `dataset.token_roles`), models (`token_roles=`, `model.token_roles`) and
  `evaluate_tpp(roles=)`. `replace(**changes)` returns a modified copy, and `predictable_mask(D)`
  returns the `(D,)` mask of non-context types. See
  [Notation → Token roles](./notation.md#token-roles).
- **`DatasetStats`**: training-set statistics used for initialization.

[`batching.py`](../torchpp/utils/batching.py)

- **`BucketBatchSampler(data_source, batch_size, shuffle=True, drop_last=False, num_buckets=10, dynamic_batch_size=False, target_total_events=None, seed=None)`**:
  groups sequences of similar length and draws a new order every epoch.
- **`collate_batch_with_truncation(mv_l, max_length=None, truncation_strategy='right', seed=None) -> BatchedMVEventData`**:
  `truncation_strategy` is `right` (keep the earliest events), `left` (keep the latest) or
  `random`. `TruncationStrategy` holds the three functions.
- `analyze_length_distribution(sequences) -> dict`, `print_length_analysis(sequences)`.

[`evaluation.py`](../torchpp/utils/evaluation.py)

- **`evaluate_tpp(model, sequences, num_event_types=None, device='cpu', roles=None, eval_batch_size=None, compute_marginals=True, integrate_marginal_via_grid=True, grid_points_per_unit_time=None, verbose=False) -> dict`**:
  the full-sequence log-likelihood per scored event (`ll_per_event`) and the next-event
  metrics over a dataset or a list of `MVEventData`, in native units. See
  [Metrics](./metrics.md) for the definitions.
- `evaluate_at_time(model, T, ts, ignored_types=None, compute_marginals=True, integrate_marginal_via_grid=True, grid_points_per_unit_time=None) -> dict`:
  the predictive quantities of an intensity model at time `T`.
- `format_metrics(metrics, prefix='') -> str`.

[`math.py`](../torchpp/utils/math.py)

- `LinearSpline`, `PiecewiseConstant`: spline evaluation and closed-form integrals.
- `inverse_softplus(x)`, `loglinear_cumulative_intensity(mu, slope, T)`, `right_limit(t, event_dtype)`.
- `numerical_integration(func, t_start, t_end, ts, method='trapezoidal', num_points=100, ...)`,
  `trapezoidal_integration(...)`, `monte_carlo_trapezoidal_integration(...)`, `monte_carlo_integration(...)`.
- `IntensityGrid` (`build`, `build_until`, `extend`, `total_ci`, `ci_at_t`, `intensity_at_t`):
  intensities on a time grid with trapezoidal prefix sums.
- `invert_CI(...)`, `bracket_monotone(...)`, `invert_monotone_newton(...)`: bracketed
  Newton inversion of the cumulative intensity.
- `baseline_spline_knot_locs(T, K, schedule='uniform', power=2.0)`,
  `interaction_spline_knot_gaps(T, K, schedule='geometric', power=2.0)`.

## `torchpp.modules`: building blocks

[`intensity.py`](../torchpp/modules/intensity.py): baseline and kernel modules for
composite Hawkes models.

- `BaselineIntensityModule(D)` (abstract): `ConstantBaselineModule`, `LogLinearBaselineModule`,
  `SplineBaselineModule(D, num_knots, delta_t, ...)`, `PiecewiseConstantBaselineModule(D, num_knots, delta_t, ...)`.
- `KernelIntensityModule(D)` (abstract): `ExpKernelModule`,
  `SplineKernelModule(D, num_knots, delta_t, ...)`, `PiecewiseConstantKernelModule(D, num_knots, delta_t, ...)`,
  `MLPKernelModule(D, emb_dim=32, time_embd_dim=32, n_hidden=64, n_mlp_layers=2, seed=None)`,
  `MLPSplineKernelModule(D, num_knots, delta_t, emb_dim=32, n_hidden=64, n_mlp_layers=2, seed=None, ...)`.

[`nn/`](../torchpp/modules/nn/): history encoders.

- `SequenceEncoder(n_embd, num_event_types)` (abstract).
- `TransformerConfig(num_event_types, n_layer=4, n_head=4, n_embd=64, dropout=0.1, token_dropout=0.0, bias=False, add_bos=False)`,
  `Transformer(config, add_bos=False)`, `TransformerEncoder(config, has_origin=False)`.
- `GRUEncoder(num_event_types, n_embd=32, num_layers=2, dropout=0.0, eps=1e-8, has_origin=False)`.
- `ContTimeLSTMEncoder(num_event_types, hidden_dim=32, bias=True, has_origin=False)` (NHP).
- `AttNHPEncoder(num_event_types, d_model=16, d_time=16, n_layers=2, n_head=2, dropout=0.0, use_norm=False, has_origin=False)`.
