# API reference

The most-used symbols are re-exported from the package root and listed
below per subpackage. Each entry links to the source so you can read the
exact signature and docstring there. For higher-level guidance see
[Architecture](./architecture.md), [Models](./models.md), and
[Data](./data.md).

```python
from torchpp import (
    # data
    MVEventData, BatchedMVEventData, DatasetStats,
    # batching
    BucketBatchSampler, TruncationStrategy,
    collate_batch_with_truncation,
    analyze_length_distribution, print_length_analysis,
    # base classes
    TemporalPointProcess, IntensityBasedTPP, DensityBasedTPP,
    # Poisson family
    PoissonProcess, ConditionalInhomogeniousPoissonProcess, SplinePoissonProcess,
    # Hawkes family
    HawkesProcess, InhibitiveHawkesProcess,
    ConstBaselineExpKernelHawkesProcess, LinearBaselineExpKernelHawkesProcess,
    SplineBaselineExpKernelHawkesProcess,
    ConstBaselineSplineKernelHawkesProcess, SplineBaselineSplineKernelHawkesProcess,
    InhibitiveConstExpHawkesProcess, InhibitiveSplineExpHawkesProcess,
    InhibitiveSplineBaselineSplineKernelHawkesProcess,
    SplineBaselineNeuralKernelHawkesProcess, SplineBaselineNeuralSplineKernelHawkesProcess,
    # Neural TPPs
    NeuralPoissonProcess, NeuralLogLinearProcess, NeuralSplineProcess,
    NeuralMLPProcess, NeuralTHPProcess, NeuralSAHPProcess,
    EasyTPPTHPProcess, EasyTPPSAHPProcess,
    # Datasets
    TPPDataset, UKBDataset, load_ukb_sequences, UKB_TIME_SCALE_FACTOR,
)
```

## `torchpp.tpp`

[`torchpp/tpp/base_tpp.py`](../torchpp/tpp/base_tpp.py)

The class tree has three levels: an abstract root holding the family-agnostic
contract, and two family bases that realize it (via an intensity, or via an
inter-event-time density).

- **`TemporalPointProcess(D, seed=42, terminating=False)`** — abstract root. Holds
  the universal predictive / likelihood / sampling contract; every method has a
  `log: bool = True` argument (log-space is the implemented primitive; `log=False`
  returns the `exp`), and every time-taking method accepts `t` of shape `(B,)` or
  `(B, Q)`.
  - `likelihood(ts, T=None, log=True) -> (B,)` — **abstract**. Full per-sequence (log-)likelihood; the canonical training objective.
  - `sample(ts, num_steps=1, rng=None, sampling_config=None) -> (MVEventData, list, list)` — **abstract**. Draw scheme set by `SamplingConfig`.
  - `density_joint(t, ts, log=True) -> (B, [Q,] D)` — joint next-event density p(t, m).
  - `density_time(t, ts, log=True) -> (B, [Q])` — time-marginal density p(t).
  - `cdf_time(t, ts, log=True) -> (B, [Q])` / `survival_time(t, ts, log=True) -> (B, [Q])` — P(t_next < t) / S(t).
  - `prob_mark(ts, log=True) -> (B, D)` — mark marginal p(m), integrated over the next-event time.
  - `prob_mark_given_time(t, ts, log=True) -> (B, [Q,] D)` — conditional mark p(m | t).
  - `density_time_given_mark(t, ts, log=True) -> (B, [Q,] D)` — mark-conditional time density p(t | m).
  - `time_mean(ts) -> (B,)` — E[t_next]; `time_quantile(u, ts) -> (B,)` — inverse `cdf_time` (median = `u=0.5`).
  - `transform_params() -> Tuple[Tensor, ...]` — **abstract**. Constrained-form parameters.

  Only `likelihood` and `sample` are hard-universal; the predictive methods have
  generic decomposition fallbacks that each family base overrides for the subset it
  roots (see [Architecture](./architecture.md)).

- **`IntensityBasedTPP(TemporalPointProcess)(D, seed=42, terminating=False, ci_config=None, sampling_config=None)`** —
  the intensity harness (all Hawkes, Baseline, Poisson, and encoder-based intensity
  models). Realizes the contract via the intensity λ and cumulative intensity Λ.
  - `intensity(t, ts) -> (B, D)` (or `(B, Q, D)` for a `(B, Q)` query) — conditional intensity, with termination masking applied.
  - `cumulative_intensity(T, ts, T_low=None, ci_config=None) -> (B, D)` — Λ over `[T_low or 0, T]`, dispatched by `CIConfig` (closed-form vs numerical quadrature).
  - `ci_config` — active `CIConfig` (explicit override else `_default_ci_config()`); `use_ci(cfg)` — context manager to force one for a block.
  - `sampling_config` / `use_sampling(cfg)` — the sampling twin of the above.
  - `inverse_marginal_cumulative_intensity(u, ts) -> (B,)` — invert $\sum_d \Lambda_d$.
  - `build_intensity_grid(t_start, t_end, ts, num_points=200, integrate_marginal_via_grid=True) -> IntensityGrid`
  - `build_intensity_grid_until(t_start, ts, points_per_unit_time=200.0, target_cdf=0.999, T_max=None, ...) -> IntensityGrid`
  - `build_integration_grid(ts, T_max=None, num_points=200, points_per_unit_time=200.0, ...) -> IntensityGrid`
  - Legacy-name delegators (kept for existing call-sites; the contract names are primary):
    `PDF -> density_joint(log=False)`, `CDF -> cdf_time(log=False)`,
    `marginal_class_distribution -> prob_mark`, `mean_next_event_time -> time_mean`,
    `inverse_CDF -> time_quantile`.

  Subclass hooks (override to define a new intensity model):
  - `_intensity_at_t(t, ts) -> (B, D)` **or** `_intensities_at_queries(query_times, ts) -> (B, Q, D)` — **one required**. Raw intensity, before termination masking (they mutually delegate).
  - `transform_params() -> Tuple[Tensor, ...]` — **required**. Constrained-form parameters.
  - `_interval_ci(ts, t_lo, t_hi) -> (B, L+1, D)` or `_closed_form_ci(ts, T_low, T) -> (B, D)` — closed-form CI when available; ignore termination internally.
  - `_default_ci_config() -> CIConfig` — per-family default; override to declare numerical integration (the base default is closed-form).
  - `_event_intensities(ts) -> (B, L)` — observed-type intensity; override for efficiency in neural models.
  - `_sequence_intensities(ts) -> (B, L, D)` — all-type intensities along the sequence.

- **`DensityBasedTPP(TemporalPointProcess)(D, seed=42, terminating=False, sampling_config=None)`** —
  the inter-event-time-density family: realizes the contract natively from a density
  `f(τ)` and a history-conditioned mark head `p(m | H)`, with no intensity. Members:
  `IntensityFreeProcess` (log-normal mixture, Shchur et al. ICLR 2020). Subclass hooks:
  `_log_density_time_given_mark`, `_log_survival_time_given_mark`, `_log_prob_mark`, `_event_log_density_sum`,
  `_draw_next`, `time_mean`. Carries the same `sampling_config` / `use_sampling` plumbing.

Config objects (frozen dataclasses in [`base_tpp.py`](../torchpp/tpp/base_tpp.py)):

- **`CIConfig(mode='numerical', placement='interval', rule='mc_trapezoidal', n_points=100, chunk_size=None)`** —
  how Λ = ∫λ is computed. `mode` is `closed_form` (analytic hooks) or `numerical`
  (quadrature); numerical adds `placement` (`interval` per inter-event interval, or
  `global` uniform over the window), `rule` (`trapezoidal` / `mc` / `mc_trapezoidal`),
  and `n_points`. Resolution order: call-arg > active `use_ci` context > instance default.
- **`SamplingConfig(mode='inversion', inversion_method='auto', ...)`** —
  how events are drawn. `inversion` (inverse-CDF; number-preserving; `inversion_method`
  ∈ `auto`/`closed_form`/`newton`/`grid`) or `thinning` (Ogata). Resolution mirrors `CIConfig`.

[`torchpp/tpp/baseline.py`](../torchpp/tpp/baseline.py)

- `PoissonProcess(D, ...)`, `ConditionalInhomogeniousPoissonProcess(D, ...)`, `SplinePoissonProcess(D, K, T_max, ...)`, `PiecewiseConstantPoissonProcess(D, K, T_max, ...)`.

[`torchpp/tpp/hawkes.py`](../torchpp/tpp/hawkes.py)

- `HawkesProcess(D, baseline, kernel, ...)` — composite ABC.
- `InhibitiveHawkesProcess(D, baseline, kernel, clamping_fn='softplus', ...)` — sum-then-clamp ABC.
- `resolve_clamping_fn(name)` — name → callable for the inhibitive clamp; valid names listed in `VALID_CLAMPING_FNS`.
- Concrete composites — see [Models](./models.md) for the full table.

[`torchpp/tpp/neural.py`](../torchpp/tpp/neural.py)

- `NeuralParametrizedTPP` — `IntensityBasedTPP` base for encoder-conditioned *intensity* TPPs (combines `NeuralEncoderMixin` with the intensity harness).
- `TransformerParametrizedTPP` — convenience subclass that builds a transformer encoder internally.
- `NHPProcess`, `AttNHPProcess` — continuous-time-LSTM / attentive intensity models (numerical CI).
- `IntensityFreeProcess` — the `DensityBasedTPP` member: a log-normal mixture over the inter-event time plus a mark head (no intensity on the contract path).
- Concrete heads — see [Models](./models.md).

## `torchpp.modules`

[`torchpp/modules/intensity.py`](../torchpp/modules/intensity.py) — reusable Hawkes building blocks. Each module exposes the same interface so it can be plugged into any composite.

- `BaselineIntensityModule` (ABC) — concrete: `ConstantBaselineModule`, `LinearBaselineModule`, `SplineBaselineModule`, `PiecewiseConstantBaselineModule`.
- `KernelIntensityModule` (ABC) — concrete: `ExpKernelModule`, `SplineKernelModule`, `PiecewiseConstantKernelModule`, `NonStationarySplineExpKernelModule`, `NeuralSplineKernelModule`, `NeuralKernelModule`.

[`torchpp/modules/nn/`](../torchpp/modules/nn/) — causal transformer encoder used by the neural and hybrid models. `TransformerConfig` exposes `n_layer`, `n_head`, `n_embd`, `dropout`, `bias`.

## `torchpp.data`

[`torchpp/data/base_dataset.py`](../torchpp/data/base_dataset.py)

- `TPPDataset` — `torch.utils.data.Dataset` over `MVEventData` items with metadata (`num_event_types`, `terminating`, …).

[`torchpp/data/ukb_data.py`](../torchpp/data/ukb_data.py)

- `UKBDataset(data_dir, split, ...)` — lazy reader over `train.bin` / `val.bin` / `test.bin`.
- `load_ukb_sequences(data_dir, split) -> list[MVEventData]` — eager loader.
- `UKB_TIME_SCALE_FACTOR` — converts scaled time back to days (`= 1 / (365.25 * 80)`).

[`torchpp/data/easytpp_data.py`](../torchpp/data/easytpp_data.py)

- `EasyTPPDataset(data_dir, dataset_name, split, ...)`.
- `load_easytpp_sequences(data_dir, split, ...) -> list[MVEventData]`.
- `num_event_types_for(name)` — `D` for the named EasyTPP variant.

[`torchpp/data/mimic_iv_data.py`](../torchpp/data/mimic_iv_data.py)

- `MIMICIVDataset(data_dir, split, top_k_event_types=None, ...)`.
- `load_mimic_iv_sequences(...)`.

## `torchpp.utils`

[`torchpp/utils/event_data.py`](../torchpp/utils/event_data.py)

- `MVEventData(time_points, event_types, covariates=None, sort=False, T_obs=None)`.
- `BatchedMVEventData(time_points, event_types, covariates=None, mv_events=None, sort=False, T_obs=None)`.
- `DatasetStats` — empirical statistics consumed by `create_model`.

[`torchpp/utils/batching.py`](../torchpp/utils/batching.py)

- `BucketBatchSampler(lengths, batch_size, num_buckets=10, drop_last=False, shuffle=True)`.
- `TruncationStrategy` — `right | left | random | stratified`.
- `collate_batch_with_truncation(batch, max_sequence_length=None, strategy='right')` — collate fn for `DataLoader`.
- `analyze_length_distribution(sequences) -> dict`, `print_length_analysis(sequences)`.

[`torchpp/utils/math.py`](../torchpp/utils/math.py)

- `LinearSpline`, `PiecewiseConstant` — differentiable building blocks.
- `inverse_softplus(x)` — initialise unconstrained parameters.
- `baseline_spline_knot_locs(T, K, schedule='uniform', power=2.0)`.
- `interaction_spline_knot_gaps(T, K, schedule='geometric', power=2.0)`.
- `loglinear_cumulative_intensity(...)` — analytical $\int \exp(at + b)\,dt$.
- `trapezoidal_integration`, `monte_carlo_trapezoidal_integration`, `monte_carlo_integration`, `numerical_integration` — quadrature helpers used by the numerical CI path.
- `IntensityGrid` — pre-computed intensity + CI prefix sums; supports `build`, `build_until`, total CI, and inversion.
- `bracket_monotone`, `invert_monotone_newton`, `invert_CI` — bracketed Newton inversion of the CI for sampling and quantile prediction.

[`torchpp/utils/evaluation.py`](../torchpp/utils/evaluation.py)

- `evaluate_at_time(model, ts, T)` — evaluate next-event marginals at `T`.
- `evaluate_tpp(model, dataloader, ...)` — top-level evaluation loop, returns the metrics dict reported by `evaluate.py`.
- `format_metrics(metrics, prefix='')` — pretty-printer.
- `TIME_SCALE_FACTOR` — UKB scaled-time → days, used by metric reporting.

## `torchpp.training`

[`torchpp/training/config.py`](../torchpp/training/config.py)

- `FitConfig`, `StandaloneEvalConfig` — top-level dataclasses.
- `BatchingConfig`, `TrainingConfig`, `ModelArchConfig`, `EvalConfig`, `WandbConfig`, `SaveConfig`, `DatasetStatsConfig`, `ProfilingConfig`, `CompileConfig` — nested groups.
- `VALID_MODEL_TYPES` — single source of truth for model names.

[`torchpp/training/config_utils.py`](../torchpp/training/config_utils.py) — YAML composition, dotted CLI parsing, OmegaConf struct-mode validation.

[`torchpp/training/model_factory.py`](../torchpp/training/model_factory.py)

- `create_model(model_type, D, arch, terminating=True, seed=None, load_path=None, dataset_stats=None) -> (TemporalPointProcess, run_name)` — single authoritative entry point for instantiating a model.

[`torchpp/training/dataset_registry.py`](../torchpp/training/dataset_registry.py) — registers each `dataset_type` with its loader and typed config schema.

[`torchpp/training/training_helpers.py`](../torchpp/training/training_helpers.py) — optimiser construction, scheduler dispatch, gradient accumulation, early stopping; used by `train.py`.
