# Changelog

## 0.1.0

Initial public release.

**Models and interface**

- All models share one interface for the intensity, cumulative intensity, likelihood,
  next-event densities and type probabilities, survival, quantiles, sampling and evaluation.
  Public methods take and return native time units. The models normalize time internally.
- We include 26 models: Poisson processes, Hawkes processes with constant, log-linear, spline or
  piecewise baselines and exponential, spline or piecewise kernels, inhibitive Hawkes
  processes, transformer-encoder neural TPPs (including THP and SAHP), NHP, AttNHP and
  IntensityFree. Two MLP-kernel Hawkes models are marked experimental.
- `baseline_values`, `kernel_values` and `branching_matrix` read out a fitted Hawkes process
  in native units.
- We support terminating sequences (each event type at most once), right-censored
  observation windows with forecasts from `T_obs`, and context event types.
- The cumulative intensity is computed in closed form where available and by numerical
  integration otherwise. Sampling uses inversion or thinning. `sample` returns
  `Sample(sequence, times, mark_probs)`.
- Arguments after the essential ones (`model_type`, `D`) are keyword-only, and `log=` is passed
  as a keyword.

**Training, checkpoints, evaluation**

- `fit()` implements the training loop. It trains on the device the model is on unless told
  otherwise.
- Checkpoints are self-describing. `save_model` / `load_model` rebuild any model exactly from its
  recorded constructor arguments and integration settings.
- `register_model` adds your own models. The integration and sampling schemes (`CIConfig`,
  `SamplingConfig`) are constructor arguments and YAML fields (`arch.ci.*`, `arch.sampling.*`).
- The training loss and the validation and test log-likelihoods share one definition of scored
  events (`num_scored_events`). `evaluate_tpp` reports the full-sequence log-likelihood per event
  (`ll_per_event`, the training objective) and next-event metrics on the same events, with or
  without a meaningful time origin, including sequences with zero or one event.
- `TokenRoles` holds the origin regime, the context types and the ignored types. One object is
  shared by datasets (`default_token_roles`), models (`token_roles=`) and `evaluate_tpp(roles=)`.
- The data containers provide `last_event_time` (distinct from `T_obs`) and
  `BatchedMVEventData.from_sequences` / `to_sequences`. Padding happens only inside batches.
- Truncation of long sequences keeps the observation window consistent (`T_obs` is never
  dropped).
- Query times, `T`, `T_low` and `T_obs` are converted to internal units exactly like event
  times, so a query at an event time never sees events tied with it. Evaluation on float32
  data with ties is now consistent with float64 and with the likelihood. This includes the
  IntensityFree diagnostic `intensity()`.
- `neural_poisson`, `neural_spline` and `neural_piecewise_constant` with `has_origin=True` but an
  encoder built without an origin raise a clear `ValueError`. Their heads give zero intensity
  from the no-history state, so the likelihood would be meaningless.

**Configuration and data**

- Training and evaluation run from the command line with composable YAML presets. There is one
  preset per model, and its family (`training/classical`, `training/neural`) sets the training
  defaults. `configs/demo.yaml` lists every field with its default.
- Each model type declares its own `arch` fields, grouped by component (`baseline`, `kernel`,
  `encoder`, `head`) next to the shared `seed`, `ci` and `sampling`. Fields that a model does
  not have are rejected with the list of valid ones.
- We include a benchmark dataset loader (`BenchmarkDataset`) and a download script. The data are
  stored under `$TORCHPPDATA/benchmarks/`.
- Taobao is downloaded from the json files of the EasyTPP Hugging Face release and stored in
  float64. We do not use the Google Drive versions, whose train/dev and test splits come from
  different preprocessings.

**Documentation**

- The documentation contains an introduction to temporal point processes, a notation reference,
  a model-selection table, metric definitions, and notebooks from TPP basics to comparing models and survival analysis.
