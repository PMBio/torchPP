# TorchPP documentation

TorchPP is a PyTorch library for multivariate temporal point processes (TPPs). We implement
classical Poisson and Hawkes processes, neural and hybrid models through a shared interface for
the intensity, cumulative intensity, likelihood, next-event predictions, sampling and
evaluation. All models also support a terminating mode, in which each event type occurs at most
once per sequence.

**Start here**

- The [introduction](./introduction.md) explains what a TPP is and how every quantity follows
  from the intensity, and it names the TorchPP call for each.
- [Getting started](./getting_started.md) shows how to install TorchPP, train and evaluate on a
  benchmark, and fit your own data.
- The [model catalog](./models.md) lists every `model_type`, its class and what it parametrizes.

**Use**

- [Notation](./notation.md) is a compact reference for the quantities, the likelihood, the
  origin, token roles, forecasting from `T_obs` and time units.
- [Data](./data.md) describes `MVEventData`, batching, the benchmark datasets and how to add
  your own dataset.
- [Configs](./configs.md) describes the YAML presets, command-line overrides and the `FitConfig`
  schema.
- [Metrics](./metrics.md) explains what the evaluation reports.

**Extend**

- [Extending](./extending.md) shows how to add a model or a dataset.
- [Architecture](./architecture.md) describes the class hierarchy, the intensity and integration
  pipeline and the terminating mode. It is written for contributors.
- The [API reference](./api_reference.md) lists the public symbols and their signatures.

The notebooks are in [`examples/`](../examples/README.md).

## Glossary

- **Event type** (also *mark*): the category of an event, an integer in `0 … D−1`. `D` is the
  number of event types.
- **Intensity** $\lambda_d(t)$: the expected rate of type-$d$ events at time $t$, given the past.
- **Cumulative intensity** $\Lambda$: the intensity integrated over time, with $S = e^{-\Lambda}$.
- **Survival** $S(t)$: the probability that no event happens before $t$.
- **Terminating**: each event type occurs at most once per sequence (survival analysis,
  competing risks).
- **Origin** (`has_origin`): whether time 0 is meaningful. Without it, the first event is given
  and scoring starts at the second.
- **`T_obs`**: the end of the observation window. It can be later than the last event.
- **Scored events**: the events whose log-intensity enters the likelihood
  ([Notation → Scored events](./notation.md#scored-events)).

## FAQ

- **My log-likelihood is `-inf`.** An event received zero intensity. Usually a terminating model
  (`terminating=True`) saw an event type twice. After a type has occurred, its intensity is zero.
  Use `terminating=False` for data where types repeat.
- **A query before `T_obs` raises an error.** Predictions are for the next event after the
  observation window. For the density between the last event and `T_obs`, pass the history without
  `T_obs` ([Notation → Forecasting](./notation.md#forecasting-from-t_obs)).
- **Can I compare log-likelihoods across settings?** Only with the same time unit and the same
  `has_origin`. Densities carry the time unit, and the two origin settings score different events.
- **Which unit are the metrics in?** The data's native time unit ([Metrics](./metrics.md)).
- **Evaluation is slow on CPU.** The grid-based metrics (RMSE, marginal accuracy) dominate the
  runtime. `eval.compute_marginal_type=false` skips them, and `eval.limit_test_size` evaluates a
  subset.

