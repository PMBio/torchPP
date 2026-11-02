# torchpp documentation

`torchpp` is a PyTorch library for **multivariate temporal point processes
(TPPs)**. Models share a uniform intensity / cumulative-intensity / likelihood
/ sampling / evaluation API across classical Hawkes, neural, and hybrid
models. Most models also support a **terminating** mode where each event type
occurs at most once per sequence.

## User documentation

Start here.

- [Getting started](./getting_started.md) — install, run a model, evaluate a
  checkpoint.
- [Architecture](./architecture.md) — class hierarchy, intensity / CI /
  likelihood pipeline, terminating mode.
- [Model catalog](./models.md) — every model exposed as a `model_type` config
  string, the underlying class, and what it parameterises.
- [Data](./data.md) — `MVEventData`, `BatchedMVEventData`, padding/masking
  conventions, available datasets.
- [Configs](./configs.md) — YAML composition, dotted CLI overrides, the typed
  `FitConfig` schema.
- [API reference](./api_reference.md) — public symbols per subpackage.
- [Extending](./extending.md) — adding a new model end-to-end.
- [Design overview](./design_overview.md) — diagrams of the class hierarchy,
  baseline ⊕ kernel and encoder ⊕ head composition, and the training pipeline.

## Datasets and experiments

- [MIMIC-IV benchmark construction](./mimic_iv_benchmark_construction.md)
- [UKB conversion](./ukb_csv_conversion.md)
- [EasyTPP hyperparameter search](./easytpp_hyperparameter_search.md)
- [UKB hyperparameter search](./ukb_real_hyperparameter_search.md)
