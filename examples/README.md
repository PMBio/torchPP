# Examples

These notebooks are self-contained. Each one builds or simulates its own small dataset, so no
dataset download is needed. Most run in under a minute, and the model comparison (06) takes a few
minutes. Open them with Jupyter (the `examples` extra installs JupyterLab).

```bash
jupyter lab examples/00_tpp_introduction.ipynb
```

We recommend reading them in order, since each one builds on the previous one.

| Notebook | What it teaches | Runtime |
|---|---|---|
| [`00_tpp_introduction.ipynb`](./00_tpp_introduction.ipynb) | What a TPP is: the intensity, cumulative intensity, survival, next-event densities, the log-likelihood and sampling, each computed with TorchPP and checked against its formula on a hand-set Hawkes process. Optional sliders (`pip install ipywidgets`) change the parameters. | ~10 s |
| [`01_quickstart.ipynb`](./01_quickstart.ipynb) | The data structures (`MVEventData`, `BatchedMVEventData`) and the interface every model shares: `density_*`, `prob_*`, `time_*`, `likelihood`, `sample`, and the intensity view, on a Poisson process. | ~5 s |
| [`02_fit_evaluate.ipynb`](./02_fit_evaluate.ipynb) | Fit a Poisson process with `fit`, recover the true rates, and score held-out data with `evaluate_tpp`. | ~10 s |
| [`03_intensity_and_sampling.ipynb`](./03_intensity_and_sampling.ipynb) | Forecasting with a Hawkes process: the intensity on a history, many sampled futures, and how an event-free observation window (`T_obs`) changes the forecast. | ~5 s |
| [`04_add_a_model.ipynb`](./04_add_a_model.ipynb) | Write your own model as an `IntensityBasedTPP` subclass (an intensity hook plus a closed-form or numerical cumulative intensity), then train and evaluate it like a built-in model. Companion to [`docs/extending.md`](../docs/extending.md). | ~10 s |
| [`05_neural_piecewise_constant.ipynb`](./05_neural_piecewise_constant.ipynb) | Build a neural TPP from a shared transformer encoder and a small head, fit it to Hawkes data and compare it with a Poisson process. Uses the base class's internal hooks (`_get_params`, `_get_params_at_queries`, …), which may change between versions. | ~25 s |
| [`06_compare_models.ipynb`](./06_compare_models.ipynb) | Simulate from a known Hawkes process, fit a Poisson process, two Hawkes processes and a neural TPP, compare them on held-out data, and recover the excitation matrix and kernel shapes. | ~2–4 min |

### Survival analysis

[`survival/`](./survival/) is a separate series on survival and competing-risks models built on
TorchPP. It covers Cox models, a spline hazard, the density-based route and a neural covariate
model. See [`survival/README.md`](./survival/README.md) for details.

For training on the benchmark datasets with `train.py`, or on your own data, see
[getting started](../docs/getting_started.md). The [introduction](../docs/introduction.md) gives
the background on the quantities.
