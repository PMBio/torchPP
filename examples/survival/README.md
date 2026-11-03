# Survival analysis with TorchPP

This notebook series builds survival models of increasing complexity on TorchPP, one step at a
time. Survival analysis is a temporal point process with one terminal event per subject. The
outcome either occurs once (uncensored) or does not occur before the end of the observation
window (right-censored). A terminating TPP models this directly. Its intensity is the hazard,
and the survival function is `S(t) = exp(-∫₀ᵗ λ)`.

The shared helpers are in [`survival_tools.py`](./survival_tools.py). They include synthetic data
generators, the Kaplan–Meier, Aalen–Johansen and concordance estimators, the fit loop and the
plots. Each notebook imports from this file and defines its model inline, so that the model code
is what you read in the notebook.

Run the notebooks from this directory, so that `survival_tools` is importable.

```bash
cd examples/survival && jupyter lab      # or: jupyter notebook
```

| Notebook | What it builds |
|---|---|
| [`01_cox_competing_risks.ipynb`](./01_cox_competing_risks.ipynb) | Cox proportional hazards (covariate-modulated constant hazard) as a terminating `IntensityBasedTPP`. The same class is then extended to competing risks. It recovers the per-cause coefficients, and its cohort-averaged cumulative incidence is compared with the Aalen–Johansen estimate. |
| [`02_spline_hazard.ipynb`](./02_spline_hazard.ipynb) | A time-varying baseline hazard `h₀(t)·exp(x·β)` with `h₀(t)` a learnable spline, fit to Weibull data. It recovers the rising hazard shape, which a constant model cannot. |
| [`03_density_route.ipynb`](./03_density_route.ipynb) | The same survival problem solved with the `DensityBasedTPP` family. The model parametrizes the density of the event time directly (a log-normal mixture) instead of a hazard, and `S(t) = 1 − CDF` follows from it. The notebook compares this with the intensity route. |
| [`04_neural_covariates.ipynb`](./04_neural_covariates.ipynb) | A small MLP maps a complex covariate vector to the model's parameters, and captures non-linear covariate effects that a linear Cox model misses. It reaches a higher held-out likelihood and concordance and recovers the true risk function. |

## Predictive queries

Survival curves, hazards and expected event times are next-event predictions: they start at the
forecast origin, which is the last event of the history (or its `T_obs`). A training subject already
carries its terminal event, so its forecast origin is that event's time, and a query at an earlier
time raises an error. To get the curve of a subject from time 0, query an empty-history batch.
This is a sequence with no events that carries only the subject's covariates. `covariate_batch(x)`
builds one, and `model_survival_curve` and `model_hazard_curve` use it. Pass `T_obs=` to condition on
survival up to that time.

Each notebook is self-contained. It uses synthetic data, runs on CPU in a few minutes or less,
and ships with cleared outputs. Run it from top to bottom to reproduce the figures.
