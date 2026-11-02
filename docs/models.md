# Model catalog

Every model is selectable from a YAML config via a `model_type` string and
constructible programmatically by importing its class. The authoritative
list lives in
[`torchpp/training/config.py::VALID_MODEL_TYPES`](../torchpp/training/config.py)
and the factory mapping in
[`torchpp/training/model_factory.py`](../torchpp/training/model_factory.py).

The third column reports the cumulative-intensity path the model uses by
default. "Analytical" means $\Lambda(t)$ has a closed form; "numerical"
means it is computed by quadrature over an `IntensityGrid` (see
[Architecture](./architecture.md#numerical-vs-analytical-ci)).

## Poisson family

| `model_type` | Class | CI | Intensity |
|---|---|---|---|
| `poisson` | `PoissonProcess` | analytical | $\lambda_d(t) = \mu_d$ — homogeneous baseline rate |
| `inhomogeneous_poisson` | `ConditionalInhomogeniousPoissonProcess` | analytical | $\lambda_d(t) = \exp(a_d t + b_d)$ — log-linear in $t$ |
| `spline_poisson` | `SplinePoissonProcess` | analytical | $\lambda_d(t) = \exp(\mathrm{spline}_d(t))$ — log-spline baseline with `spline_K` knots |

Defined in [`torchpp/tpp/baseline.py`](../torchpp/tpp/baseline.py).

## Hawkes family

A Hawkes intensity sums a baseline and a kernel:

$$\lambda_d(t) = \mu_d(t) + \sum_{i: t_i < t} \phi_{m_i, d}(t - t_i).$$

Each composite below picks one
[`BaselineIntensityModule`](../torchpp/modules/intensity.py) and one
[`KernelIntensityModule`](../torchpp/modules/intensity.py); the CI is
analytical when both modules support it.

| `model_type` | Class | CI | Baseline + kernel |
|---|---|---|---|
| `hawkes` | `ConstBaselineExpKernelHawkesProcess` | analytical | constant $\mu_d$ + exponential kernel |
| `linear_exp_hawkes` | `LinearBaselineExpKernelHawkesProcess` | analytical | linear $\mu_d(t) = a_d t + b_d$ + exponential kernel |
| `spline_exp_hawkes` | `SplineBaselineExpKernelHawkesProcess` | analytical | spline baseline (`spline_K` knots) + exponential kernel |
| `const_spline_kernel` | `ConstBaselineSplineKernelHawkesProcess` | analytical | constant baseline + spline kernel (`spline_kernel_K` knots) |
| `spline_spline_kernel` | `SplineBaselineSplineKernelHawkesProcess` | analytical | spline baseline + spline kernel |

Inhibitive variants wrap $\mu(t) + \kappa(t)$ in a positivity-enforcing
non-linearity selected by `arch.inhibitive_clamping_fn`
(`softplus`, `relu`, `square`, `exp`; full list in
[`hawkes.py::VALID_CLAMPING_FNS`](../torchpp/tpp/hawkes.py)). The
non-linearity makes the integral non-decomposable, so CI is numerical.

| `model_type` | Class | CI | Baseline + kernel |
|---|---|---|---|
| `inhibitive_const_exp_hawkes` | `InhibitiveConstExpHawkesProcess` | numerical | constant + exponential, sum-then-clamp |
| `inhibitive_spline_exp_hawkes` | `InhibitiveSplineExpHawkesProcess` | numerical | spline + exponential, sum-then-clamp |
| `inhibitive_spline_spline_kernel` | `InhibitiveSplineBaselineSplineKernelHawkesProcess` | numerical | spline + spline, sum-then-clamp |

Hybrid Hawkes + neural kernel — baseline is a classical spline; the kernel
is a learned function of the elapsed time and the source/target type pair
output by a transformer encoder.

| `model_type` | Class | CI | Components |
|---|---|---|---|
| `neural_hawkes` | `SplineBaselineNeuralKernelHawkesProcess` | numerical | spline baseline + transformer-conditioned kernel |
| `neural_spline_hawkes` | `SplineBaselineNeuralSplineKernelHawkesProcess` | numerical | spline baseline + transformer-conditioned spline kernel |

Defined in [`torchpp/tpp/hawkes.py`](../torchpp/tpp/hawkes.py).

## Neural TPPs

All neural TPPs share a causal transformer encoder
([`torchpp/modules/nn/`](../torchpp/modules/nn/)) and differ only in the
intra-event intensity head. The encoder runs once per batch and its output
parameterises $\lambda_d(t)$ between observed events.

| `model_type` | Class | CI | Head |
|---|---|---|---|
| `neural_poisson` | `NeuralPoissonProcess` | analytical | constant $\lambda_d$ per inter-event interval |
| `neural_log_linear` | `NeuralLogLinearProcess` | analytical | $\lambda_d(t) = \exp(a_d \cdot \tau + b_d)$ in elapsed time $\tau$ |
| `neural_spline` | `NeuralSplineProcess` | analytical | log-spline in elapsed time, `spline_kernel_K` knots |
| `neural_mlp` | `NeuralMLPProcess` | numerical | MLP-parameterised intensity in elapsed time |
| `neural_thp` | `NeuralTHPProcess` | numerical | Transformer Hawkes Process head |
| `neural_sahp` | `NeuralSAHPProcess` | numerical | Self-Attentive Hawkes Process head |

Defined in [`torchpp/tpp/neural.py`](../torchpp/tpp/neural.py).

### EasyTPP-faithful replicas

Re-implementations that match the [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess)
benchmark's heads exactly (numerical CI, identical hyper-parameter
semantics). Kept separate from `neural_thp` / `neural_sahp` so the latter
can use our analytical-CI infrastructure where possible.

| `model_type` | Class |
|---|---|
| `easytpp_thp` | `EasyTPPTHPProcess` |
| `easytpp_sahp` | `EasyTPPSAHPProcess` |

## Picking a model

- **Sanity check / lower bound** — `poisson`, `inhomogeneous_poisson`.
- **Classical excitatory dynamics, fast** — `hawkes`, `spline_exp_hawkes`.
- **Inhibition / negative dependencies** — any `inhibitive_*` model.
- **Flexible classical baseline** — `spline_spline_kernel`.
- **Best classical paper baselines on UKB / MIMIC-IV** — see the replica
  scripts under [`scripts/`](../scripts/) for the exact `model_type`
  used per dataset.
- **Neural TPP, analytical CI** — `neural_log_linear`, `neural_spline`.
- **Neural TPP, EasyTPP comparison** — `easytpp_thp`, `easytpp_sahp`.

## Hyper-parameters that affect every model

These come from `ModelArchConfig` in
[`torchpp/training/config.py`](../torchpp/training/config.py):

- `spline_K` — number of knots for spline baselines.
- `spline_kernel_K` — number of knots for spline kernels and neural
  spline/piecewise heads.
- `spline_T_factor` — multiplier on `DatasetStats.sequence_time_p95` that
  sets the spline anchor; the default ensures the spline covers
  essentially the full observation window.
- `num_ci_integration_steps` — grid points for numerical CI.
- `inhibitive_clamping_fn` — the clamp used by inhibitive models.
- `n_layer`, `n_embd`, `n_head` — transformer geometry for neural / hybrid
  models.

The canonical hyper-parameters used in the paper are stored per
`(dataset, model_type)` in
[`scripts/tuning_*_best_params.sh`](../scripts/) and applied automatically
by the replica wrappers.
