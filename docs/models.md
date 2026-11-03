# Model catalog

Every model can be selected in a YAML config by its `model_type` string, or constructed in
Python by importing its class. The registry in
[`torchpp/training/registry.py`](../torchpp/training/registry.py) defines every `model_type`
together with its class, description and builder. `VALID_MODEL_TYPES` and `create_model` are
derived from it. `hawkes` is accepted as a short alias of `const_exp_hawkes`.

The Λ column reports how the model computes the cumulative intensity Λ by default.
"Analytical" means that $\Lambda(t)$ has a closed form, and "numerical" means that it is
computed by quadrature (see
[Architecture](./architecture.md#numerical-vs-closed-form-cumulative-intensity)).

Classical models are named `<baseline>_<kernel>_hawkes`, or `<baseline>_poisson` if they have
no kernel. The baselines are `const`, `loglinear`, `spline` and `piecewise`, and the kernels are
`exp`, `spline`, `piecewise`, `mlp` and `mlpspline`. The prefix `inhibitive_` marks models that
apply the positivity map to the sum of baseline and kernel. Neural heads on the shared
transformer encoder are named `neural_<head>`, and published architectures use their acronym.

## Which model to use

All models support terminating sequences (`terminating=True`) and both origin settings. The cost
is relative and assumes the same sequence length. "Low" means that training takes seconds to
minutes on a CPU, and "high" means that realistic data needs a GPU.

| Family | Example `model_type` | Depends on history | Inhibition | Closed-form Λ | Cost | Use when |
|---|---|---|---|---|---|---|
| Poisson | `poisson`, `spline_poisson` | no | no | yes | low | a lower bound, or rates that change only with time |
| Additive Hawkes | `const_exp_hawkes`, `spline_spline_hawkes` | yes, pairwise | no | yes | low–medium | events trigger later events, and you want readable kernels and a branching matrix |
| Inhibitive Hawkes | `inhibitive_spline_spline_hawkes` | yes, pairwise | yes | no (quadrature) | medium | some event types make others less likely |
| Neural, closed-form head | `neural_spline`, `neural_log_linear` | yes, learned | yes | yes | medium | interactions beyond pairwise, with an exact likelihood |
| Neural, numerical head | `thp`, `sahp`, `neural_mlp` | yes, learned | yes | no (quadrature) | medium–high | comparison with published transformer TPPs |
| Recurrent / attention | `nhp`, `attnhp` | yes, learned | yes | no (quadrature) | high | comparison with NHP / AttNHP |
| Density-based | `intensity_free` | yes, learned | yes | density instead of Λ | low–medium | fast, flexible next-event times when no intensity is needed |

"Inhibition" means that an event can lower the intensity of another type. Hawkes models are
interpretable (see [Reading a fitted model](#reading-a-fitted-model)). Neural models are more
flexible but have no readable parameters.

## Poisson processes

| `model_type` | Class | Λ | Intensity |
|---|---|---|---|
| `poisson` | `PoissonProcess` | analytical | $\lambda_d(t) = \mu_d$ |
| `loglinear_poisson` | `LogLinearPoissonProcess` | analytical | $\lambda_d(t) = \exp(a_d + b_d t)$ |
| `spline_poisson` | `SplinePoissonProcess` | analytical | linear spline over positive knot heights (`arch.baseline.num_knots` knots) |
| `piecewise_poisson` | `PiecewisePoissonProcess` | analytical | piecewise-constant over `arch.baseline.num_knots` time bins |

Defined in [`torchpp/tpp/baseline.py`](../torchpp/tpp/baseline.py).

## Hawkes processes

A Hawkes intensity is the sum of a baseline and a kernel (Hawkes, 1971):

$$\lambda_d(t) = \mu_d(t) + \sum_{i: t_i < t} \phi_{d, m_i}(t - t_i),$$

where $\phi_{d,j}$ is the effect of a type-$j$ event on the intensity of type $d$.
The exponential kernel is $\phi_{d,j}(\Delta t) = \alpha_{d,j}\,\beta_{d,j}\,e^{-\beta_{d,j}\Delta t}$.
Here $\alpha_{d,j} \ge 0$ is the expected number of type-$d$ events that one type-$j$ event
triggers, and $1/\beta_{d,j}$ is how long the effect lasts. Spline and piecewise-constant kernels
replace this shape with a flexible one over `arch.kernel.num_knots` knots. All `(D, D)` arrays
are indexed `[target d, source j]`.

Each composite model combines one
[`BaselineIntensityModule`](../torchpp/modules/intensity.py) and one
[`KernelIntensityModule`](../torchpp/modules/intensity.py). Λ is analytical when both modules
support it.

| `model_type` | Class | Λ | Baseline + kernel |
|---|---|---|---|
| `const_exp_hawkes` | `ConstBaselineExpKernelHawkesProcess` | analytical | constant $\mu_d$ + exponential kernel |
| `loglinear_exp_hawkes` | `LogLinearBaselineExpKernelHawkesProcess` | analytical | $\mu_d(t) = \exp(a_d + b_d t)$ + exponential kernel |
| `spline_exp_hawkes` | `SplineBaselineExpKernelHawkesProcess` | analytical | spline baseline (`arch.baseline.num_knots` knots) + exponential kernel |
| `piecewise_exp_hawkes` | `PiecewiseBaselineExpKernelHawkesProcess` | analytical | piecewise-constant baseline (`arch.baseline.num_knots` bins) + exponential kernel |
| `const_spline_hawkes` | `ConstBaselineSplineKernelHawkesProcess` | analytical | constant baseline + spline kernel (`arch.kernel.num_knots` knots) |
| `const_piecewise_hawkes` | `ConstBaselinePiecewiseKernelHawkesProcess` | analytical | constant baseline + piecewise-constant kernel (`arch.kernel.num_knots` bins) |
| `spline_spline_hawkes` | `SplineBaselineSplineKernelHawkesProcess` | analytical | spline baseline + spline kernel |

Inhibitive variants allow negative kernel values and apply a positivity map to
$\mu(t) + \sum \phi$. The map is selected by `arch.clamping_fn` (`softplus`, `relu` or `exp`,
see [`hawkes.py::VALID_CLAMPING_FNS`](../torchpp/tpp/hawkes.py)). With the map, the integral is
no longer decomposable, so Λ is numerical.

| `model_type` | Class | Λ | Baseline + kernel |
|---|---|---|---|
| `inhibitive_const_exp_hawkes` | `InhibitiveConstBaselineExpKernelHawkesProcess` | numerical | constant + exponential |
| `inhibitive_spline_exp_hawkes` | `InhibitiveSplineBaselineExpKernelHawkesProcess` | numerical | spline + exponential |
| `inhibitive_spline_spline_hawkes` | `InhibitiveSplineBaselineSplineKernelHawkesProcess` | numerical | spline + spline |

The two models with MLP kernels are experimental. Their kernel is a small MLP of the
source-type embedding (and of a sinusoidal encoding of the elapsed time), and they have no
history encoder. These models pass the shared contract tests but have not been benchmarked.

| `model_type` | Class | Λ | Kernel |
|---|---|---|---|
| `inhibitive_spline_mlp_hawkes` | `InhibitiveSplineBaselineMLPKernelHawkesProcess` | numerical | MLP of `[type embedding; sin(Δt)]` |
| `inhibitive_spline_mlpspline_hawkes` | `InhibitiveSplineBaselineMLPSplineKernelHawkesProcess` | numerical | spline in Δt whose knot heights an MLP predicts from the type embedding |

Defined in [`torchpp/tpp/hawkes.py`](../torchpp/tpp/hawkes.py).

### Reading a fitted model

All values are in native time units and indexed `[target d, source j]`.

```python
model.baseline_values(t)    # (Q, D)     μ_d(t)
model.kernel_values(dt)     # (Q, D, D)  φ_{d,j}(Δt)
model.branching_matrix()    # (D, D)     ∫ φ_{d,j}: expected type-d events triggered by one type-j event
```

For the exponential kernel $\phi_{d,j}(\Delta t) = \alpha_{d,j}\beta_{d,j} e^{-\beta_{d,j}\Delta t}$,
the branching matrix is $\alpha$. For inhibitive models, these values are the signed terms inside
the positivity map and not rates, and the branching matrix is their signed integral. Kernels
without a closed-form integral (MLP) need `branching_matrix(t_max=...)`. The Poisson models only
have `baseline_values`.

## Neural TPPs on the shared transformer encoder

These models share a causal transformer encoder
([`torchpp/modules/nn/`](../torchpp/modules/nn/)) and differ in the intensity head.
The encoder runs once per batch, and its output parametrizes $\lambda_d$ between observed
events. $\tau$ denotes the time since the last event. You can construct these models with a
`TransformerConfig`, or with `D=` to use the default encoder sizes.

| `model_type` | Class | Λ | Head |
|---|---|---|---|
| `neural_poisson` | `NeuralPoissonProcess` | analytical | constant $\lambda_d$ per inter-event interval |
| `neural_log_linear` | `NeuralLogLinearProcess` | analytical | $\lambda_d(\tau) = \exp(a_d + b_d \tau)$ |
| `neural_spline` | `NeuralSplineProcess` | analytical | linear spline in $\tau$ over positive knot heights, `arch.head.num_bins` knots |
| `neural_piecewise_constant` | `NeuralPiecewiseConstantProcess` | analytical | piecewise-constant in $\tau$, `arch.head.num_bins` bins |
| `neural_mlp` | `NeuralMLPProcess` | numerical | MLP of the encoder state and $\tau$ |
| `thp` | `THPProcess` | numerical | Transformer Hawkes Process head (Zuo et al., 2020) |
| `sahp` | `SAHPProcess` | numerical | Self-Attentive Hawkes Process head (Zhang et al., 2020) |

The THP and SAHP heads use a scaled-softplus intensity with a numerical Λ, and they run on the
shared TorchPP transformer encoder.

Defined in [`torchpp/tpp/neural.py`](../torchpp/tpp/neural.py).

### No-history state and origin regimes

Neural models have no learned prior parameter. Instead, the state before any event has been
seen comes from the data.

- With an origin (`has_origin=True`), the encoder prepends a learned `<bos>` embedding before
  the first event. The no-history state is then `encoder(<bos>)`, the first real event is
  scored, and the gap `[0, t_first]` is modeled.
- Without an origin (`has_origin=False`, as for the benchmark datasets), the first event *is*
  the origin. It is conditioned on but not scored, so the likelihood begins at the second
  event. The no-history column is neither scored nor integrated.

`has_origin` is a property of the dataset and applies to every model. Likelihoods are
therefore comparable within one regime but not across the two. See
[Notation → Origin](./notation.md#origin-first-event-conditioning) and
[Notation → Token roles](./notation.md#token-roles).

## Other neural TPPs

| `model_type` | Class | Λ | Model |
|---|---|---|---|
| `intensity_free` | `IntensityFreeProcess` | density | log-normal mixture over the inter-event time + categorical mark head (Shchur et al., 2020) |
| `nhp` | `NHPProcess` | numerical | Neural Hawkes Process, continuous-time LSTM (Mei & Eisner, 2017) |
| `attnhp` | `AttNHPProcess` | numerical | Attentive Neural Hawkes Process, continuous-time attention (Yang et al., 2022) |

In terminating mode, a type can no longer fire in IntensityFree once it has occurred. Let
$A = \sum_{d \text{ still active}} p(d \mid H)$ be the share of the type distribution on the
active types. The restricted process then has survival $S(\tau)^A$ and time density
$A\,f(\tau)\,S(\tau)^{A-1}$, and the type distribution is renormalized to $p(d \mid H)/A$ over
the active types. This holds because all per-type hazards share the factor $f/S$, so dropping
types scales the total hazard by $A$. $A = 1$ gives the non-terminating model, and $A = 0$
(every type has occurred) means that there is no next event.

`intensity_free` is a [`DensityBasedTPP`](./architecture.md#class-hierarchy). Its contract is
computed from an inter-event-time density and a mark head, and no intensity is used on the
contract path. A diagnostic hazard is exposed for plotting only. The encoder is set by
`arch.encoder.type` and is either a GRU (as in Shchur et al.) or the shared transformer. `nhp`
and `attnhp` follow the original models. The origins of the code are listed in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

All three are defined in [`torchpp/tpp/neural.py`](../torchpp/tpp/neural.py).

## Hyper-parameters

Each model type has its own architecture config
([`torchpp/training/arch.py`](../torchpp/training/arch.py)). The full table is in
[Configs → Architecture fields](./configs.md#architecture-fields). The sections are described
below.

- `arch.baseline.{num_knots, T_factor}` sets the knots of spline and piecewise baselines. The
  extent is `T_factor × sequence_time_p95` of the training data.
- `arch.kernel.{num_knots, T_factor, knot_schedule, knot_power}` sets the knots of spline and
  piecewise kernels. Their extent is `T_factor × inter_event_time_p95`. The MLP kernels have
  their own `arch.kernel` fields (`emb_dim`, `n_hidden`, `n_layers`, …).
- `arch.head.{num_bins, T_factor, knot_schedule, knot_power}` sets the knots over the time since
  the last event for `neural_spline` (which also has `smoothness_reg`) and
  `neural_piecewise_constant`. The extent follows the same rule as for kernels. `neural_mlp` and
  `intensity_free` have their own `arch.head` fields.
- `create_model` turns these settings into knot spacings. Baselines get equal gaps
  `baseline_delta_t = extent / (num_knots − 1)`. Kernels and heads get `num_knots − 1` (or
  `num_bins − 1`) gaps that start at 0 and follow `knot_schedule`, which is either `uniform` or
  `geometric` (denser near 0). Without dataset statistics, the `T_factor` values are used
  directly as extents.
- When you construct a class directly, `baseline_delta_t` and `kernel_delta_t` are these
  spacings in the model's internal time unit, which is native time divided by `time_scale`
  (see [Architecture → Time units](./architecture.md#time-units)). With the default
  `time_scale=1`, this is the native unit.
- `arch.ci.*` sets any `CIConfig` field (rule, placement, `n_points`, `chunk_size`), and
  `arch.sampling.*` sets any `SamplingConfig` field. Numerically integrated models use 100
  quadrature points per interval unless `arch.ci.n_points` is set. `neural_mlp`, the MLP-kernel
  models and the inhibitive Hawkes models train with random quadrature nodes (`mc_trapezoidal`).
  THP, SAHP, NHP and AttNHP use fixed nodes (`trapezoidal`), as in their original
  implementations. With fixed nodes, a flexible head can learn intensity spikes between the
  nodes that the survival term never sees, which inflates the training likelihood. Setting
  `arch.ci.rule=mc_trapezoidal` switches to random nodes. Predictions and evaluation always use
  fixed nodes, so they are deterministic.
- `arch.clamping_fn` sets the positivity map of the inhibitive models.
- `arch.encoder.*` sets the encoder geometry (`n_layers`, `n_heads`, `n_embd` and `dropout` for
  the transformer, `hidden_dim` for NHP).

## References

- A. G. Hawkes. Spectra of some self-exciting and mutually exciting point processes. *Biometrika*, 1971.
- H. Mei and J. Eisner. The Neural Hawkes Process. *NeurIPS*, 2017.
- S. Zuo, H. Jiang, Z. Li, T. Zhao, H. Zha. Transformer Hawkes Process. *ICML*, 2020.
- Q. Zhang, A. Lipani, O. Kirnap, E. Yilmaz. Self-Attentive Hawkes Process. *ICML*, 2020.
- O. Shchur, M. Biloš, S. Günnemann. Intensity-Free Learning of Temporal Point Processes. *ICLR*, 2020.
- C. Yang, H. Mei, J. Eisner. Transformer Embeddings of Irregularly Spaced Events and Their Applications. *ICLR*, 2022.
