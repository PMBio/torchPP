# Architecture

This page explains how `torchpp` represents temporal point processes
internally: the abstract base, the four "verbs" every model exposes
(intensity, cumulative intensity, likelihood, sampling), the modular
Hawkes composition, and how terminating processes are handled.

## Class hierarchy

```
TemporalPointProcess            (torchpp/tpp/base_tpp.py — abstract root: the contract)
├── IntensityBasedTPP           (torchpp/tpp/base_tpp.py — the intensity harness)
│   ├── BaselineTPP             (torchpp/tpp/baseline.py)
│   │   ├── PoissonProcess
│   │   ├── ConditionalInhomogeniousPoissonProcess
│   │   ├── SplinePoissonProcess
│   │   └── PiecewiseConstantPoissonProcess
│   ├── HawkesProcess           (torchpp/tpp/hawkes.py, ABC: baseline + kernel)
│   │   ├── ConstBaselineExpKernelHawkesProcess
│   │   ├── LinearBaselineExpKernelHawkesProcess
│   │   ├── SplineBaselineExpKernelHawkesProcess
│   │   ├── ConstBaselineSplineKernelHawkesProcess
│   │   ├── SplineBaselineSplineKernelHawkesProcess
│   │   └── …                   (full list in models.md)
│   ├── InhibitiveHawkesProcess (sum-then-clamp; numerical CI)
│   │   ├── InhibitiveConstExpHawkesProcess
│   │   ├── InhibitiveSplineExpHawkesProcess
│   │   ├── InhibitiveSplineBaselineSplineKernelHawkesProcess
│   │   ├── SplineBaselineNeuralKernelHawkesProcess
│   │   └── SplineBaselineNeuralSplineKernelHawkesProcess
│   └── NeuralParametrizedTPP   (torchpp/tpp/neural.py)
│       ├── TransformerParametrizedTPP
│       │   ├── NeuralPoissonProcess
│       │   ├── NeuralLogLinearProcess
│       │   ├── NeuralSplineProcess
│       │   ├── NeuralPiecewiseConstantProcess
│       │   ├── NeuralMLPProcess
│       │   ├── NeuralTHPProcess
│       │   ├── NeuralSAHPProcess
│       │   ├── EasyTPPTHPProcess
│       │   └── EasyTPPSAHPProcess
│       ├── NHPProcess
│       └── AttNHPProcess
└── DensityBasedTPP             (torchpp/tpp/base_tpp.py — inter-event-density family)
    └── IntensityFreeProcess    (torchpp/tpp/neural.py — log-normal mixture)
```

`TemporalPointProcess` is the abstract `nn.Module` root: it holds only the
family-agnostic contract (likelihood, sampling, the next-event predictive
distributions). `IntensityBasedTPP` realizes that contract through an intensity
λ and cumulative intensity Λ; `DensityBasedTPP` realizes it directly from an
inter-event-time density. Every concrete model is a `torch.nn.Module` and can
be trained with the standard PyTorch optimisation loop.

## What every model exposes

The uniform API is the **predictive / likelihood / sampling contract**, not the
intensity: every model — intensity-based or density-based — exposes the same
next-event decomposition, named by axis (`density_*` for the continuous time
axis, `prob_*` for the discrete mark axis). Each distribution method takes a
`log: bool = True` argument (log-space is the implemented primitive; `log=False`
returns the `exp`) and accepts time queries `t` of shape `(B,)` or `(B, Q)`.
Only `likelihood` and `sample` are abstract; the rest have generic fallbacks
from the decomposition identities that each family base overrides for the subset
it roots. The intensity family realizes the contract from
$\lambda_d(t \mid \mathcal{H}_t)$ (the instantaneous rate of a type-$d$ event
given history), shown in the last column below.

| Method | Returns | Shape | Intensity-family realization |
|---|---|---|---|
| `likelihood(ts, T=None, log=True)` | per-sequence log-likelihood | `(B,)` | $\sum_i \log \lambda_{m_i}(t_i) - \sum_d \Lambda_d(T)$ |
| `density_joint(t, ts, log=True)` | next-event joint density p(t, m) | `(B, [Q,] D)` | $\lambda_d(t)\,S(t)$ |
| `density_time(t, ts, log=True)` | time-marginal density p(t) | `(B, [Q])` | $\sum_d \lambda_d(t)\,S(t)$ |
| `cdf_time(t, ts, log=True)` / `survival_time(...)` | P(t_next < t) / S(t) | `(B, [Q])` | $1 - S(t)$ / $S(t)$ |
| `prob_mark(ts, log=True)` | next-event class probs p(m) | `(B, D)` | $\int_{t_N}^{\infty} \lambda_d(s)\,S(s)\,ds$, normalised |
| `prob_mark_given_time(t, ts, log=True)` | conditional mark p(m \| t) | `(B, [Q,] D)` | $\lambda_d(t) / \sum_{d'} \lambda_{d'}(t)$ |
| `time_mean(ts)` | expected next-event time | `(B,)` | $t_N + \int_{t_N}^{T_\max} S(s)\,ds$ |
| `time_quantile(u, ts)` | quantile (median = `u=0.5`) | `(B,)` | grid- or Newton-based inverse of `cdf_time` |
| `sample(ts, num_steps, ...)` | generated future events | `MVEventData` | inverse-CDF rollout (`SamplingConfig`) |

`B` = batch size, `D` = number of event types, `L` = padded sequence length,
`Q` = number of query times. $S(s) = \exp(-\sum_d \Delta\Lambda_d(s))$ is the
survival function from the last observed event. The legacy names `PDF`, `CDF`,
`marginal_class_distribution`, `mean_next_event_time`, `inverse_CDF` remain as
thin aliases of `density_joint`, `cdf_time`, `prob_mark`, `time_mean`,
`time_quantile` respectively.

## Subclass contract

An intensity model subclasses `IntensityBasedTPP` and overrides two abstract
methods:

- **`_intensity_at_t(t, ts) -> (B, D)`** (or the multi-query
  `_intensities_at_queries(query_times, ts) -> (B, Q, D)`) — the raw intensity
  function before termination masking. The base class wraps it in
  [`intensity()`](../torchpp/tpp/base_tpp.py) and applies the termination mask
  there. (The two hooks mutually delegate, so overriding either one suffices.)
- **`transform_params() -> Tuple[Tensor, ...]`** — return the constrained
  form of the model's parameters. Raw parameters are usually stored
  unconstrained and mapped through `softplus`/`exp` in this method to
  enforce positivity. Use
  [`torchpp.utils.math.inverse_softplus`](../torchpp/utils/math.py) for
  initialisation.

Concrete models *should* (when tractable) override:

- **`_closed_form_ci(ts, T_low, T) -> (B, D)`** (or the per-interval
  `_interval_ci(ts, t_lo, t_hi)`) — the closed-form cumulative intensity over
  `[T_low or 0, T]`, selected when the model's `CIConfig` is `closed_form` (the
  default). The base class folds termination into the per-type upper limit
  `min(T, t_term_d)`, so these hooks never need to inspect termination times. A
  model with no closed form instead declares numerical integration via
  `_default_ci_config()` (see below).
- **`_event_intensities(ts) -> (B, L)`** — direct computation of the
  observed-type intensity at every position, when this can be done without
  materialising the full `(B, L, D)` tensor (neural models override this for
  efficiency). The base class handles padding, log-clamping, and the
  terminating-repeat penalty inside `likelihood`.

A density model subclasses `DensityBasedTPP` instead, and roots the family by
implementing the density hooks (`_log_density_time_given_mark`, `_log_survival_time_given_mark`,
`_log_prob_mark`, `_event_log_density_sum`, `_draw_next`, `time_mean`) — no
intensity is involved.

## Numerical vs analytical CI

The cumulative-intensity scheme is a value object, `CIConfig`, resolved
call-arg > active `use_ci(cfg)` context > the model's `_default_ci_config()`.
`cumulative_intensity` dispatches on it:

- **Closed-form** (`CIConfig(mode="closed_form")`) — dispatches to the model's
  analytic per-interval hook (`_interval_ci` / `_closed_form_ci`). Used by all
  Hawkes and Baseline composites and the closed-form neural heads.
- **Numerical** (`CIConfig(mode="numerical", ...)`) — quadrature via
  [`numerical_integration`](../torchpp/utils/math.py). `placement` picks
  event-knotted per-interval integration (`interval`) or a fixed budget of
  points over the whole window (`global`); `rule` ∈ `{"trapezoidal", "mc",
  "mc_trapezoidal"}` and `n_points` (default `100`) set the quadrature. Used when
  the intensity has no closed-form integral, e.g. `InhibitiveHawkesProcess`, the
  neural-kernel composites, and the numerical neural heads (NHP, AttNHP, MLP,
  EasyTPP THP/SAHP).

Each family sets its default via `_default_ci_config()` (closed-form for the
majority; numerical families declare their scheme explicitly). Wrapping a block
in `with model.use_ci(cfg):` redirects every CI call inside without touching the
model — this is how the evaluation harness forces a high-precision scheme.

For numerical-CI inversion (sampling, quantile prediction) the base class builds
an [`IntensityGrid`](../torchpp/utils/math.py) once and inverts via linear
interpolation in O(1) per query, instead of repeating numerical integration in a
Newton loop.

## Hawkes composition

Hawkes models in `torchpp/tpp/hawkes.py` are assembled from two reusable
building blocks defined in
[`torchpp/modules/intensity.py`](../torchpp/modules/intensity.py):

- `BaselineIntensityModule` — exogenous component $\mu_d(t)$. Concrete
  subclasses: `ConstantBaselineModule`, `LinearBaselineModule`,
  `SplineBaselineModule`, `PiecewiseConstantBaselineModule`.
- `KernelIntensityModule` — endogenous self-exciting component
  $\sum_{i: t_i < t} \phi_{m_i, d}(t - t_i)$. Concrete subclasses:
  `ExpKernelModule`, `SplineKernelModule`, `PiecewiseConstantKernelModule`,
  `NonStationarySplineExpKernelModule`, `NeuralSplineKernelModule`,
  `NeuralKernelModule`.

```
HawkesProcess(D, baseline=Baseline(...), kernel=Kernel(...))
  intensity(t)   = baseline(t) + kernel(t)
  CI([0, T])     = baseline.CI(0, T) + kernel.CI(0, T)   (analytical when both modules support it)
```

`InhibitiveHawkesProcess` wraps the *sum* in a positivity-enforcing function
$\mathrm{clamp}(\mu(t) + \kappa(t))$ — `softplus`, `relu`, `square`, or
`exp` (full list in
[`hawkes.py::VALID_CLAMPING_FNS`](../torchpp/tpp/hawkes.py)). Because the
clamp is non-linear, the CI no longer factors and is computed numerically.
This is what unlocks "inhibitive" behaviour where some kernel weights are
negative.

The bottom of `hawkes.py` is a list of thin compositions selecting specific
baseline/kernel pairs. Add a new composition by writing one such class and
registering it; see [Extending](./extending.md).

## Neural TPPs

`NeuralParametrizedTPP` (in
[`torchpp/tpp/neural.py`](../torchpp/tpp/neural.py)) couples a sequence
encoder (the causal transformer in
[`torchpp/modules/nn/`](../torchpp/modules/nn/)) with a parametric
intra-event intensity head. The encoder produces, per event position, a
vector of head parameters; the head defines $\lambda_d(t)$ for $t$ between
that event and the next.

`TransformerParametrizedTPP` adds per-batch parameter caching: the
transformer runs once per batch, and intensity / CI queries at arbitrary
times look up the parameters of the current inter-event interval. **Any
change to the caching or lookup logic must preserve causal semantics for
simultaneous events and time queries between observed events** — validate
with the targeted neural-model tests, not just the generic matrix.

Concrete heads:

- `NeuralPoissonProcess` — constant intensity per interval.
- `NeuralLogLinearProcess` — log-linear intensity, analytical CI.
- `NeuralSplineProcess` — log-spline intensity, analytical CI.
- `NeuralPiecewiseConstantProcess` — piecewise-constant intensity.
- `NeuralMLPProcess` — MLP-parameterised intensity, numerical CI.
- `NeuralTHPProcess` / `NeuralSAHPProcess` — re-implementations of THP /
  SAHP with our infrastructure.
- `EasyTPPTHPProcess` / `EasyTPPSAHPProcess` — faithful EasyTPP replicas
  (numerical CI, kept for one-to-one comparison with the EasyTPP benchmark).

## Terminating mode

Setting `terminating=True` enforces that each event type can occur at most
once per sequence (e.g. first-occurrence disease data). The computation lives
as pure functions of the event data in
[`torchpp/utils/event_data.py`](../torchpp/utils/event_data.py); the model
carries the `terminating`/`D` config and reaches them through thin method
wrappers (`_get_termination_mask`, `_get_termination_times`, …). Termination
enters the math in three places:

- `_get_termination_mask(ts, t)` returns `(B, D)` with `True` where the type
  has *not* yet occurred strictly before $t$. `intensity()` multiplies by
  this mask, so $\lambda_d(t) = 0$ once type $d$ has fired.
- `_get_termination_times(ts)` returns `(B, D)` with each type's first-
  occurrence time, or `inf` for types that haven't fired. The closed-form CI
  path clamps the per-type integration upper bound to
  $\min(T, t_{\text{term},d})$; the numerical CI path inserts these
  termination times as exact grid knots.
- `likelihood` detects sequences with a repeated type and sets their
  log-likelihood to $-\infty$ (via the positive term), since such sequences
  have probability zero under a terminating model.

Subclasses do **not** need to handle termination themselves — they
implement the intensity/CI hooks as if the process were non-terminating, and
the base class composes the masking and effective upper limits.

## Tensor shape conventions

| Quantity | Shape |
|---|---|
| Time inputs (`t`, `T`) | `(B,)` |
| Per-batch, per-type | `(B, D)` |
| Per-event | `(B, L)` — padding marked by `event_types == -1` |
| Per-event, per-type | `(B, L, D)` |
| Per-sequence likelihood | `(B,)` |

Padding is `-1` for event types; padded times are `max_time + 1.0` per
sequence. Always derive masks from
[`BatchedMVEventData.event_mask`](../torchpp/utils/event_data.py) /
`time_points_before(t)` rather than relying on raw padded values.
`event_types_idx_safe` exists to avoid invalid indexing on padded entries
(it replaces `-1` with `0`); the result is then masked out by `event_mask`.

## Numerical-stability notes

- Intensities are clamped to `min=1e-12` before `log` in the positive
  likelihood term.
- `LinearSpline` ([`torchpp/utils/math.py`](../torchpp/utils/math.py))
  guards against `inf * 0` via conditional `torch.where`.
- CI inversion uses bracketed Newton's method (`invert_CI`) for analytical
  CI models, and grid-based linear interpolation for numerical-CI models.
- For repeated CI queries during marginal computations, `IntensityGrid`
  caches a single intensity-evaluation pass and exposes O(1) lookup.

## File map (where things live)

| Concern | Location |
|---|---|
| Abstract base, likelihood, sampling | [`torchpp/tpp/base_tpp.py`](../torchpp/tpp/base_tpp.py) |
| Poisson family | [`torchpp/tpp/baseline.py`](../torchpp/tpp/baseline.py) |
| Hawkes composites | [`torchpp/tpp/hawkes.py`](../torchpp/tpp/hawkes.py) |
| Neural TPPs | [`torchpp/tpp/neural.py`](../torchpp/tpp/neural.py) |
| Baseline/kernel modules | [`torchpp/modules/intensity.py`](../torchpp/modules/intensity.py) |
| Causal transformer | [`torchpp/modules/nn/`](../torchpp/modules/nn/) |
| Event containers, padding/masks | [`torchpp/utils/event_data.py`](../torchpp/utils/event_data.py) |
| Bucket/dynamic batching | [`torchpp/utils/batching.py`](../torchpp/utils/batching.py) |
| Math utilities | [`torchpp/utils/math.py`](../torchpp/utils/math.py) |
| Evaluation metrics | [`torchpp/utils/evaluation.py`](../torchpp/utils/evaluation.py) |
| Datasets | [`torchpp/data/`](../torchpp/data/) |
| Config schema, factory | [`torchpp/training/`](../torchpp/training/) |
