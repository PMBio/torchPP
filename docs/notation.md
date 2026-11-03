# Notation and conventions

This page lists the quantities that every TorchPP model computes and the conventions behind
them. Other pages link here instead of repeating these definitions. For a step-by-step derivation of each
quantity, read the [introduction](./introduction.md) first.

## Event sequences

$D$ is the number of event types. A sequence is a list of events
$(t_1, m_1), \dots, (t_N, m_N)$ with times $t_1 \le t_2 \le \dots \le t_N$ and event types
$m_i \in \{0, \dots, D-1\}$ (also called marks), observed on the window
$[0, T_\mathrm{obs}]$ with $T_\mathrm{obs} \ge t_N$ (`MVEventData.T_obs`, default $t_N$).
$d$ denotes a generic type, $m_i$ the observed type of event $i$. $\mathcal{H}_t$ is the
history of events strictly before $t$. All quantities below are conditional on the history.

Two times recur below. The **window start** $t_s$ of the likelihood is 0 or $t_1$ (see
[Origin](#origin-first-event-conditioning)). The **forecast origin** $t_0$ of next-event
predictions is the last event or $T_\mathrm{obs}$ (see [Forecasting](#forecasting-from-t_obs)).

## Intensity, cumulative intensity, survival

| Symbol | Meaning | Method |
|---|---|---|
| $\lambda_d(t \mid \mathcal{H}_t)$ | conditional intensity of type $d$ | `intensity(t, ts)` |
| $\Lambda_d(a, b) = \int_a^b \lambda_d(s)\,ds$ | cumulative intensity, $a$ = `T_low`, $b$ = `T` | `cumulative_intensity(T, ts, T_low)` |
| $S(t) = \exp\big(-\sum_d \Lambda_d(t_0, t)\big)$ | survival: no event in $(t_0, t]$ | `survival_time(t, ts)` |
| $F(t) = 1 - S(t)$ | next-event CDF | `cdf_time(t, ts)` |
| $p(t, d) = \lambda_d(t)\,S(t)$ | joint next-event density | `density_joint(t, ts)` |
| $p(t) = \sum_d \lambda_d(t)\,S(t)$ | time-marginal density | `density_time(t, ts)` |
| $p(d \mid t) = \lambda_d(t) / \sum_{d'} \lambda_{d'}(t)$ | type given the time | `prob_mark_given_time(t, ts)` |
| $p(d) = \int_{t_0}^\infty p(s, d)\,ds$ | type marginal | `prob_mark(ts)` |

Density-based models (IntensityFree) define $p(t \mid d)$ and $p(d)$
directly and expose the same methods without an intensity. All distribution methods take
`log=True` by default.

Integrals that start at an event use the right limit $\lambda(t_i^+)$, the intensity that
already includes the event at $t_i$.

## Likelihood

For a non-terminating sequence the log-likelihood is

$$
\log L = \sum_{i \in \mathcal{I}} \log \lambda_{m_i}(t_i) - \sum_d \Lambda_d(t_s, T_\mathrm{obs}),
$$

where $\mathcal{I}$ is the set of [scored events](#scored-events) and $t_s$ the window
start (see [Origin](#origin-first-event-conditioning)). `likelihood(ts)` returns it per
sequence with shape `(B,)`, and `loss(ts)` is its negative.

In terminating mode (`terminating=True`), each type occurs at most once. After type $d$
fires at $t_{\mathrm{term},d}$ its intensity is 0, so $\Lambda_d$ is integrated only up to
$\min(T_\mathrm{obs}, t_{\mathrm{term},d})$. A sequence that repeats a predictable type has
log-likelihood $-\infty$.

## Origin (first-event conditioning)

The origin regime is the `has_origin` field of the data's `TokenRoles` (next section).

- `has_origin=False` is the default and is common for benchmarks with an arbitrary time
  origin. The first event is the origin. It and any events tied with it are conditioned on,
  not scored, and the window starts at $t_s = t_1$.
- `has_origin=True` means that sequences start at a real origin $t=0$ (e.g. admission or
  birth). Every event is scored and the window starts at $t_s = 0$. Transformer-encoder models
  prepend a learned `<bos>` embedding for the empty history. This embedding is internal to the
  encoder, never part of the data, and does not change $D$.

Likelihoods are comparable within one regime but not across the two.

## Token roles

`TokenRoles` collects how time 0 and each event type are treated.

```python
from torchpp import TokenRoles

roles = TokenRoles(has_origin=False, context_types=(), ignored_types=())
```

- `has_origin` sets whether time 0 is a real origin (previous section).
- `context_types` are types that are read as history but never predicted. They are not
  scored, not integrated, never sampled, and may repeat in terminating mode. All other types
  are **predictable**. They are scored in the positive term, integrated in the survival and,
  in terminating mode, part of the competing-risks set.
- `ignored_types` are types that are excluded from the evaluation targets only. They stay in
  the history and in the training likelihood.

One `TokenRoles` object is passed through the library.

- A dataset declares its defaults as the class attribute `default_token_roles`. A run
  overrides the type lists with `dataset.context_types` and `dataset.ignored_types`, and
  `dataset.token_roles` is the result.
- `create_model(..., token_roles=dataset.token_roles)` (or `token_roles=` on any model
  class) builds the model with these roles. `model.token_roles` returns them, and checkpoints
  store them (context types as the `model.predictable_mask` buffer).
- `evaluate_tpp(model, sequences)` uses `model.token_roles`, and `roles=` may change only
  `ignored_types`.

## Scored events

The scored events are the events whose density enters the likelihood. These are the real
events of predictable types, without the first event and the events tied with it when there
is no origin. `model.num_scored_events(ts)` counts them per sequence. Every "per event" number
uses this count: the training loss, the validation log-likelihood and the test
`ll_per_event`. The next-event evaluation predicts exactly these events (see
[Metrics](./metrics.md)).

## Forecasting from T_obs

Next-event predictions start at the forecast origin $t_0 = \max(t_N, T_\mathrm{obs})$.
When $T_\mathrm{obs} > t_N$, the model conditions on no event in $(t_N, T_\mathrm{obs}]$.
Survival, densities and mark probabilities are then taken from $t_0$, queries need
$t \ge t_0$, and `sample` draws events after $t_0$. When the next-event distribution has
probability mass at infinity (the event may never occur), `time_mean` and `time_quantile`
return `inf` and `sample` stops.

## Time units

Public methods take and return times in the data's native units. Internally each model
works in scaled units $t / s$ with $s$ = `model.time_scale`, which `create_model` sets to
the mean inter-event time of the training data (default 1). Densities therefore carry a
$-\log s$ term per scored event, so likelihoods and metrics are reported in native units.
Survival, CDF and mark probabilities are unitless and unaffected.
