# Extending torchpp

This page describes how to add a new model from start to finish. The steps are the same for a new
Hawkes composition, a new neural head or an entirely standalone TPP.

## 1. Decide what to subclass

Pick the smallest base class that already does what you need.

- For a standalone parametric intensity TPP, subclass
  [`IntensityBasedTPP`](../torchpp/tpp/base_tpp.py) directly, for example next to
  `PoissonProcess` in [`torchpp/tpp/baseline.py`](../torchpp/tpp/baseline.py).
  `TemporalPointProcess` itself is the contract root shared by all families. Subclass it only for
  an entirely new family.
- For a Hawkes-style composite, subclass [`HawkesProcess`](../torchpp/tpp/hawkes.py) and choose a
  `BaselineIntensityModule` and a `KernelIntensityModule`. Most of the work is already done, and
  the new class is usually 20-40 lines long.
- For an inhibitive Hawkes composite, subclass `InhibitiveHawkesProcess`. The CI is numerical, so
  you mainly select the modules and the clamp.
- For a transformer-conditioned TPP, subclass
  [`TransformerParametrizedTPP`](../torchpp/tpp/neural.py) and define the intensity head.
- For an inter-event-density TPP without an intensity, subclass
  [`DensityBasedTPP`](../torchpp/tpp/base_tpp.py) and root the family with the density hooks
  (`_log_density_time_given_mark` / `_log_survival_time_given_mark` / `_log_prob_mark` /
  `_event_log_density_sum` / `_draw_next` / `time_mean`), as `IntensityFreeProcess` does.

If your model needs new shared infrastructure, such as a new spline variant or a new kernel
module, put it in [`torchpp/modules/intensity.py`](../torchpp/modules/intensity.py) or
[`torchpp/utils/math.py`](../torchpp/utils/math.py) so that it can be reused. We prefer reuse over
duplication, and many utilities for shapes and integration already exist.

## 2. Implement the required hooks

```python
class MyTPP(IntensityBasedTPP):
    def __init__(self, D, ..., terminating=False):
        super().__init__(D=D, terminating=terminating)   # closed-form CI by default
        # Store *unconstrained* parameters; transform_params() applies
        # the constraint (softplus / exp) to enforce positivity.
        # Use inverse_softplus(initial_value) to seed unconstrained params
        # so that softplus(raw) starts at the desired value.
        self.raw_mu = nn.Parameter(inverse_softplus(torch.full((D,), 0.1)))

    def transform_params(self):
        return (F.softplus(self.raw_mu),)        # (D,)

    def _intensities_at_queries(self, query_times, ts):
        (mu,) = self.transform_params()               # (D,)
        B, Q = query_times.shape
        return mu.view(1, 1, -1).expand(B, Q, -1)      # (B, Q, D)

    def _closed_form_ci(self, ts, T_low, T):
        # Optional but recommended: closed-form Λ over [T_low or 0, T].
        # _effective_T folds in the terminating per-type upper limit min(T, t_term_d).
        (mu,) = self.transform_params()
        hi = self._effective_T(T, ts)             # (B, D)
        lo = self._effective_T(T_low, ts) if T_low is not None else 0.0
        return mu.unsqueeze(0) * (hi - lo)        # (B, D)
```

Only one hook is required.

- `_intensities_at_queries(query_times, ts) -> (B, Q, D)` returns the raw intensity (before
  termination masking) at arbitrary query times. The base class wraps it in `intensity()` and
  applies the termination mask. A single-query evaluation is computed as
  `_intensities_at_queries(t.unsqueeze(1), ts).squeeze(1)`. The hook is enforced at subclass
  creation.

The example above uses `transform_params() -> Tuple[Tensor, ...]`. This is an *optional*
convention that the classical Hawkes and Baseline models follow to return their parameters in
constrained form. It is not part of the base contract and is not required.

A closed-form CI hook (`_closed_form_ci`, or the per-interval `_interval_ci`) is optional, but we
strongly recommend it whenever a closed-form integral exists. It is selected by the default
`CIConfig(mode="closed_form")`. Without such a hook, declare numerical integration by overriding
`_default_ci_config()` to return a
`CIConfig(mode="numerical", placement=..., rule=..., n_points=...)`. This is slower, but it is the
only option when the intensity has no closed-form integral.

For neural and transformer-based heads, also consider overriding
`_event_intensities(ts) -> (B, L)`. It lets you compute the intensities of the observed types
directly from the cached encoder output, without materializing the full `(B, L, D)` tensor.

## 3. Respect the terminating contract

If the model should work with `terminating=True`, **do not inspect the termination state inside
the intensity or CI hooks**. The base class already does the following:

- It multiplies the output of `_intensities_at_queries` by the termination mask in `intensity()`.
- It clamps the per-type upper limit of the closed-form CI hook to `min(T, t_term_d)` (via
  `_effective_T`).
- It inserts the termination times as exact knots in the numerical CI grid.
- It penalizes sequences with a repeated **predictable** type inside `likelihood` by setting their
  positive term to $-\infty$. Context (read-only) types, declared via `context_types`, are outside
  the competing-risks set. They may repeat, they are never suppressed after they occur, and they
  never enter the survival competition. The role mask is applied first, and termination then acts
  within the predictable subset.

If your model really changes its functional form under termination (which is rare), document
this clearly and add tests in `tests/models/test_terminating_values.py`.

## 4. Register the new model

In your own code, you register the class with `register_model`.

```python
import torchpp

@torchpp.register_model("my_tpp", description="My TPP")
class MyTPP(torchpp.IntensityBasedTPP):
    def __init__(self, D, *, seed=None, terminating=False, time_scale=1.0,
                 token_roles=None, ci_config=None, sampling_config=None):
        super().__init__(D, seed=seed, terminating=terminating, time_scale=time_scale,
                         token_roles=token_roles, ci_config=ci_config,
                         sampling_config=sampling_config)
        ...

model = torchpp.create_model("my_tpp", D=3)
```

The default builder calls
`MyTPP(D, terminating=..., seed=..., time_scale=..., token_roles=...)`. A model with its own
settings declares an architecture dataclass and a builder that reads it.

```python
from dataclasses import dataclass
from torchpp.training.arch import IntensityArchConfig

@dataclass
class MyTPPArch(IntensityArchConfig):
    num_basis: int = 8

torchpp.register_model(
    "my_tpp", MyTPP, config_cls=MyTPPArch,
    build=lambda arch, ctx: MyTPP(ctx.D, num_basis=arch.num_basis, **ctx.kw),
)
model = torchpp.create_model("my_tpp", D=3, arch={"num_basis": 16})
```

The `arch:` section in YAML then accepts exactly these fields, plus the shared `seed`, `ci` and
`sampling`. `ctx.baseline_spacing(section)` and `ctx.knot_gaps(section, num_knots)` convert knot
sections into spacings that are scaled by the dataset statistics, as in the built-in models. The
model then works with `create_model`, with `save_model` / `load_model` (constructor arguments are
recorded automatically) and with the command line, for example
`python train.py -c default_fit -c data/taxi model_module=my_pkg.my_tpp model_type=my_tpp
training_preset=classical`. `model_module` is imported before the config is validated, and a
model without a preset needs `training_preset`.

Inside the library, you add one entry to `MODEL_REGISTRY` in
[`torchpp/training/registry.py`](../torchpp/training/registry.py). The entry contains the class,
a one-line description, its architecture dataclass in
[`torchpp/training/arch.py`](../torchpp/training/arch.py) (reuse a section such as
`BaselineKnotsConfig` where it fits), a builder (`(arch, ctx) -> model`) and a run-name rule.
`VALID_MODEL_TYPES`, `create_model` and the config validation are all derived from the registry.

For the tests, add a small configuration of the model to `_BUILDERS` in
[`tests/helpers.py`](../tests/helpers.py). Importing the helpers fails until every registered
model has an entry, and the parametrized test matrix then covers the new model.

If the model should be selectable with the `-c model/<name>` shorthand, add
[`configs/model/<name>.yaml`](../configs/model/) with at least the following content:

```yaml
# configs/model/my_tpp.yaml
model_type: my_tpp

arch:
  num_basis: 8                  # the fields of the model's architecture dataclass
```

If your model exports user-facing classes, also add them to the re-exports in
[`torchpp/__init__.py`](../torchpp/__init__.py) and to the [API reference](./api_reference.md).

## 5. Test it

The parametrized model tests take their list of models from the registry, so a new registry entry
with its `_BUILDERS` configuration is covered automatically. The full matrix is large, so we
recommend running the targeted files with `-k`.

```bash
# Smoke + likelihood / CI consistency
pytest tests/contract/test_model_contract.py -k "my_tpp" -v
pytest tests/contract/test_intensity_consistency.py -k "my_tpp" -v

# Terminating behaviour, if applicable
pytest tests/models/test_terminating_values.py -k "my_tpp" -v

# If you added new spline / kernel infrastructure
pytest tests/unit/intensity/test_spline_kernel.py -v
```

For neural heads, also run the targeted tests for neural models and simultaneous events
(`tests/models/test_neural_simultaneous.py`). The generic matrix does not exercise the
causal-cache lookup carefully.

## 6. Numerical-stability checklist

- Clamp intensities before they are passed to `log`. The base class already does this in the
  positive term of `likelihood`.
- Initialize unconstrained parameters with `inverse_softplus`, so that `softplus(raw)` starts at
  a meaningful value.
- For an analytical CI that involves exponentials of large arguments, prefer `expm1` / `log1p`
  and clamp the arguments before the exponential to avoid overflow.
- For any new spline or piecewise-constant component, look at how `LinearSpline` and
  `PiecewiseConstant` handle inf-times-zero cases before writing your own.

## 7. Document it

- Add a short entry to [`docs/models.md`](./models.md) that describes the model in one sentence
  and points to the class.
- If the change introduces broader infrastructure, such as a new module type or a new dataset
  loader, add a note under `docs/` and link it from [`docs/index.md`](./index.md).

## Adding a dataset

A dataset is a `TPPDataset` subclass that is registered under a `dataset_type` string, in the
same way as the models are registered. This takes two steps.

1. Subclass `TPPDataset` ([`torchpp/data/base_dataset.py`](../torchpp/data/base_dataset.py)) and
   implement the classmethod `load(cls, cfg, root, split) -> TPPDataset`. It receives the
   `dataset:` config dict, the `$TORCHPPDATA` root and the split name (`"train"` / `"val"` /
   `"test"`). It returns a dataset whose `__getitem__` yields `MVEventData` items with times in
   **raw native units**. The model owns the time normalization, so do not pre-scale the times
   (see [Architecture → Time units](./architecture.md#time-units)). Declare the fixed properties
   of the data and the task as ClassVars.
   - `num_event_types` and `terminating` give the vocabulary size `D` and whether each type
     occurs at most once.
   - `default_token_roles` (`TokenRoles`) specifies whether sequences carry a real origin
     (`has_origin`), which types are context types (read but never predicted) and which types the
     evaluation ignores. The type lists can be overridden per run via `dataset.context_types` and
     `dataset.ignored_types` ([Notation → Token roles](./notation.md#token-roles)).
   - `cfg_keys` (a frozenset) lists the `dataset.*` keys that the dataset accepts. Any other key
     raises an error.

2. Register it with the `@register_dataset("<name>")` decorator
   ([`torchpp/training/dataset_registry.py`](../torchpp/training/dataset_registry.py)). The name
   becomes the `dataset_type` string that `load_split` dispatches on. A `data/<name>.yaml` preset
   can then select the dataset with `dataset_type: <name>`, together with the `dataset:` block
   that its `load()` expects.

[`BenchmarkDataset`](../torchpp/data/benchmark_data.py) is a worked example.
[Data → Your own dataset](./data.md#your-own-dataset) shows a minimal one and explains how
`train.py` finds it.

## Common pitfalls

- Returning shape `(B, Q)` from `_intensities_at_queries` instead of `(B, Q, D)`. The base class
  strictly expects per-type intensities.
- Inspecting the termination state in subclass code. This is the job of the base class, and doing
  it a second time masks the intensity twice.
- Registering the model without a test configuration. Importing `tests/helpers.py` fails until
  `_BUILDERS` has an entry for every registered model.
- Skipping `inverse_softplus` for initialization. Setting `raw_mu = 0.1` gives
  `softplus(0.1) ≈ 0.74`, not `0.1`.
- Implementing the CI by numerical quadrature when an analytical form exists. The effect is
  subtle, but it adds a 2-10× cost during training, and the gradients through the trapezoidal
  grid are noisier.
