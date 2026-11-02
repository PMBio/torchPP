# Extending torchpp

This page walks through adding a new model end-to-end. The mechanics are
the same whether you're adding a new Hawkes composition, a new neural
head, or an entirely standalone TPP.

## 1. Decide what to subclass

Pick the minimal base that already does what you need:

- **Standalone parametric intensity TPP** — subclass
  [`IntensityBasedTPP`](../torchpp/tpp/base_tpp.py) directly, e.g.
  alongside `PoissonProcess` in
  [`torchpp/tpp/baseline.py`](../torchpp/tpp/baseline.py).
  (`TemporalPointProcess` itself is the family-agnostic contract root — subclass
  it only for a wholly new family.)
- **Hawkes-style composite** — subclass
  [`HawkesProcess`](../torchpp/tpp/hawkes.py) and pick a
  `BaselineIntensityModule` + `KernelIntensityModule`. Most of the work
  is already done; the new class is usually 20-40 lines.
- **Inhibitive Hawkes composite** — subclass `InhibitiveHawkesProcess`.
  CI is numerical, so you mainly select the modules and the clamp.
- **Transformer-conditioned TPP** — subclass
  [`TransformerParametrizedTPP`](../torchpp/tpp/neural.py) and define the
  intensity head.
- **Inter-event-density TPP** (no intensity) — subclass
  [`DensityBasedTPP`](../torchpp/tpp/base_tpp.py) and root the family with the
  density hooks (`_log_density_time_given_mark` / `_log_survival_time_given_mark` / `_log_prob_mark` /
  `_event_log_density_sum` / `_draw_next` / `time_mean`), as
  `IntensityFreeProcess` does.

If your model needs new shared infrastructure (a new spline variant, a
new kernel module), put it in
[`torchpp/modules/intensity.py`](../torchpp/modules/intensity.py) or
[`torchpp/utils/math.py`](../torchpp/utils/math.py) so it can be reused.
Reuse is preferred over duplication; many shape and integration utilities
already exist.

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

    def _intensity_at_t(self, t, ts):
        (mu,) = self.transform_params()           # (D,)
        return mu.unsqueeze(0).expand(t.shape[0], -1)  # (B, D)

    def _closed_form_ci(self, ts, T_low, T):
        # Optional but recommended: closed-form Λ over [T_low or 0, T].
        # _effective_T folds in the terminating per-type upper limit min(T, t_term_d).
        (mu,) = self.transform_params()
        hi = self._effective_T(T, ts)             # (B, D)
        lo = self._effective_T(T_low, ts) if T_low is not None else 0.0
        return mu.unsqueeze(0) * (hi - lo)        # (B, D)
```

Two hooks are mandatory:

- `_intensity_at_t(t, ts) -> (B, D)` — raw intensity, before termination
  masking. The base class wraps this in `intensity()` and applies the
  termination mask. (You may instead override the multi-query
  `_intensities_at_queries(query_times, ts) -> (B, Q, D)`; the two delegate to
  each other, so exactly one is required.)
- `transform_params() -> Tuple[Tensor, ...]` — return parameters in their
  constrained form.

A closed-form CI hook (`_closed_form_ci`, or the per-interval `_interval_ci`) is
optional but strongly recommended whenever you have a closed-form integral —
it is selected by the default `CIConfig(mode="closed_form")`. Without one,
declare numerical integration by overriding `_default_ci_config()` to return a
`CIConfig(mode="numerical", placement=..., rule=..., n_points=...)` (slower, but
the only option when the intensity has no closed-form integral).

For neural / transformer-based heads, also consider overriding
`_event_intensities(ts) -> (B, L)` — it lets you compute observed-type
intensities directly from the cached encoder output without materialising
the full `(B, L, D)` tensor.

## 3. Respect the terminating contract

If the model is meant to be usable with `terminating=True`, **do not
inspect termination state inside the intensity or CI hooks**. The
base class:

- multiplies `_intensity_at_t`'s output by the termination mask in
  `intensity()`,
- clamps the closed-form CI hook's per-type upper limit to
  `min(T, t_term_d)` (via `_effective_T`),
- inserts termination times as exact knots in the numerical CI grid,
- penalises sequences with repeated types inside `likelihood` (their
  positive term is set to $-\infty$).

If your model genuinely changes its functional form under termination
(rare), document this clearly and add tests in
`tests/contract/test_terminating_contract.py`.

## 4. Register the new model

Three places need updating to make the model selectable from a YAML
config:

1. **Factory** — import and instantiate in
   [`torchpp/training/model_factory.py::create_model`](../torchpp/training/model_factory.py).
   Add the description to `_MODEL_DESCRIPTIONS`. If the run-name
   convention needs new arch knobs, extend `_build_run_name`.
2. **Validation list** — add the `model_type` string to
   [`torchpp/training/config.py::VALID_MODEL_TYPES`](../torchpp/training/config.py).
3. **Test matrix** — add it to
   [`tests/conftest.py::MODEL_TYPES`](../tests/conftest.py) so the
   parameterised matrix covers it.

If the model should be selectable via `-c model/<name>` shorthand, add
[`configs/model/<name>.yaml`](../configs/model/) with at minimum:

```yaml
# configs/model/my_tpp.yaml
model_type: my_tpp

arch:
  spline_K: 5         # or any model-specific knobs
```

If your model exports user-facing classes, also add them to the
re-exports in [`torchpp/__init__.py`](../torchpp/__init__.py) and to the
[API reference](./api_reference.md).

## 5. Test it

The repo's test fixtures are in
[`tests/conftest.py`](../tests/conftest.py). Adding a `model_type` to
`MODEL_TYPES` exercises every parameterised test against your model
automatically; the resulting matrix grows quickly, so the targeted files
are usually a faster feedback loop:

```bash
# Smoke + likelihood / CI consistency
pytest tests/contract/test_model_contract.py -k "my_tpp" -v
pytest tests/contract/test_intensity_consistency.py -k "my_tpp" -v

# Terminating behaviour, if applicable
pytest tests/contract/test_terminating_contract.py -k "my_tpp" -v

# If you added new spline / kernel infrastructure
pytest tests/unit/intensity/test_spline_kernel.py -v
```

For neural heads, run the targeted neural / simultaneous-events tests too
(`tests/models/test_neural_simultaneous.py`) since the generic matrix does not
exercise the causal-cache lookup carefully.

## 6. Numerical-stability checklist

- Clamp intensities used in `log` (the base class already does this in the
  positive term of `likelihood`).
- Initialise unconstrained parameters with `inverse_softplus` so that
  `softplus(raw)` starts at a meaningful value.
- For analytical CI involving exponentials of large arguments, prefer
  `expm1` / `log1p` and clamp arguments before the exponential to avoid
  overflow.
- For any new spline / piecewise-constant component, look at how
  `LinearSpline` and `PiecewiseConstant` handle inf-times-zero patterns
  before rolling your own.

## 7. Document it

- Add a short entry to [`docs/models.md`](./models.md) explaining the
  model in one sentence and pointing at the class.
- If the change introduces broader infrastructure (a new module type, a
  new dataset loader, etc.), add a note under `docs/` and link it from
  [`docs/index.md`](./index.md).

## Common pitfalls

- **Returning shape `(B,)` from `_intensity_at_t`** instead of `(B, D)`.
  The base class strictly expects per-type intensities.
- **Inspecting termination state in subclass code.** It's the base
  class's job; doing it twice double-masks the intensity.
- **Forgetting to register the model in all three places.** Symptom:
  YAML composition succeeds but the factory raises `ValueError: Unknown
  model_type`.
- **Skipping `inverse_softplus` for initialisation.** Setting
  `raw_mu = 0.1` gives `softplus(0.1) ≈ 0.74`, not `0.1`.
- **Implementing CI via numerical quadrature when an analytical form
  exists.** Subtle but adds a 2-10× cost during training, and the
  gradients through the trapezoidal grid are noisier.
