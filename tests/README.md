# Tests

We organize the tests in three layers. Each property is tested in the layer where it belongs,
instead of running every check over all 26 models.

```
unit/         modules in isolation — instantiate directly, no model factory, CPU-stable
  intensity/    baseline + kernel modules; spline knots/tail; kernel init
  nn/           encoders, heads, LogNormalMixture
  utils/        IntensityGrid, integration, inversion
  config/       YAML compose, registry, CLI args
contract/     base-class API / relation / shape contracts over the ARCHETYPE set
models/       per-FAMILY numerical (CI-VALUE) correctness — kept fanned over the matrix
```

We use two model tiers. The relation, shape and identity contracts follow the same code path
for every model. The `contract/` layer therefore runs them over 6 representative archetypes
(`helpers.ARCHETYPES`), one for each distinct base-class branch: analytical CI, numerical CI,
step intensity, encoder-backed, API-divergent log-normal and log-linear. Checks of numerical
values (for example, that a family's CI equals its closed form or a fine-grid trapezoid) stay in
`models/`. They run over the full matrix and are never reduced to archetypes.
`models/test_registry_smoke.py` constructs every model in `MODEL_TYPES` and computes one finite
likelihood, so a newly registered model is always tested.

## Run

```bash
# Fast device-stable gate (unit + contract), runnable anywhere
TPP_TEST_DEVICE=cpu pytest -m "unit or contract"

# Per-family numerical checks (heavier; full matrix)
pytest -m models

# One model family across the matrix
pytest tests/ -k "const_exp_hawkes"

# A single layer / file
pytest tests/unit/intensity/
```

`TPP_TEST_DEVICE` (`cpu` or `cuda`) overrides the default device, which is CUDA if available.
The layer markers (`unit`, `contract`, `models`) are applied automatically by directory, so
`pytest -m contract`, for example, selects one layer.

## Shared code

- `helpers.py` contains the constants, the `create_model` factory, `MODEL_TYPES`,
  `ARCHETYPES`, the skip-sets for numerical properties and the event-data builders.
  Import values and the factory from here (`from helpers import …`).
- `conftest.py` contains only fixtures. These are the RNG seeding (autouse), `device`, the
  per-model fixtures (`model`/`model_type`/`model_pair`), the archetype fixtures
  (`representative_model`/`archetype`/…) and the event-data fixtures.

## Adding a model

Add a small configuration to `helpers._BUILDERS`, keyed on the new registry entry. Importing
`helpers` fails until every registered model has one. The registry smoke test and the
per-family tests in `models/` then include the new model automatically. Add a representative to
`ARCHETYPES` only if it covers a base-class branch that none of the existing six cover.
