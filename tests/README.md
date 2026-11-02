# Tests

Four layers, so each property is asserted at the right altitude instead of
fanning every check over all ~29 models:

```
unit/         modules in isolation — instantiate directly, no model factory, CPU-stable
  intensity/    baseline + kernel modules; spline knots/tail; kernel init
  nn/           encoders, heads, LogNormalMixture
  utils/        IntensityGrid, integration, inversion
  config/       YAML compose, registry, CLI args
contract/     base-class API / relation / shape contracts over the ARCHETYPE set
models/       per-FAMILY numerical (CI-VALUE) correctness — kept fanned over the matrix
integration/  pointer to scripts/equivalence (full-train; not collected by pytest)
```

**Why two model tiers.** Relation/shape/identity contracts take an identical
code path for every model, so the `contract/` layer runs them over 6
representative **archetypes** (`helpers.ARCHETYPES`) — one per distinct
base-class branch (analytical CI, numerical CI, step intensity, encoder-backed,
API-divergent log-normal, log-linear). Genuine *numerical value* checks ("this
family's CI equals its closed form / fine-grid trapezoid / EasyTPP") stay in
`models/`, fanned over the full matrix — that is the trust chain and is never
reduced. `models/test_registry_smoke.py` fans all of `MODEL_TYPES` (construct +
one finite likelihood), so a newly registered model is always exercised.

## Run

```bash
# Fast device-stable gate (unit + contract), runnable anywhere
TPP_TEST_DEVICE=cpu pytest -m "unit or contract"

# Per-family numerical checks (heavier; full matrix)
pytest -m models

# One model family across the matrix
pytest tests/ -k "hawkes"

# A single layer / file
pytest tests/unit/intensity/
```

`TPP_TEST_DEVICE` (`cpu`/`cuda`) overrides the default device (cuda-if-available).
Markers (`unit`, `contract`, `models`, `integration`, `numerical`, `slow`,
`gpu`, `equivalence`) are auto-applied by directory.

## Shared code

- `helpers.py` — constants, the `create_model` factory, `MODEL_TYPES`,
  `ARCHETYPES`, the numerical-property skip-sets, and the event-data builders.
  Import values/factory from here (`from helpers import …`).
- `conftest.py` — fixtures only: RNG seeding (autouse), `device`, the per-model
  fan (`model`/`model_type`/`model_pair`) and the archetype fan
  (`representative_model`/`archetype`/…), plus the event-data fixtures.

## Adding a model

Add a branch to `helpers.create_model` keyed on the new `VALID_MODEL_TYPES`
entry. The registry smoke test and the per-family `models/` fan pick it up
automatically; add a representative to `ARCHETYPES` only if it exercises a
base-class branch none of the existing six cover.
