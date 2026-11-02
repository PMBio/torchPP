# MIMIC-IV Benchmark Construction

## Scope

This note records the construction of the two recurring MIMIC-IV benchmarks in `TorchPP`:

- `hospital_flow`
- `icu_interventions`

MIMIC-IV is available to credentialed users via PhysioNet. No MIMIC-derived data
or models are distributed with this repository. Starting from the raw
`mimic-iv-3.1.zip` archive, a one-command pipeline:

1. builds both generated datasets under `$TORCHPPDATA/mimic_iv/`,
2. writes train/dev/test splits in the native `TorchPP` format,
3. and writes analysis reports under `tmp/mimic_iv_reports/`.

## One-command reproduction

Preferred command:

```bash
uv run python scripts/data_prep/recreate_mimic_iv_benchmarks.py \
  --zip-path <path>/mimic-iv-3.1.zip
```

This runs both steps below:

```bash
uv run python scripts/data_prep/prepare_mimic_iv_datasets.py \
  --zip-path <path>/mimic-iv-3.1.zip

uv run python scripts/data_prep/analyze_mimic_iv_datasets.py
```

Useful flags:

- `--force-data` rebuilds already generated datasets.
- `--skip-prepare` runs only the analysis on existing generated data.
- `--skip-analysis` regenerates the `.pt` files without writing the reports.
- `--out-root <dir>` sets the output directory (default `$TORCHPPDATA/mimic_iv`).
- `--float32` stores event times as float32 instead of float64 (see the presets below).

## Generated files

Dataset outputs live under the shared TorchPP data root:

```text
$TORCHPPDATA/mimic_iv/
  hospital_flow/
    train.pt
    dev.pt
    test.pt
    metadata.pt
  icu_interventions/
    train.pt
    dev.pt
    test.pt
    metadata.pt
```

The analysis step writes:

```text
tmp/mimic_iv_reports/
  hospital_flow_summary.json
  hospital_flow_type_frequencies.csv
  hospital_flow_topk.csv
  icu_interventions_summary.json
  icu_interventions_type_frequencies.csv
  icu_interventions_topk.csv
  icu_interventions_category_frequencies.csv
  mimic_iv_report.md
```

## Task definitions

### `hospital_flow`

- Sequence unit: one hospital admission (`hadm_id`)
- Source table: `mimic-iv-3.1/hosp/transfers.csv.gz`
- Event time: transfer `intime`
- Event type: `careunit`, with `EVENTTYPE::<eventtype>` fallback when `careunit` is missing
- Preprocessing: consecutive duplicate care-unit states are collapsed
- Process type: recurring, not terminating

This dataset models movement between care locations during one admission. It is the cleaner and sparser of the two MIMIC benchmarks.

### `icu_interventions`

- Sequence unit: one ICU stay (`stay_id`)
- Source tables:
  - `mimic-iv-3.1/icu/icustays.csv.gz`
  - `mimic-iv-3.1/icu/inputevents.csv.gz`
  - `mimic-iv-3.1/icu/procedureevents.csv.gz`
  - `mimic-iv-3.1/icu/d_items.csv.gz`
- Event time: event `starttime`
- Event type: raw ICU `itemid`
- Preprocessing: events are kept only if `starttime` lies within the ICU stay window `[intime, outtime]`
- Process type: recurring, not terminating

This dataset is the dense clinical benchmark. It keeps the raw intervention vocabulary on purpose so later benchmark variants can trim or merge it in a controlled way.

## Shared preprocessing rules

Both datasets currently use the same general policy:

- subject-level split by `subject_id`, with an `80 / 10 / 10` train/dev/test partition,
- minimum retained sequence length `2`,
- timestamps converted to days relative to the start of the local observation window,
- strictly increasing timestamps enforced with tiny forward jitter when needed,
- sequence times rescaled by dividing through the mean training-set observation window.

The final scaling factors in natural units are stored in `metadata.pt` and surfaced by the dataset loader.

## Loader and config integration

The generated data are loaded through the `mimic_iv` dataset type in `torchpp/data/mimic_iv_data.py`.

Current config presets:

- `configs/data/mimic_iv_hospital_flow.yaml`
- `configs/data/mimic_iv_icu_interventions.yaml`
- `configs/data/mimic_iv_icu_interventions_top75.yaml`
- `configs/data/mimic_iv_icu_interventions_top116.yaml` — reads `$TORCHPPDATA/mimic_iv`; the paper's
  hyperparameter search used float32 event times (build with `--float32`)
- `configs/data/mimic_iv_icu_interventions_top116_f64.yaml` — the same cohort with
  event times stored in float64 (build it with
  `prepare_mimic_iv_datasets.py --datasets icu_interventions --out-root $TORCHPPDATA/mimic_iv_f64`)
- `configs/data/mimic_iv_icu_interventions_top116_f64_dequant.yaml` — the float64 cohort with
  dequantized event times (used for all MIMIC-IV results of the paper; see below)

### Dequantized event times

Event times are recorded to the minute, and events of the same minute are ordered by itemid and
separated by the `1e-6`-day jitter above. `scripts/data_prep/make_mimic_iv_dequantized.py` removes this
structure from the float64 cohort: it recovers the recorded minute of every event, adds an offset drawn
uniformly within that minute (truncated at the ICU `outtime`, so `T_obs` is unchanged; seed 0), and
re-sorts the events of each stay. Stays, splits, type ids and the top-116 vocabulary are unchanged.
`scripts/data_prep/verify_mimic_iv_dequantized.py` checks these invariants against the source.

```bash
python scripts/data_prep/make_mimic_iv_dequantized.py --src-root mimic_iv_f64 --out-root mimic_iv_f64_dequant --seed 0
python scripts/data_prep/verify_mimic_iv_dequantized.py --orig-root mimic_iv_f64 --new-root mimic_iv_f64_dequant
```

The `mimic_iv` loader also supports an on-load ICU vocabulary remap through
`dataset.top_k_event_types=<K>`. For `icu_interventions`, this keeps the top-`K`
itemids ranked by train-split event frequency and maps the rest to a single
`OTHER` class. The resulting number of event types is therefore `K + 1`.

So the datasets can be selected with standard config composition, for example:

```bash
uv run python train.py -c default_fit -c data/mimic_iv_hospital_flow -c model/hawkes
uv run python train.py -c default_fit -c data/mimic_iv_icu_interventions -c model/hawkes
uv run python train.py -c default_fit -c data/mimic_iv_icu_interventions_top75 -c model/hawkes
uv run python train.py -c default_fit -c data/mimic_iv_icu_interventions_top116 -c model/hawkes
```

## Benchmark statistics

`analyze_mimic_iv_datasets.py` reports sequence counts, sequence lengths,
observation windows, time-scale factors, and top-K coverage of the ICU
intervention vocabulary on the train split. The ICU vocabulary has a heavy long
tail, so it is treated as a family of benchmark variants (top-K itemids plus an
`OTHER` class) rather than a single task.

## Recommended benchmark variants

### `hospital_flow`

Keep the raw vocabulary.

Reason: the vocabulary is already compact and sequences are short.

### `icu_interventions`

Use three variants, each answering a different question.

1. Raw benchmark: all itemids.
   Use this when you want the highest-fidelity ICU event stream and can tolerate a heavy long tail.

2. Compact benchmark: top-`116` itemids plus `OTHER`.
   This preserves at least `95%` of train events while removing the worst of the tail without introducing manual clinical clustering.

3. Category benchmark: collapse itemids to the `d_items.category` groups.
   Use this as a coarse ablation to test whether a model is learning fine-grained intervention structure or only broad clinical regimes.

The analysis script computes the exact top-K and tail-mass tables so these cutoffs can be revised later without touching the extraction code.

## Non-goals

The pipeline intentionally does **not** do any of the following yet:

- merge semantically similar ICU itemids by hand,
- collapse sparse items into custom clinical superclasses,
- drop events by dosage, amount, or route,
- build patient-level longitudinal admission sequences across multiple hospitalizations.

Those are benchmark-design decisions layered on top of this extraction step, not part of the base reproducible dataset build.

## Relevant scripts

- `scripts/data_prep/prepare_mimic_iv_datasets.py`: extraction from the raw zip archive into generated `.pt` splits
- `scripts/data_prep/analyze_mimic_iv_datasets.py`: type-frequency, sequence-length, and top-K analysis on the generated datasets
- `scripts/data_prep/recreate_mimic_iv_benchmarks.py`: one-command wrapper that runs both steps
- `scripts/tuning/tune_mimic_iv_icu_combined_local.sh`: staged local multi-GPU tuning launcher for the ICU benchmark variants
- `scripts/tuning/tuning_mimic_iv_best_params.sh`: stage-1 results file for the MIMIC tuning launcher

The tuning launcher defaults to the compact `top116` ICU benchmark:

```bash
bash scripts/tuning/tune_mimic_iv_icu_combined_local.sh --dry-run --stage1
```

To target the `top75` variant instead, override the dataset config and tuning key:

```bash
DATA_CONFIG=data/mimic_iv_icu_interventions_top75 \
DATASET_KEY=mimic_iv_icu_top75 \
bash scripts/tuning/tune_mimic_iv_icu_combined_local.sh --dry-run --stage1
```
