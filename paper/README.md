# Paper tables and figures

This directory regenerates the result tables and Figure 2 of *Flexible Intensities Matter* (NeurIPS 2026)
from the released per-seed metrics in `results/`. Run all commands from the repository root.

```bash
python paper/make_tables.py               # -> paper/tables/*.tex
python paper/make_flexibility_figure.py   # -> paper/figures/flexibility_decomposition.{pdf,png}
```

UK Biobank results are not released. Tables that contain UKB results are rendered without the UKB part:
the UKB column of `main_medical.tex` and the UKB rows of `app_iftpp_encoder.tex` are omitted, and the
full UKB table is not produced.

## Data

- **Taobao** is the EasyTPP release on the Hugging Face hub (`easytpp/taobao`), stored in float64 under the
  data root `easytpp_hf_f64` (`scripts/data_prep/download_easytpp_datasets.py --json_dir ... --dtype float64`,
  see the main README). The root selects the time constants computed from its training split
  (`scripts/data_prep/compute_easytpp_time_constants.py`); the run generators pass
  `dataset.easytpp_root=easytpp_hf_f64` for Taobao.
- **MIMIC-IV** (`mimic_iv`) is the ICU-interventions task with dequantized float64 event times: each event is
  placed uniformly within its recorded minute (truncated at the ICU outtime, seed 0;
  `scripts/data_prep/make_mimic_iv_dequantized.py`, data config `mimic_iv_icu_interventions_top116_f64_dequant`).
  It needs a credentialed MIMIC-IV 3.1 download; only aggregate metrics are released for it. NeuralConstTPP uses
  a Softplus head on MIMIC-IV (`arch.neural_poisson_link=softplus`).

## Result files (`results/`)

| File | Content | Produced by |
|---|---|---|
| `main_runs_perseed.csv` | Main runs: one row per model × dataset × seed, with the full configuration (`arch.*`, `training.*`, `batching.*`, `eval.*`) and the closed-form test metrics (joint / time / type log-likelihood, Acc\|t). | `scripts/tuning/generate_main_run_commands.sh` |
| `grid_eval_perseed.csv` | Grid-based metrics of the same checkpoints: marginal Acc, log p(d), RMSE (predictive mean; median for IntensityFree). `rmse_divergent` marks seeds whose integrated mean diverged. | `ACTION=eval bash scripts/tuning/generate_main_run_commands.sh` |
| `intensity_free_gru_perseed.csv` | IntensityFree with a GRU encoder. | `IFTPP_ENCODER=gru MODELS=intensity_free bash scripts/tuning/generate_main_run_commands.sh` |
| `knot_sweep.csv` | Knot-count sweep, mean and std over 5 seeds per model × dataset × K. | `scripts/tuning/generate_ksweep_commands.sh`; MIMIC-IV: `PROFILE=mimic … --datasets=mimic_iv_icu_interventions_top116` |
| `ns_control.csv` | Non-stationary Hawkes control on MIMIC-IV, mean and std over 5 seeds. | `scripts/tuning/generate_ns_mimic_commands.sh` |
| `binned_time_ll.csv` | Binned time log-likelihood log P(t ∈ bin) of the main runs (`run=main`) and the knot sweep (`run=knot_sweep`), one row per run × bin width (`delta`, in the time unit of the dataset; `centered`: bins centered on the grid points; `bin_anchor`: absolute time, previous event or recorded minute), with the continuous log p(t) of the same targets (`cont_all`) and the split into tied and other targets (`tie_definition`). | `paper/binned/` |
| `mimic_same_minute_split.csv` | MIMIC-IV main runs: log-likelihoods and Acc\|t of all targets, of the targets recorded in the same minute as the previous event, and of the others. | `paper/binned/mimic_tie_split.py` |
| `time_resolution.csv` | Recorded time resolution, bin widths and fraction of zero gaps of the evaluated test events. | inspection of the test splits |
| `integration_points.csv` | Inhibitive Hawkes, seed 0: test metrics with 100 integration points (the reported setting) and with 10,000 (Taobao) or 3,000 (MIMIC-IV). | `evaluate.py … arch.num_ci_integration_steps=N` (below) |
| `tick_repro.csv` | Acc\|t (%) of EasyTPP's Hawkes baseline refitted with `tick`. | `paper/tick/tick_hawkes_repro.py` |
| `calibration.csv` | Time-rescaling KS distance and mark ECE, seed 0. | `paper/calibration/` |

## Checkpoints (`checkpoints_manifest.csv`)

One row per released checkpoint (standard benchmarks: 370 main runs + 25 IntensityFree-GRU runs): file,
sha256, generator job, architecture overrides and the expected closed-form test metrics. Unzip the
checkpoint archive into `<dir>` and set `TORCHPPMODELS=<dir>`; `ACTION=eval bash
scripts/tuning/generate_main_run_commands.sh` then finds every file under `<dir>/main_run/`.

## Tables

| Table | File | Metric source |
|---|---|---|
| Main results (Table 1) | `main_std.tex` | `main_runs_perseed.csv` |
| Medical results (MIMIC-IV part) | `main_medical.tex` | `main_runs_perseed.csv` |
| Joint / time / type log-likelihood, Acc\|t | `app_std_{joint,time,type,acct}.tex` | `main_runs_perseed.csv` |
| Marginal Acc, log p(d), RMSE | `app_std_{acc,ptype,rmse}.tex` | `grid_eval_perseed.csv` |
| Full MIMIC-IV results | `app_mimic_full.tex` | both per-seed files |
| IntensityFree encoder | `app_iftpp_encoder.tex` | `main_runs_perseed.csv`, `intensity_free_gru_perseed.csv` |
| Knot sweep | `app_ksweep_all.tex`, `app_ksweep_mimic.tex` | `knot_sweep.csv` |
| Non-stationarity control | `app_ns_control.tex` | `ns_control.csv`, `knot_sweep.csv`, `main_runs_perseed.csv` |
| tick reproduction | `app_tick_repro.tex` | `tick_repro.csv`, `main_runs_perseed.csv` |
| Calibration | `app_calibration.tex` | `calibration.csv` |
| Time resolution | `app_time_resolution.tex` | `time_resolution.csv` |
| Binned log p(t), main models | `app_binned_main.tex` | `binned_time_ll.csv` |
| Binned log p(t), knot sweep | `app_binned_ksweep.tex` | `binned_time_ll.csv`, `knot_sweep.csv` |
| MIMIC-IV same-minute split | `app_mimic_ties.tex` | `mimic_same_minute_split.csv`, `binned_time_ll.csv` |

Figure 2 (`make_flexibility_figure.py`) reads `knot_sweep.csv` and `binned_time_ll.csv`.

The equivalence check against EasyTPP is run with `scripts/equivalence/`.

## Running the experiments

Each generator prints one `[JOB]` block per run. Run them with the local runner, for example:

```bash
bash scripts/tuning/tune_local_generic.sh scripts/tuning/generate_main_run_commands.sh --models=hawkes --datasets=taxi
```

The knot sweep sets no checkpoint path itself; run it with `MODEL_SUBDIR=knot_sweep` so that its checkpoints
land in `$TORCHPPMODELS/knot_sweep/`, where the binned-likelihood jobs look for them.

**Binned time likelihood and MIMIC-IV same-minute split.** `paper/binned/make_binned_commands.py` emits one
`binned_time_ll.py` job per main-run and knot-sweep checkpoint, with the bins of each dataset (Taxi 1 s,
StackOverflow 2^-13, Retweet 1, Amazon 0.01, Taobao 1e-4 anchored at the previous event, MIMIC-IV 1 minute
anchored at the recorded minute, plus coarser and centered bins). With `--mimic-dumps` it also emits the per-item
evaluation dumps of the MIMIC-IV main runs:

```bash
bash scripts/tuning/tune_local_generic.sh paper/binned/generate_binned_commands.sh --out-dir binned_out --mimic-dumps
python paper/binned/collect_binned.py <job logs> --out paper/results/binned_time_ll.csv
python paper/binned/mimic_tie_split.py --dir binned_out --out paper/results/mimic_same_minute_split.csv
```

`collect_binned.py` reads the `BINNED_METRICS_JSON` line that every binned job prints to its log.

**Integration points.** The inhibitive Hawkes processes integrate their cumulative intensity on a grid of 100
points. The check re-evaluates seed 0 with more points, using the evaluation command of the main-run generator
(`ACTION=eval`) without `-c eval/grid_metrics`, with `eval.compute_marginal_type=false` and
`arch.num_ci_integration_steps=10000` (Taobao, all 500 test sequences) or `3000` (MIMIC-IV, the first 200 test
stays truncated to 316 events: `eval.limit_test_size=200 eval.max_sequence_length=316`).

**tick reproduction.** `tick` needs its own environment (see `paper/tick/requirements.txt`):

```bash
python paper/tick/tick_hawkes_repro.py --pkl-root <easytpp-gdrive> --pt-root $TORCHPPDATA/easytpp_hf_f64 --out tick_repro.csv
```

`--pkl-root` is the EasyTPP Google Drive folder (`taxi/`, `so_v2/`, `retweet/`, `amazon/`); `--pt-root` holds the
Taobao splits written by `scripts/data_prep/download_easytpp_datasets.py --json_dir` (root `easytpp_hf_f64`).

**Calibration.** Evaluates trained checkpoints (seed 0) of Const./Exp Hawkes, Spline/Spline Hawkes,
NeuralConstTPP and NeuralSplineTPP:

```bash
python paper/calibration/run_calibration.py --checkpoint-dir <checkpoints> --out calib_easytpp
python paper/calibration/run_calibration.py --checkpoint-dir <checkpoints> --out calib_mimic \
    --datasets mimic_iv_icu --max-events 5000
python paper/calibration/make_report.py calib_*/calibration_raw.json --out paper/results/calibration.csv
```
