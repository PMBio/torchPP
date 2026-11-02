# Flexible Intensities Matter

This branch reproduces the NeurIPS 2026 paper *Flexible Intensities Matter: A Comprehensive
Re-evaluation of Hawkes Processes and Neural Temporal Point Processes*.

It contains **TorchPP**, a PyTorch library for multivariate temporal point processes (TPPs):
classical Hawkes, neural and hybrid models behind one intensity / cumulative-intensity /
likelihood / sampling / evaluation interface, with support for terminating and right-censored
sequences. The code on this branch is the version used for the paper. The library continues on
the `main` branch.

## Install

```bash
git clone -b neurips2026 https://github.com/PMBio/torchPP.git torchpp-neurips2026 && cd torchpp-neurips2026
uv sync
```

Python 3.11 or 3.12 (the paper runs used 3.11). PyTorch is installed from the `pytorch-cu128` index on Linux and Windows.

Two environment variables set the storage locations:

- `TORCHPPDATA` (default `~/data/torchpp/`): dataset splits.
- `TORCHPPMODELS` (default `models/` in the repository): checkpoint root. The run generators
  write to subdirectories of it, e.g. `$TORCHPPMODELS/main_run/`.

## Quickstart

```bash
# download the EasyTPP benchmarks (Taxi, Retweet, Amazon, StackOverflow) to $TORCHPPDATA/easytpp/
python scripts/data_prep/download_easytpp_datasets.py

# train Spline/Spline Hawkes on Taxi
python train.py -c easytpp/default -c model/spline_spline_kernel -c training/transformer \
    -c data/easytpp_taxi save.path=models/ssk_taxi.pth

# evaluate the checkpoint
python evaluate.py -c default_eval -c data/easytpp_taxi -c model/spline_spline_kernel \
    model_path=models/ssk_taxi.pth
```

The evaluation also computes grid-based metrics (RMSE, marginal accuracy), which are slow
on CPU; `eval.compute_marginal_type=false` skips them and `eval.limit_test_size=<N>` limits
the test set.

`python train.py ... --print` shows the merged configuration without training. Configuration is
a composable YAML stack (`-c preset`, later wins) plus dotted `key=value` overrides; see
[docs/configs.md](docs/configs.md).

## Models

Paper names and the corresponding `model_type` (config `configs/model/<model_type>.yaml`) and class:

| Paper name | `model_type` | Class |
|---|---|---|
| Homogeneous Poisson | `poisson` | `PoissonProcess` |
| Spline Poisson | `spline_poisson` | `SplinePoissonProcess` |
| Const./Exp Hawkes | `hawkes` | `ConstBaselineExpKernelHawkesProcess` |
| Spline/Exp Hawkes | `spline_exp_hawkes` | `SplineBaselineExpKernelHawkesProcess` |
| Const./Spline Hawkes | `const_spline_kernel` | `ConstBaselineSplineKernelHawkesProcess` |
| Spline/Spline Hawkes | `spline_spline_kernel` | `SplineBaselineSplineKernelHawkesProcess` |
| Inhib. Const./Exp Hawkes | `inhibitive_const_exp_hawkes` | `InhibitiveConstExpHawkesProcess` |
| Inhib. Spline/Spline Hawkes | `inhibitive_spline_spline_kernel` | `InhibitiveSplineBaselineSplineKernelHawkesProcess` |
| NeuralConstTPP | `neural_poisson` | `NeuralPoissonProcess` |
| THP | `easytpp_thp` | `EasyTPPTHPProcess` |
| SAHP | `easytpp_sahp` | `EasyTPPSAHPProcess` |
| NHP | `nhp` | `NHPProcess` |
| IntensityFree | `intensity_free` | `IntensityFreeProcess` |
| NeuralSplineTPP | `neural_spline` | `NeuralSplineProcess` |
| Non-stationary Spline/Spline Hawkes | `spline_ns_spline_kernel` | `SplineBaselineNSSplineKernelHawkesProcess` |

All valid types are listed in `torchpp/training/config.py::VALID_MODEL_TYPES`; see
[docs/models.md](docs/models.md).

NeuralConstTPP uses an exponential rate head on the standard benchmarks and a Softplus head on MIMIC-IV
(`arch.neural_poisson_link=softplus`), as in the paper. The `main` branch uses a softplus head throughout
and cannot load the checkpoints of this branch.

## Datasets

| Dataset | Access | Preparation |
|---|---|---|
| EasyTPP Taxi, StackOverflow, Retweet, Amazon | public (Hugging Face) | `scripts/data_prep/download_easytpp_datasets.py` |
| EasyTPP Taobao | public (Hugging Face, json files) | same script with `--json_dir`, float64 (below) |
| MIMIC-IV ICU interventions | credentialed (PhysioNet) | `scripts/data_prep/prepare_mimic_iv_datasets.py` (below) |
| UK Biobank | approved UKB application | `scripts/data_prep/prepare_ukb_cluster_dataset.py` |

**Taobao.** The paper uses the EasyTPP release of Taobao on the Hugging Face hub
([`easytpp/taobao`](https://huggingface.co/datasets/easytpp/taobao), 17 event types), stored in float64
under the data root `easytpp_hf_f64`. This root selects the time constants computed from its training split
(`torchpp/data/easytpp_data.py`), and the run generators pass `dataset.easytpp_root=easytpp_hf_f64` for Taobao:

```bash
mkdir -p tbhf && for s in train dev test; do
  curl -L -o tbhf/$s.json https://huggingface.co/datasets/easytpp/taobao/resolve/main/$s.json; done
python scripts/data_prep/download_easytpp_datasets.py --json_dir tbhf --name taobao --num_types 17 \
    --dtype float64 --out_root $TORCHPPDATA/easytpp_hf_f64
```

The json files used for the paper have the md5 sums `8b69ec879f22492995b6d36b32c8ebca` (train),
`6716d9f225613541f951b43124af9638` (dev) and `4e7a74b45d3a5afbd6d5902868411337` (test); the converted splits
hold 1,300 / 200 / 500 sequences with 73,483 / 11,472 / 28,455 events.

**MIMIC-IV.** Download MIMIC-IV 3.1 from [PhysioNet](https://physionet.org/content/mimiciv/3.1/)
and build the top-116 ICU cohort in float64. All MIMIC-IV results of the paper (main runs, knot sweep,
non-stationarity control) use its dequantized version, in which each event is placed uniformly within its
recorded minute (truncated at the ICU outtime, seed 0):

```bash
# float64 cohort (preset data/mimic_iv_icu_interventions_top116_f64)
python scripts/data_prep/prepare_mimic_iv_datasets.py --zip-path /path/to/mimic-iv-3.1.zip \
    --datasets icu_interventions --out-root $TORCHPPDATA/mimic_iv_f64
# dequantized event times (preset data/mimic_iv_icu_interventions_top116_f64_dequant), and its checks
python scripts/data_prep/make_mimic_iv_dequantized.py --src-root mimic_iv_f64 --out-root mimic_iv_f64_dequant --seed 0
python scripts/data_prep/verify_mimic_iv_dequantized.py --orig-root mimic_iv_f64 --new-root mimic_iv_f64_dequant
# hyperparameter search (preset data/mimic_iv_icu_interventions_top116): float32 event times
python scripts/data_prep/prepare_mimic_iv_datasets.py --zip-path /path/to/mimic-iv-3.1.zip \
    --datasets icu_interventions --float32 --out-root $TORCHPPDATA/mimic_iv
```

The binned likelihood also reads `mimic_iv_f64` (the recorded minutes).

`scripts/data_prep/recreate_mimic_iv_benchmarks.py` builds the same data (same `--out-root` and
`--float32` options) and also writes the vocabulary reports. Construction details:
[docs/mimic_iv_benchmark_construction.md](docs/mimic_iv_benchmark_construction.md). Following the
PhysioNet policy on derived data, no MIMIC-derived data or checkpoints are released; the
configurations and aggregate metrics are.

**UK Biobank.** The UKB results cannot be reproduced without an approved UK Biobank
application. No UKB data, UKB-derived data, UKB-trained models or UKB logs are released. The
preparation code is provided ([docs/ukb_csv_conversion.md](docs/ukb_csv_conversion.md)); the
token vocabulary and ICD-10 mapping come from the
[Delphi repository](https://github.com/gerstung-lab/Delphi), whose public synthetic data also
works with the `data/ukb_simulated_*` presets.

## Reproducing the paper

All runs are defined by command generators in `scripts/tuning/`, executed with the local
multi-GPU runner:

```bash
# main results: 15 models x (5 EasyTPP datasets + MIMIC-IV) x 5 seeds
# (the 14 models of the paper tables plus neural_log_linear, which is not reported)
GPUS=0,1 bash scripts/tuning/tune_local_generic.sh scripts/tuning/generate_main_run_commands.sh

# grid-based metrics (RMSE, marginal Acc, log p(d)) on the saved checkpoints
ACTION=eval GPUS=0 bash scripts/tuning/tune_local_generic.sh scripts/tuning/generate_main_run_commands.sh

# knot-count sweep: EasyTPP, then MIMIC-IV (checkpoints in $TORCHPPMODELS/knot_sweep/)
MODEL_SUBDIR=knot_sweep bash scripts/tuning/tune_local_generic.sh scripts/tuning/generate_ksweep_commands.sh
MODEL_SUBDIR=knot_sweep PROFILE=mimic bash scripts/tuning/tune_local_generic.sh \
    scripts/tuning/generate_ksweep_commands.sh --datasets=mimic_iv_icu_interventions_top116

# non-stationarity control (MIMIC-IV)
bash scripts/tuning/tune_local_generic.sh scripts/tuning/generate_ns_mimic_commands.sh

# binned time likelihood and the MIMIC-IV same-minute split, on the checkpoints above
bash scripts/tuning/tune_local_generic.sh paper/binned/generate_binned_commands.sh --out-dir binned_out --mimic-dumps
```

The calibration, binned-likelihood and `tick` experiments and the collection of their results are described
in [paper/README.md](paper/README.md).

Weights & Biases logging is set by `WANDB_MODE` (`offline` by default, `online` or `disabled`);
`online` needs `wandb login` and uses `WANDB_ENTITY` if set.

Running a generator directly prints its commands. Every generator takes `--filter=PATTERN` (keep
jobs whose name contains PATTERN) and `--dry-run`, and rejects unknown flags. The main-run
generator also takes `--models=` and `--datasets=`; the knot sweep takes `--datasets=` and the
environment variables `MODELS`, `K_VALUES` and `PROFILE` (see the header of each script):

```bash
bash scripts/tuning/generate_main_run_commands.sh --models=hawkes --datasets=taxi
```

The hyperparameter searches are in
`scripts/tuning/tune_*_combined_local.sh`
([docs/easytpp_hyperparameter_search.md](docs/easytpp_hyperparameter_search.md)); the selected
values are in `scripts/tuning/tuning_*_best_params.sh`. The comparison against EasyTPP's
implementations is in `scripts/equivalence/`.

**Checkpoints** for the standard benchmarks will be published on Zenodo (contents and checksums: [paper/checkpoints_manifest.csv](paper/checkpoints_manifest.csv); usage: [paper/README.md](paper/README.md#checkpoints-checkpoints_manifestcsv)).

**Tables.** `paper/` holds the aggregate per-seed results and the script that renders the
paper's tables from them; see [paper/README.md](paper/README.md).

## Tests

```bash
pytest tests/unit -q                          # fast
pytest tests/ -k "poisson and not neural"     # the model matrix, scoped to one model
```

The model tests are parameterised over `VALID_MODEL_TYPES`; the full matrix takes a while, so
scope it with `-k`.

## Layout

```
torchpp/     library: tpp/ (models), modules/ (intensity + neural blocks), data/, utils/, training/
configs/     composable YAML presets (model / data / training / eval / batching / wandb)
scripts/     data_prep/, tuning/ (run generators + runner), equivalence/
paper/       aggregate results, table/figure generation and the calibration, binned-likelihood and tick experiments
tests/       unit, contract and model tests
train.py     training entry point
evaluate.py  evaluation of a saved checkpoint
```

## Citation

```bibtex
@inproceedings{mehrtens2026flexible,
  title     = {Flexible Intensities Matter: A Comprehensive Re-evaluation of Hawkes
               Processes and Neural Temporal Point Processes},
  author    = {Mehrtens, Hendrik and B{\"u}ttner, Florian and Stegle, Oliver},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE). Some modules are adapted from EasyTPP (Apache-2.0) and nanoGPT/Delphi (MIT);
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
