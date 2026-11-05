<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img alt="TorchPP" src="docs/assets/logo.svg" width="420">
  </picture>
</p>

# TorchPP

TorchPP is a PyTorch library for marked temporal point processes (TPPs): models of event
sequences in continuous time, where each event has a time and a type. We implement classical
Poisson and Hawkes processes, neural TPPs and hybrids through a shared interface for the
intensity, cumulative intensity, likelihood, next-event predictions, sampling and evaluation, so
they can be compared directly.

**Why TorchPP**

- **One interface for intensity-based and density-based models.** Hawkes processes, transformer
  and recurrent neural TPPs, and models that parametrize the next-event density directly are
  available under one unified interface.
- **Handling of boilerplate.** We implement a wrapper class for event series that handles the
  padding, combining, sequence-level covariates and time units. Models automatically convert to
  numerically stable time units internally. We implement and test all the relations for TPPs,
  implementing generic samplers and numeric integrators.
- **Flexible models with exact likelihoods.** We support flexible parametrizations, like
  spline-based Poisson processes, Hawkes kernels or neural network heads.
- **Terminating and censored sequences.** Each event type can be restricted to occur at most once
  (survival analysis, competing risks), and observation windows can end after the last event.
- **Classical and neural models side by side**, including adapted THP, SAHP, NHP, AttNHP and
  IntensityFree, and tools to read a fitted Hawkes process (baselines, kernels, branching matrix).

```python
import torch
from torchpp import MVEventData, create_model, fit, evaluate_tpp
from torchpp.training import default_fit_config

# Simulate 200 sequences from a 2-type Hawkes process (every model can generate data).
truth = create_model("const_exp_hawkes", D=2, seed=1)
with torch.no_grad():
    seqs = [truth.sample(MVEventData(torch.zeros(1), torch.tensor([i % 2])), num_steps=30, rng=i).sequence
            for i in range(200)]
train, test = seqs[:160], seqs[160:]

# Fit a flexible spline-kernel Hawkes process, then score next-event prediction on held-out data.
model = create_model("spline_spline_hawkes", D=2, seed=0)
result = fit(model, train, test, default_fit_config(num_steps=300, learning_rate=0.05))
for name, m in [("fitted", result.model), ("true", truth)]:
    r = evaluate_tpp(m, test, compute_marginals=False)
    print(name, {k: round(r[k], 3) for k in ["ll_per_event", "cond_top1_accuracy"]})
```

```
fitted {'ll_per_event': 0.834, 'cond_top1_accuracy': 0.708}
true {'ll_per_event': 0.851, 'cond_top1_accuracy': 0.708}
```

`ll_per_event` is the test log-likelihood per scored event, and `cond_top1_accuracy` is the
accuracy of the predicted type of the next event. Both are described in
[docs/metrics.md](docs/metrics.md).

If you are new to temporal point processes, we recommend starting with the
[introduction](docs/introduction.md). It explains the intensity, survival function, densities and
likelihood, how they follow from each other, and which TorchPP call computes each of them. The
other entry points are:

- [Getting started](docs/getting_started.md) covers installation, a first training run and
  using your own data.
- [Models](docs/models.md) explains which model to use when.
- The [notebooks](examples/README.md) cover [TPP basics](examples/00_tpp_introduction.ipynb),
  [comparing models](examples/06_compare_models.ipynb), intensities and sampling, and adding a
  model. There is also a [survival-analysis series](examples/survival/README.md).

In TorchPP, event types are categorical. Sequences can carry static covariates, but the built-in
models do not use them. The survival notebooks show how to write models that do. Continuous or
spatial marks are not (yet) supported. We are internally working on including diffusion and flow
models in the future.

## Install

```bash
git clone https://github.com/PMBio/torchPP.git torchpp && cd torchpp
uv sync --all-extras   # or: pip install -e ".[all]"
```

The core install needs only PyTorch and a few small packages. There are three optional extras.
`wandb` adds experiment logging, `examples` adds matplotlib and JupyterLab for the notebooks, and
`data` adds Hugging Face `datasets` for the benchmark download script. `all` installs all three.
The development tools (pytest) are installed with `uv sync --group dev` or
`pip install pytest pytest-xdist`. TorchPP requires Python ≥ 3.11 and is tested with PyTorch 2.9.

PyTorch is installed from PyPI. On a CPU-only machine, install the CPU build first with
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

`train.py`, `evaluate.py` and the `configs/` presets are part of the repository, not of the
installed package, so you need a clone to use the command line.

The storage locations are set by two environment variables.

- `TORCHPPDATA` (default `~/data/torchpp/`) holds the dataset splits.
- `TORCHPPMODELS` (default `models/` in the repository) is where `train.py` writes checkpoints
  when no `save.path` is given.

## Python API in more detail

```python
import torch
from torchpp import BatchedMVEventData, MVEventData, ConstBaselineExpKernelHawkesProcess

# Two sequences with D=2 event types, observed on [0, T_obs].
seqs = [
    MVEventData(torch.tensor([0.4, 1.1, 1.3, 2.7]), torch.tensor([0, 1, 1, 0]), T_obs=3.0),
    MVEventData(torch.tensor([0.2, 0.9, 2.2]), torch.tensor([1, 0, 1]), T_obs=3.0),
]
batch = BatchedMVEventData(mv_events=seqs)

model = ConstBaselineExpKernelHawkesProcess(D=2, seed=0)
opt = torch.optim.Adam(model.parameters(), lr=0.05)
for _ in range(200):
    opt.zero_grad()
    loss = model.loss(batch).sum() / model.num_scored_events(batch).sum() + model.regularization(batch)
    loss.backward()
    opt.step()

with torch.no_grad():
    t = batch.last_event_time + 0.1
    print(model.intensity(t, batch))       # λ_d(t | history), shape (B, D)
    print(model.likelihood(batch))         # log-likelihood per sequence, shape (B,)
    future = model.sample(seqs[0], num_steps=3).sequence  # the history plus 3 sampled events
    print(future.time_points, future.event_types)
```

For real datasets, `torchpp.fit` adds mini-batching, a learning-rate schedule, gradient
clipping, validation and early stopping, and returns the model with the best validation weights.
`save_model` and `load_model` save and load a model together with its configuration. When
`create_model` is given `dataset_stats`, it sets the model's time unit (the mean inter-event time)
and a data-driven initialization, as `train.py` does.

```python
from torchpp import DatasetStats, create_model, fit, load_model, save_model
from torchpp.training import default_fit_config

stats = DatasetStats.from_sequences(seqs, D=2)
model = create_model("const_exp_hawkes", D=2, dataset_stats=stats, seed=0)
config = default_fit_config(num_steps=100, learning_rate=0.05, eval_freq=50)
result = fit(model, seqs, seqs, config)  # validation set: use held-out sequences in practice
save_model("hawkes.pth", result.model)    # model type, D and architecture are recorded
model = load_model("hawkes.pth")
```

`fit(..., verbose=True)` prints progress. `fit` trains on the device the model is on. To train
on a GPU, call `model.to("cuda")` first or pass `device="cuda"`. If your own data is a table with
one row per event, see [Getting started](docs/getting_started.md#your-own-data). The notebooks in
`examples/` ([overview](examples/README.md)) cover the basics of TPPs, the data structures and
the model interface, fitting and evaluation, comparing models, intensities and sampling, adding a
model, and survival analysis.

## Command line

```bash
# download the benchmark datasets to $TORCHPPDATA/benchmarks/ (needs the `data` extra)
python scripts/data_prep/download_benchmark_datasets.py

# train Spline/Spline Hawkes on Taxi: a short 3000-step run (the model's default preset trains up
# to 40k steps with early stopping)
python train.py -c default_fit -c model/spline_spline_hawkes -c data/taxi training.num_steps=3000 \
    save.path=models/ssk_taxi.pth

# evaluate the checkpoint (model type, architecture and dataset are read from it)
python evaluate.py -c default_eval model_path=models/ssk_taxi.pth
```

Training writes the checkpoint (the weights plus the model type, architecture, dataset and seed),
the resolved configuration (`ssk_taxi.yaml`) and the final metrics (`ssk_taxi.metrics.json`).
`evaluate.py` writes `ssk_taxi.eval.json`. Relative paths in `save.path` and `model_path` are
resolved against the directory you run the command from.

The evaluation reports next-event metrics in native time units
([docs/metrics.md](docs/metrics.md)). Its grid-based metrics (RMSE, marginal accuracy) are slow
on CPU, and `eval.compute_marginal_type=false` skips them. By default, the first 1000 test
sequences are used. `-c eval/full` evaluates all of them.

The configuration is a composable stack of YAML presets (`-c preset`, later presets win) plus
dotted `key=value` overrides. `--print` shows the merged configuration without training, and
`--help` lists the presets. `default_fit` is the base. A model preset adds the training defaults
of its model family, and a data preset (`-c data/<name>`) selects the dataset.
`configs/demo.yaml` lists every field with its default. The details are in
[docs/configs.md](docs/configs.md). Weights & Biases logging is off by default and is turned on
with `-c wandb/offline` or `-c wandb/online`.

## Models

Each model has a `model_type`, and its configuration is in `configs/model/<model_type>.yaml`.
The table below lists a selection of the models.

| Family | `model_type` | Class | Intensity |
|---|---|---|---|
| Poisson | `poisson` | `PoissonProcess` | constant rate per event type |
| | `spline_poisson` | `SplinePoissonProcess` | linear-spline rate over time |
| Hawkes (Hawkes, 1971) | `const_exp_hawkes` | `ConstBaselineExpKernelHawkesProcess` | constant baseline + exponential kernel |
| | `spline_exp_hawkes` | `SplineBaselineExpKernelHawkesProcess` | spline baseline + exponential kernel |
| | `const_spline_hawkes` | `ConstBaselineSplineKernelHawkesProcess` | constant baseline + spline kernel |
| | `spline_spline_hawkes` | `SplineBaselineSplineKernelHawkesProcess` | spline baseline + spline kernel |
| Inhibitive Hawkes | `inhibitive_const_exp_hawkes` | `InhibitiveConstBaselineExpKernelHawkesProcess` | constant + exponential, clamped to stay positive |
| | `inhibitive_spline_spline_hawkes` | `InhibitiveSplineBaselineSplineKernelHawkesProcess` | spline + spline, clamped to stay positive |
| Neural (transformer encoder) | `neural_poisson` | `NeuralPoissonProcess` | constant rate per inter-event interval |
| | `neural_spline` | `NeuralSplineProcess` | linear spline in the time since the last event |
| | `thp` | `THPProcess` | Transformer Hawkes Process (Zuo et al., 2020) |
| | `sahp` | `SAHPProcess` | Self-Attentive Hawkes Process (Zhang et al., 2020) |
| Neural (recurrent / attention) | `nhp` | `NHPProcess` | Neural Hawkes Process, continuous-time LSTM (Mei & Eisner, 2017) |
| | `attnhp` | `AttNHPProcess` | Attentive Neural Hawkes Process (Yang et al., 2022) |
| Density-based | `intensity_free` | `IntensityFreeProcess` | log-normal mixture over inter-event times (Shchur et al., 2020) |

All 26 types are defined in the model registry (`torchpp/training/registry.py`,
`VALID_MODEL_TYPES`) and are described in [docs/models.md](docs/models.md). To add a model, see
[docs/extending.md](docs/extending.md).

## Datasets

| Dataset | Access | Preparation |
|---|---|---|
| Taxi, StackOverflow, Retweet, Amazon, Taobao, Earthquake, Volcano | public (Hugging Face) | `scripts/data_prep/download_benchmark_datasets.py` |

The download sources and attribution are listed in
[docs/data.md](docs/data.md#benchmark-datasets).

**Taobao** comes from the EasyTPP Hugging Face release (`easytpp/taobao`) and is stored in
float64. Avoid the Google Drive versions of Taobao: their train/dev and test splits do not match
(see [docs/data.md](docs/data.md#benchmark-datasets)).

Every model can also generate synthetic data. `model.sample(...)` extends a history, so a fitted
or hand-parametrized model generates data with known ground truth.

Any list of `MVEventData` works with the Python API. To use your own data with `train.py`,
subclass `TPPDataset` and register it, as described in
[docs/data.md](docs/data.md#your-own-dataset).

## Tests

```bash
pytest tests/unit tests/contract -q           # fast
pytest tests/ -k "poisson and not neural"     # the model matrix, scoped to one model
```

The model tests are parametrized over `VALID_MODEL_TYPES`. The full matrix takes a while, so we
recommend scoping it with `-k`. See [CONTRIBUTING.md](CONTRIBUTING.md) for more.

## Layout

```
torchpp/     library: tpp/ (models), modules/ (intensity + neural blocks), data/, utils/, training/
configs/     composable YAML presets (model / data / training / eval / batching / wandb)
scripts/     data_prep/ (dataset download and preparation)
examples/    notebooks
docs/        documentation (start at docs/index.md; definitions in docs/notation.md)
tests/       unit, contract and model tests
train.py     training entry point
evaluate.py  evaluation of a saved checkpoint
```

## Citation

If you use TorchPP, please cite the paper in which it was introduced,
[Flexible Intensities Matter](https://arxiv.org/abs/2611.06348) (NeurIPS 2026), and the software as
given in [CITATION.cff](CITATION.cff). The code for the paper is on the
[`neurips2026`](https://github.com/PMBio/torchPP/tree/neurips2026) branch.

```bibtex
@inproceedings{mehrtens2026flexible,
  title     = {Flexible Intensities Matter: A Comprehensive Re-evaluation of Hawkes
               Processes and Neural Temporal Point Processes},
  author    = {Mehrtens, Hendrik and B{\"u}ttner, Florian and Stegle, Oliver},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2611.06348}
}
```

## License

TorchPP is released under the MIT license (see [LICENSE](LICENSE)). Some modules are adapted
from EasyTPP (Apache-2.0) and nanoGPT/Delphi (MIT). Their notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
