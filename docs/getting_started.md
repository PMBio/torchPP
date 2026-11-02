# Getting started

This page gets you from a fresh checkout to a trained, evaluated model. For
the dataset-specific preparation (UKB, MIMIC-IV, EasyTPP) see the project
[`README.md`](../README.md), which is the canonical source for download and
preprocessing instructions.

## Install

The repository uses [`uv`](https://docs.astral.sh/uv/) for dependency
management. The virtual environment lives in `.venv/`.

```bash
uv sync
```

PyTorch is pulled from the `pytorch-cu128` index on Linux/Windows. Python
≥ 3.11 is required.

Two environment variables control I/O paths and default to sensible
locations:

- `TORCHPPDATA` (default `~/data/torchpp/`) — root of dataset splits.
- `TORCHPPMODELS` (default `<repo>/models/`) — where checkpoints are written.

## Run a training

The unified entry point is [`train.py`](../train.py). Configuration is
composed from any number of YAML presets via `-c <path>` plus dotted
`key=value` overrides. See [Configs](./configs.md) for the full composition
rules.

```bash
# EasyTPP — Hawkes with spline baseline + spline kernel on taxi
python train.py \
    -c easytpp/default \
    -c model/spline_spline_kernel \
    -c training/transformer \
    dataset.dataset_name=taxi

# UKB — inhibitive const-baseline exp-kernel Hawkes
python train.py \
    -c default_fit \
    -c model/inhibitive_const_exp_hawkes \
    -c data/ukb_real_full
```

`--print` (or `-p`) renders the merged config and exits without training,
which is the fastest way to confirm what a complex `-c` stack actually
produces.

A 30-second smoke test that confirms the data path is wired correctly:

```bash
python train.py -c default_fit -c model/hawkes -c data/easytpp_taxi \
    training.num_steps=50 eval.limit_test_size=100 device=cpu
```

Valid `model_type` strings are catalogued in [Model catalog](./models.md) and
listed authoritatively in [`torchpp/training/config.py::VALID_MODEL_TYPES`](../torchpp/training/config.py).

## Evaluate a saved checkpoint

[`evaluate.py`](../evaluate.py) is the standalone evaluation entry point. It
re-reads the same dataset config the model was trained on and reports the
joint test log-likelihood plus next-event metrics:

```bash
python evaluate.py \
    -c default_eval \
    -c data/ukb_real_full \
    -c eval/full \
    model_path=models/<...>.pth \
    model_type=spline_exp_hawkes
```

## Use the library programmatically

`torchpp.__init__` re-exports the most commonly used symbols, so most user
code only imports from the top level:

```python
import torch
from torchpp import (
    BatchedMVEventData,
    MVEventData,
    SplineBaselineExpKernelHawkesProcess,
)

# Toy multivariate sequence: 3 event types, two sequences of length 4 and 6.
seqs = [
    MVEventData(
        time_points=torch.tensor([0.10, 0.34, 0.57, 0.91]),
        event_types=torch.tensor([0, 1, 0, 2]),
    ),
    MVEventData(
        time_points=torch.tensor([0.05, 0.22, 0.40, 0.55, 0.75, 0.95]),
        event_types=torch.tensor([2, 1, 0, 2, 1, 0]),
    ),
]
batch = BatchedMVEventData(mv_events=seqs)

model = SplineBaselineExpKernelHawkesProcess(
    D=3,                # number of event types
    K=5,                # spline knots for the baseline
    T_max=1.0,          # spline anchor (full observation window)
    terminating=False,
)

# Per-sequence log-likelihood, shape (B,)
log_lik = model.likelihood(batch)
print(log_lik)

# Sample 5 future events from the first sequence:
new_seq, times, type_dists = model.sample(seqs[0], num_steps=5)
```

The shared API of every TPP class is documented in
[Architecture](./architecture.md) and in
[`torchpp/tpp/base_tpp.py`](../torchpp/tpp/base_tpp.py).

## Tests

```bash
pytest tests/contract/test_model_contract.py tests/contract/test_terminating_contract.py
```

The matrix is parameterised over every entry in
[`tests/conftest.py::MODEL_TYPES`](../tests/conftest.py), which mirrors the
paper's models. Filter with `-k` to scope:

```bash
pytest tests/ -k "hawkes" -v
pytest tests/contract/test_intensity_consistency.py -v
pytest tests/unit/intensity/test_spline_kernel.py -v
```
