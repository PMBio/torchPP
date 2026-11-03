# Getting started

This page goes from a fresh checkout to a trained and evaluated model, on a benchmark dataset or
on your own data. If you are new to point processes, we recommend reading the
[introduction](./introduction.md) first.

## Install

```bash
uv sync --all-extras        # or: pip install -e ".[all]"
uv sync --group dev         # pytest, for the tests
```

The [README](../README.md#install) has the details on CPU-only PyTorch, the extras and the
environment variables. Datasets are stored in `$TORCHPPDATA` (default `~/data/torchpp/`) and
checkpoints in `$TORCHPPMODELS` (default `models/` in the repository).

## Train and evaluate on a benchmark

**1. Download one dataset.** This needs the `data` extra.

```bash
python scripts/data_prep/download_benchmark_datasets.py --datasets taxi
```

**2. Run a smoke test.** It runs 10 steps (about 10 seconds on CPU) and checks that the data and
the code work.

```bash
python train.py -c default_fit -c model/const_exp_hawkes -c data/taxi -c training/smoke \
    eval.limit_test_size=100 eval.compute_marginal_type=false device=cpu
```

Ten steps only check that everything runs. The printed metrics are not meaningful yet.

**3. Start a real run.** A model preset sets the training defaults of its model family (learning
rate, schedule, and up to 40k steps with early stopping).

```bash
python train.py -c default_fit -c model/spline_spline_hawkes -c data/taxi \
    save.path=models/ssk_taxi.pth
```

`--print` shows the merged configuration without training. Every model type has a preset
`configs/model/<model_type>.yaml`, listed in the [model catalog](./models.md). The rules for
combining configurations are described in [Configs](./configs.md).

**4. Evaluate the checkpoint.** The checkpoint stores the model type, architecture and dataset,
so its path is enough.

```bash
python evaluate.py -c default_eval -c eval/full model_path=models/ssk_taxi.pth
```

The metrics are written next to the checkpoint (`models/ssk_taxi.eval.json`) and explained in
[Metrics](./metrics.md). In Python, `torchpp.load_model(path)` rebuilds the model.

## Your own data

Most event data starts as a table with one row per event. You turn each sequence into an
`MVEventData`, split the sequences, and fit a model. The example below reads a CSV file with
pandas (`pip install pandas`).

```python
import random
import pandas as pd
import torch
import torchpp as tpp
from torchpp.training import default_fit_config

df = pd.read_csv("events.csv")                 # columns: seq_id, time, type (0 .. D-1)
D = int(df["type"].max()) + 1
seqs = [
    tpp.MVEventData(
        torch.tensor(g["time"].to_numpy(), dtype=torch.float64),
        torch.tensor(g["type"].to_numpy()),
        T_obs=float(g["time"].max()),          # end of observation; see below
    )
    for _, g in df.sort_values(["seq_id", "time"]).groupby("seq_id")
]

random.Random(0).shuffle(seqs)
n = len(seqs)
train, val, test = seqs[: int(0.7 * n)], seqs[int(0.7 * n) : int(0.85 * n)], seqs[int(0.85 * n) :]

stats = tpp.DatasetStats.from_sequences(train, D=D)   # time unit and initial values
model = tpp.create_model("spline_spline_hawkes", D, dataset_stats=stats, seed=0)
result = tpp.fit(model, train, val, default_fit_config(num_steps=2000, learning_rate=0.05))  # GPU: model.to("cuda") first, or fit(..., device="cuda")
print(tpp.evaluate_tpp(result.model, test))
tpp.save_model("my_model.pth", result.model)
```

Without pandas, you can build the same lists from any `(seq_id, time, type)` rows. Group the
rows per sequence, sort them by time, and pass the times and types as tensors.

Three choices describe your data. You pass them to `create_model`, with `has_origin` given inside
`token_roles=TokenRoles(...)` and `terminating` also passed to `DatasetStats.from_sequences`.
Use the same values for every model you compare.

- **`T_obs`** is the time at which the observation of a sequence ended. If you know that nothing
  happened after the last event until some time, for example the end of follow-up, set `T_obs`
  to that time. The quiet tail then counts in the likelihood, and forecasts start after it. If
  you don't know, leave the default, which is the time of the last event.
- **`has_origin`** says whether time 0 means something, like admission or birth. In that case,
  set `token_roles=TokenRoles(has_origin=True)`. Every event is then predicted, including the
  wait for the first one. If time 0 is arbitrary (a calendar date or the start of a log file),
  keep the default `False`. The first event is then given, and prediction starts at the second.
- **`terminating`** says whether each event type can happen at most once per sequence, as with
  first diagnoses, a single failure or death. In that case, set `terminating=True`. With one
  type this is survival analysis, which is covered in
  [`examples/survival/`](../examples/survival/).

Times can be in any unit, and results are reported in that unit. Keep timestamps small by
shifting each sequence to start near 0, so that float32 data does not lose resolution. We
recommend learning rates of about 0.05–0.1 for Poisson and Hawkes models and 0.001–0.01 for
neural models. To use your dataset with `train.py`, register it as described in
[Data](./data.md#your-own-dataset).

## Tests

```bash
pytest tests/unit tests/contract -q            # fast
pytest tests/ -k "const_exp_hawkes and not inhibitive" -v   # one model across the matrix (-k matches substrings)
```

The model tests run over every registered model, so we recommend scoping them with `-k`. See
[CONTRIBUTING](../CONTRIBUTING.md) for more.
