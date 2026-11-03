# Data

This page covers the in-memory event representations of `torchpp` (`MVEventData`,
`BatchedMVEventData`), the batching strategies, and the dataset loaders under `torchpp/data/`.

## `MVEventData`

`MVEventData` holds a single multivariate event sequence. It inherits from the shared
`EventSequence` base, which is itself a
[`tensordict.TensorDict`](https://pytorch.org/tensordict/), so it serializes and moves between
devices like a normal tensor container.

```python
from torchpp import MVEventData
import torch

seq = MVEventData(
    time_points=torch.tensor([0.10, 0.34, 0.57, 0.91]),  # shape (L,), float
    event_types=torch.tensor([0, 1, 0, 2]),               # shape (L,), int
    covariates={"sex": torch.tensor([1.0]), "age": torch.tensor([62.0])},  # optional, 1-D each
    sort=False,    # set True to sort events by time
    T_obs=1.0,     # optional observation window end; defaults to last event time
)
```

`__init__` runs the following checks and raises a `ValueError` or `TypeError` that names the
problem:

- `time_points` and `event_types` must be 1-D and of equal length. Lists and NumPy arrays are
  converted.
- Times must be non-decreasing. Unsorted input raises an error unless `sort=True`, in which case
  it is sorted.
- `event_types` must be non-negative integers.
- `T_obs`, if given, must not lie before the last event. An earlier value is raised to the last
  event time with a warning. A `T_obs` after the last event means that the sequence was observed
  without events up to `T_obs`, and predictions take this into account
  ([Notation → Forecasting](./notation.md#forecasting-from-t_obs)).
- Covariates are a `dict[str, 1-D Tensor]` of static per-sequence features.

Defined in [`torchpp/utils/event_data.py`](../torchpp/utils/event_data.py).

## `BatchedMVEventData`

`BatchedMVEventData` is a right-padded batch of `MVEventData`. It is the input type accepted by
every TPP model.

```python
from torchpp import BatchedMVEventData

batch = BatchedMVEventData.from_sequences([seq1, seq2, seq3])
seqs = batch.to_sequences()   # back to unpadded single sequences (T_obs and covariates kept)
# or, equivalently, from raw tensors:
batch = BatchedMVEventData(
    time_points=[seq1.time_points, seq2.time_points, seq3.time_points],
    event_types=[seq1.event_types, seq2.event_types, seq3.event_types],
)
```

### Padding conventions

| Field | Padding value |
|---|---|
| `event_types` | `-1` (`BatchedMVEventData.EVENT_PADDING_VALUE`) |
| `time_points` | `max(last_event_time across batch) + PADDING_TIME_OFFSET` (`= 1.0`) |

Padded times are guaranteed to sort *after* every real event in the batch, so masks built on
`time_points < t` behave correctly. Padding exists only inside a batch, and a single
`MVEventData` rejects type `-1`. Use `event_mask` or `real_counts` (below) to tell real events
from padding, and use `last_event_time` or `T_obs` instead of reading padded times.

### Properties / helpers

| Property | Shape | What it returns |
|---|---|---|
| `time_points` | `(B, L)` | padded event times |
| `event_types` | `(B, L)` | padded event types (`-1` for padding) |
| `event_mask` | `(B, L)` | `True` where the entry is a real event |
| `event_types_idx_safe` | `(B, L)` | `event_types` with `-1 → 0`, safe for `gather` / `one_hot` |
| `real_counts` | `(B,)` | number of real events per sequence |
| `seq_lengths` | `list[int]` | original (unpadded) lengths |
| `last_event_time` | `(B,)` | last real-event time per sequence (0.0 if empty) |
| `T_obs` | `(B,)` | per-sequence observation window end |
| `covariates` | `dict[str, (B, dim)]` | stacked per-sequence covariates |

| Method | Returns | Use case |
|---|---|---|
| `time_points_before(t)` | `(B, L)` bool | mask of real events with `t_i < t` |
| `count_events_before(t)` | `(B,)` long | how many real events are before `t` |
| `event_time_before(t)` | `(B,)` | last real event time strictly before `t`, or `0.0` |
| `causal_indices` | `(B, L)` long | per-position index into a causal parameter cache, where simultaneous events share a column |
| `one_hot_event_types(D)` | `(B, L, D)` bool | one-hot with padding rows zeroed |
| `causal_mask()` | `(B, L, L)` bool | strict causal mask (`time[q] > time[k]`) |
| `attention_mask()` | `(B, L, L)` bool | inclusive causal mask (`time[q] >= time[k]`), used by the transformer encoder |

Neural TPPs use `causal_indices` to look up cached inter-event-interval parameters.
Simultaneous events deliberately share an index, so that they all see the same "before"
parameters.

## Batching strategies

The module [`torchpp/utils/batching.py`](../torchpp/utils/batching.py) provides the following
batching tools.

- `BucketBatchSampler` groups sequences of similar length to minimize padding. This is useful
  when sequence lengths vary a lot.
- Dynamic batching caps each batch by its total event count (`target_total_events`) instead of
  its number of sequences. It is configured with `BatchingConfig.dynamic_batch_size=True`.
- Truncation is applied to over-long sequences when `max_sequence_length` is set. Each strategy
  keeps a contiguous stretch of events with a valid observation window.
  - `right` keeps the first events, and the window ends at the last kept event.
  - `left` keeps the last events and the original `T_obs`.
  - `random` keeps a stretch at a random position, which is re-drawn each epoch. Its window
    ends at its last kept event unless the stretch reaches the end of the sequence.

  Without an origin, the first kept event is the conditioned-on start, so a stretch from the
  middle of a sequence is scored correctly. With an origin (`has_origin=True`), `left` and
  `random` drop the start of the time axis and print a warning. Use `right` in that case.
- `collate_batch_with_truncation(...)` is a drop-in `collate_fn` for
  `torch.utils.data.DataLoader` that produces a `BatchedMVEventData`.
- `analyze_length_distribution(...)` and `print_length_analysis(...)` give a quick diagnostic of
  the sequence lengths in a dataset.

The high-level configuration options live in `BatchingConfig` in
[`torchpp/training/config.py`](../torchpp/training/config.py).

## Datasets

All dataset classes implement `TPPDataset`
([`torchpp/data/base_dataset.py`](../torchpp/data/base_dataset.py)), a
`torch.utils.data.Dataset` whose `__getitem__` returns an `MVEventData`. The splits are stored
under `$TORCHPPDATA` (default `~/data/torchpp/`).

### Benchmark datasets

[`BenchmarkDataset`](../torchpp/data/benchmark_data.py) (`dataset_type: benchmark`) reads the
standard TPP benchmarks, which are recurrent and non-terminating. Each benchmark has a preset
`-c data/<name>`.

| Name | D | Download |
|---|---|---|
| `taxi` | 10 | Hugging Face |
| `stackoverflow` | 22 | Hugging Face |
| `retweet` | 3 | Hugging Face |
| `amazon` | 16 | Hugging Face |
| `earthquake` | 7 | Hugging Face |
| `volcano` | 1 | Hugging Face |
| `taobao` | 17 | Hugging Face (json files, see below) |

We use the preprocessed splits distributed by the
[EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess) project on the
[`easytpp` Hugging Face organization](https://huggingface.co/easytpp) and in its
[Google Drive folder](https://drive.google.com/drive/u/0/folders/1f8k82-NL6KFKuNMsUwozmbzDSFycYvz7).
TorchPP downloads and converts them. The datasets themselves come from their original
publications.

```bash
pip install "datasets>=4.8"   # or the `data` extra
python scripts/data_prep/download_benchmark_datasets.py                   # all seven datasets
python scripts/data_prep/download_benchmark_datasets.py --datasets taxi   # just one
```

For Taobao, the Hugging Face `datasets` loader only shows a preview. The script therefore
downloads `train.json`, `dev.json` and `test.json` directly and stores the times in float64.
Do not use the Taobao versions in the Google Drive folder. In `tb_v8`, the train/dev splits and
the test split come from two different preprocessings. The training gaps are on average about
17 times longer than the test gaps, and the two time fields disagree for 8% of the training
events. Stored in float32, the absolute times of `tb_v8` also fall on a 1/64 grid. The Hugging Face
release equals `tb_v7` with every sequence shifted to start at zero, and its test split is the
same as in `tb_v8`. You can convert local json files with
`--json_dir /path/to/json --name taobao --num_types 17`.

The splits are stored as `$TORCHPPDATA/benchmarks/<name>/{train,dev,test}.pt`, where `dev` is
the validation split. Data downloaded with an earlier version under `$TORCHPPDATA/easytpp/` can
be moved there. `num_event_types_for(name)` returns `D`. The `.pt` and `.pkl` files are
pickles, so only load them from sources you trust.

### Your own dataset

There are two ways to use your own dataset.

1. From Python, build a list of `MVEventData` for each split and pass it to
   `torchpp.fit(model, train_seqs, val_seqs, config)` and `evaluate_tpp(model, test_seqs)`.
   No registration is needed.
2. For `train.py`, subclass `TPPDataset` and register it.

   ```python
   from torchpp.data import TPPDataset
   from torchpp.training.dataset_registry import register_dataset

   @register_dataset("my_data")
   class MyDataset(TPPDataset):
       terminating = False              # each type at most once?
       cfg_keys = frozenset({"path"})   # accepted dataset.* keys; others raise

       def __init__(self, seqs, D):
           self.seqs, self._D = seqs, D

       @property
       def num_event_types(self):
           return self._D

       def __len__(self):
           return len(self.seqs)

       def __getitem__(self, i):
           return self.seqs[i]

       @classmethod
       def load(cls, cfg, root, split):  # split is "train", "val" or "test"
           seqs = ...                     # read root / cfg["path"] / split, raw native-unit times
           return cls(seqs, D=...)
   ```

   `root` is `$TORCHPPDATA`. `cfg_keys` lists the dataset's own `dataset.*` keys, and an empty
   `cfg_keys` accepts any key. The keys `context_types`, `ignored_types` and
   `limit_{train,val,test}_size` are always accepted. For the limit keys, the registry keeps the
   first `n` sequences of a split unless the loader declares the limit key itself. The optional
   class attribute `default_token_roles` sets the origin regime, context types and ignored types
   ([Notation → Token roles](./notation.md#token-roles)). With `dataset_module=<module>`,
   `train.py` and `evaluate.py` import the module (from the working directory or the Python
   path) before the dataset is resolved. A preset then selects the dataset.

   ```yaml
   # configs/data/my_data.yaml
   dataset_module: my_data     # my_data.py in the working directory
   dataset_type: my_data
   dataset:
     path: my_data
   ```

   `python scripts/data_prep/collect_dataset_stats.py` prints the `DatasetStats` of the
   registered datasets.

## Time units

Datasets emit raw event times in their native units, and there is no dataset-level scaling. The
model owns the time normalization ([Notation → Time units](./notation.md#time-units)).
`DatasetStats.sequence_time_p95` gives a quick sense of the time scale of a dataset.

## `DatasetStats`

`DatasetStats` is a small `dataclass` that is computed from the training set and used by
`create_model(...)` to seed data-driven defaults. These are
`arch.baseline.T_factor × sequence_time_p95` for the spline anchors, a kernel decay calibrated
from inter-event time percentiles, and an $\alpha$ initialized from pair-occurrence counts.
`DatasetStats` is defined at the top of
[`torchpp/utils/event_data.py`](../torchpp/utils/event_data.py).
