# Data

This page covers `torchpp`'s in-memory event representations
(`MVEventData`, `BatchedMVEventData`), batching strategies, and the
dataset loaders shipped under `torchpp/data/`.

## `MVEventData`

A single multivariate event sequence. Inherits from
[`tensordict.TensorDict`](https://pytorch.org/tensordict/) so it serialises
and moves between devices like a normal tensor container.

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

Constraints checked in `__init__`:

- `time_points` and `event_types` are 1-D, equal length.
- `time_points` is floating point; `event_types` is integer.
- `T_obs`, if provided, is silently clamped to `>= time_points[-1]`.
- Covariates are a `dict[str, 1-D Tensor]`. Each value is a per-sequence
  static feature vector — broadcasted over events in covariate-aware
  models.

Defined in [`torchpp/utils/event_data.py`](../torchpp/utils/event_data.py).

## `BatchedMVEventData`

A right-padded batch of `MVEventData`. This is the input type accepted by
every TPP model.

```python
from torchpp import BatchedMVEventData

batch = BatchedMVEventData(mv_events=[seq1, seq2, seq3])
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
| `time_points` | `max(last_event_time across batch) + MAX_TIME_OFFSET` (`= 1.0`) |

Padded times are guaranteed to sort *after* every real event in the batch,
so masks built on `time_points < t` behave correctly. Always derive masks
from one of the helper properties below — never rely on the raw padded
values.

### Properties / helpers

| Property | Shape | What it returns |
|---|---|---|
| `time_points` | `(B, L)` | padded event times |
| `event_types` | `(B, L)` | padded event types (`-1` for padding) |
| `event_mask` | `(B, L)` | `True` where the entry is a real event |
| `event_types_idx_safe` | `(B, L)` | `event_types` with `-1 → 0`, safe for `gather` / `one_hot` |
| `real_counts` | `(B,)` | number of real events per sequence |
| `seq_lengths` | `list[int]` | original (unpadded) lengths |
| `max_time` | `(B,)` | last real-event time per sequence (0.0 if empty) |
| `T_obs` | `(B,)` | per-sequence observation window end |
| `covariates` | `dict[str, (B, dim)]` | stacked per-sequence covariates |

| Method | Returns | Use case |
|---|---|---|
| `time_points_before(t)` | `(B, L)` bool | mask of real events with `t_i < t` |
| `count_events_before(t)` | `(B,)` long | how many real events are before `t` |
| `event_time_before(t)` | `(B,)` | last real event time strictly before `t`, or `0.0` |
| `causal_indices` | `(B, L)` long | per-position index into a causal parameter cache; simultaneous events share a column |
| `one_hot_event_types(D)` | `(B, L, D)` bool | one-hot with padding rows zeroed |
| `causal_mask()` | `(B, L, L)` bool | strict causal mask (`time[q] > time[k]`) |
| `attention_mask()` | `(B, L, L)` bool | inclusive causal mask (`time[q] >= time[k]`); used by the transformer encoder |

`causal_indices` is the hook neural TPPs use to look up cached
inter-event-interval parameters. Simultaneous events deliberately share an
index so that they all see the same "before" parameters.

## Batching strategies

[`torchpp/utils/batching.py`](../torchpp/utils/batching.py) provides:

- **`BucketBatchSampler`** — groups sequences of similar length to
  minimise padding. Used for training on UKB / MIMIC-IV where length
  variance is high.
- **Dynamic batching** — caps each batch by total event count
  (`target_total_events`) instead of sequence count. Configured via
  `BatchingConfig.dynamic_batch_size=True`.
- **`TruncationStrategy`** — `right` / `left` / `random` / `stratified`.
  Applied to over-long sequences when `max_sequence_length` is set.
- **`collate_batch_with_truncation(...)`** — drop-in `collate_fn` for
  `torch.utils.data.DataLoader` that produces a `BatchedMVEventData`.
- **`analyze_length_distribution(...)`** /
  **`print_length_analysis(...)`** — quick diagnostic of sequence
  lengths in a dataset.

The high-level configuration knobs live in `BatchingConfig` in
[`torchpp/training/config.py`](../torchpp/training/config.py).

## Datasets

All dataset classes implement `TPPDataset`
([`torchpp/data/base_dataset.py`](../torchpp/data/base_dataset.py)), a thin
`torch.utils.data.Dataset` whose `__getitem__` returns an `MVEventData`.
Splits live under `$TORCHPPDATA` (default `~/data/torchpp/`); see the
[README](../README.md) for the prep scripts that produce them.

| Dataset | Loader | Notes |
|---|---|---|
| UK Biobank | [`UKBDataset`](../torchpp/data/ukb_data.py) (`load_ukb_sequences`) | Lazy `torch.utils.data.Dataset` over `.bin` files. First-occurrence disease data → terminating mode. Time scaled to `t_days / 365.25 / 80`, so `1.0 ≈ 80 years`. Use `UKB_TIME_SCALE_FACTOR` to convert back. |
| EasyTPP × 5 | [`EasyTPPDataset`](../torchpp/data/easytpp_data.py) (`load_easytpp_sequences`) | Taxi / StackOverflow / Retweet / Amazon / Taobao. Recurrent (non-terminating). |
| MIMIC-IV ICU | [`MIMICIVDataset`](../torchpp/data/mimic_iv_data.py) (`load_mimic_iv_sequences`) | Top-K `itemid` selection via `top_k_event_types`. Recurrent. |

`num_event_types_for(...)` in `easytpp_data.py` looks up `D` for each named
EasyTPP variant so YAML configs can stay dataset-agnostic.

## Time scaling

UKB events are stored as days since birth and scaled to
$[0, \approx 1]$ before being passed to models. The factor is

```python
UKB_TIME_SCALE_FACTOR = 1.0 / (365.25 * 80)   # 1 unit ≈ 80 years
```

`evaluate.py` and the metrics in
[`torchpp/utils/evaluation.py`](../torchpp/utils/evaluation.py) use
`TIME_SCALE_FACTOR` to convert predicted times back to days for reporting
human-readable error metrics. EasyTPP and MIMIC-IV use their own scaling
defined in their respective loaders; check `DatasetStats.sequence_time_p95`
for a quick sense of the working time scale.

## `DatasetStats`

A small `dataclass` produced from the training set and consumed by
`create_model(...)` to seed data-driven defaults
(`spline_T_factor × sequence_time_p95` for spline anchors, kernel decay
calibrated from inter-event time percentiles, $\alpha$ initialised from
pair-occurrence counts). Defined at the top of
[`torchpp/utils/event_data.py`](../torchpp/utils/event_data.py).
