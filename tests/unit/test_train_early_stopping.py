"""train.py validation aggregation + early-stopping NaN guards (Fix 2).

A single NaN val batch used to poison the summed val LL (NaN), which the
early-stopping rule then counted as "no improvement", burning patience and
spuriously early-stopping (this truncated NHP-MIMIC).  The fixes:

- ``_aggregate_val_ll`` excludes non-finite per-sequence contributions (skip +
  count), so one bad batch cannot NaN the whole number; it stays a plain
  per-event mean when every contribution is finite.
- ``_early_stopping_step`` treats a non-finite val LL as an uninformative eval:
  it neither updates the best nor increments the no-improvement counter.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# train.py lives at the repo root (not a package); make it importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import train  # noqa: E402


class _StubBatch:
    """Minimal stand-in exposing the two attributes _aggregate_val_ll touches."""

    def __init__(self, ll, counts):
        self._ll = torch.as_tensor(ll, dtype=torch.float64)
        self._counts = torch.as_tensor(counts, dtype=torch.long)

    def to(self, _device):
        return self

    @property
    def real_counts(self):
        return self._counts


class _StubModel:
    def likelihood(self, batch):
        return batch._ll


# ---------------------------------------------------------------------------
# _aggregate_val_ll
# ---------------------------------------------------------------------------


def test_aggregate_skips_nonfinite_and_counts_them():
    """A NaN in one sequence is excluded from both the LL sum and the event count."""
    b_bad = _StubBatch([-1.0, float("nan"), -2.0], [10, 5, 20])
    b_ok = _StubBatch([-3.0, -4.0], [30, 40])
    val, n_nonfinite = train._aggregate_val_ll(_StubModel(), [b_bad, b_ok], device="cpu")
    expected = (-1.0 - 2.0 - 3.0 - 4.0) / (10 + 20 + 30 + 40)
    assert n_nonfinite == 1
    assert math.isfinite(val)
    assert abs(val - expected) < 1e-12


def test_aggregate_all_finite_is_plain_per_event_mean():
    """With all-finite input the aggregate is exactly the plain per-event mean
    (number-preserving on healthy inputs)."""
    b1 = _StubBatch([-3.0, -4.0], [30, 40])
    b2 = _StubBatch([-1.0, -2.0], [10, 20])
    val, n_nonfinite = train._aggregate_val_ll(_StubModel(), [b1, b2], device="cpu")
    plain = (-3.0 - 4.0 - 1.0 - 2.0) / (30 + 40 + 10 + 20)
    assert n_nonfinite == 0
    assert abs(val - plain) < 1e-15


def test_aggregate_all_nonfinite_returns_nan():
    """When every contribution is non-finite the eval is fully uninformative (NaN)."""
    b = _StubBatch([float("nan"), float("inf")], [10, 20])
    val, n_nonfinite = train._aggregate_val_ll(_StubModel(), [b], device="cpu")
    assert n_nonfinite == 2
    assert math.isnan(val)


# ---------------------------------------------------------------------------
# _early_stopping_step
# ---------------------------------------------------------------------------


def test_nonfinite_val_does_not_burn_patience_or_corrupt_best():
    """A NaN val LL leaves best-val untouched and does NOT increment the counter."""
    best, improved, counter, triggered = train._early_stopping_step(
        current_val_ll=float("nan"),
        best_val_ll=-4.2,
        early_stop_counter=2,
        min_delta=0.0,
        patience=3,
    )
    assert best == -4.2  # best not corrupted
    assert improved is False
    assert counter == 2  # patience NOT burned
    assert triggered is False


def test_finite_improvement_updates_best_and_resets_counter():
    best, improved, counter, triggered = train._early_stopping_step(
        current_val_ll=-3.0,
        best_val_ll=-4.0,
        early_stop_counter=2,
        min_delta=0.0,
        patience=3,
    )
    assert best == -3.0
    assert improved is True
    assert counter == 0
    assert triggered is False


def test_genuine_plateau_still_early_stops():
    """A finite non-improving val still burns patience and triggers at the limit."""
    counter = 0
    best = -3.0
    triggered = False
    for _ in range(3):
        best, improved, counter, triggered = train._early_stopping_step(
            current_val_ll=-3.5,  # worse than best -> no improvement
            best_val_ll=best,
            early_stop_counter=counter,
            min_delta=0.0,
            patience=3,
        )
        assert improved is False
    assert counter == 3
    assert triggered is True


def test_transient_nan_between_finite_evals_preserves_progress():
    """A NaN eval in the middle of a plateau does not advance the counter, so a
    later finite eval still needs the full remaining patience."""
    best = -3.0
    counter = 0
    # one finite non-improving eval
    best, _, counter, triggered = train._early_stopping_step(-3.5, best, counter, 0.0, 2)
    assert counter == 1 and not triggered
    # transient NaN: counter frozen
    best, _, counter, triggered = train._early_stopping_step(float("nan"), best, counter, 0.0, 2)
    assert counter == 1 and not triggered
    # another finite non-improving eval -> now hits patience
    best, _, counter, triggered = train._early_stopping_step(-3.5, best, counter, 0.0, 2)
    assert counter == 2 and triggered
