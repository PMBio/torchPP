"""IntensityFree inter-event standardisation is wired from data (Fix 1).

``arch.mean/std_log_inter_time`` default to (0.0, 1.0), which makes the log-normal
mixture skip its affine transform — i.e. IFTPP runs un-standardised.  The factory
now passes ``DatasetStats.mean/std_log_inter_time`` (mean/std of log positive
inter-event gaps on the scaled gaps the model sees) when available, so the affine
transform is active and calibrated to the data.
"""

from __future__ import annotations

import torch

from torchpp.training.config import ModelArchConfig
from torchpp.training.model_factory import create_model
from torchpp.utils.event_data import DatasetStats, MVEventData
from torchpp.modules.nn.lognormal_mixture import LogNormalMixtureDistribution


def _stats_from_gaps(gap_pattern, n_seq=8, D=3, terminating=False) -> DatasetStats:
    """DatasetStats from sequences whose consecutive gaps follow ``gap_pattern``."""
    gaps = torch.tensor(gap_pattern, dtype=torch.float64)
    seqs = []
    for _ in range(n_seq):
        t = torch.cumsum(gaps, dim=0)
        types = torch.arange(len(gaps)) % D
        seqs.append(MVEventData(t, types.long()))
    return DatasetStats.from_sequences(seqs, D=D, terminating=terminating)


def test_standardize_off_by_default_ignores_stats():
    """DEFAULT (iftpp_standardize=False): even with dataset_stats present the model runs
    un-standardised (0,1) — the pipeline pre-scales times, so standardisation is redundant
    and empirically harmful (retweet). This is the shipped default."""
    stats = _stats_from_gaps([15.0, 20.0, 25.0, 30.0])
    arch = ModelArchConfig()  # iftpp_standardize defaults False
    model, _ = create_model("intensity_free", D=3, arch=arch, terminating=False, seed=0, dataset_stats=stats)
    assert (model.mean_log_inter_time, model.std_log_inter_time) == (0.0, 1.0)
    params = torch.zeros(1, 3 * model.K + model.D)
    dist, _ = model._unpack(params, torch.float32)
    assert len(dist.transforms) == 1  # affine skipped (un-standardised)


def test_far_from_unit_gaps_yield_nonunit_stats_and_active_affine():
    """OPT-IN (iftpp_standardize=True): model receives non-(0,1) mean/std from data and the
    log-normal mixture applies its affine transform (2 transforms: affine + exp)."""
    # gaps ~ e^3 => log-gaps mean ≈ 3, well away from 0.
    stats = _stats_from_gaps([15.0, 20.0, 25.0, 30.0])
    assert abs(stats.mean_log_inter_time) > 1.0
    assert stats.std_log_inter_time > 0.0

    arch = ModelArchConfig(iftpp_standardize=True)  # opt in (default is off)
    assert arch.mean_log_inter_time == 0.0 and arch.std_log_inter_time == 1.0

    model, _ = create_model("intensity_free", D=3, arch=arch, terminating=False, seed=0, dataset_stats=stats)

    # The model received the data-driven stats, not the arch defaults.
    assert model.mean_log_inter_time == stats.mean_log_inter_time
    assert model.std_log_inter_time == stats.std_log_inter_time
    assert (model.mean_log_inter_time, model.std_log_inter_time) != (0.0, 1.0)

    # The mixture the model builds applies the affine transform.
    params = torch.zeros(1, 3 * model.K + model.D)
    dist, _ = model._unpack(params, torch.float32)
    assert isinstance(dist, LogNormalMixtureDistribution)
    assert len(dist.transforms) == 2  # AffineTransform + ExpTransform (affine active)


def test_no_dataset_stats_falls_back_to_arch_defaults():
    """Without dataset_stats the model uses the arch defaults (0.0, 1.0) => no affine."""
    arch = ModelArchConfig()
    model, _ = create_model("intensity_free", D=3, arch=arch, terminating=False, seed=0, dataset_stats=None)
    assert (model.mean_log_inter_time, model.std_log_inter_time) == (0.0, 1.0)
    params = torch.zeros(1, 3 * model.K + model.D)
    dist, _ = model._unpack(params, torch.float32)
    assert len(dist.transforms) == 1  # ExpTransform only (affine skipped)


def test_easytpp_scale_near_unit_stats_still_load_and_run():
    """EasyTPP-style already-near-(0,1) scaled data still builds a working model."""
    # log-gaps drawn near N(0, 1): gaps = exp(z), z ~ standard normal.
    torch.manual_seed(0)
    z = torch.randn(200, dtype=torch.float64)
    gaps = z.exp().clamp(min=1e-3)
    stats = _stats_from_gaps(gaps.tolist(), n_seq=4)
    # Near (0, 1) — mean small, std near 1.
    assert abs(stats.mean_log_inter_time) < 0.5
    assert 0.5 < stats.std_log_inter_time < 1.5

    arch = ModelArchConfig()
    model, _ = create_model("intensity_free", D=3, arch=arch, terminating=False, seed=0, dataset_stats=stats)

    # A forward likelihood pass runs and is finite.
    from torchpp.utils.event_data import BatchedMVEventData

    t = torch.tensor([0.2, 0.9, 1.5, 2.4], dtype=torch.float64)
    ty = torch.tensor([0, 1, 2, 0])
    batch = BatchedMVEventData([t], [ty])
    ll = model.likelihood(batch)
    assert torch.isfinite(ll).all()


def test_stale_dataset_stats_cache_loads_with_defaults():
    """A cache written before the new fields existed (missing keys) still loads,
    defaulting the new fields; unknown extra keys are dropped."""
    from torchpp.training.training_helpers import _dataset_stats_from_state, _dataset_stats_to_state

    stats = _stats_from_gaps([15.0, 20.0, 25.0, 30.0])
    state = _dataset_stats_to_state(stats)

    # Simulate a stale cache: drop the new keys and add a since-removed key.
    stale = {k: v for k, v in state.items() if k not in ("mean_log_inter_time", "std_log_inter_time")}
    stale["some_removed_field"] = 123

    restored = _dataset_stats_from_state(stale)
    assert restored.mean_log_inter_time == 0.0  # defaulted
    assert restored.std_log_inter_time == 1.0  # defaulted
    assert restored.D == stats.D  # other fields intact

    # A fresh (complete) cache round-trips exactly.
    restored2 = _dataset_stats_from_state(_dataset_stats_to_state(stats))
    assert restored2.mean_log_inter_time == stats.mean_log_inter_time
    assert restored2.std_log_inter_time == stats.std_log_inter_time
