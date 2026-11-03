"""Time-unit wiring guards for the model factory.

The dataset emits raw native-unit times and the *model* owns time
normalisation: ``model_factory.create_model`` stamps ``time_scale`` from the training
mean inter-event time and initialises data-driven modules from *scaled* statistics.
These are synthetic (no dataset files needed) guards for that wiring.
"""

import torch

from torchpp.training.model_factory import create_model
from torchpp.utils.event_data import DatasetStats, MVEventData


def _synth_stats(D: int = 3, terminating: bool = False) -> DatasetStats:
    # A handful of sequences with an obviously non-unit mean inter-event time.
    seqs = [
        MVEventData(torch.tensor([2.0, 5.0, 9.0, 14.0]), torch.tensor([0, 1, 2, 0])),
        MVEventData(torch.tensor([3.0, 8.0, 12.0]), torch.tensor([1, 0, 2])),
    ]
    return DatasetStats.from_sequences(seqs, D, terminating=terminating)


def test_rescale_time_units():
    """rescale_time scales time-like fields, inverse-scales rates, leaves the rest."""
    s = _synth_stats()
    f = 0.25
    r = s.rescale_time(f)
    assert torch.allclose(r.mean_first_occ_time, s.mean_first_occ_time * f)
    assert torch.allclose(r.mean_rate, s.mean_rate / f)
    assert r.mean_inter_event_time == s.mean_inter_event_time * f
    assert r.sequence_time_p95 == s.sequence_time_p95 * f
    assert r.inter_event_time_p95 == s.inter_event_time_p95 * f
    # dimensionless / structural fields unchanged
    assert torch.allclose(r.type_distribution, s.type_distribution)
    assert r.mean_events_per_sequence == s.mean_events_per_sequence
    assert r.D == s.D


def test_create_model_stamps_time_scale_from_stats():
    """create_model sets time_scale = training mean inter-event time (raw units)."""
    stats = _synth_stats()
    arch = None
    for mt in ["const_exp_hawkes", "neural_log_linear", "nhp", "intensity_free"]:
        model = create_model(mt, D=3, arch=arch, terminating=False, seed=0, dataset_stats=stats)
        assert abs(float(model.time_scale) - stats.mean_inter_event_time) < 1e-6, mt


def test_create_model_defaults_to_unit_scale_without_stats():
    """No dataset_stats → time_scale stays 1.0 (the no-op path)."""
    model = create_model("const_exp_hawkes", D=3, terminating=False, seed=0, dataset_stats=None)
    assert float(model.time_scale) == 1.0
