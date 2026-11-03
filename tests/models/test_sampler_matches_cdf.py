"""A model's sampler must agree with its own CDF.

A sampler that forgets the scaled-core/native-shell conversion draws gaps in scaled time and
adds them to native timestamps, inflating every inter-event time by about 1/time_scale. Two
things hide such a bug, so keep them in mind when adding sampling code:

1. **It is inert at `time_scale == 1`**, the default for a model built without dataset stats.
2. `test_time_scale_equivalence` pins that sampling *scales covariantly* (s vs 2s on 2x data), which
   stays true when the sampler is uniformly wrong at every scale. Covariance is not correctness.

The invariant here is self-contained: draw from the model, push the draws through the model's own
`cdf_time`, and the result must be Uniform(0,1). It needs no ground-truth data and no reference
implementation, so it holds for trained and untrained models alike.
"""

import numpy as np
import pytest
import torch

from torchpp.training.model_factory import create_model
from torchpp.utils.event_data import BatchedMVEventData, DatasetStats, MVEventData

D = 4
N_CTX = 48
# The regime that matters: a dataset-derived scale far from 1, as model_factory assigns from the data.
TIME_SCALES = [1.0, 0.0354985486453492]


def _histories(n, mean_gap, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        gaps = mean_gap * (0.5 + torch.rand(10, generator=g))
        out.append(MVEventData(torch.cumsum(gaps, 0).double(),
                               torch.randint(D, (10,), generator=g)))
    return out


def _build(model_type, arch, mean_gap):
    stats = DatasetStats.from_sequences(_histories(40, mean_gap, seed=7), D, terminating=False)
    model = create_model(model_type, D, arch=arch, terminating=False, seed=0, dataset_stats=stats)
    return model.eval()


@pytest.mark.parametrize("time_scale", TIME_SCALES)
@pytest.mark.parametrize("model_type,arch", [
    ("intensity_free", {"encoder": {"type": "gru", "n_layers": 1, "n_embd": 16}, "head": {"num_mix_components": 8}}),
    ("poisson", None),
    ("nhp", {"encoder": {"hidden_dim": 16}}),
])
def test_sampled_events_are_uniform_under_the_models_own_cdf(model_type, arch, time_scale):
    model = _build(model_type, arch, mean_gap=time_scale)
    ctxs = _histories(N_CTX, time_scale, seed=3)
    batch = BatchedMVEventData([c.time_points for c in ctxs], [c.event_types for c in ctxs])

    with torch.no_grad():
        drawn = torch.stack([model.sample(c, num_steps=1, rng=1000 + i)[0].time_points[-1]
                             for i, c in enumerate(ctxs)]).float()
        pit = model.cdf_time(drawn, batch, log=False).double().numpy()

    pit = np.clip(pit, 0.0, 1.0)
    assert np.isfinite(pit).all(), f"{model_type}: non-finite PIT values"

    # Kolmogorov-Smirnov against Uniform(0,1).  The 1% critical value is ~1.63/sqrt(n); the bound
    # below is deliberately loose so this fails only on a real defect, never on sampling noise.
    # (The defect it was written for produced D ~ 0.5 with 73% of the mass above 0.9.)
    ks = np.max(np.abs(np.sort(pit) - (np.arange(1, len(pit) + 1) - 0.5) / len(pit)))
    assert ks < 0.30, (
        f"{model_type} @ time_scale={time_scale}: sampler disagrees with its own CDF "
        f"(KS={ks:.3f}, PIT mean={pit.mean():.3f}, median={np.median(pit):.3f}). "
        "A sampler drawing in scaled time must de-scale its output — see "
        "IntensityBasedTPP.sample / DensityBasedTPP.sample."
    )
