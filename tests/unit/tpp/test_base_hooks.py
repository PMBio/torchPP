"""Base-class hook mechanics: the ``__init_subclass__`` intensity-
hook guard and the ``use_ci`` config context.

The guard rejects a concrete model that overrides neither the leaf nor the flat
intensity hook, while overriding *either* alone is accepted (the other falls back to
it) — these take no real model. The ``use_ci`` test pins the CI-config resolution
order: the context overrides the instance default, an explicit call-arg still wins
over the context, and the instance default is restored on exit.
"""

from __future__ import annotations

import pytest
import torch

from torchpp.tpp.base_tpp import CIConfig, IntensityBasedTPP
from torchpp.utils.event_data import BatchedMVEventData

from helpers import create_model, DEVICE


def _batch(device="cpu"):
    # No simultaneous events; distinct interval interiors per sequence.
    return BatchedMVEventData(
        [torch.tensor([0.5, 1.2, 2.0, 3.1], dtype=torch.float64), torch.tensor([0.3, 0.9, 1.5], dtype=torch.float64)],
        [torch.tensor([0, 2, 1, 0]), torch.tensor([1, 1, 2])],
    ).to(device)


def _model(model_type):
    """Model (native float32 heads, float64 batch — the standard setup) + T.

    Neural heads manage their own float32/float64 casting internally, so we do not
    ``.double()`` the model; the float64 batch keeps the time arithmetic precise.
    """
    m = create_model(model_type, terminating=False)
    m.eval()
    ts = _batch(DEVICE)
    return m, ts, ts.max_time + 0.5


# ---------------------------------------------------------------------------
# __init_subclass__ guard
# ---------------------------------------------------------------------------


def test_guard_rejects_model_overriding_neither_hook():
    with pytest.raises(TypeError, match="must implement _intensity_at_t or _intensities_at_queries"):

        class _Bad(IntensityBasedTPP):  # noqa: D401 — concrete but defines no intensity hook
            def transform_params(self):
                return ()


def test_guard_accepts_overriding_only_flat():
    # Overriding the flat hook alone is enough; the leaf falls back to it.
    class _OnlyFlat(IntensityBasedTPP):
        def transform_params(self):
            return ()

        def _intensities_at_queries(self, query_times, ts):
            B, Q = query_times.shape
            return torch.zeros(B, Q, self.D, dtype=query_times.dtype)

    m = _OnlyFlat(D=2, seed=0)
    ts = _batch()
    leaf = m._intensity_at_t(torch.tensor([1.0, 1.0], dtype=torch.float64), ts)  # via base default
    assert leaf.shape == (2, 2) and torch.equal(leaf, torch.zeros(2, 2, dtype=torch.float64))


def test_guard_accepts_overriding_only_leaf():
    class _OnlyLeaf(IntensityBasedTPP):
        def transform_params(self):
            return ()

        def _intensity_at_t(self, t, ts):
            return torch.ones(t.shape[0], self.D, dtype=t.dtype)

    m = _OnlyLeaf(D=2, seed=0)
    ts = _batch()
    flat = m._intensities_at_queries(torch.zeros(2, 3, dtype=torch.float64), ts)  # via base default loop
    assert flat.shape == (2, 3, 2) and torch.equal(flat, torch.ones(2, 3, 2, dtype=torch.float64))


# ---------------------------------------------------------------------------
# use_ci context + call-arg resolution order
# ---------------------------------------------------------------------------


def test_use_ci_context_and_resolution_order():
    m, ts, T = _model("nhp")
    default = m.ci_config
    assert default.placement == "interval"
    glob = CIConfig("numerical", placement="global", rule="trapezoidal", n_points=32)
    with m.use_ci(glob):
        assert m.ci_config == glob  # context overrides instance default
        # an explicit call-arg still wins over the context
        with m.fix_context(ts), torch.no_grad():
            a = m.cumulative_intensity(T, ts, T_low=ts.max_time)  # uses glob
            b = m.cumulative_intensity(
                T, ts, T_low=ts.max_time,
                ci_config=CIConfig("numerical", placement="interval", rule="trapezoidal", n_points=32),
            )
        assert a.shape == b.shape
    assert m.ci_config == default  # restored on exit


def test_use_ci_redirects_inversion_path():
    """use_ci must reach the inversion/grid path choice, not just cumulative_intensity.

    A closed-form model inverts analytically by default (no grid build); forcing a
    numerical CIConfig via use_ci must switch inverse_CDF to the grid path.  This pins
    the CIConfig-D fix: those consumers now read self.ci_config, not the raw
    use_analytical_ci flag (which ignored the context).
    """
    m, ts, _ = _model("hawkes")  # closed-form CI family
    u = torch.full((ts.time_points.shape[0],), 0.5, dtype=torch.float64, device=DEVICE)

    calls = {"n": 0}
    orig = m.build_intensity_grid_until

    def spy(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    m.build_intensity_grid_until = spy
    with torch.no_grad():
        m.inverse_CDF(u, ts)  # default closed_form -> analytical Newton, no grid
        assert calls["n"] == 0, "closed-form default must not build a grid"
        with m.use_ci(CIConfig("numerical", placement="global", rule="trapezoidal", n_points=64)):
            m.inverse_CDF(u, ts)  # forced numerical -> grid path
        assert calls["n"] >= 1, "use_ci(numerical) must redirect inverse_CDF to the grid path"
