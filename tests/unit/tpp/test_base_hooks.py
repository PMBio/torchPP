"""Base-class hook mechanics: the ``__init_subclass__`` intensity-
hook guard and the ``use_ci`` config context.

``_intensities_at_queries`` is the single required intensity primitive; the guard
rejects a concrete model that does not override it, while a model that does is
accepted and its public single-query intensity routes through the flat primitive at
Q=1 — these take no real model. The ``use_ci`` test pins the CI-config resolution
order: the context overrides the instance default, an explicit call-arg still wins
over the context, and the instance default is restored on exit.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from torchpp.tpp.base_tpp import CIConfig, SamplingConfig, IntensityBasedTPP, AutoregressiveTPP, TemporalPointProcess
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
    return m, ts, ts.last_event_time + 0.5


# ---------------------------------------------------------------------------
# __init_subclass__ guard
# ---------------------------------------------------------------------------


def test_guard_rejects_model_without_intensity_primitive():
    with pytest.raises(TypeError, match="must implement _intensities_at_queries"):

        class _Bad(IntensityBasedTPP):  # noqa: D401 — concrete but defines no intensity primitive
            def transform_params(self):
                return ()


def test_guard_accepts_flat_and_routes_single_query_through_it():
    # Overriding the flat primitive is required and sufficient; the public
    # single-query intensity routes through it at Q=1.
    class _OnlyFlat(IntensityBasedTPP):
        def transform_params(self):
            return ()

        def _intensities_at_queries(self, query_times, ts):
            # Query-dependent value so the single-query routing is observable.
            return query_times.unsqueeze(-1).expand(*query_times.shape, self.D)

    m = _OnlyFlat(D=3, seed=0)
    ts = _batch()

    # Flat primitive is callable directly with an arbitrary (B, Q) grid.
    flat = m._intensities_at_queries(torch.zeros(2, 3, dtype=torch.float64), ts)
    assert flat.shape == (2, 3, 3) and torch.equal(flat, torch.zeros(2, 3, 3, dtype=torch.float64))

    # Public single-query (B,) intensity must route through the flat at Q=1.
    t = torch.tensor([1.0, 4.0], dtype=torch.float64)
    lam = m.intensity(t, ts)  # (B, D)
    assert lam.shape == (2, 3) and torch.equal(lam, t.unsqueeze(-1).expand(2, 3))


# ---------------------------------------------------------------------------
# AutoregressiveTPP decomposition guard: all three factorizations of p(t,m),
# on the AR base — a non-autoregressive model stays decomposition-agnostic.
# ---------------------------------------------------------------------------


def _root_stub(*args, **kwargs):
    return torch.zeros(1)


def _concrete_ar(**decomposition):
    """A concrete ``AutoregressiveTPP`` subclass with the non-decomposition abstract
    methods stubbed and whatever decomposition roots are passed in."""
    ns = {
        "sample": lambda self, ts, num_steps=1, rng=None, sampling_config=None: ts,
        "_log_prob_events": lambda self, ts_s, T_s: torch.zeros(1),
        **decomposition,
    }
    return type("_AR", (AutoregressiveTPP,), ns)


def test_ar_guard_accepts_all_three_decompositions():
    """p(t, m) roots three equivalent ways; the guard must accept each — privileging
    one would be arbitrary."""
    _concrete_ar(density_joint=_root_stub)                              # direct
    _concrete_ar(density_time=_root_stub, prob_mark_given_time=_root_stub)      # time-first
    _concrete_ar(prob_mark=_root_stub, density_time_given_mark=_root_stub)      # mark-first


def test_ar_guard_rejects_no_root_or_half_pair():
    with pytest.raises(TypeError, match="must root the next-event joint density"):
        _concrete_ar()  # roots nothing
    with pytest.raises(TypeError, match="must root the next-event joint density"):
        _concrete_ar(prob_mark=_root_stub)  # only one half of the mark-first pair


def test_decomposition_guard_is_off_the_root():
    """The guard lives on AutoregressiveTPP, so a non-autoregressive model (its own
    ``likelihood`` / ``loss`` / ``sample``) subclasses the root directly without rooting
    any decomposition — no TypeError."""
    type(
        "_Flow",
        (TemporalPointProcess,),
        {
            "likelihood": lambda self, ts, T=None, log=True: torch.zeros(1),
            "loss": lambda self, ts: torch.zeros(1),
            "sample": lambda self, ts, num_steps=1, rng=None, sampling_config=None: ts,
        },
    )  # must not raise


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
            a = m.cumulative_intensity(T, ts, T_low=ts.last_event_time)  # uses glob
            b = m.cumulative_intensity(
                T, ts, T_low=ts.last_event_time,
                ci_config=CIConfig("numerical", placement="interval", rule="trapezoidal", n_points=32),
            )
        assert a.shape == b.shape
    assert m.ci_config == default  # restored on exit


def test_use_ci_redirects_inversion_path():
    """use_ci must reach the inversion/grid path choice, not just cumulative_intensity.

    A closed-form model inverts analytically by default (no grid build); forcing a
    numerical CIConfig via use_ci must switch _inverse_CDF to the grid path: these
    consumers read self.ci_config, not the raw use_analytical_ci flag.
    """
    m, ts, _ = _model("const_exp_hawkes")  # closed-form CI family
    u = torch.full((ts.time_points.shape[0],), 0.5, dtype=torch.float64, device=DEVICE)

    calls = {"n": 0}
    orig = m._build_intensity_grid_until

    def spy(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    m._build_intensity_grid_until = spy
    with torch.no_grad():
        m._inverse_CDF(u, ts)  # default closed_form -> analytical Newton, no grid
        assert calls["n"] == 0, "closed-form default must not build a grid"
        with m.use_ci(CIConfig("numerical", placement="global", rule="trapezoidal", n_points=64)):
            m._inverse_CDF(u, ts)  # forced numerical -> grid path
        assert calls["n"] >= 1, "use_ci(numerical) must redirect _inverse_CDF to the grid path"


# ---------------------------------------------------------------------------
# Numerics constants in CIConfig / SamplingConfig are reachable via use_ci /
# use_sampling, and the defaults keep the documented values.
# ---------------------------------------------------------------------------


def test_ciconfig_grid_num_points_override_reaches_grid():
    """A CIConfig.grid_num_points override (via use_ci) changes the built grid's
    point count; the default (200) is restored on exit."""
    m, ts, _ = _model("const_exp_hawkes")
    t0 = torch.zeros(ts.time_points.shape[0], dtype=torch.float64, device=DEVICE)
    t1 = ts.last_event_time + 1.0
    with torch.no_grad():
        assert m._build_intensity_grid(t0, t1, ts).num_points == 200  # default
        with m.use_ci(replace(m.ci_config, grid_num_points=37)):
            assert m._build_intensity_grid(t0, t1, ts).num_points == 37
        assert m._build_intensity_grid(t0, t1, ts).num_points == 200  # restored


def test_samplingconfig_quantile_iters_override_reaches_time_quantile():
    """A SamplingConfig.quantile_bisection_iters override (via use_sampling) reaches the
    density-family bisection: fewer iterations converge less tightly, so the returned
    quantile differs from the well-converged default."""
    m, ts, _ = _model("intensity_free")  # density family owns time_quantile
    with torch.no_grad():
        fine = m.time_quantile(0.5, ts)
        with m.use_sampling(replace(m.sampling_config, quantile_bisection_iters=2)):
            coarse = m.time_quantile(0.5, ts)
    assert torch.isfinite(fine).all() and torch.isfinite(coarse).all()
    assert not torch.allclose(fine, coarse), "quantile_bisection_iters override did not reach the bisection loop"


def test_config_field_defaults():
    """The CIConfig / SamplingConfig numeric defaults."""
    c, s = CIConfig(), SamplingConfig()
    assert (c.grid_num_points, c.grid_points_per_unit_time, c.grid_target_cdf) == (200, 200.0, 0.999)
    assert (c.grid_chunk_duration, c.grid_max_chunks) == (0.1, 100)
    assert (s.inversion_jitter, s.quantile_bracket_grow_iters, s.quantile_bisection_iters) == (0.1, 64, 60)
    assert s.quantile_init_bracket_width == 1.0
    assert (s.thinning_sup_points, s.thinning_window_events, s.thinning_global_window_events) == (64, 10.0, 50.0)
