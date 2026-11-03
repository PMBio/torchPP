"""Regression tests for numerical edge cases of the intensity / density contract.

Each test pins one behaviour on a small, fully specified model: quadrature at event
discontinuities, thinning bounds, the IntensityFree full-history column, context types,
ignored types on the evaluation grid, defective (mass-at-infinity) distributions, ties
with the first event, compact kernel support, deterministic predictive integrals, and
forecasting from the end of the observation window ``T_obs``.
"""

import math
from dataclasses import replace

import pytest
import torch
from torchpp.utils.event_data import TokenRoles

from helpers import create_model
from torchpp.modules.intensity import ConstantBaselineModule, ConstantBaselineParams, ExpKernelParams
from torchpp.modules.intensity import MLPSplineKernelModule
from torchpp.tpp.base_tpp import CIConfig, SamplingConfig
from torchpp.training.model_factory import create_model as build_model
from torchpp.tpp.baseline import LogLinearPoissonParams, LogLinearPoissonProcess, PoissonProcess, PoissonProcessParams
from torchpp.tpp.hawkes import ConstBaselineExpKernelHawkesProcess
from torchpp.utils.evaluation import _build_prediction_items, evaluate_at_time
from torchpp.utils.event_data import BatchedMVEventData, MVEventData

DT = torch.float64


def _seq(times, types, dtype=DT, T_obs=None):
    return MVEventData(torch.tensor(times, dtype=dtype), torch.tensor(types, dtype=torch.long), T_obs=T_obs)


def _batch(*seqs):
    return BatchedMVEventData(mv_events=list(seqs))


def _hawkes(mu, alpha, beta, **kw):
    D = len(mu)
    return ConstBaselineExpKernelHawkesProcess(
        D=D,
        baseline_params=ConstantBaselineParams(mu=torch.tensor(mu, dtype=DT)),
        kernel_params=ExpKernelParams(alpha=torch.tensor(alpha, dtype=DT), beta=torch.tensor(beta, dtype=DT)),
        **kw,
    )


# ---------------------------------------------------------------------------
# Quadrature at event discontinuities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", ["thp", "neural_mlp", "attnhp", "inhibitive_const_exp_hawkes"])
def test_interval_quadrature_converges_second_order(model_type):
    """Event-knotted trapezoid: the first node of each interval takes the post-jump intensity,
    so the likelihood error shrinks ~100x per 10x nodes (second order), not ~10x."""
    torch.manual_seed(0)
    model = create_model(model_type, d=2).cpu().eval()
    ts = _batch(_seq([0.2, 0.5, 1.1, 1.7, 2.4], [0, 1, 0, 0, 1]), _seq([0.3, 0.9, 2.0], [1, 1, 0]))

    def ll(P):
        model.ci_config = CIConfig("numerical", "interval", "trapezoidal", n_points=P)
        with torch.no_grad():
            return model.likelihood(ts)

    ref = ll(4001)
    e_coarse = (ll(11) - ref).abs().max().item()
    e_fine = (ll(101) - ref).abs().max().item()
    assert e_coarse > 0
    assert e_fine < e_coarse / 40, (e_coarse, e_fine)


def test_global_window_starting_at_event_is_dtype_stable():
    """A global-placement window that starts exactly at an event integrates the post-jump
    intensity regardless of the query dtype (float32 vs float64 end time)."""
    model = _hawkes([0.3, 0.2], [[0.8, 0.1], [0.2, 0.6]], [[1.0, 1.0], [1.0, 1.0]])
    model.ci_config = CIConfig("numerical", "global", "trapezoidal", n_points=200)
    ts = _batch(_seq([0.2, 0.7, 1.5], [0, 1, 0]))
    t1 = ts.time_points[:, 0]
    ci64 = model.cumulative_intensity(torch.tensor([4.0], dtype=torch.float64), ts, T_low=t1)
    ci32 = model.cumulative_intensity(torch.tensor([4.0], dtype=torch.float32), ts, T_low=t1)
    assert (ci64.double() - ci32.double()).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# Thinning
# ---------------------------------------------------------------------------


def test_thinning_matches_cdf_after_excitatory_jump():
    """With a tiny baseline the post-event peak dominates; thinning must bound the
    right-limit intensity, not the pre-jump value at the last event."""
    model = _hawkes([1e-3], [[2.0]], [[1.0]])
    hist = _seq([1.0], [0])
    cfg = SamplingConfig(mode="thinning")
    gaps = []
    for seed in range(300):
        out, _, _ = model.sample(hist, num_steps=1, rng=seed, sampling_config=cfg)
        gaps.append(float(out.time_points[-1]) - 1.0)
    empirical = sum(g < 5.0 for g in gaps) / len(gaps)
    expected = 1.0 - math.exp(-(1e-3 * 5.0 + 2.0 * (1.0 - math.exp(-5.0))))  # ≈ 0.865
    assert abs(empirical - expected) < 0.07, (empirical, expected)


# ---------------------------------------------------------------------------
# IntensityFree full-history column
# ---------------------------------------------------------------------------


def test_intensity_free_full_history_column_float32_large_times():
    """The predictive mark/time law conditions on ALL events also when float32 times are
    large enough that t_last + 1e-6 rounds back to t_last."""
    torch.manual_seed(0)
    model = create_model("intensity_free", d=3).cpu().eval()
    times, types = [1000.2, 1000.8, 1002.0, 1002.3], [0, 2, 1, 0]
    b64 = _batch(_seq(times, types, dtype=torch.float64))
    b32 = _batch(_seq(times, types, dtype=torch.float32))
    with torch.no_grad():
        p64 = model.prob_mark(b64, log=False)
        p32 = model.prob_mark(b32, log=False)
        q = torch.tensor([1003.0], dtype=torch.float64)
        d64 = model.density_joint(q, b64)
        d32 = model.density_joint(q, b32)
    assert (p64 - p32.double()).abs().max().item() < 1e-5
    assert (d64 - d32.double()).abs().max().item() < 1e-3


# ---------------------------------------------------------------------------
# Context types
# ---------------------------------------------------------------------------


def test_context_types_closed_form_quantile_after_last_event():
    model = PoissonProcess(params=PoissonProcessParams(mu=torch.tensor([5.0, 1.0, 1.0], dtype=DT)), token_roles=TokenRoles(context_types=[0]))
    ts = _batch(_seq([1.0, 2.0], [1, 2]))
    q = model.time_quantile(0.5, ts)
    assert abs(q.item() - (2.0 + math.log(2.0) / 2.0)) < 1e-6


@pytest.mark.parametrize("mode", ["inversion", "thinning"])
def test_context_types_never_sampled(mode):
    hawkes = _hawkes([2.0, 0.5, 0.5], [[0.1] * 3] * 3, [[1.0] * 3] * 3, token_roles=TokenRoles(context_types=[0]))
    poisson = PoissonProcess(params=PoissonProcessParams(mu=torch.tensor([5.0, 1.0, 1.0], dtype=DT)), token_roles=TokenRoles(context_types=[0]))
    hist = _seq([0.5, 1.0], [1, 2])
    cfg = SamplingConfig(mode=mode)
    for model in (hawkes, poisson):
        out, _, dists = model.sample(hist, num_steps=20, rng=0, sampling_config=cfg)
        assert not (out.event_types[2:] == 0).any()
        assert all(float(d[0]) == 0.0 for d in dists)


# ---------------------------------------------------------------------------
# Ignored types on the evaluation grid
# ---------------------------------------------------------------------------


def test_evaluate_at_time_ignored_types_grow_grid_on_retained_process():
    model = PoissonProcess(params=PoissonProcessParams(mu=torch.tensor([10.0, 0.1, 0.1], dtype=DT)))
    ts = _batch(_seq([0.5, 1.0], [1, 2]))
    res = evaluate_at_time(model, torch.tensor([2.0], dtype=DT), ts, ignored_types=[0])
    assert abs(res["mean_time"].item() - (1.0 + 5.0)) < 0.05
    assert abs(res["median_time"].item() - (1.0 + math.log(2.0) / 0.2)) < 0.05


# ---------------------------------------------------------------------------
# Mass at infinity
# ---------------------------------------------------------------------------


def _defective_model():
    # λ(t) = exp(log 0.5 − t): Λ(∞) − Λ(0.5) = 0.5·e^{−0.5} ≈ 0.30 → S(∞) ≈ 0.74.
    return LogLinearPoissonProcess(
        params=LogLinearPoissonParams(mu=torch.tensor([math.log(0.5)], dtype=DT), slope=torch.tensor([-1.0], dtype=DT))
    )


def test_defective_distribution_mean_and_median_are_inf():
    model = _defective_model()
    ts = _batch(_seq([0.5], [0]))
    assert torch.isinf(model.time_mean(ts)).all()
    assert torch.isinf(model.time_quantile(0.5, ts)).all()
    assert torch.isfinite(model.time_quantile(0.1, ts)).all()


def test_defective_distribution_sampling_stops_instead_of_huge_times():
    model = _defective_model()
    hist = _seq([0.5], [0])
    n_stopped = 0
    for seed in range(40):
        out, times, _ = model.sample(hist, num_steps=3, rng=seed)
        assert all(math.isfinite(float(t)) and float(t) < 1e3 for t in out.time_points)
        n_stopped += len(times) < 3
    assert n_stopped > 0


# ---------------------------------------------------------------------------
# Ties with the first event (no origin)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", ["neural_spline", "const_exp_hawkes"])
def test_events_tied_with_first_event_are_conditioned_on(model_type):
    """Without an origin the whole tied first block is the conditioned-on origin: the
    second event at t1 is not scored from the empty-history state."""
    torch.manual_seed(0)
    model = create_model(model_type, d=2).cpu().eval()
    tied = _batch(_seq([0.5, 0.5, 1.0], [0, 1, 0]))
    with torch.no_grad():
        ll = model.likelihood(tied)
    assert torch.isfinite(ll).all() and ll.item() > -50.0
    assert model._scored_mask(tied).tolist() == [[False, False, True]]


def test_prediction_items_skip_targets_tied_with_first_event():
    seq = _seq([0.5, 0.5, 1.0, 1.0], [0, 1, 0, 1])
    items = _build_prediction_items([seq], has_origin=False)
    assert [float(it["target_time"]) for it in items] == [1.0, 1.0]
    items_origin = _build_prediction_items([seq], has_origin=True)
    assert len(items_origin) == 4  # with an origin every event is a target, the first included


# ---------------------------------------------------------------------------
# Compact support of the MLP-spline kernel
# ---------------------------------------------------------------------------


def test_mlp_spline_kernel_is_zero_past_the_tail_knot():
    torch.manual_seed(0)
    kernel = MLPSplineKernelModule(D=2, num_knots=3, delta_t=0.5, seed=0).double()
    with torch.no_grad():
        kernel.mlp[-1].bias.fill_(1.0)
    tail = float(kernel.knot_locs_with_tail[-1])
    ts = _batch(_seq([0.0], [0]))
    far = torch.tensor([tail + 3.0], dtype=DT)
    with torch.no_grad():
        lam = kernel.intensity(far, ts)
        ci_tail = kernel._cumulative_intensity(torch.full((1, 2), tail, dtype=DT), ts)
        ci_far = kernel._cumulative_intensity(torch.full((1, 2), tail + 3.0, dtype=DT), ts)
    assert torch.equal(lam, torch.zeros_like(lam))
    assert torch.allclose(ci_tail, ci_far)


# ---------------------------------------------------------------------------
# Constant baseline initialisation
# ---------------------------------------------------------------------------


def test_unconstrained_constant_baseline_keeps_its_init():
    mu = torch.tensor([0.3, 1.7], dtype=DT)
    module = ConstantBaselineModule(D=2, params=ConstantBaselineParams(mu=mu), constrained=False)
    assert torch.allclose(module.transform_params()[0].double(), mu)


# ---------------------------------------------------------------------------
# Deterministic predictive integrals
# ---------------------------------------------------------------------------


def test_predictive_methods_are_deterministic_under_mc_training_rule():
    model = build_model("inhibitive_const_exp_hawkes", D=2, seed=0, terminating=False)
    model.eval()
    assert model.ci_config.rule == "mc_trapezoidal"
    ts = _batch(_seq([0.2, 0.7, 1.5], [0, 1, 0]))
    q = torch.tensor([2.5], dtype=DT)
    with torch.no_grad():
        a = model.survival_time(q, ts)
        b = model.survival_time(q, ts)
        c = model.density_joint(q, ts)
        d = model.density_joint(q, ts)
    assert torch.equal(a, b) and torch.equal(c, d)


# ---------------------------------------------------------------------------
# Forecasting from the end of the observation window
# ---------------------------------------------------------------------------


def test_poisson_forecast_conditions_on_T_obs():
    model = PoissonProcess(params=PoissonProcessParams(mu=torch.tensor([1.0, 1.0], dtype=DT)))
    ts = _batch(_seq([1.0], [0], T_obs=3.0))
    assert abs(model.time_quantile(0.5, ts).item() - (3.0 + math.log(2.0) / 2.0)) < 1e-6
    assert abs(model.time_mean(ts).item() - 3.5) < 1e-2
    assert abs(model.survival_time(torch.tensor([3.0], dtype=DT), ts, log=False).item() - 1.0) < 1e-9
    hist = _seq([1.0], [0], T_obs=3.0)
    for seed in range(20):
        out, _, _ = model.sample(hist, num_steps=1, rng=seed)
        assert float(out.time_points[-1]) > 3.0


def test_hawkes_forecast_survival_is_conditional_on_T_obs():
    model = _hawkes([0.3, 0.2], [[0.8, 0.1], [0.2, 0.6]], [[1.0, 1.0], [1.0, 1.0]])
    hist_open = _batch(_seq([0.2, 0.7, 1.5], [0, 1, 0]))
    hist_cens = _batch(_seq([0.2, 0.7, 1.5], [0, 1, 0], T_obs=2.0))
    t = torch.tensor([3.0], dtype=DT)
    expected = model.survival_time(t, hist_open, log=True) - model.survival_time(
        torch.tensor([2.0], dtype=DT), hist_open, log=True
    )
    assert torch.allclose(model.survival_time(t, hist_cens, log=True), expected, atol=1e-10)
    assert model.time_quantile(0.5, hist_cens).item() > 2.0
    for mode in ("inversion", "thinning"):
        out, _, _ = model.sample(_seq([0.2, 0.7, 1.5], [0, 1, 0], T_obs=2.0), num_steps=2, rng=0,
                                 sampling_config=SamplingConfig(mode=mode))
        assert float(out.time_points[3]) > 2.0


def test_intensity_free_forecast_conditions_on_T_obs():
    torch.manual_seed(0)
    model = create_model("intensity_free", d=3).cpu().eval()
    times, types = [0.2, 0.8, 1.1], [0, 2, 1]
    open_ = _batch(_seq(times, types))
    cens = _batch(_seq(times, types, T_obs=1.6))
    t = torch.tensor([2.0], dtype=DT)
    with torch.no_grad():
        expected = model.survival_time(t, open_) - model.survival_time(torch.tensor([1.6], dtype=DT), open_)
        assert torch.allclose(model.survival_time(t, cens), expected, atol=1e-8)
        assert torch.allclose(model.prob_mark(cens), model.prob_mark(open_), atol=1e-8)
        assert model.time_quantile(0.5, cens).item() > 1.6
        assert model.time_mean(cens).item() > 1.6
        out, _, _ = model.sample(_seq(times, types, T_obs=1.6), num_steps=1, rng=0)
    assert float(out.time_points[-1]) > 1.6


# ---------------------------------------------------------------------------
# Memory: forecast windows and point estimates
# ---------------------------------------------------------------------------


def test_forecast_window_integrates_only_the_final_interval():
    """A density query after the last event integrates [T_obs, t] only: the default interval
    hook is evaluated on one interval, not on all L+1 (which kept GBs of activations alive)."""
    model = create_model("inhibitive_spline_mlp_hawkes", d=3, seed=0, terminating=False).cpu()
    model.ci_config = replace(model.ci_config, mode="numerical", placement="interval")
    batch = _batch(_seq([0.2, 0.5, 0.9, 1.4, 2.0, 2.3], [0, 1, 2, 0, 1, 2]))
    n_queries = []
    original = model._intensities_at_queries

    def counting(q, ts):
        n_queries.append(q.shape[1])
        return original(q, ts)

    model._intensities_at_queries = counting
    model.density_joint(torch.tensor([2.8], dtype=DT), batch, log=False)
    P = model._predictive_ci_config().n_points
    assert max(n_queries) <= P  # interval CI: one interval × P nodes (intensity at t is 1 query)


def test_point_estimates_carry_no_autograd_graph():
    """time_quantile / time_mean come from grid inversion or bisection: no graph is kept."""
    model = create_model("inhibitive_spline_mlp_hawkes", d=3, seed=0, terminating=False).cpu()
    batch = _batch(_seq([0.2, 0.9, 1.4], [0, 1, 2]))
    assert not model.time_quantile(0.5, batch).requires_grad
    assert not model.time_mean(batch).requires_grad


# ---------------------------------------------------------------------------
# Grid targets, per-row grid integrals, terminating IntensityFree mean, limits at infinity
# ---------------------------------------------------------------------------


def test_grid_target_near_one_stays_finite_in_float32():
    """A target CDF near 1 is converted to a cumulative intensity in float64; float32 rounds it
    to inf and the grid would grow to ~1e18."""
    import warnings

    model = build_model("thp", D=2, seed=0).eval()
    ts = _batch(_seq([0.1, 0.3], [0, 1], dtype=torch.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with torch.no_grad():
            t = model._inverse_CDF(torch.tensor([0.9995]), ts)
    assert torch.isfinite(t).all() and float(t) < 1e3


@pytest.mark.parametrize("model_type", ["thp", "const_exp_hawkes"])
def test_grid_mean_and_marks_do_not_depend_on_other_rows(model_type):
    """Each row integrates the grid only up to its own stopping point: a fully terminated row
    (which never reaches the target) must not change the other rows' results."""
    model = build_model(model_type, D=3, seed=0, terminating=True).eval()
    row = _seq([0.2, 0.4], [1, 0])
    terminated = _seq([0.1, 0.5, 0.9], [0, 1, 2])
    with torch.no_grad():
        alone_mean, alone_marks = model.time_mean(_batch(row))[0], model.prob_mark(_batch(row), log=False)[0]
        both = _batch(terminated, row)
        mean, marks = model.time_mean(both), model.prob_mark(both, log=False)
    assert torch.isinf(mean[0])
    assert torch.allclose(mean[1], alone_mean, rtol=1e-6)
    assert torch.allclose(marks[1], alone_marks, atol=1e-6)


def test_intensity_free_terminating_mean_matches_survival_integral():
    """Terminating IntensityFree: E[t] = t_last + ∫ S(τ)^A dτ under the exact log-normal mixture
    law (A = remaining mark mass); a fully terminated row has no next event (inf)."""
    model = build_model("intensity_free", D=3, seed=0, terminating=True).double().eval()
    ts = _batch(_seq([0.2, 0.4], [1, 0]), _seq([0.1, 0.5, 0.9], [0, 1, 2]))
    from torchpp.tpp.neural import _LOG_SCALE_MAX, _LOG_SCALE_MIN

    with torch.no_grad():
        mean = model.time_mean(ts)
        scaled = model._scaled_ts(ts)
        A = model._log_remaining_mark_mass(scaled).exp()
        params = model._full_history_params(scaled)
    K = model.K
    loc = params[:, :K]
    scale = params[:, K : 2 * K].clamp(_LOG_SCALE_MIN, _LOG_SCALE_MAX).exp()
    logw = torch.log_softmax(params[:, 2 * K : 3 * K], dim=-1)
    tau = torch.logspace(-10, 10, 200001, dtype=DT)
    z = (loc[0].unsqueeze(0) - tau.log().unsqueeze(1)) / scale[0].unsqueeze(0)
    log_S = torch.logsumexp(logw[0].unsqueeze(0) + torch.special.log_ndtr(z), dim=1)
    expected = 0.4 + torch.trapezoid(torch.exp(A[0] * log_S), tau)
    assert torch.allclose(mean[0], expected, rtol=1e-4)
    assert torch.isinf(mean[1])


@pytest.mark.parametrize("model_type", ["poisson", "const_exp_hawkes", "inhibitive_const_exp_hawkes", "nhp", "thp", "intensity_free"])
def test_survival_and_cdf_at_infinity(model_type):
    """S(∞) = 1 for a row with no possible next event, 0 otherwise; never NaN."""
    model = build_model(model_type, D=3, seed=0, terminating=True).eval()
    ts = _batch(_seq([0.1, 0.5, 0.9], [0, 1, 2], dtype=torch.float32), _seq([0.2, 0.4], [1, 0], dtype=torch.float32))
    inf = torch.full((2,), float("inf"))
    with torch.no_grad():
        S = model.survival_time(inf, ts, log=False)
        F = model.cdf_time(inf, ts, log=False)
        f = model.density_time(inf, ts, log=False)
    assert torch.equal(S.float(), torch.tensor([1.0, 0.0]))
    assert torch.allclose(F.float(), torch.tensor([0.0, 1.0]))
    assert torch.equal(f.float(), torch.zeros(2))


# ---------------------------------------------------------------------------
# Thinning with a narrow late peak
# ---------------------------------------------------------------------------


def _spike_model():
    from torchpp.modules.intensity import SplineKernelParams
    from torchpp.tpp.hawkes import ConstBaselineSplineKernelHawkesProcess

    heights = torch.tensor([[[0.01, 0.01, 0.01, 5.0, 0.01, 0.01]]], dtype=DT)
    return ConstBaselineSplineKernelHawkesProcess(
        1,
        kernel_num_knots=6,
        kernel_delta_t=torch.tensor([2.9, 0.05, 0.05, 0.05, 0.05], dtype=DT),
        baseline_params=ConstantBaselineParams(mu=torch.tensor([0.1], dtype=DT)),
        kernel_params=SplineKernelParams(heights, torch.zeros(6, dtype=DT), torch.tensor(0.05, dtype=DT)),
    ).eval()


def test_thinning_bound_contains_the_knots_of_a_spline_kernel():
    """The bound scan includes the kernel's knots, so a peak narrower than the scan grid
    (height 5 at lag 3, width 0.1) is still dominated."""
    model = _spike_model()
    ts = _batch(_seq([0.0], [0]))
    t = model._forecast_origin(ts)
    with torch.no_grad():
        lam_bar, _ = model._thinning_upper_bound(ts, t, SamplingConfig(mode="thinning"))
        peak = model._intensity_scaled(torch.tensor([[3.0]], dtype=DT), ts).sum()
    assert float(lam_bar) >= float(peak)


def test_thinning_matches_cdf_with_a_narrow_late_peak():
    """KS of thinning draws against the exact CDF on the spike model (n = 600)."""
    model = _spike_model()
    hist = _seq([0.0], [0])
    cfg = SamplingConfig(mode="thinning")
    gaps = []
    with torch.no_grad():
        for seed in range(600):
            _, times, _ = model.sample(hist, num_steps=1, rng=seed, sampling_config=cfg)
            gaps.append(float(times[0]))
        F = model.cdf_time(torch.tensor([sorted(gaps)], dtype=DT), _batch(hist), log=False)[0]
    n = len(gaps)
    i = torch.arange(1, n + 1, dtype=DT)
    ks = float(torch.maximum(i / n - F, F - (i - 1) / n).max())
    assert ks < 1.63 / math.sqrt(n), ks  # 1% critical value
