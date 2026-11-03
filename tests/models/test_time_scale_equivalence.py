"""Forward-identity roundtrip test for the model time scale.

The model owns a ``time_scale`` buffer ``s``; it normalises event times to ``t/s``
internally and de-normalises its outputs back to native units. Fixed weights +
matched *internal* times must therefore give outputs related by a fixed table:

    * intensity λ   — rate, per unit time      → λ_A(t) = k · λ_B(k·t)
    * cumulative Λ, survival S, cdf, prob_mark — dimensionless → identical
    * density (log) — per unit time            → log p_A(t) = log p_B(k·t) + log k
    * likelihood    — one density factor/scored event → ll_A − ll_B = n_scored · log k
    * time_mean / time_quantile — times        → q_B = k · q_A

Run A uses ``s = 1`` on data ``τ``; Run B uses ``s = k`` on data ``k·τ`` (same
weights). Both see internal times ``τ``. ``k`` is a power of two so the ×k / ÷k
conversions are exact in IEEE-754 — the relations hold to float precision, which
proves the normalise-in / de-normalise-out bookkeeping is bug-free (a scaling bug
is off by a *factor* of k, far above any tolerance).
"""

import math

import pytest
import torch

from helpers import SEED, create_model
from torchpp.tpp.base_tpp import IntensityBasedTPP
from torchpp.utils.event_data import BatchedMVEventData, MVEventData

K = 2.0  # power-of-two scale factor (exact ×/÷)
ATOL = 1e-4  # generous; scaling bugs are off by a factor of K

# ScaledSoftplus (numerical), closed-form, and density controls; terminating both ways.
# The three closed-form neural heads (log_linear / spline / piecewise_constant) are
# included to guard the positive-term cache seam: their ``_sequence_intensities`` must
# read the content-key-validated cache (``_get_cached_params``) so the native-primed
# cache is never applied to the scaled relative times — otherwise ll is off by a factor
# at ``time_scale != 1`` (a bug the earlier omission of these three let through).
#
# ``inhibitive_spline_mlpspline_hawkes`` guards the *kernel* cache seam: its ``fix_context`` MLP
# cache is primed on the scaled batch and content-keyed, so the scaled core hits it
# instead of tripping the wrong-batch guard — the model was uncomputable at ``s != 1``.
_MODELS = [
    "nhp",
    "thp",
    "neural_mlp",
    "const_exp_hawkes",
    "piecewise_poisson",
    "intensity_free",
    "neural_log_linear",
    "neural_spline",
    "neural_piecewise_constant",
    "inhibitive_spline_mlpspline_hawkes",
]

# Distinct event types per sequence (no repeat → valid terminating history) AND at
# least one type left un-occurred per sequence, so under `terminating` the next-event
# density is dominated by a live type (a genuine density that scales) rather than the
# scale-invariant zero-intensity floor of a fully-terminated process.
_BASE_TIMES = [
    [0.2, 0.8],
    [0.1, 1.2],
]
_BASE_TYPES = [
    [0, 1],  # type 2 still active
    [2, 0],  # type 1 still active
]


def _batch(scale: float) -> BatchedMVEventData:
    times = [torch.tensor([x * scale for x in seq], dtype=torch.float64) for seq in _BASE_TIMES]
    types = [torch.tensor(seq, dtype=torch.long) for seq in _BASE_TYPES]
    return BatchedMVEventData(times, types).to("cpu")


def _set_scale(model, s: float) -> None:
    with torch.no_grad():
        model.time_scale.fill_(float(s))


@pytest.mark.parametrize("terminating", [False, True])
@pytest.mark.parametrize("model_type", _MODELS)
def test_forward_identity_roundtrip(model_type, terminating):
    """A fixed model at (s=1, τ) and (s=k, k·τ) must agree per the unit table."""
    model = create_model(model_type, terminating=terminating).to("cpu")
    model.eval()

    tsA, tsB = _batch(1.0), _batch(K)
    B = tsA.time_points.shape[0]
    tA = torch.full((B,), 3.0, dtype=torch.float64)
    tB = tA * K
    T_A = torch.full((B,), 3.0, dtype=torch.float64)
    T_B = T_A * K
    # the density-factor Jacobian is one −log s per SCORED event (predictable-typed,
    # minus the no-origin conditioned-on first event), not per valid event.
    n = model._scored_mask(tsA).sum(dim=1).double()
    is_intensity = isinstance(model, IntensityBasedTPP)

    with torch.no_grad():
        # --- Run A: s = 1 ---
        # Reseed before each run so any stochastic quadrature (e.g. mc_trapezoidal CI)
        # draws the SAME nodes in both runs; the identity is then exact, not MC-noisy.
        torch.manual_seed(SEED)
        _set_scale(model, 1.0)
        llA = model.likelihood(tsA, T=T_A, log=True)
        dtA = model.density_time(tA, tsA, log=True)
        svA = model.survival_time(tA, tsA, log=True)
        cdA = model.cdf_time(tA, tsA, log=True)
        pmA = model.prob_mark(tsA, log=True)
        tmA = model.time_mean(tsA)
        tqA = model.time_quantile(0.5, tsA)
        if is_intensity:
            iA = model.intensity(tA, tsA)
            ciA = model.cumulative_intensity(T_A, tsA)

        # --- Run B: s = k, data k·τ (identical internal times) ---
        torch.manual_seed(SEED)
        _set_scale(model, K)
        llB = model.likelihood(tsB, T=T_B, log=True)
        dtB = model.density_time(tB, tsB, log=True)
        svB = model.survival_time(tB, tsB, log=True)
        cdB = model.cdf_time(tB, tsB, log=True)
        pmB = model.prob_mark(tsB, log=True)
        tmB = model.time_mean(tsB)
        tqB = model.time_quantile(0.5, tsB)
        if is_intensity:
            iB = model.intensity(tB, tsB)
            ciB = model.cumulative_intensity(T_B, tsB)

    logk = math.log(K)
    # likelihood: differ by exactly n_scored · log k (one density-factor Jacobian per scored event)
    assert torch.allclose(llA - llB, n * logk, atol=ATOL), f"{model_type}: likelihood Jacobian"
    # log density: differ by log k per query
    assert torch.allclose(dtA - dtB, torch.full_like(dtA, logk), atol=ATOL), f"{model_type}: density_time"
    # dimensionless: identical
    assert torch.allclose(svA, svB, atol=ATOL), f"{model_type}: survival"
    assert torch.allclose(cdA, cdB, atol=ATOL), f"{model_type}: cdf"
    assert torch.allclose(pmA, pmB, atol=ATOL), f"{model_type}: prob_mark"
    # times: scale by exactly k
    assert torch.allclose(tmB, K * tmA, atol=ATOL, rtol=1e-4), f"{model_type}: time_mean"
    assert torch.allclose(tqB, K * tqA, atol=ATOL, rtol=1e-4), f"{model_type}: time_quantile"
    if is_intensity:
        assert torch.allclose(iA, K * iB, atol=ATOL), f"{model_type}: intensity rate"
        assert torch.allclose(ciA, ciB, atol=ATOL), f"{model_type}: cumulative_intensity"


@pytest.mark.parametrize("model_type", ["const_exp_hawkes", "nhp"])
def test_evaluate_at_time_scale_covariance(model_type):
    """The eval path (``evaluate_at_time``) must be scale-covariant end-to-end.

    Guards the grid-scale seam: ``evaluate_at_time`` builds the integration grid on
    the scaled batch and converts grid-derived TIMES back to native units.  A fixed
    model at (s, τ) vs (s·k, k·τ): grid-derived times scale by k, intensity is a rate
    (÷ scale, so ``i_A = k·i_B``), and CI / marginal mark dist are dimensionless.
    """
    from torchpp.utils.evaluation import evaluate_at_time

    model = create_model(model_type, terminating=False).to("cpu")
    model.eval()
    tsA, tsB = _batch(1.0), _batch(K)
    B = tsA.time_points.shape[0]
    TA = torch.full((B,), 3.0, dtype=torch.float64)

    with torch.no_grad():
        torch.manual_seed(SEED)
        _set_scale(model, 1.0)
        rA = evaluate_at_time(model, TA, tsA, compute_marginals=True)
        torch.manual_seed(SEED)
        _set_scale(model, K)
        rB = evaluate_at_time(model, TA * K, tsB, compute_marginals=True)

    assert torch.allclose(rA["intensity"], K * rB["intensity"], atol=ATOL), f"{model_type}: eval intensity"
    assert torch.allclose(rA["ci"], rB["ci"], atol=ATOL), f"{model_type}: eval ci"
    assert torch.allclose(rB["mean_time"], K * rA["mean_time"], atol=ATOL, rtol=1e-4), f"{model_type}: eval mean_time"
    assert torch.allclose(rB["median_time"], K * rA["median_time"], atol=ATOL, rtol=1e-4), f"{model_type}: eval median_time"
    assert torch.allclose(
        rA["marginal_type_dist"], rB["marginal_type_dist"], atol=ATOL
    ), f"{model_type}: eval marginal_type_dist"


@pytest.mark.parametrize("with_T_obs", [False, True])
@pytest.mark.parametrize("model_type", _MODELS)
def test_full_contract_roundtrip(model_type, with_T_obs):
    """Every remaining public method follows the same unit table, also with ``T_obs`` past the last event."""
    model = create_model(model_type, terminating=False).to("cpu")
    model.eval()
    T_obs = [1.5, 2.0] if with_T_obs else None

    def batch(scale):
        times = [torch.tensor([x * scale for x in seq], dtype=torch.float64) for seq in _BASE_TIMES]
        types = [torch.tensor(seq, dtype=torch.long) for seq in _BASE_TYPES]
        obs = [x * scale for x in T_obs] if T_obs else None
        return BatchedMVEventData(times, types, T_obs=obs).to("cpu")

    def one(scale):
        obs = T_obs[0] * scale if T_obs else None
        return MVEventData(
            torch.tensor([x * scale for x in _BASE_TIMES[0]], dtype=torch.float64),
            torch.tensor(_BASE_TYPES[0], dtype=torch.long),
            T_obs=obs,
        )

    tsA, tsB = batch(1.0), batch(K)
    tA = torch.tensor([[2.5, 3.0], [2.2, 4.0]], dtype=torch.float64)  # (B, Q), after every T_obs
    T_lowA = torch.tensor([0.5, 0.3], dtype=torch.float64)
    T_A = torch.full((2,), 3.0, dtype=torch.float64)
    has_intensity = hasattr(model, "intensity")

    def run(s, ts, k):
        torch.manual_seed(SEED)
        _set_scale(model, s)
        out = {
            "dj": model.density_joint(tA * k, ts, log=True),
            "pmt": model.prob_mark_given_time(tA * k, ts, log=True),
            "dtm": model.density_time_given_mark(tA * k, ts, log=True),
        }
        if has_intensity:
            out["iq"] = model.intensity(tA * k, ts)
        if isinstance(model, IntensityBasedTPP):
            out["ci_low"] = model.cumulative_intensity(T_A * k, ts, T_low=T_lowA * k)
        return out

    with torch.no_grad():
        A = run(1.0, tsA, 1.0)
        B = run(K, tsB, K)
        torch.manual_seed(SEED)
        _set_scale(model, 1.0)
        sA = model.sample(one(1.0), num_steps=2, rng=0)
        torch.manual_seed(SEED)
        _set_scale(model, K)
        sB = model.sample(one(K), num_steps=2, rng=0)

    logk = math.log(K)
    assert torch.allclose(A["dj"] - B["dj"], torch.full_like(A["dj"], logk), atol=ATOL), f"{model_type}: density_joint"
    assert torch.allclose(A["pmt"], B["pmt"], atol=ATOL), f"{model_type}: prob_mark_given_time"
    finite = torch.isfinite(A["dtm"])
    assert torch.allclose(
        (A["dtm"] - B["dtm"])[finite], torch.full_like(A["dtm"], logk)[finite], atol=ATOL
    ), f"{model_type}: density_time_given_mark"
    if has_intensity:
        assert torch.allclose(A["iq"], K * B["iq"], atol=ATOL), f"{model_type}: intensity (B, Q)"
    if "ci_low" in A:
        assert torch.allclose(A["ci_low"], B["ci_low"], atol=ATOL), f"{model_type}: cumulative_intensity T_low"
    tA_s, tB_s = sA[0].time_points, sB[0].time_points
    assert torch.allclose(tB_s, K * tA_s, atol=ATOL, rtol=1e-4), f"{model_type}: sample times"
    assert torch.equal(sA[0].event_types, sB[0].event_types), f"{model_type}: sample types"
