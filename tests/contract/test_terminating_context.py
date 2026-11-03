"""Terminating mode composes correctly with context/``<bos>`` roles.

Context types (the injected ``<bos>`` origin, read-only side-info) are OUTSIDE the
competing-risks set: never scored, never integrated, never suppressed-after-fire,
excluded from the terminating active set.  The role mask is applied FIRST, then the
terminating semantics act only within the predictable subset.  These tests exercise a
EHR-like regime — ``terminating=True``, ``has_origin=True``, one ``<bos>`` context
type — and assert the composition; the default all-True / no-context path is
covered by the terminating suite in ``tests/models/test_terminating*``, so it is not
re-derived here.
"""

import pytest
import torch

from helpers import create_model, DEVICE, SEED
from torchpp.utils.event_data import BatchedMVEventData

D_TOTAL = 4  # predictable {0, 1, 2} + one context/<bos> type
BOS = 3  # the context (read-only) type index

# Families spanning the two likelihood bases: intensity (poisson closed-form, hawkes
# kernel) and density (intensity_free).  The repeat-gate / competing-risks assertions
# are family-agnostic; the closed-form value checks use poisson only.
_FAMILIES = ["poisson", "const_exp_hawkes", "intensity_free"]


def _ctx_model(model_type, terminating=True, has_origin=True):
    """An EHR-like model: terminating, has_origin, type ``BOS`` marked context."""
    m = create_model(model_type, d=D_TOTAL, seed=SEED, terminating=terminating)
    mask = torch.ones(D_TOTAL, dtype=torch.bool, device=m.predictable_mask.device)
    mask[BOS] = False
    m.predictable_mask.copy_(mask)
    m.has_origin = has_origin
    m.eval()
    return m


def _batch(times, types, T):
    return BatchedMVEventData(
        [torch.tensor(times, dtype=torch.float64, device=DEVICE)],
        [torch.tensor(types, dtype=torch.long, device=DEVICE)],
        T_obs=[T],
    ).to(DEVICE)


# ---------------------------------------------------------------------------
# Repeated-type gate — scoped to predictable types (fix 1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", _FAMILIES)
def test_repeated_context_type_does_not_produce_neg_inf(model_type):
    """A repeated *context* type is not an impossible event: the likelihood stays finite.

    ``<bos>`` occurs at t=0 and again at t=1.0 (a repeated context type); the predictable
    types 0 and 1 each occur once.  No predictable type repeats, so the terminating
    first-occurrence gate must NOT fire.
    """
    m = _ctx_model(model_type)
    batch = _batch([0.0, 0.5, 1.0, 1.5], [BOS, 0, BOS, 1], 3.0)
    with torch.no_grad():
        assert bool(m._has_repeated_event_types(batch).item()) is False
        ll = m.likelihood(batch, log=True)
    assert torch.isfinite(ll).all(), f"repeated context type must not give -inf: {ll}"


@pytest.mark.parametrize("model_type", _FAMILIES)
def test_repeated_predictable_type_still_impossible(model_type):
    """A repeated *predictable* terminating type is impossible: the likelihood is -inf.

    Type 0 occurs at t=0.5 and again at t=1.0; ``<bos>`` provides the origin context.
    """
    m = _ctx_model(model_type)
    batch = _batch([0.0, 0.5, 1.0], [BOS, 0, 0], 3.0)
    with torch.no_grad():
        assert bool(m._has_repeated_event_types(batch).item()) is True
        ll = m.likelihood(batch, log=True)
    assert torch.isneginf(ll).all(), f"repeated predictable type must give -inf: {ll}"


# ---------------------------------------------------------------------------
# Context excluded from the scored objective (positive term + survival Σ_d Λ_d)
# ---------------------------------------------------------------------------


def test_context_never_scored_or_integrated_poisson():
    """Closed-form check that ``<bos>`` is neither scored nor integrated (poisson).

    Poisson has a constant, history-independent intensity, so the terminating +
    has_origin likelihood is exactly

        log μ_0 + log μ_1 − (μ_0·min(T,t_term0) + μ_1·min(T,t_term1) + μ_2·T)

    with the ``<bos>`` (context, type 3) column absent from Σ_d Λ_d and its event
    contributing 0 to the positive term.  Hand-derived from the transformed rates.
    """
    m = _ctx_model("poisson")
    # <bos>@0 (context), type 0 @0.5 (terminates 0.5), type 1 @1.5 (terminates 1.5); T=3.
    batch = _batch([0.0, 0.5, 1.5], [BOS, 0, 1], 3.0)
    T = 3.0
    (mu,) = m.transform_params()  # (D,) constant rates
    mu = mu.double()
    expected = (
        torch.log(mu[0]) + torch.log(mu[1])
        - (mu[0] * 0.5 + mu[1] * 1.5 + mu[2] * T)  # type 3 (<bos>) excluded from the survival sum
    )
    with torch.no_grad():
        ll = m.likelihood(batch, log=True)[0].double()
        # The positive term alone: <bos> scores 0, so it equals log μ_0 + log μ_1.
        pos = m._event_log_intensity_sum(m._scaled_ts(batch))[0].double()
    assert torch.isclose(ll, expected, atol=1e-9), f"LL {ll} != hand-derived {expected}"
    assert torch.isclose(pos, torch.log(mu[0]) + torch.log(mu[1]), atol=1e-9), (
        f"positive term {pos} != log μ_0 + log μ_1 (context <bos> must score 0)"
    )


def test_context_inert_to_scored_objective_poisson():
    """Adding a context event leaves the poisson likelihood unchanged.

    Poisson has no history-dependent excitation, so a read-only ``<bos>`` — never
    scored, never integrated, never suppressing a predictable type — cannot move the
    likelihood.  (Hawkes/neural DO let context condition the state, tested elsewhere.)
    """
    m = _ctx_model("poisson")
    base = _batch([0.0, 0.5, 1.5], [BOS, 0, 1], 3.0)
    plus_ctx = _batch([0.0, 0.5, 1.0, 1.5], [BOS, 0, BOS, 1], 3.0)  # extra <bos>@1.0
    with torch.no_grad():
        assert torch.isclose(
            m.likelihood(base, log=True)[0], m.likelihood(plus_ctx, log=True)[0], atol=1e-9
        )


# ---------------------------------------------------------------------------
# IFTPP: the terminating active set / survival correction exclude context (fixes 2, 3, 4)
# ---------------------------------------------------------------------------


def test_iftpp_active_mark_mass_excludes_context():
    """IFTPP ``A`` (remaining-mass scale factor) is over predictable-active types only.

    ``_log_remaining_mark_mass`` (the predictive-path ``A`` feeding the reduced survival
    ``S^A`` / density ``A·f·S^{A-1}``) must be the log-sum of the predictable-renormalized
    mark law over still-active types — context types carry zero mass — and so must agree
    with the training-path ``A`` used in ``_event_log_density_sum``.
    """
    m = _ctx_model("intensity_free")
    # <bos>@0 (context) + type 0 @0.5 (predictable, terminated); types 1,2 still active.
    batch = _batch([0.0, 0.5], [BOS, 0], 3.0)
    with torch.no_grad():
        log_A = m._log_remaining_mark_mass(batch)
        # Reference: predictable-renormalized marks, summed over still-active types.
        t_query = m._last_event_time(batch) + 1e-6
        _, raw = m._unpack(m._get_params_at_time(t_query, batch), dtype=torch.float64)
        pred = m._predictable_mark_log_probs(raw)  # context -> -inf
        active = m._get_termination_mask(batch)  # (1, D)
        A_ref = (pred.exp() * active.to(torch.float64)).sum(-1)
        A_raw = (raw.exp() * active.to(torch.float64)).sum(-1)  # the value without the role mask (wrong)
    assert torch.allclose(log_A.exp(), A_ref, atol=1e-9), f"A {log_A.exp()} != predictable-renorm ref {A_ref}"
    assert not torch.allclose(log_A.exp(), A_raw, atol=1e-6), "test premise: raw-mark A must differ (context leak)"


def test_iftpp_prob_mark_zero_on_context_and_occurred():
    """IFTPP mark law: zero mass on context AND on occurred predictable types; active predictable
    types renormalise to 1.  Context is never in the active set whether or not it has fired."""
    m = _ctx_model("intensity_free")
    batch = _batch([0.0, 0.5], [BOS, 0], 3.0)  # <bos> fired, type 0 fired; types 1,2 active
    with torch.no_grad():
        pm = m.prob_mark(batch, log=False)[0]  # (D,)
    assert pm[BOS].item() == 0.0, f"context <bos> must get 0 mark mass: {pm}"
    assert pm[0].item() == 0.0, f"occurred predictable type 0 must get 0 mass: {pm}"
    assert pm[1] > 0.0 and pm[2] > 0.0, f"active predictable types must keep mass: {pm}"
    assert abs(float(pm.sum()) - 1.0) < 1e-6, f"active-predictable mass must renormalise to 1: {pm.sum()}"


def test_iftpp_no_next_degeneracy_with_context():
    """All predictable types terminated (context present) ⇒ no next event ⇒ survival ≡ 1.

    The ``_log_survival_marginal`` no_next branch must key on the *predictable* mark law
    being fully exhausted, not on the raw all-type law (which the context column would
    keep alive)."""
    m = _ctx_model("intensity_free")
    # <bos>@0 + every predictable type {0,1,2} fired: no predictable next event possible.
    batch = _batch([0.0, 0.5, 1.0, 1.5], [BOS, 0, 1, 2], 3.0)
    with torch.no_grad():
        q = batch.T_obs + 0.5
        log_surv = m.survival_time(q, batch, log=True)
        lpm = m._log_prob_mark(m._scaled_ts(batch))
    assert torch.isneginf(lpm).all(), f"a fully-terminated predictable set must give all -inf marks: {lpm}"
    assert torch.isclose(log_surv[0], torch.zeros(1, dtype=torch.float64, device=DEVICE)[0], atol=1e-9), (
        f"no next event ⇒ log S = 0 (S ≡ 1): {log_surv}"
    )


def test_context_not_suppressed_is_structural_not_terminating():
    """A context type is excluded from competing risks whether or not it has 'fired'.

    Marking type BOS as context makes its mark mass zero regardless of its termination
    state, so a repeated ``<bos>`` neither reactivates nor re-suppresses anything: the
    predictable active set is identical with one vs. two ``<bos>`` events.
    """
    m = _ctx_model("intensity_free")
    one = _batch([0.0, 0.5], [BOS, 0], 3.0)
    two = _batch([0.0, 0.5, 1.0], [BOS, 0, BOS], 3.0)  # extra <bos> after type 0
    with torch.no_grad():
        pm_one = m.prob_mark(one, log=False)[0]
        pm_two = m.prob_mark(two, log=False)[0]
    # Both give zero context mass and identical *support* over predictable types (1, 2 active).
    assert pm_one[BOS].item() == 0.0 and pm_two[BOS].item() == 0.0
    assert (pm_two[[1, 2]] > 0).all() and pm_two[0].item() == 0.0
