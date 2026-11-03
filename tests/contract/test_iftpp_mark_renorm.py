"""Predictable-renormalized IFTPP mark head.

The IntensityFree mark categorical (log-softmax over all ``D`` types) is restricted to
**predictable** types: context/``<bos>`` columns are masked to ``-inf`` and the head
renormalises over the predictable columns, so it puts exactly zero probability on a type
that is never predicted.  Applied in both ``_log_prob_mark`` (the predictive mark law) and
the per-event term of ``_event_log_density_sum`` (the scored ``log p(m|H)``), sharing one
``_predictable_mark_log_probs`` helper.

Two invariants gate the change:

* **Gate A — survival unchanged.** IFTPP's inter-event survival is mark-independent
  (``S(t|m,H) = S(τ)`` for every ``m``) and the mark law sums to 1 before *and* after the
  restriction, so ``S(t|H) = Σ_m p(m)·S(τ) = S(τ)`` is invariant.  ``_log_survival_marginal``
  is left untouched and its value must not move (to machine epsilon).
* **Gate B — mark renorm correctness.** With a context type present the head assigns it
  exactly zero mass and renormalises over predictable types; with no context types the head
  equals the all-``D`` log-softmax exactly (inert).
"""

from __future__ import annotations

import pytest
import torch

from torchpp.modules.nn.encoder import GRUEncoder
from torchpp.tpp.neural import IntensityFreeProcess
from torchpp.utils.event_data import BatchedMVEventData

D_PRED = 3  # predictable types 0..2
BOS = D_PRED  # context/origin type index
D_ALL = D_PRED + 1
_EPS = 1e-15  # ~a few float64 ULP at unit magnitude


def _iftpp(d, seed=7):
    torch.manual_seed(seed)
    enc = GRUEncoder(num_event_types=d, n_embd=16, num_layers=1, dropout=0.0)
    return IntensityFreeProcess(enc, num_mix_components=4, seed=seed).double().eval()


def _origin_batch():
    """Sequences that start with a <bos> (context) event at t=0."""
    s1_t = torch.tensor([0.0, 0.4, 0.9, 1.5, 2.1, 2.8], dtype=torch.float64)
    s1_ty = torch.tensor([BOS, 0, 1, 2, 0, 1], dtype=torch.long)
    s2_t = torch.tensor([0.0, 0.6, 1.2, 1.7, 2.4, 3.0], dtype=torch.float64)
    s2_ty = torch.tensor([BOS, 2, 0, 1, 2, 0], dtype=torch.long)
    return BatchedMVEventData([s1_t, s2_t], [s1_ty, s2_ty], T_obs=[3.5, 3.5])


def _plain_batch():
    """No context types — the default all-True-mask regime."""
    s1_t = torch.tensor([0.3, 0.9, 1.6, 2.4], dtype=torch.float64)
    s1_ty = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    s2_t = torch.tensor([0.5, 1.2, 2.0], dtype=torch.float64)
    s2_ty = torch.tensor([1, 2, 0], dtype=torch.long)
    return BatchedMVEventData([s1_t, s2_t], [s1_ty, s2_ty], T_obs=[3.5, 3.5])


# ---------------------------------------------------------------------------
# Gate A — the time-marginal survival is mark-independent and untouched
# ---------------------------------------------------------------------------


def test_survival_is_mark_independent_and_invariant_to_mark_restriction():
    """``_log_survival_marginal`` equals the pure per-mark ``log S(τ)`` and does not move
    when a context type is masked out of the head (Gate A, the key gate).

    ``S(t|H) = Σ_m p(m)·S(t|m,H)``; for IFTPP every ``S(t|m,H)`` is the same ``S(τ)`` and
    ``Σ_m p(m) = 1`` both before and after the restriction, so the marginal survival is the
    mark-independent ``S(τ)`` — the mark renormalization cannot alter it.
    """
    model = _iftpp(D_ALL)
    ts = model._scaled_ts(_origin_batch())
    t = torch.tensor([[3.1, 3.4], [3.2, 3.5]], dtype=torch.float64)  # (B, Q), ≥ last event

    with torch.no_grad():
        log_S = model._log_survival_time_given_mark(t, ts)[..., 0]  # pure time survival (any mark)
        surv_all_true = model._log_survival_marginal(t, ts)  # default all-True mask
        assert torch.allclose(surv_all_true, log_S, atol=_EPS), "survival must be the mark-independent S(τ)"

        # Restrict the head to predictable types; the marginal survival must not move.
        model.has_origin = True
        model.predictable_mask[BOS] = False
        surv_masked = model._log_survival_marginal(t, ts)

    assert torch.allclose(surv_masked, log_S, atol=_EPS)
    delta = (surv_masked - surv_all_true).abs().max().item()
    assert delta <= _EPS, f"survival moved by {delta:.2e} (> ~1 ULP) — mark restriction leaked into the time law"


def test_full_survival_paths_untouched_by_mark_restriction():
    """The public survival / CDF (native shell) are invariant to the mark restriction."""
    model = _iftpp(D_ALL)
    batch = _origin_batch()
    t = torch.tensor([3.7, 3.8], dtype=torch.float64)  # after T_obs = 3.5
    with torch.no_grad():
        s0 = model.survival_time(t, batch, log=True)
        c0 = model.cdf_time(t, batch, log=True)
        model.has_origin = True
        model.predictable_mask[BOS] = False
        s1 = model.survival_time(t, batch, log=True)
        c1 = model.cdf_time(t, batch, log=True)
    assert torch.allclose(s1, s0, atol=_EPS) and torch.allclose(c1, c0, atol=_EPS)


# ---------------------------------------------------------------------------
# Gate B — mark renorm correctness (context present) + inertness (no context)
# ---------------------------------------------------------------------------


def test_mark_head_zeros_context_and_renormalises_over_predictable():
    """With a context type present, ``_log_prob_mark`` gives it exactly zero probability
    (``-inf`` log-prob) and sums to 1 over the predictable types."""
    model = _iftpp(D_ALL)
    model.has_origin = True
    model.predictable_mask[BOS] = False
    ts = model._scaled_ts(_origin_batch())
    with torch.no_grad():
        lm = model._log_prob_mark(ts)  # (B, D_ALL)

    assert torch.isneginf(lm[:, BOS]).all(), "context type must get exactly zero mass (-inf)"
    pred = [d for d in range(D_ALL) if d != BOS]
    mass = lm[:, pred].exp().sum(dim=-1)
    assert torch.allclose(mass, torch.ones_like(mass), atol=_EPS), "predictable mass must sum to 1"


def test_scored_per_event_mark_uses_same_renorm_as_log_prob_mark():
    """The per-event ``log p(m|H)`` in ``_event_log_density_sum`` and the predictive
    ``_log_prob_mark`` are the *same* predictable-renormalized distribution (consistency).

    Both route through ``_predictable_mark_log_probs``; here we check the operation is
    applied at both sites — the per-event mark matrix zeros the context column and each row
    sums to 1 over predictable types, and ``_log_prob_mark`` equals the helper applied to the
    full-history mark column.
    """
    model = _iftpp(D_ALL)
    model.has_origin = True
    model.predictable_mask[BOS] = False
    ts = model._scaled_ts(_origin_batch())
    pred = [d for d in range(D_ALL) if d != BOS]

    with torch.no_grad():
        # Per-event marks, as gathered inside _event_log_density_sum.
        full = model._full_params(ts)
        col_idx = ts.causal_indices
        idx = col_idx.unsqueeze(-1).expand(*col_idx.shape, full.shape[-1])
        _, raw = model._unpack(full.gather(1, idx), dtype=torch.float64)  # (B, L, D_ALL)
        per_event = model._predictable_mark_log_probs(raw)
        assert torch.isneginf(per_event[:, :, BOS]).all()
        row_mass = per_event[:, :, pred].exp().sum(dim=-1)  # (B, L)
        assert torch.allclose(row_mass, torch.ones_like(row_mass), atol=_EPS)

        # _log_prob_mark IS the helper applied to the full-history mark column.
        t_query = model._last_event_time(ts) + 1e-6
        _, raw_full = model._unpack(model._get_params_at_time(t_query, ts), dtype=torch.float64)
        assert torch.equal(model._log_prob_mark(ts), model._predictable_mark_log_probs(raw_full))


def test_mark_renorm_inert_under_all_true_mask():
    """No context types → the head is bit-identical to the raw ``_unpack`` log-softmax over
    all ``D`` (the unrenormalized distribution).  The all-True fast path returns the input object."""
    model = _iftpp(D_PRED)  # all-True mask by construction
    ts = model._scaled_ts(_plain_batch())
    with torch.no_grad():
        t_query = model._last_event_time(ts) + 1e-6
        _, raw = model._unpack(model._get_params_at_time(t_query, ts), dtype=torch.float64)
        assert torch.equal(model._log_prob_mark(ts), raw), "all-True mask must be a no-op on the mark head"


def test_context_mark_restriction_raises_scored_predictable_mark_terms():
    """Objective check: removing context mass raises the scored predictable ``log p(m|H)``
    (the density sum increases), while the plain (no-context) density sum is unchanged.

    Sanity that the renorm actually bites where it should and nowhere else.
    """
    # Plain model: density sum identical with an explicit all-True mask (inert).
    plain = _iftpp(D_PRED)
    ts_p = plain._scaled_ts(_plain_batch())
    with torch.no_grad():
        base = plain._event_log_density_sum(ts_p)
    plain.predictable_mask.copy_(torch.ones(D_PRED, dtype=torch.bool))
    with torch.no_grad():
        assert torch.equal(plain._event_log_density_sum(ts_p), base)

    # Origin model: masking BOS moves mass onto predictable types → larger scored log p(m).
    model = _iftpp(D_ALL)
    ts = model._scaled_ts(_origin_batch())
    with torch.no_grad():
        before = model._event_log_density_sum(ts)
        model.has_origin = True
        model.predictable_mask[BOS] = False
        after = model._event_log_density_sum(ts)
    assert (after > before).all(), "removing context mark mass must raise the scored density sum"
