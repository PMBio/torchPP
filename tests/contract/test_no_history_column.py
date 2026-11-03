"""No-history column + no-origin condition-on-first.

There is no learned per-head prior — ``_full_params`` column 0 is a zeros column —
so the "no-events" state is realized by ``encoder(<bos>)`` (with an origin) or is never
used (without one).  Two regimes, switched by the data/task flag ``has_origin``:

* **origin (EHR, ``<bos>`` at t=0):** the first real event is scored and the gap ``[0, t₁]``
  is integrated; ``<bos>`` is a context-masked, zero-width row that contributes nothing.
* **no origin:** the first event is *conditioned on* — dropped from the positive
  term and the survival window starts at ``t₁`` — matching ``utils/evaluation.py``.

Gate A (``test_bos_embedding_gradient_flows_unattenuated``) is the load-bearing one: it
encodes the lesson that sank a prior attempt (a raw post-``ln_f`` prior vector received
badly attenuated gradients).  ``<bos>`` is now an ENCODER-INTERNAL learned vector
(``bos_emb``) for the attention encoders (transformer / AttNHP): it is routed through the
attention stack (+ ``ln_f`` for the transformer) as an ordinary token, so its gradient must
be O(1) relative to the type-embedding rows.  The recurrent encoders (GRU / CT-LSTM) realize
the origin as a deterministic zero initial state with no learned ``<bos>`` parameter, so the
gradient regression applies to the attention encoders only.
"""

from __future__ import annotations

import pytest
import torch

from helpers import create_model, DEVICE
from torchpp.modules.nn.transformer import TransformerConfig
from torchpp.modules.nn.attnhp import AttNHPEncoder
from torchpp.tpp.neural import NeuralPoissonProcess, AttNHPProcess
from torchpp.utils.event_data import BatchedMVEventData, MVEventData, TokenRoles
from torchpp.utils.evaluation import _build_prediction_items


D_PRED = 3  # predictable event types (0..2); <bos> is encoder-internal, NOT a data token


# ---------------------------------------------------------------------------
# has_origin model + batch builders.  <bos> is encoder-internal (bos_emb), so the
# models are built with ``has_origin=True`` and the batches carry only real events —
# the origin never appears as an event type.
# ---------------------------------------------------------------------------


def _transformer_origin_model():
    torch.manual_seed(7)
    cfg = TransformerConfig(num_event_types=D_PRED, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    return NeuralPoissonProcess(config=cfg, seed=7, token_roles=TokenRoles(has_origin=True)).to(DEVICE)


def _attnhp_origin_model():
    torch.manual_seed(7)
    enc = AttNHPEncoder(num_event_types=D_PRED, d_model=16, d_time=8, n_layers=1, n_head=2, has_origin=True)
    return AttNHPProcess(enc, seed=7, token_roles=TokenRoles(has_origin=True)).to(DEVICE)


# (builder, type-embedding table getter, bos_emb getter).  F2 is attention-encoders
# only (the encoders that own a learned <bos> vector).
_F2_MODELS = {
    "transformer_poisson": (
        _transformer_origin_model,
        lambda m: m.encoder.transformer.mods.event_emb.weight,
        lambda m: m.encoder.transformer.bos_emb,
    ),
    "attnhp": (
        _attnhp_origin_model,
        lambda m: m.encoder.type_emb.weight,
        lambda m: m.encoder.bos_emb,
    ),
}


def _origin_batch():
    """A has_origin batch of real predictable events (no data-level <bos> token)."""
    s1_t = torch.tensor([0.4, 0.9, 1.5, 2.1, 2.8], dtype=torch.float64, device=DEVICE)
    s1_ty = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long, device=DEVICE)
    s2_t = torch.tensor([0.6, 1.2, 1.7, 2.4, 3.0], dtype=torch.float64, device=DEVICE)
    s2_ty = torch.tensor([2, 0, 1, 2, 0], dtype=torch.long, device=DEVICE)
    return BatchedMVEventData([s1_t, s2_t], [s1_ty, s2_ty], T_obs=[3.5, 3.5]).to(DEVICE)


# ===========================================================================
# Gate A — the <bos> embedding gradient must flow unattenuated
# ===========================================================================


@pytest.mark.parametrize("name", list(_F2_MODELS))
def test_bos_embedding_gradient_flows_unattenuated(name):
    """After a few optimizer steps, ‖grad[bos_emb]‖ is O(1) vs the median type-embedding row.

    The encoder-internal ``<bos>`` is routed through the attention stack (+ ``ln_f`` for the
    transformer) as an ordinary token, so its learned ``bos_emb`` gradient is comparable to
    any type-embedding row's — NOT the ~50× attenuation a raw post-``ln_f`` prior suffered.
    """
    build, type_emb_of, bos_emb_of = _F2_MODELS[name]
    model = build()
    model.train()
    batch = _origin_batch()

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(6):
        opt.zero_grad()
        loss = -model.likelihood(batch).mean()
        loss.backward()
        if step < 5:
            opt.step()

    type_grad = type_emb_of(model).grad  # (num_types + 1, n_embd)
    bos_grad = bos_emb_of(model).grad  # (n_embd,)
    assert type_grad is not None, f"{name}: type embedding received no gradient"
    assert bos_grad is not None, f"{name}: bos_emb received no gradient at all"

    # Median over type rows that actually received gradient (types present in the batch);
    # the never-indexed padding row stays ~0 and would bias the median downward.
    row_norms = type_grad.norm(dim=1)  # (rows,)
    active = row_norms > 0
    median = row_norms[active].median()
    bos_norm = bos_grad.norm()

    assert bos_norm > 0, f"{name}: bos_emb grad is zero — gradient is not reaching it"
    ratio = (bos_norm / median).item()
    assert 0.1 <= ratio <= 10.0, (
        f"{name}: bos_emb grad-norm {bos_norm.item():.3e} is {ratio:.2f}× the median "
        f"type-embedding row {median.item():.3e} — outside the O(1) band [0.1, 10] "
        f"(attenuation regression)"
    )


# ===========================================================================
# Anti-leak (I6) — the origin state must not see the first real event
# ===========================================================================


def _origin_batch_variant(first_types, first_times):
    """``_origin_batch`` with the two sequences' first event type/time overridden."""
    s1_t = torch.tensor([first_times[0], 0.9, 1.5, 2.1, 2.8], dtype=torch.float64, device=DEVICE)
    s1_ty = torch.tensor([first_types[0], 1, 2, 0, 1], dtype=torch.long, device=DEVICE)
    s2_t = torch.tensor([first_times[1], 1.2, 1.7, 2.4, 3.0], dtype=torch.float64, device=DEVICE)
    s2_ty = torch.tensor([first_types[1], 0, 1, 2, 0], dtype=torch.long, device=DEVICE)
    return BatchedMVEventData([s1_t, s2_t], [s1_ty, s2_ty], T_obs=[3.5, 3.5]).to(DEVICE)


def test_transformer_bos_params_do_not_leak_first_event():
    """Position-0 (``<bos>``) params are ``head(bos-state)``.  ``<bos>`` is placed strictly
    before the first event and (inclusive mask) attends only to itself, so with ``t₁ > 0``
    (⇒ ``t_bos ≡ 0`` fixed) they cannot depend on the first event's own type OR time."""
    model = _transformer_origin_model()
    model.eval()
    base = _origin_batch_variant([0, 2], [0.4, 0.6])
    # Retype AND retime the first event of both sequences (times kept > 0 ⇒ t_bos unchanged).
    mutated = _origin_batch_variant([1, 1], [0.15, 0.20])
    p0 = model._full_params(base)[:, 0].detach()
    p0_mut = model._full_params(mutated)[:, 0].detach()
    assert torch.allclose(p0, p0_mut, atol=1e-6), "<bos> position-0 params leaked the first real event"


def test_attnhp_first_event_intensity_does_not_leak_its_own_type():
    """AttNHP scores the first event by re-attending to events strictly before ``t₁`` (= just
    ``<bos>``), so the first event's own type cannot enter its own intensity."""
    model = _attnhp_origin_model()
    model.eval()
    base = _origin_batch_variant([0, 2], [0.4, 0.6])
    mutated = _origin_batch_variant([1, 1], [0.4, 0.6])  # first type changed, first time identical
    lam = model._sequence_intensities(base)[:, 0, :].detach()
    lam_mut = model._sequence_intensities(mutated)[:, 0, :].detach()
    assert torch.allclose(lam, lam_mut, atol=1e-6), "AttNHP first-event intensity leaked its own type"


# ===========================================================================
# Gate B — train↔eval consistency for the no-origin (condition-on-first) regime
# ===========================================================================


def test_no_origin_scored_set_matches_eval_targets():
    """The training positive term scores exactly the events ``evaluation.py`` targets.

    ``_build_prediction_items`` targets indices ``1..len-1`` (the first event is history
    only).  For a no-origin, all-predictable batch the training ``_scored_mask`` must be
    True at exactly those positions and False at position 0 — a structural (index-set)
    match, not a value match.
    """
    times = torch.tensor([0.3, 0.8, 1.4, 2.0, 2.6], dtype=torch.float64, device=DEVICE)
    types = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long, device=DEVICE)
    seq = MVEventData(times, types).to(DEVICE)
    batch = BatchedMVEventData([times], [types], T_obs=[3.0]).to(DEVICE)

    model = create_model("poisson", seed=1)  # has_origin=False by default (no-origin regime)
    assert model.has_origin is False

    scored = model._scored_mask(batch)[0]  # (L,)
    train_scored = set(torch.nonzero(scored, as_tuple=False).flatten().tolist())

    eval_targets = {item["history_length"] for item in _build_prediction_items([seq])}
    # history_length == the target index (history = events[:idx], target = events[idx]).
    assert eval_targets == set(range(1, len(seq))), "eval targets must be indices 1..len-1"
    assert train_scored == eval_targets, (
        f"training scored set {sorted(train_scored)} != eval target set {sorted(eval_targets)}"
    )
    assert 0 not in train_scored, "position 0 (conditioned-on first event) must be dropped"


def test_no_origin_survival_starts_at_first_event():
    """No-origin survival lower bound is t₁ (condition on the first event), not 0."""
    times = torch.tensor([0.7, 1.3, 2.2], dtype=torch.float64, device=DEVICE)
    types = torch.tensor([0, 1, 2], dtype=torch.long, device=DEVICE)
    batch = BatchedMVEventData([times], [types], T_obs=[3.0]).to(DEVICE)

    # log-linear: the zeros no-history column maps to a nonzero rate exp(0)=1, so the
    # [0, t₁] gap carries real mass iff T_low is NOT applied — a sharp probe of the window.
    model = create_model("neural_log_linear", seed=2)
    model.eval()
    with model.fix_context(model._scaled_ts(batch)):
        T = torch.tensor([3.0], dtype=torch.float64, device=DEVICE)
        t1 = batch.time_points[:, 0]  # first event time
        ci_from0 = model._cumulative_intensity_scaled(T, model._scaled_ts(batch), T_low=None).sum()
        ci_from_t1 = model._cumulative_intensity_scaled(T, model._scaled_ts(batch), T_low=t1).sum()
    excluded = (ci_from0 - ci_from_t1).item()
    assert excluded > 1e-3, "the [0, t₁] zeros-column mass must be nonzero (else the probe is vacuous)"
    # And the likelihood path uses the t₁ lower bound: recompute the scaled LL and check it
    # equals positive − Σ_d Λ_d over [t₁, T] (not over [0, T]).
    with model.fix_context(model._scaled_ts(batch)):
        ll = model._likelihood_scaled(model._scaled_ts(batch), torch.tensor([3.0], dtype=torch.float64, device=DEVICE))
        pos = model._event_log_intensity_sum(model._scaled_ts(batch))
        ll_from_t1 = pos - ci_from_t1
    assert torch.allclose(ll, ll_from_t1, atol=1e-6), "no-origin LL must integrate survival from t₁"


# ===========================================================================
# Gate C — invariants across families
# ===========================================================================

_INVARIANT_MODELS = ["poisson", "neural_poisson", "neural_log_linear", "nhp", "intensity_free"]


@pytest.mark.parametrize("model_type", _INVARIANT_MODELS)
def test_likelihood_finite_and_grads_flow(model_type):
    """Well-fit params → finite LL; every trainable parameter receives a gradient."""
    times = torch.tensor([0.3, 0.9, 1.6, 2.4], dtype=torch.float64, device=DEVICE)
    types = torch.tensor([0, 1, 2, 0], dtype=torch.long, device=DEVICE)
    batch = BatchedMVEventData([times], [types], T_obs=[3.0]).to(DEVICE)

    model = create_model(model_type, seed=3)
    model.train()
    ll = model.likelihood(batch)
    assert torch.isfinite(ll).all(), f"{model_type}: non-finite likelihood"

    (-ll.sum()).backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"{model_type}: no gradient reached {missing}"
    total = sum(float(p.grad.abs().sum()) for _, p in model.named_parameters() if p.grad is not None)
    assert total > 0, f"{model_type}: all gradients are exactly zero"


@pytest.mark.parametrize("model_type", ["neural_poisson", "nhp", "intensity_free"])
def test_decomposition_identity(model_type):
    """joint_ll ≈ time_ll + cond_type_ll (the log-space next-event decomposition)."""
    times = torch.tensor([0.3, 0.9, 1.6], dtype=torch.float64, device=DEVICE)
    types = torch.tensor([0, 1, 2], dtype=torch.long, device=DEVICE)
    batch = BatchedMVEventData([times], [types]).to(DEVICE)

    model = create_model(model_type, seed=3)
    model.eval()
    t = torch.tensor([2.3], dtype=torch.float64, device=DEVICE)
    with torch.no_grad():
        joint = model.density_joint(t, batch, log=True)  # (B, D)
        time_ll = model.density_time(t, batch, log=True)  # (B,)
        cond = model.prob_mark_given_time(t, batch, log=True)  # (B, D)
    lhs = joint
    rhs = time_ll.unsqueeze(-1) + cond
    assert torch.allclose(lhs, rhs, atol=1e-5), f"{model_type}: joint != time + cond|time"


def test_origin_first_real_event_is_scored_and_survival_from_zero():
    """has_origin: the first real event IS scored (not conditioned away) and the survival
    integrates from 0 (the [0, t₁] gap is real).  ``<bos>`` is encoder-internal — no data token."""
    model = _transformer_origin_model()
    model.eval()
    batch = _origin_batch()
    scored = model._scored_mask(batch)
    # Every real event is scored under an origin, including the first (position 0) — the
    # no-origin condition-on-first drop does NOT apply.
    assert (scored[:, 0] == True).all(), "first real event must be scored under an origin"
    assert int(scored.sum()) == int(batch.event_mask.sum()), "all real (predictable) events scored"

    # has_origin uses no T_low: the survival integrates from 0.
    with model.fix_context(model._scaled_ts(batch)):
        T = batch.T_obs.to(torch.float64)
        ci = model._cumulative_intensity_scaled(T, model._scaled_ts(batch), T_low=None)
    assert torch.isfinite(ci).all() and (ci >= 0).all()
    ll = model.likelihood(batch)
    assert torch.isfinite(ll).all()
