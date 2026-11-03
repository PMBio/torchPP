"""NHP numerics guards: finite on pathological batches, unchanged on healthy ones.

Three guards, each far outside the healthy operating range so they are
no-ops on well-behaved inputs:

- CT-LSTM encoder: right-limit states are finite-clamped (a bare ``nan_to_num``
  maps +inf to the largest float, which causes ``inf - inf`` in the decay
  ``c_bar + (c - c_bar)·exp(-δ·Δt)``).
- ``ScaledSoftplus``: ``log_beta`` is lower-bounded so ``1/beta`` cannot overflow.
- ``NHPProcess``: the intensity is clamped to a finite max before log / CI.
"""

from __future__ import annotations

import torch

import torchpp.modules.nn.encoder as encoder_mod
import torchpp.tpp.neural as neural_mod
from torchpp.modules.nn.encoder import ContTimeLSTMEncoder
from torchpp.tpp.neural import NHPProcess
from torchpp.utils.event_data import BatchedMVEventData


def _extreme_batch(D=5, L=316):
    """Long, float64, near-zero-gap sequence (ICU-record-like) with a few larger gaps."""
    gaps = torch.full((L,), 1e-7, dtype=torch.float64)
    gaps[::20] = 2.0
    t = torch.cumsum(gaps, dim=0)
    types = (torch.arange(L) % D).long()
    return BatchedMVEventData([t], [types])


def _build_nhp(D=5, hidden=16, seed=0, weight_scale=1.0, log_beta=None):
    torch.manual_seed(seed)
    enc = ContTimeLSTMEncoder(num_event_types=D, hidden_dim=hidden).double()
    model = NHPProcess(enc, terminating=False).double()
    if weight_scale != 1.0 or log_beta is not None:
        with torch.no_grad():
            if weight_scale != 1.0:
                for p in model.parameters():
                    p.mul_(weight_scale)
            if log_beta is not None:
                model.softplus.log_beta.fill_(log_beta)
    return model


def test_extreme_batch_forward_backward_stays_finite():
    """Pathological batch (long seq, near-zero gaps, large weights, driven-negative
    log_beta) yields a finite loss and finite gradients (no NaN grad)."""
    model = _build_nhp(weight_scale=3.0, log_beta=-1000.0)
    batch = _extreme_batch()

    ll = model.likelihood(batch)
    loss = -ll.sum()
    assert torch.isfinite(loss).all(), "loss is non-finite on the extreme batch"

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    for g in grads:
        assert torch.isfinite(g).all(), "non-finite gradient on the extreme batch"


def test_encoder_states_finite_and_bounded_on_extreme_batch():
    """The CT-LSTM encoder emits finite, in-bound states even on the extreme batch."""
    model = _build_nhp(weight_scale=3.0, log_beta=-1000.0)
    batch = _extreme_batch()
    with torch.no_grad():
        states = model.encoder(batch)  # (B, L, 4H)
    assert torch.isfinite(states).all()
    assert states.abs().max().item() <= encoder_mod._CT_LSTM_STATE_BOUND


def test_normal_batch_unchanged_vs_unguarded():
    """On a normal benchmark-scale batch the guards are no-ops: the likelihood is
    identical to the computation with the clamps disabled."""
    model = _build_nhp(seed=1)  # untouched weights, log_beta = 0
    t = torch.tensor([0.2, 0.5, 1.1, 1.8, 2.4, 3.0], dtype=torch.float64)
    ty = torch.tensor([0, 1, 2, 3, 4, 0]).long()
    batch = BatchedMVEventData([t], [ty])

    ll_fixed = model.likelihood(batch)

    # Disable the two tunable bounds; the log_beta clamp is an identity here
    # (log_beta = 0 > -30), so this is the unguarded computation.
    assert model.softplus.log_beta.min().item() > -30.0
    orig_int = neural_mod._NHP_INTENSITY_MAX
    orig_state = encoder_mod._CT_LSTM_STATE_BOUND
    try:
        neural_mod._NHP_INTENSITY_MAX = float("inf")
        encoder_mod._CT_LSTM_STATE_BOUND = float("inf")
        ll_prefix = model.likelihood(batch)
    finally:
        neural_mod._NHP_INTENSITY_MAX = orig_int
        encoder_mod._CT_LSTM_STATE_BOUND = orig_state

    assert torch.equal(ll_fixed, ll_prefix)
    assert torch.isfinite(ll_fixed).all()
