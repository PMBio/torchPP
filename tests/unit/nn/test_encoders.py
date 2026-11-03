"""Direct unit tests for the sequence encoders in ``torchpp.modules.nn``.

Each encoder is constructed with small dims and fed a ``BatchedMVEventData``
from :func:`build_test_data`. We assert the documented contract directly:

- output SHAPE is ``(B, L, n_embd)``;
- CAUSALITY: per each encoder's documented convention, the encoding at an early
  position does not depend on a strictly-later event (verified by perturbing a
  future event type and checking earlier positions are unchanged);
- PADDING: trailing padding positions (type ``-1``) leave the outputs at valid
  positions unchanged (padded vs. unpadded);
- finiteness.

Causality conventions (from the source):
- ``GRUEncoder``       — auto-regressive RNN: position ``l`` sees events ``<= l``.
- ``ContTimeLSTMEncoder`` — CT-LSTM recurrence: position ``l`` sees events ``<= l``.
- ``TransformerEncoder`` — causal self-attention with ``>=`` time mask: position
  ``l`` attends to events at time ``<= t_l`` (includes simultaneous).
- ``AttNHPEncoder``    — ``forward`` is a per-event embedding (no mixing); the
  intensity states come from ``states_at_queries`` under a *strictly before*
  query convention, tested separately here.
"""

from __future__ import annotations

import torch

from helpers import DEVICE, D, build_test_data
from torchpp.modules.nn.encoder import GRUEncoder, ContTimeLSTMEncoder
from torchpp.modules.nn.transformer import TransformerConfig, TransformerEncoder
from torchpp.modules.nn.attnhp import AttNHPEncoder
from torchpp.utils.event_data import BatchedMVEventData


HIDDEN = 16


def _build(enc_name: str):
    """Construct an encoder by short name, on DEVICE, with small dims."""
    if enc_name == "gru":
        return GRUEncoder(num_event_types=D, n_embd=HIDDEN, num_layers=1, dropout=0.0).to(DEVICE).eval()
    if enc_name == "ctlstm":
        return ContTimeLSTMEncoder(num_event_types=D, hidden_dim=HIDDEN).to(DEVICE).eval()
    if enc_name == "transformer":
        cfg = TransformerConfig(num_event_types=D, n_layer=2, n_head=2, n_embd=HIDDEN, dropout=0.0)
        return TransformerEncoder(cfg).to(DEVICE).eval()
    raise ValueError(enc_name)


def _expected_embd(enc_name: str) -> int:
    if enc_name == "ctlstm":
        return 4 * HIDDEN
    return HIDDEN


# ---------------------------------------------------------------------------
# Shape + finiteness (forward-based encoders)
# ---------------------------------------------------------------------------


class TestEncoderShapeFinite:
    def _check(self, enc_name: str):
        enc = _build(enc_name)
        ts = build_test_data(DEVICE)["batch_multi"]  # B=2, padded L
        B, L = ts.time_points.shape
        out = enc(ts)
        assert out.shape == (B, L, _expected_embd(enc_name))
        assert torch.isfinite(out).all()

    def test_gru(self):
        self._check("gru")

    def test_ctlstm(self):
        self._check("ctlstm")

    def test_transformer(self):
        self._check("transformer")

    def test_empty_sequence_zero_length(self):
        """L==0 batches return an empty (B, 0, n_embd) tensor for all encoders."""
        empty = build_test_data(DEVICE)["empty_batch"]
        for name in ("gru", "ctlstm", "transformer"):
            out = _build(name)(empty)
            assert out.shape[0] == empty.time_points.shape[0]
            assert out.shape[1] == 0
            assert out.shape[2] == _expected_embd(name)


# ---------------------------------------------------------------------------
# Causality: perturbing a future event must not change earlier-position outputs
# ---------------------------------------------------------------------------


def _perturb_future(ts: BatchedMVEventData, pos: int) -> BatchedMVEventData:
    """Copy ``ts`` but change the event TYPE at ``pos`` (a strictly-later event).

    Times are untouched so the only difference is the identity of a future event.
    """
    times = [ts.time_points[i].clone() for i in range(ts.time_points.shape[0])]
    types = [ts.event_types[i].clone() for i in range(ts.event_types.shape[0])]
    # Flip the type at pos to a different valid type.
    old = int(types[0][pos].item())
    types[0][pos] = (old + 1) % D
    return BatchedMVEventData(times, types).to(DEVICE)


class TestEncoderCausality:
    def _check(self, enc_name: str, split: int):
        """Positions ``< split`` must be identical after perturbing event ``split``."""
        ts = build_test_data(DEVICE)["batch_single"]  # single seq, 4 events, no padding
        enc = _build(enc_name)
        base = enc(ts)
        pert = enc(_perturb_future(ts, split))
        # Earlier positions unchanged; the perturbed (and later) positions may move.
        torch.testing.assert_close(base[:, :split], pert[:, :split], rtol=1e-5, atol=1e-6)

    def test_gru(self):
        # RNN: state at l<split independent of the type fed at >=split.
        self._check("gru", split=2)

    def test_ctlstm(self):
        # CT-LSTM recurrence: same causal guarantee as the GRU.
        self._check("ctlstm", split=2)

    def test_transformer(self):
        # Causal attention with strictly-increasing times here, so position l<split
        # attends only to events at time <= t_l, none of which is the perturbed one.
        self._check("transformer", split=2)


# ---------------------------------------------------------------------------
# Padding: trailing pad positions don't change outputs at valid positions
# ---------------------------------------------------------------------------


class TestEncoderPadding:
    def _check(self, enc_name: str):
        data = build_test_data(DEVICE)
        padded = data["padded_batch"]  # 4 real events + 2 pads (type -1)
        n_valid = int(padded.event_mask[0].sum().item())
        # Unpadded counterpart: same 4 real events, no trailing pads.
        unpadded = BatchedMVEventData(
            [padded.time_points[0, :n_valid].clone()],
            [padded.event_types[0, :n_valid].clone()],
        ).to(DEVICE)
        enc = _build(enc_name)
        out_pad = enc(padded)
        out_unpad = enc(unpadded)
        torch.testing.assert_close(
            out_pad[:, :n_valid], out_unpad[:, :n_valid], rtol=1e-5, atol=1e-6
        )
        assert torch.isfinite(out_pad).all()

    def test_gru(self):
        self._check("gru")

    def test_ctlstm(self):
        self._check("ctlstm")

    def test_transformer(self):
        self._check("transformer")


# ---------------------------------------------------------------------------
# AttNHPEncoder — forward is a per-event embedding; states come from queries.
# ---------------------------------------------------------------------------


def _attnhp(d=D):
    return (
        AttNHPEncoder(num_event_types=d, d_model=HIDDEN, d_time=8, n_layers=2, n_head=2, dropout=0.0)
        .to(DEVICE)
        .eval()
    )


class TestAttNHPEncoder:
    def test_forward_shape_finite(self):
        """forward(ts) -> (B, L, d_model + d_time) per-event embedding."""
        enc = _attnhp()
        ts = build_test_data(DEVICE)["batch_multi"]
        B, L = ts.time_points.shape
        out = enc(ts)
        assert out.shape == (B, L, enc.n_embd)
        assert out.shape[2] == HIDDEN + 8
        assert torch.isfinite(out).all()

    def test_states_at_queries_shape_finite(self):
        """states_at_queries -> (B, Q, d_model * n_head)."""
        enc = _attnhp()
        ts = build_test_data(DEVICE)["batch_single"]
        emb = enc(ts)
        q = ts.time_points + 0.05  # query just after each event
        states = enc.states_at_queries(emb, ts.time_points, ts.event_mask, q)
        assert states.shape == (ts.time_points.shape[0], q.shape[1], enc.state_dim)
        assert torch.isfinite(states).all()

    def test_query_causality_strictly_before(self):
        """A query state attends only to events strictly before its time.

        Perturbing a later event's type leaves the state of an earlier query
        (placed before that event) unchanged.
        """
        enc = _attnhp()
        ts = build_test_data(DEVICE)["batch_single"]  # times 0.2,0.8,2.0,2.3
        # Query at t=0.9 sees events at 0.2,0.8 (strictly before), not 2.0/2.3.
        q = torch.tensor([[0.9]], dtype=torch.float64, device=DEVICE)

        emb = enc(ts)
        base = enc.states_at_queries(emb, ts.time_points, ts.event_mask, q)

        pert = _perturb_future(ts, pos=2)  # change the t=2.0 event type
        emb_p = enc(pert)
        state_p = enc.states_at_queries(emb_p, pert.time_points, pert.event_mask, q)

        torch.testing.assert_close(base, state_p, rtol=1e-5, atol=1e-6)

    def test_query_padding_invariance(self):
        """Trailing padded events don't change the query states at valid times."""
        enc = _attnhp()
        data = build_test_data(DEVICE)
        padded = data["padded_batch"]
        n_valid = int(padded.event_mask[0].sum().item())
        unpadded = BatchedMVEventData(
            [padded.time_points[0, :n_valid].clone()],
            [padded.event_types[0, :n_valid].clone()],
        ).to(DEVICE)

        q = torch.tensor([[1.0, 2.4]], dtype=torch.float64, device=DEVICE)

        s_pad = enc.states_at_queries(enc(padded), padded.time_points, padded.event_mask, q)
        s_unpad = enc.states_at_queries(
            enc(unpadded), unpadded.time_points, unpadded.event_mask, q
        )
        torch.testing.assert_close(s_pad, s_unpad, rtol=1e-5, atol=1e-6)
        assert torch.isfinite(s_pad).all()
