# ContTimeLSTMCell is adapted from EasyTPP (https://github.com/ant-research/EasyTemporalPointProcess), Copyright 2022
# The EasyTPP Authors, Apache License 2.0; modified for TorchPP. See THIRD_PARTY_NOTICES.md.
"""Abstract base class for sequence encoders used by neural TPPs.

Any encoder that maps a :class:`~torchpp.utils.event_data.BatchedMVEventData`
history to a dense context embedding can be plugged into
:class:`~torchpp.tpp.neural.NeuralParametrizedTPP` by subclassing
:class:`SequenceEncoder`.

Contract
--------
- ``n_embd``           — dimensionality of the output embedding vectors.
- ``num_event_types``  — number of distinct event types the encoder was built for.
- ``forward(ts)``      — maps a batched event sequence to ``(B, L, n_embd)``
                         context embeddings, where ``B`` is the batch size and
                         ``L`` is the (padded) sequence length.

The output at position ``l`` should encode the history up to and including
event ``l`` (i.e. causal / auto-regressive context), so that the TPP head can
read off parameters valid in the interval *after* event ``l``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from torchpp.utils.event_data import BatchedMVEventData
from torchpp.utils.init_record import records_init

# Finite bound for CT-LSTM right-limit states: far above healthy values (O(tens)) yet small
# enough that ``c - c_bar`` in the decay ``c_bar + (c - c_bar)·exp(-δ·Δt)`` stays finite
# (mapping inf to the largest float would make it ``inf - inf``).
_CT_LSTM_STATE_BOUND = 1e30


class SequenceEncoder(nn.Module, ABC):
    """Abstract base class for history encoders in neural TPPs.

    Subclasses implement :meth:`forward`, which maps a batch of shape ``(B, L)`` to context
    embeddings of shape ``(B, L, n_embd)``, or ``(B, L + 1, n_embd)`` when the encoder prepends
    an origin token (then :attr:`prepends_origin` is ``True``).

    Args:
        n_embd: Dimensionality of the output context embedding.
        num_event_types: Number of distinct event types the encoder handles.
    """

    _tpp_recorded = True  # constructor arguments are recorded for checkpoints
    prepends_origin: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        records_init(cls)


    def __init__(self, n_embd: int, num_event_types: int) -> None:
        super().__init__()
        self._n_embd = n_embd
        self._num_event_types = num_event_types

    @property
    def n_embd(self) -> int:
        """Output embedding dimensionality."""
        return self._n_embd

    @property
    def num_event_types(self) -> int:
        """Number of distinct event types."""
        return self._num_event_types

    @abstractmethod
    def forward(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Encode event history into context embeddings.

        Args:
            ts: Batched event sequences of shape ``(B, L)``.

        Returns:
            Context tensor of shape ``(B, L, n_embd)``.  Position ``l``
            should reflect the history up to and including event ``l``.
        """


def _inter_event_deltas(ts: BatchedMVEventData) -> torch.Tensor:
    """Per-position inter-event time ``dt_l = t_l - t_{l-1}`` (``dt_0 = t_0``).

    Shape ``(B, L)``, non-negative.  Padding positions are left as-is; they are
    never selected as the causal column of a real event downstream.
    """
    tp = ts.time_points  # (B, L)
    if tp.shape[1] == 0:
        return tp
    prev = torch.cat([torch.zeros_like(tp[:, :1]), tp[:, :-1]], dim=1)  # (B, L)
    return (tp - prev).clamp(min=0.0)


class GRUEncoder(SequenceEncoder):
    """Recurrent history encoder used by IntensityFree (Shchur et al., 2020).

    At each step the GRU consumes the
    concatenation of ``log(dt + eps)`` and the event-type embedding, and the
    hidden state after step ``l`` is the causal context for predicting the
    distribution of the next inter-event time.
    """

    def __init__(
        self,
        num_event_types: int,
        n_embd: int = 32,
        num_layers: int = 2,
        dropout: float = 0.0,
        eps: float = 1e-8,
        has_origin: bool = False,
    ) -> None:
        super().__init__(n_embd=n_embd, num_event_types=num_event_types)
        self.eps = eps
        # has_origin: emit the zero-hidden initial state as position 0, so head(position 0)
        # = the head biases = the before-any-event prior.  No new parameter (deterministic
        # zeros -> no RNG draw), so the has_origin=False path is unaffected.
        self.has_origin = bool(has_origin)
        self.prepends_origin = self.has_origin
        # +1 row for the padding type (event_types_idx_safe maps padding -> 0).
        self.type_emb = nn.Embedding(num_event_types + 1, n_embd)
        self.rnn = nn.GRU(
            input_size=1 + n_embd,
            hidden_size=n_embd,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, ts: BatchedMVEventData) -> torch.Tensor:
        B, L = ts.time_points.shape
        w = self.type_emb.weight
        if L == 0:
            n0 = 1 if self.has_origin else 0
            return torch.zeros(B, n0, self.n_embd, device=w.device, dtype=w.dtype)
        param_dtype = w.dtype
        dt = _inter_event_deltas(ts).to(param_dtype)  # (B, L)
        log_dt = torch.log(dt + self.eps).unsqueeze(-1)  # (B, L, 1)
        type_emb = self.type_emb(ts.event_types_idx_safe)  # (B, L, n_embd)
        rnn_in = torch.cat([log_dt, type_emb], dim=-1)  # (B, L, 1 + n_embd)
        context, _ = self.rnn(rnn_in)  # (B, L, n_embd)
        context = torch.nan_to_num(context, nan=0.0)
        if self.has_origin:
            h0 = torch.zeros(B, 1, self.n_embd, device=context.device, dtype=context.dtype)
            context = torch.cat([h0, context], dim=1)  # (B, L+1, n_embd)
        return context


class ContTimeLSTMCell(nn.Module):
    """Continuous-time LSTM cell of the Neural Hawkes Process (Mei & Eisner, 2017).

    Produces the right-limit state
    ``(c, c_bar, delta, o)`` after an event, and decays it continuously between
    events via ``c(t) = c_bar + (c - c_bar) * exp(-delta * dt)``,
    ``h(t) = o * tanh(c(t))``.
    """

    def __init__(self, hidden_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear_layer = nn.Linear(2 * hidden_dim, 7 * hidden_dim, bias=bias)
        self.softplus = nn.Softplus()

    def forward(self, x_i, hidden_ti_minus, ct_ti_minus, c_bar_im1):
        x_i_ = torch.cat((x_i, hidden_ti_minus), dim=-1)
        i_i, i_bar_i, f_i, f_bar_i, z_i, o_i, delta_i = self.linear_layer(x_i_).chunk(7, dim=-1)
        i_i = torch.sigmoid(i_i)
        i_bar_i = torch.sigmoid(i_bar_i)
        f_i = torch.sigmoid(f_i)
        f_bar_i = torch.sigmoid(f_bar_i)
        z_i = torch.tanh(z_i)
        o_i = torch.sigmoid(o_i)
        delta_i = self.softplus(delta_i)
        c_i = f_i * ct_ti_minus + i_i * z_i
        c_bar_i = f_bar_i * c_bar_im1 + i_bar_i * z_i
        return c_i, c_bar_i, delta_i, o_i

    @staticmethod
    def decay(c_i, c_bar_i, delta_i, o_i, dtime):
        """Continuous decay of the cell/hidden state over ``dtime`` (Eq. 7)."""
        c_t = c_bar_i + (c_i - c_bar_i) * torch.exp(-delta_i * dtime)
        h_t = o_i * torch.tanh(c_t)
        return c_t, h_t


class ContTimeLSTMEncoder(SequenceEncoder):
    """Neural Hawkes Process encoder: a continuous-time LSTM over the history.

    Runs the CT-LSTM recurrence and emits, at each position ``l``, the
    **right-limit** state ``(c, c_bar, delta, o)`` *after* event ``l`` — the state
    that governs the intensity in the interval following event ``l``.  Output
    dimensionality is therefore ``n_embd = 4 * hidden_dim``; the NHP model decays
    this state to a query time and applies its own intensity head.
    """

    def __init__(
        self, num_event_types: int, hidden_dim: int = 32, bias: bool = True, has_origin: bool = False
    ) -> None:
        super().__init__(n_embd=4 * hidden_dim, num_event_types=num_event_types)
        self.hidden_dim = hidden_dim
        # has_origin: emit the zero initial state (c, c_bar, delta, o = 0) as position 0.
        # No new parameter (deterministic zeros -> no RNG draw), so the has_origin=False
        # path is unaffected; the zeros column matches the no-history column.
        self.has_origin = bool(has_origin)
        self.prepends_origin = self.has_origin
        self.type_emb = nn.Embedding(num_event_types + 1, hidden_dim)
        self.cell = ContTimeLSTMCell(hidden_dim, bias=bias)

    def forward(self, ts: BatchedMVEventData) -> torch.Tensor:
        B, L = ts.time_points.shape
        H = self.hidden_dim
        dev = self.type_emb.weight.device
        dtype = self.type_emb.weight.dtype
        if L == 0:
            n0 = 1 if self.has_origin else 0
            return torch.zeros(B, n0, 4 * H, device=dev, dtype=dtype)
        dt = _inter_event_deltas(ts).to(dtype)  # (B, L)
        emb = self.type_emb(ts.event_types_idx_safe)  # (B, L, H)
        c = torch.zeros(B, H, device=dev, dtype=dtype)
        c_bar = torch.zeros(B, H, device=dev, dtype=dtype)
        delta = torch.zeros(B, H, device=dev, dtype=dtype)
        o = torch.zeros(B, H, device=dev, dtype=dtype)
        states = []
        for i in range(L):
            c_d, h_d = self.cell.decay(c, c_bar, delta, o, dt[:, i].unsqueeze(-1))
            c, c_bar, delta, o = self.cell(emb[:, i, :], h_d, c_d, c_bar)
            states.append(torch.cat((c, c_bar, delta, o), dim=-1))  # (B, 4H)
        out = torch.stack(states, dim=1)  # (B, L, 4H)
        # Finite-clamp the states (NaN → 0, ±inf → ±bound) so the downstream decay
        # ``c_bar + (c - c_bar)·exp(-δ·Δt)`` stays finite.  A bare nan_to_num maps
        # +inf → the largest float, and ``c - c_bar`` then overflows to inf; a real
        # finite bound keeps the difference representable.  No effect on healthy
        # (finite, in-range) states.
        out = torch.nan_to_num(
            out, nan=0.0, posinf=_CT_LSTM_STATE_BOUND, neginf=-_CT_LSTM_STATE_BOUND
        )
        if self.has_origin:
            z = torch.zeros(B, 1, 4 * H, device=dev, dtype=dtype)
            out = torch.cat([z, out], dim=1)  # (B, L+1, 4H)
        return out
