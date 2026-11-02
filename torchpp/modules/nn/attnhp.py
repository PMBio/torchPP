# Adapted from EasyTPP (https://github.com/ant-research/EasyTemporalPointProcess), Copyright 2022 The EasyTPP Authors,
# Apache License 2.0; modified for TorchPP. See THIRD_PARTY_NOTICES.md.
"""Attentive Neural Hawkes Process encoder (Yang et al., ICLR 2022).

Faithful port of the EasyTPP reference (``torch_attnhp.py`` + the attention
layers in ``torch_baselayer.py``).  Unlike THP/NHP, AttNHP has no per-event
state that is simply decayed: the intensity at a query time is obtained by
re-running the attention stack with a temporal embedding of *that query time*,
attending to the events that precede it.  This module exposes
:meth:`AttNHPEncoder.states_at_queries`, which the :class:`AttNHPProcess` TPP
calls to evaluate the intensity at arbitrary query times under torchPP's
"strictly before t" causal convention.

Masking convention (matches the ported ``attention``): ``mask == 1`` blocks a
key, ``mask == 0`` attends to it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from torchpp.utils.event_data import BatchedMVEventData
from .encoder import SequenceEncoder


def _attention(query, key, value, mask, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask > 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)


class MultiHeadAttention(nn.Module):
    """Single-projection multi-head attention (no output linear), as in EasyTPP."""

    def __init__(self, n_head: int, d_input: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.linears = nn.ModuleList([nn.Linear(d_input, d_model) for _ in range(3)])
        self.dropout = nn.Dropout(p=dropout)

    def _proj(self, lin, x):
        nb = x.size(0)
        return lin(x).view(nb, -1, self.n_head, self.d_k).transpose(1, 2)  # (nb, n_head, S, d_k)

    def project_kv(self, x):
        """Projected key/value for ``x`` (nb, S, d_input); each -> (nb, n_head, S, d_k)."""
        return self._proj(self.linears[1], x), self._proj(self.linears[2], x)

    def attend(self, query, key, value, mask):
        """Attention with pre-projected ``key``/``value`` (nb, n_head, S, d_k); ``query`` raw."""
        q = self._proj(self.linears[0], query)
        if mask is not None:
            mask = mask.unsqueeze(1)  # broadcast over heads
        x = _attention(q, key, value, mask=mask, dropout=self.dropout)
        return x.transpose(1, 2).contiguous().view(q.size(0), -1, self.n_head * self.d_k)

    def forward(self, query, key, value, mask):
        k, v = self._proj(self.linears[1], key), self._proj(self.linears[2], value)
        return self.attend(query, k, v, mask)


class EncoderLayer(nn.Module):
    """Self-attention layer without residual/FFN, as used by AttNHP."""

    def __init__(self, self_attn: MultiHeadAttention):
        super().__init__()
        self.self_attn = self_attn

    def forward(self, x, mask):
        return self.self_attn(x, x, x, mask)


class AttNHPEncoder(SequenceEncoder):
    """Attentive continuous-time encoder.

    ``forward(ts)`` returns the per-event embedding ``[tanh(type_emb); time_emb]``
    of shape ``(B, L, d_model + d_time)`` (used as the cached representation).
    The intensity-bearing states are produced by :meth:`states_at_queries`.
    """

    def __init__(
        self,
        num_event_types: int,
        d_model: int = 16,
        d_time: int = 16,
        n_layers: int = 2,
        n_head: int = 2,
        dropout: float = 0.0,
        use_norm: bool = False,
    ):
        super().__init__(n_embd=d_model + d_time, num_event_types=num_event_types)
        self.d_model = d_model
        self.d_time = d_time
        self.n_layers = n_layers
        self.n_head = n_head
        self.use_norm = use_norm
        self.type_emb = nn.Embedding(num_event_types + 1, d_model)
        div_term = torch.exp(torch.arange(0, d_time, 2) * -(math.log(10000.0) / d_time)).reshape(1, 1, -1)
        self.register_buffer("div_term", div_term)
        self.heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [EncoderLayer(MultiHeadAttention(1, d_model + d_time, d_model, dropout)) for _ in range(n_layers)]
                )
                for _ in range(n_head)
            ]
        )
        if use_norm:
            self.norm = nn.LayerNorm(d_model)

    @property
    def state_dim(self) -> int:
        return self.d_model * self.n_head

    def temporal_embedding(self, time: torch.Tensor) -> torch.Tensor:
        """Sinusoidal temporal embedding. ``time`` (B, S) -> (B, S, d_time)."""
        dtype = self.type_emb.weight.dtype
        time = time.to(dtype)
        pe = torch.zeros(*time.shape, self.d_time, device=time.device, dtype=dtype)
        arg = time.unsqueeze(-1) * self.div_term.to(dtype)
        pe[..., 0::2] = torch.sin(arg)
        pe[..., 1::2] = torch.cos(arg)
        return pe

    def forward(self, ts: BatchedMVEventData) -> torch.Tensor:
        B, L = ts.time_points.shape
        dtype = self.type_emb.weight.dtype
        if L == 0:
            return torch.zeros(B, 0, self.n_embd, device=self.type_emb.weight.device, dtype=dtype)
        time_emb = self.temporal_embedding(ts.time_points)  # (B, L, d_time)
        type_emb = torch.tanh(self.type_emb(ts.event_types_idx_safe))  # (B, L, d_model)
        return torch.cat([type_emb, time_emb], dim=-1)  # (B, L, d_model + d_time)

    def encode_events(
        self,
        event_emb: torch.Tensor,  # (B, L, d_model + d_time)
        ev_times: torch.Tensor,  # (B, L)
        ev_mask: torch.Tensor,  # (B, L) bool — real events
    ) -> dict:
        """Run the query-independent event recurrence once.

        Events never attend to queries, so their per-(head, layer) representations
        do not depend on the query times. Cache the projected event key/value at
        each step so the (possibly batch-folded) query pass cross-attends to them
        instead of re-encoding the events.
        """
        dtype = self.type_emb.weight.dtype
        ev_times = ev_times.to(dtype)
        time_emb = event_emb[..., self.d_model :]  # (B, L, d_time)
        # event -> event: causal, attend to real events at time <= own time (incl. simultaneous)
        ee = (~(ev_mask.unsqueeze(1) & (ev_times.unsqueeze(1) <= ev_times.unsqueeze(2)))).to(dtype)  # (B, L, L)

        kv = []
        emb = event_emb  # carried across heads, as in EasyTPP
        for h in range(self.n_head):
            for li in range(self.n_layers):
                mha = self.heads[h][li].self_attn
                k_e, v_e = mha.project_kv(emb)  # each (B, n_head, L, d_k)
                kv.append((k_e, v_e))
                enc_events = mha.attend(emb, k_e, v_e, ee)  # (B, L, d_model)
                emb = torch.cat([enc_events, time_emb], dim=-1)  # refresh event repr
        return {"kv": kv, "ev_times": ev_times, "ev_mask": ev_mask, "B": event_emb.shape[0], "L": event_emb.shape[1]}

    def states_from_cache(self, cache: dict, query_times: torch.Tensor) -> torch.Tensor:
        """Attentive state for each query, cross-attending to the cached events.

        ``query_times`` is ``(B', Q)``; ``B'`` must be a multiple of the cache batch
        ``B`` (the quadrature fold replicates the batch). Cached event key/values are
        expanded to ``B'`` as views, so the event projections run only once.
        Returns ``(B', Q, d_model * n_head)``.
        """
        dtype = self.type_emb.weight.dtype
        Bq, Q = query_times.shape
        B, L = cache["B"], cache["L"]
        f = Bq // B
        query_times = query_times.to(dtype)
        query_time_emb = self.temporal_embedding(query_times)  # (B', Q, d_time)

        ev_times = cache["ev_times"].repeat_interleave(f, dim=0)  # (B', L)
        ev_mask = cache["ev_mask"].repeat_interleave(f, dim=0)  # (B', L)
        # query -> event: real events strictly before the query time; query -> query: diagonal
        qe = ~(ev_mask.unsqueeze(1) & (ev_times.unsqueeze(1) < query_times.unsqueeze(2)))  # (B', Q, L)
        qq = ~torch.eye(Q, dtype=torch.bool, device=query_times.device).unsqueeze(0).expand(Bq, Q, Q)
        bot = torch.cat([qe, qq], dim=2).to(dtype)  # (B', Q, L+Q)

        init_cur_layer = torch.zeros(Bq, Q, self.d_model, device=query_times.device, dtype=dtype)
        head_outputs = []
        for h in range(self.n_head):
            cur_layer = init_cur_layer
            for li in range(self.n_layers):
                mha = self.heads[h][li].self_attn
                k_e, v_e = cache["kv"][h * self.n_layers + li]  # (B, n_head, L, d_k)
                k_e = k_e.repeat_interleave(f, dim=0)  # (B', n_head, L, d_k)
                v_e = v_e.repeat_interleave(f, dim=0)
                layer_q = torch.cat([cur_layer, query_time_emb], dim=-1)  # (B', Q, De)
                k_q, v_q = mha.project_kv(layer_q)  # each (B', n_head, Q, d_k)
                k = torch.cat([k_e, k_q], dim=2)  # (B', n_head, L+Q, d_k)
                v = torch.cat([v_e, v_q], dim=2)
                q_out = mha.attend(layer_q, k, v, bot)  # (B', Q, d_model)
                cur_layer = torch.tanh(q_out) + cur_layer
                if self.use_norm:
                    cur_layer = self.norm(cur_layer)
            head_outputs.append(cur_layer)
        return torch.cat(head_outputs, dim=-1)  # (B', Q, d_model * n_head)

    def states_at_queries(
        self,
        event_emb: torch.Tensor,  # (B, L, d_model + d_time)
        ev_times: torch.Tensor,  # (B, L)
        ev_mask: torch.Tensor,  # (B, L) bool — real events
        query_times: torch.Tensor,  # (B, Q)
    ) -> torch.Tensor:
        """Convenience wrapper: encode the events, then query them. Returns ``(B, Q, state_dim)``."""
        return self.states_from_cache(self.encode_events(event_emb, ev_times, ev_mask), query_times)
