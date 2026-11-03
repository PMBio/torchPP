# Adapted from EasyTPP (https://github.com/ant-research/EasyTemporalPointProcess), Copyright 2022 The EasyTPP Authors,
# Apache License 2.0; modified for TorchPP. See THIRD_PARTY_NOTICES.md.
"""Attentive Neural Hawkes Process encoder (Yang et al., ICLR 2022).

Unlike THP/NHP, AttNHP has no per-event
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

from torchpp.utils.event_data import BatchedMVEventData, bos_time
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
    """Single-projection multi-head attention (no output linear)."""

    def __init__(self, n_head: int, d_input: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_head ({n_head}).")
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
        has_origin: bool = False,
    ):
        super().__init__(n_embd=d_model + d_time, num_event_types=num_event_types)
        self.d_model = d_model
        self.d_time = d_time
        self.n_layers = n_layers
        self.n_head = n_head
        self.use_norm = use_norm
        # has_origin: <bos> is an encoder-internal event (added inside encode_events), NOT a
        # forward column — forward keeps emitting (B, L, De), so ``prepends_origin`` is False
        # and NeuralModel._full_params still prepends its zeros dummy at col 0.
        self.has_origin = bool(has_origin)
        self.prepends_origin = False
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
        # RNG-order guard: the <bos> d_model embedding is a NEW parameter, created ONLY under
        # has_origin and AFTER all weight init above, so the has_origin=False init is unchanged.
        if self.has_origin:
            self.bos_emb = nn.Parameter(torch.empty(d_model))
            nn.init.normal_(self.bos_emb, mean=0.0, std=0.02)

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

    def _prepend_bos(self, event_emb, ev_times, ev_mask):
        """Prepend the learned ``<bos>`` event (position 0) strictly before every real event.

        ``<bos>`` embedding is ``[bos_emb (d_model); temporal_embedding(t_bos) (d_time)]``,
        placed at ``t_bos`` (:func:`bos_time`) so the causal event/query masks give it to
        every real event and query while it attends only to itself.  Returns the augmented
        ``(event_emb, ev_times, ev_mask)`` with sequence length ``L+1``.
        """
        dtype = self.type_emb.weight.dtype
        B = event_emb.shape[0]
        t_bos = bos_time(ev_times, ev_mask).to(dtype)  # (B,)
        bos_type = self.bos_emb.to(dtype).view(1, 1, -1).expand(B, 1, -1)  # (B, 1, d_model)
        bos_time_emb = self.temporal_embedding(t_bos.unsqueeze(1))  # (B, 1, d_time)
        bos_full = torch.cat([bos_type, bos_time_emb], dim=-1)  # (B, 1, De)
        event_emb = torch.cat([bos_full, event_emb], dim=1)  # (B, L+1, De)
        ev_times = torch.cat([t_bos.unsqueeze(1), ev_times], dim=1)  # (B, L+1)
        bos_mask = torch.ones(B, 1, dtype=ev_mask.dtype, device=ev_mask.device)
        ev_mask = torch.cat([bos_mask, ev_mask], dim=1)  # (B, L+1)
        return event_emb, ev_times, ev_mask

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
        if self.has_origin:
            # has_origin: prepend a learned <bos> event strictly before the first real event.
            # Every real event and every query then attends to it; it attends only to itself.
            event_emb, ev_times, ev_mask = self._prepend_bos(event_emb, ev_times, ev_mask)
        time_emb = event_emb[..., self.d_model :]  # (B, L, d_time)
        # event -> event: causal, attend to real events at time <= own time (incl. simultaneous)
        ee = (~(ev_mask.unsqueeze(1) & (ev_times.unsqueeze(1) <= ev_times.unsqueeze(2)))).to(dtype)  # (B, L, L)

        kv = []
        emb = event_emb  # carried across heads
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
