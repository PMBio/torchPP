# The transformer encoder is adapted from Delphi (https://github.com/gerstung-lab/Delphi, Copyright (c) 2024 Gerstung Lab,
# MIT), which builds on nanoGPT (https://github.com/karpathy/nanoGPT, Copyright (c) 2022 Andrej Karpathy, MIT).
# The THP and SAHP intensity heads are adapted from EasyTPP (https://github.com/ant-research/EasyTemporalPointProcess,
# Copyright 2022 The EasyTPP Authors, Apache License 2.0). All modified for TorchPP. See THIRD_PARTY_NOTICES.md.

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from torchpp.utils.event_data import BatchedMVEventData
from torchpp.modules.nn.encoder import SequenceEncoder


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of event times followed by a linear map, ``(B, L) -> (B, L, n_embd)``."""

    def __init__(self, n_embd: int, norm_factor: float = 1.0, max_wavelen: float = 10000.0):
        super().__init__()
        if n_embd % 2 != 0:
            raise ValueError(f"n_embd must be even, got {n_embd}.")
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(max_wavelen) / n_embd))
        self.register_buffer("div_term", div_term)
        self.n_embd = n_embd
        self.linear = torch.nn.Linear(n_embd, n_embd, bias=False)

        self.norm_factor = norm_factor

    def forward(self, x: torch.Tensor):
        """Embed event times ``x`` of shape ``(B, L)``."""
        t = x / self.norm_factor
        y = torch.zeros(x.shape[0], x.shape[1], self.n_embd, device=x.device)
        y[..., 0::2] = torch.sin(t * self.div_term)
        y[..., 1::2] = torch.cos(t * self.div_term)
        y = self.linear(y)

        return y


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head}).")
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, attn_mask):
        B, T, C = x.size()
        # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T_total) -> (B, nh, T, T_total)
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(attn_mask.unsqueeze(1) == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        # Rows where every key was masked produce NaN after softmax (softmax of all-(-inf)).
        # Replace those NaNs with 0 so the attention output is a zero vector instead of NaN.
        att = torch.nan_to_num(att, nan=0.0)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T_total) x (B, nh, T_total, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att, (k, v)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask):
        y, att, new_kv = self.attn(self.ln_1(x), attn_mask)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, att


@dataclass
class TransformerConfig:
    num_event_types: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.1
    token_dropout: float = 0.0
    bias: bool = False
    add_bos: bool = False  # has_origin: prepend a learned <bos> at an encoder-internal L->L+1


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig, add_bos: bool = False):
        super().__init__()
        self.config = config
        self.add_bos = bool(add_bos or getattr(config, "add_bos", False))

        self.mods = nn.ModuleDict(
            dict(
                event_emb=nn.Embedding(config.num_event_types + 1, config.n_embd),  # Add an event type for padding
                time_emb=TimeEmbedding(n_embd=config.n_embd),
                token_drop=nn.Dropout(config.token_dropout),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        # Created after the weight init above, so models without an origin draw the same
        # initial weights for a given seed.
        if self.add_bos:
            self.bos_emb = nn.Parameter(torch.empty(config.n_embd))
            torch.nn.init.normal_(self.bos_emb, mean=0.0, std=0.02)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ts: BatchedMVEventData):

        event_types = (
            ts.event_types + 1
        )  # Shift event types by 1 so padding (0) is separate from the first event type (1).
        tok_emb = self.mods.event_emb(event_types)  # (B, L, n_embd)

        if self.add_bos:
            # has_origin: prepend the learned <bos> (position 0 = the before-any-event
            # state) placed strictly before the first real event, so the augmented
            # inclusive mask has <bos> attend only to itself and every real event attend
            # to it.  Output is (B, L+1, n_embd); NOT a D+1-typed batch.
            aug_times, attn_mask = ts.bos_augmented_times_and_mask()  # (B, L+1), (B, L+1, L+1)
            bos = self.bos_emb.to(tok_emb.dtype).view(1, 1, -1).expand(tok_emb.shape[0], 1, -1)
            tok_emb = torch.cat([bos, tok_emb], dim=1)  # (B, L+1, n_embd)
            time_emb = self.mods.time_emb(aug_times.unsqueeze(-1))  # (B, L+1, n_embd)
        else:
            attn_mask = ts.attention_mask(mask_padding=True)
            time_emb = self.mods.time_emb(ts.time_points.unsqueeze(-1))  # (B, L, n_embd)

        x = self.mods.token_drop(tok_emb) * (1 - self.config.token_dropout)
        x = x + time_emb
        x = self.mods.drop(x)

        att = []
        for i, block in enumerate(self.mods.h):
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.mods.ln_f(x)
        att = torch.stack(att)

        misc = dict()
        misc["attn_mask"] = attn_mask
        misc["attn"] = att

        return x, misc


class PoissonHead(torch.nn.Module):
    """Per-type Poisson rate head: ``λ_d = softplus(W·h)``.

    Softplus keeps the rate strictly positive with linear (not exponential)
    tails, so large pre-activations grow the intensity at worst linearly rather
    than blowing it up towards ``inf``.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear = torch.nn.Linear(config.n_embd, config.num_event_types)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns rates for each event type. Shape: (B, L, D)."""
        return F.softplus(self.linear(x))


class LogLinearHead(torch.nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear_b = torch.nn.Linear(config.n_embd, config.num_event_types)
        self.linear_s = torch.nn.Linear(config.n_embd, config.num_event_types)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (b, s) stacked on a new axis. Shape: (B, L, 2, D)."""
        b = self.linear_b(x)  # (B, L, D)
        s = self.linear_s(x)  # (B, L, D)
        return torch.stack([b, s], dim=2)  # (B, L, 2, D)


class SplineHead(torch.nn.Module):
    def __init__(self, config: TransformerConfig, num_spline_bins: int = 5):
        super().__init__()
        self.linear = torch.nn.Linear(config.n_embd, config.num_event_types * num_spline_bins)
        self.num_spline_bins = num_spline_bins
        self.num_event_types = config.num_event_types

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns spline heights for each event type. Shape: (B, L, D, K)."""
        B, T, C = x.size()
        out = self.linear(x)  # (B, T, D*K)
        out = out.view(B, T, self.num_event_types, self.num_spline_bins)  # (B, T, D, K)
        return torch.exp(out)


class ScaledSoftplus(torch.nn.Module):
    """Per-mark softplus with learnable beta scaling.

    Each event type d has its own scale parameter ``beta_d = exp(log_beta_d)``.
    The activation is::

        f_d(x) = log(1 + exp(beta_d * x)) / beta_d

    with a linear passthrough for large ``beta_d * x`` to avoid overflow.
    Initialised with ``log_beta = 0`` ==> ``beta = 1`` (standard softplus).
    """

    def __init__(self, num_marks: int, threshold: float = 20.0):
        super().__init__()
        self.threshold = threshold
        self.log_beta = torch.nn.Parameter(torch.zeros(num_marks))  # (D,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-mark scaled softplus. x shape: (..., num_marks)."""
        # Lower-bound log_beta so 1/beta cannot overflow if the parameter is driven
        # to -inf during training (beta -> 0).  -30 is far below the healthy range
        # (log_beta ~ 0), so beta >= e^-30 ≈ 9.4e-14 and 1/beta <= 1.06e13 stays
        # finite; healthy log_beta is untouched (clamp is identity in-range).
        beta = self.log_beta.clamp(min=-30.0).exp()
        beta_x = beta * x
        # Clamp at the linear-passthrough threshold (a lower cap would make softplus saturate
        # below it). ``exp(threshold)`` is finite in float32, and the ``> threshold`` values are
        # discarded by ``where``; the clamp only guards their gradient.
        return torch.where(
            beta_x <= self.threshold,
            torch.log1p(beta_x.clamp(max=self.threshold).exp()) / beta,
            x,
        )


class THPHead(torch.nn.Module):
    """THP intensity head (Zuo et al., 2020).

    The encoder output ``h`` is projected to per-type values via a linear layer.
    Intensity before activation::

        state_d(t) = w_d * t_rel + linear(h)_d + b_d

    where ``w_d`` (factor_intensity_decay) and ``b_d`` (factor_intensity_base)
    are global learnable parameters per type.

    The final intensity is ``ScaledSoftplus(state)`` — applied in the model,
    not here.

    Output shape from ``forward``: ``(B, L, D)`` — the ``linear(h)`` projection.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        D = config.num_event_types
        self.layer_intensity_hidden = torch.nn.Linear(config.n_embd, D)
        # Learnable per-type base and decay, xavier-initialised.
        self.factor_intensity_base = torch.nn.Parameter(torch.empty(1, D))
        self.factor_intensity_decay = torch.nn.Parameter(torch.empty(1, D))
        torch.nn.init.xavier_normal_(self.factor_intensity_base)
        torch.nn.init.xavier_normal_(self.factor_intensity_decay)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project encoder context to per-type hidden values. (B, L, D)."""
        return self.layer_intensity_hidden(x)


class SAHPHead(torch.nn.Module):
    """SAHP state-decay intensity head (Zhang et al., 2020).

    From the encoder output ``h``, produces three per-type quantities::

        mu_d  = GELU(W_mu @ h)       — asymptotic intensity (can be negative!)
        eta_d = GELU(W_eta @ h)      — initial intensity   (can be negative!)
        gamma_d = Softplus(W_gamma @ h)  — decay rate (> 0)

    Linear layers have no bias.
    ``state = mu + (eta - mu) * exp(-gamma * dt)``; the final intensity is
    ``ScaledSoftplus(state)``, applied in the model.

    Output shape from ``forward``: ``(B, L, 3, D)`` — stacked (mu, eta, gamma).
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        D = config.num_event_types
        self.mu_head = torch.nn.Sequential(
            torch.nn.Linear(config.n_embd, D, bias=False),
            torch.nn.GELU(),
        )
        self.eta_head = torch.nn.Sequential(
            torch.nn.Linear(config.n_embd, D, bias=False),
            torch.nn.GELU(),
        )
        self.gamma_head = torch.nn.Sequential(
            torch.nn.Linear(config.n_embd, D, bias=False),
            torch.nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (mu, eta, gamma) stacked. Shape: (B, L, 3, D)."""
        mu = self.mu_head(x)  # (B, L, D)
        eta = self.eta_head(x)  # (B, L, D)
        gamma = self.gamma_head(x)  # (B, L, D)
        return torch.stack([mu, eta, gamma], dim=2)  # (B, L, 3, D)


class MLPIntensityHead(torch.nn.Module):
    """MLP head that predicts per-type intensity from context encoding and query time.

    The intensity function is fully parameterised by a small MLP that receives
    the transformer context ``h`` concatenated with a sinusoidal encoding of
    the elapsed time ``t_rel = t - t_prev``.  This gives maximum flexibility
    for the inter-event intensity shape — the MLP can represent arbitrary
    non-negative intensity functions (when followed by ScaledSoftplus).

    Since the transformer output is cached, only the lightweight MLP runs
    per query point during numerical integration.

    Forward signature::

        forward(h, t_rel) -> raw_intensity   # (..., D)

    Args:
        config: Transformer config (provides ``n_embd`` and ``num_event_types``).
        n_hidden: Width of each hidden layer.
        n_mlp_layers: Number of hidden layers (excluding the output projection).
        time_embd_dim: Dimension of the sinusoidal time encoding.
            Defaults to ``n_embd``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        n_hidden: int = 128,
        n_mlp_layers: int = 2,
        time_embd_dim: int | None = None,
    ):
        super().__init__()
        D = config.num_event_types
        n_embd = config.n_embd
        if time_embd_dim is None:
            time_embd_dim = n_embd
        self.time_embd_dim = time_embd_dim
        if time_embd_dim % 2 != 0:
            raise ValueError(f"time_embd_dim must be even, got {time_embd_dim}.")

        # Sinusoidal wavelengths (same schedule as the backbone TimeEmbedding).
        div_term = torch.exp(torch.arange(0, time_embd_dim, 2) * (-math.log(10000.0) / time_embd_dim))
        self.register_buffer("div_term", div_term)

        # MLP: [h ; time_enc(t_rel)] -> D
        input_dim = n_embd + time_embd_dim
        layers: list[nn.Module] = []
        for i in range(n_mlp_layers):
            layers.append(nn.Linear(input_dim if i == 0 else n_hidden, n_hidden))
            layers.append(nn.GELU())
        layers.append(nn.Linear(n_hidden, D))
        self.mlp = nn.Sequential(*layers)

    def time_encode(self, t_rel: torch.Tensor) -> torch.Tensor:
        """Sinusoidal encoding of relative time.

        Args:
            t_rel: Elapsed time since last event, arbitrary shape ``(...)``.

        Returns:
            Encoding of shape ``(..., time_embd_dim)``.
        """
        t = t_rel.unsqueeze(-1)  # (..., 1)
        enc = torch.zeros(*t_rel.shape, self.time_embd_dim, device=t_rel.device, dtype=t_rel.dtype)
        enc[..., 0::2] = torch.sin(t * self.div_term)
        enc[..., 1::2] = torch.cos(t * self.div_term)
        return enc

    def forward(self, h: torch.Tensor, t_rel: torch.Tensor) -> torch.Tensor:
        """Compute raw (pre-softplus) intensity.

        Args:
            h: Context encoding from the transformer, shape ``(..., n_embd)``.
            t_rel: Relative time since the last event, shape ``(...)``.

        Returns:
            Raw intensity values, shape ``(..., D)``.
        """
        time_enc = self.time_encode(t_rel.to(h.dtype))  # (..., time_embd_dim)
        x = torch.cat([h, time_enc], dim=-1)  # (..., n_embd + time_embd_dim)
        return self.mlp(x)  # (..., D)


# ---------------------------------------------------------------------------
# SequenceEncoder implementation
# ---------------------------------------------------------------------------


class TransformerEncoder(SequenceEncoder):
    """Causal transformer encoder that satisfies the :class:`SequenceEncoder` contract.

    Wraps :class:`Transformer` and exposes its context embeddings as the
    ``(B, L, n_embd)`` output required by
    :class:`~torchpp.tpp.neural.NeuralParametrizedTPP`.

    Args:
        config: :class:`TransformerConfig` describing the transformer architecture.
        has_origin: When True, prepend a learned ``<bos>`` internally so ``forward``
            returns ``(B, L+1, n_embd)`` (position 0 = the before-any-event state).
    """

    def __init__(self, config: TransformerConfig, has_origin: bool = False) -> None:
        super().__init__(n_embd=config.n_embd, num_event_types=config.num_event_types)
        self.config = config
        self.has_origin = bool(has_origin or getattr(config, "add_bos", False))
        # forward emits the (B, L+1, ...) origin column directly (no zeros-dummy prepend).
        self.prepends_origin = self.has_origin
        self.transformer = Transformer(config, add_bos=self.has_origin)

    def forward(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Run the transformer and return ``(B, L, n_embd)`` context embeddings.

        NaN values in the output (e.g. from fully-masked padding positions) are
        replaced with zeros so that downstream heads receive a neutral signal.

        Args:
            ts: Batched event sequences of shape ``(B, L)``.

        Returns:
            Context tensor of shape ``(B, L, n_embd)``.
        """
        x, _ = self.transformer(ts)
        return torch.nan_to_num(x, nan=0.0)
