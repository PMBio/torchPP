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

TIME_ENCODING = 1.0


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int, norm_factor: float = 1.0 / TIME_ENCODING, max_wavelen: float = 10000.0):
        super().__init__()
        assert n_embd % 2 == 0
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(max_wavelen) / n_embd))
        self.register_buffer("div_term", div_term)
        self.n_embd = n_embd
        self.linear = torch.nn.Linear(n_embd, n_embd, bias=False)

        self.norm_factor = norm_factor

    def forward(self, x: torch.Tensor):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        time_years = x / self.norm_factor
        y = torch.zeros(x.shape[0], x.shape[1], self.n_embd, device=x.device)
        y[..., 0::2] = torch.sin(time_years * self.div_term)  # * (1-self.div_term)
        y[..., 1::2] = torch.cos(time_years * self.div_term)  # * (1-self.div_term)
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
        assert config.n_embd % config.n_head == 0
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
    num_event_types: int = 1268
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 120
    dropout: float = 0.1
    token_dropout: float = 0.0
    bias: bool = False


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

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
        time_points = ts.time_points
        attn_mask = ts.attention_mask(mask_padding=True)

        tok_emb = self.mods.event_emb(event_types)  # (B, L, n_embd)
        time_emb = self.mods.time_emb(time_points.unsqueeze(-1))  # (B, L, n_embd)
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
    """Per-type Poisson rate head: ``λ_d = link(W·h)``.

    ``link="exp"`` (default): logit clamped at log(1e6);
    ``link="softplus"``: strictly positive with linear tails and a bounded
    gradient. The learned prior rate stays ``exp(prior)`` for both links.
    """

    _LINKS = ("exp", "softplus")

    def __init__(self, config: TransformerConfig, link: str = "exp"):
        super().__init__()
        if link not in self._LINKS:
            raise ValueError(f"PoissonHead link must be one of {self._LINKS}, got {link!r}")
        self.link = link
        self.linear = torch.nn.Linear(config.n_embd, config.num_event_types)
        # Learned prior: rates valid before the first event. Shape (D,).
        self.prior = torch.nn.Parameter(torch.zeros(config.num_event_types))
        torch.nn.init.normal_(self.prior, std=0.02)

    def get_prior(self) -> torch.Tensor:
        """Return prior rates, shape (D,), in linear (positive) space."""
        return torch.exp(self.prior)

    # Upper bound on the pre-exp logit (rate <= exp(_LOGIT_MAX) ≈ 1e6): guards the
    # unbounded exp against catastrophic finite blow-ups.  Far above healthy logits
    # (rates are O(1e0–1e2)), so the clamp is number-preserving on well-behaved input.
    _LOGIT_MAX = math.log(1e6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns rates for each event type. Shape: (B, L, D)."""
        if self.link == "softplus":
            return F.softplus(self.linear(x))
        return torch.exp(self.linear(x).clamp(max=self._LOGIT_MAX))


class LogLinearHead(torch.nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear_b = torch.nn.Linear(config.n_embd, config.num_event_types)
        self.linear_s = torch.nn.Linear(config.n_embd, config.num_event_types)
        # Learned prior: (b, s) parameters valid before the first event.
        # Stacked as (2, D) — index 0 = b, index 1 = s.
        self.prior = torch.nn.Parameter(torch.zeros(2, config.num_event_types))
        torch.nn.init.normal_(self.prior, std=0.02)
        # Bias s toward 0.3 so the prior intensity grows slightly before training.
        with torch.no_grad():
            self.prior[1].fill_(0.3)

    def get_prior(self) -> torch.Tensor:
        """Return prior (b, s) stacked, shape (2, D). b is raw; s is raw."""
        return self.prior  # (2, D) — no constraint on b or s

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
        # Learned prior: heights valid before the first event. Shape (D, K).
        self.prior = torch.nn.Parameter(torch.zeros(config.num_event_types, num_spline_bins))
        torch.nn.init.normal_(self.prior, std=0.02)

    def get_prior(self) -> torch.Tensor:
        """Return prior spline heights, shape (D, K), in linear (positive) space."""
        return torch.exp(self.prior)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns spline heights for each event type. Shape: (B, L, D, K)."""
        B, T, C = x.size()
        out = self.linear(x)  # (B, T, D*K)
        out = out.view(B, T, self.num_event_types, self.num_spline_bins)  # (B, T, D, K)
        return torch.exp(out)


# PiecewiseConstantHead is structurally identical to SplineHead — same
# (B, L, D, K) output of positive heights — but represents piecewise-constant
# levels rather than linear-spline knot heights.
PiecewiseConstantHead = SplineHead


class THPHead(torch.nn.Module):
    """Head for Transformer Hawkes Process (THP).

    Outputs per-interval intercept ``b`` (from the transformer context) in
    log-space, and a single **global** learned decay ``w`` per event type.
    Intensity in interval l: λ_d(t) = softplus(b_d + w_d * t_rel)
    where t_rel = t - t_interval_start.

    This matches the THP formulation (Zuo et al., ICML 2020) where the decay
    is a fixed learned parameter rather than being predicted per event.

    Output shapes:
        _get_params(ts) → (B, L, D)   — intercepts b (raw / unconstrained)
        _get_prior()    → (D,)         — prior intercept
        factor_decay    → (D,)         — global log-decay (learnable, not per-event)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear_b = torch.nn.Linear(config.n_embd, config.num_event_types)
        # Global learned slope per type — initialised to log(0.3) so the
        # initial w = exp(log_decay) ≈ 0.3, keeping intensity growth modest.
        self.log_decay = torch.nn.Parameter(
            torch.full((config.num_event_types,), fill_value=-1.2)  # exp(-1.2) ≈ 0.3
        )
        # Learned prior intercept, shape (D,).
        self.prior = torch.nn.Parameter(torch.zeros(config.num_event_types))
        torch.nn.init.normal_(self.prior, std=0.02)

    def get_prior(self) -> torch.Tensor:
        """Return prior intercept, shape (D,). Raw (unconstrained)."""
        return self.prior  # (D,)

    def get_decay(self) -> torch.Tensor:
        """Return positive decay per type, shape (D,)."""
        return torch.exp(self.log_decay)  # (D,) — always positive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-interval intercepts b. Shape: (B, L, D). Raw (unconstrained)."""
        return self.linear_b(x)  # (B, L, D)


class SAHPHead(torch.nn.Module):
    """Head for Self-Attentive Hawkes Process (SAHP).

    Outputs (mu, eta, gamma) — the three parameters of the mean-reverting
    decay intensity:

        λ_d(t) = mu_d + (eta_d - mu_d) * exp(-gamma_d * t_rel)

    where t_rel = t - t_interval_start and all three are predicted by the
    transformer per event.

    Output shapes:
        _get_params(ts) → (B, L, 3, D)   stacked (mu, eta, gamma)
        _get_prior()    → (3, D)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear_mu = torch.nn.Linear(config.n_embd, config.num_event_types)
        self.linear_eta = torch.nn.Linear(config.n_embd, config.num_event_types)
        self.linear_gamma = torch.nn.Linear(config.n_embd, config.num_event_types)
        # Prior: stacked (mu, eta, gamma), shape (3, D).
        self.prior = torch.nn.Parameter(torch.zeros(3, config.num_event_types))
        torch.nn.init.normal_(self.prior, std=0.02)

    def get_prior(self) -> torch.Tensor:
        """Return prior (mu, eta, gamma) stacked. Shape (3, D). Raw."""
        return self.prior  # (3, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (mu, eta, gamma) stacked. Shape: (B, L, 3, D). Raw."""
        mu = self.linear_mu(x)  # (B, L, D) — raw, will be exp'd to get positive
        eta = self.linear_eta(x)  # (B, L, D) — raw
        gamma = self.linear_gamma(x)  # (B, L, D) — raw, will be softplus'd
        return torch.stack([mu, eta, gamma], dim=2)  # (B, L, 3, D)


# ---------------------------------------------------------------------------
# SequenceEncoder implementation
# ---------------------------------------------------------------------------


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
        # Clamp at the linear-passthrough threshold, not below it: a cap under
        # ``threshold`` (the old ``log(1e5)≈11.51``) left a dead zone on ``(11.51, threshold]``
        # where softplus wrongly saturated.  ``exp(threshold)`` is finite in float32, and the
        # ``> threshold`` values are discarded by ``where`` (the clamp only guards their gradient).
        return torch.where(
            beta_x <= self.threshold,
            torch.log1p(beta_x.clamp(max=self.threshold).exp()) / beta,
            x,
        )


class EasyTPPTHPHead(torch.nn.Module):
    """Head replicating the EasyTPP THP intensity parameterisation.

    The encoder output ``h`` is projected to per-type values via a linear layer.
    Intensity before activation::

        state_d(t) = w_d * t_rel + linear(h)_d + b_d

    where ``w_d`` (factor_intensity_decay) and ``b_d`` (factor_intensity_base)
    are global learnable parameters per type — matching EasyTPP exactly.

    The final intensity is ``ScaledSoftplus(state)`` — applied in the model,
    not here.

    Output shape from ``forward``: ``(B, L, D)`` — the ``linear(h)`` projection.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        D = config.num_event_types
        self.layer_intensity_hidden = torch.nn.Linear(config.n_embd, D)
        # Learnable per-type base and decay — xavier init like EasyTPP.
        self.factor_intensity_base = torch.nn.Parameter(torch.empty(1, D))
        self.factor_intensity_decay = torch.nn.Parameter(torch.empty(1, D))
        torch.nn.init.xavier_normal_(self.factor_intensity_base)
        torch.nn.init.xavier_normal_(self.factor_intensity_decay)
        # Learned prior (before any events), shape (D,).
        self.prior = torch.nn.Parameter(torch.zeros(D))
        torch.nn.init.normal_(self.prior, std=0.02)

    def get_prior(self) -> torch.Tensor:
        return self.prior  # (D,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project encoder context to per-type hidden values. (B, L, D)."""
        return self.layer_intensity_hidden(x)


class EasyTPPSAHPHead(torch.nn.Module):
    """Head replicating the EasyTPP SAHP state-decay parameterisation.

    From the encoder output ``h``, produces three per-type quantities::

        mu_d  = GELU(W_mu @ h)       — asymptotic intensity (can be negative!)
        eta_d = GELU(W_eta @ h)      — initial intensity   (can be negative!)
        gamma_d = Softplus(W_gamma @ h)  — decay rate (> 0)

    Linear layers have **no bias**, matching the original EasyTPP code.
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
        # Prior: raw (mu, eta, gamma), shape (3, D).
        # get_prior() applies the same activations as forward() so that
        # the prior lives in the same space as head outputs.
        self.prior = torch.nn.Parameter(torch.zeros(3, D))
        torch.nn.init.normal_(self.prior, std=0.02)

    def get_prior(self) -> torch.Tensor:
        """Return prior (mu, eta, gamma) with activations matching forward().

        mu/eta go through GELU, gamma through Softplus — same as the head
        projections, so the prior is in the same functional space.
        """
        mu = torch.nn.functional.gelu(self.prior[0])  # (D,)
        eta = torch.nn.functional.gelu(self.prior[1])  # (D,)
        gamma = torch.nn.functional.softplus(self.prior[2])  # (D,)
        return torch.stack([mu, eta, gamma], dim=0)  # (3, D)

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
        assert time_embd_dim % 2 == 0, "time_embd_dim must be even"

        # Sinusoidal wavelengths (same schedule as the backbone TimeEmbedding).
        div_term = torch.exp(torch.arange(0, time_embd_dim, 2) * (-math.log(10000.0) / time_embd_dim))
        self.register_buffer("div_term", div_term)
        self.norm_factor = 1.0 / TIME_ENCODING

        # MLP: [h ; time_enc(t_rel)] -> D
        input_dim = n_embd + time_embd_dim
        layers: list[nn.Module] = []
        for i in range(n_mlp_layers):
            layers.append(nn.Linear(input_dim if i == 0 else n_hidden, n_hidden))
            layers.append(nn.GELU())
        layers.append(nn.Linear(n_hidden, D))
        self.mlp = nn.Sequential(*layers)

        # Learned prior embedding (context before any events).
        self.prior = nn.Parameter(torch.zeros(n_embd))
        nn.init.normal_(self.prior, std=0.02)

    def time_encode(self, t_rel: torch.Tensor) -> torch.Tensor:
        """Sinusoidal encoding of relative time.

        Args:
            t_rel: Elapsed time since last event, arbitrary shape ``(...)``.

        Returns:
            Encoding of shape ``(..., time_embd_dim)``.
        """
        t = t_rel.unsqueeze(-1) / self.norm_factor  # (..., 1)
        enc = torch.zeros(*t_rel.shape, self.time_embd_dim, device=t_rel.device, dtype=t_rel.dtype)
        enc[..., 0::2] = torch.sin(t * self.div_term)
        enc[..., 1::2] = torch.cos(t * self.div_term)
        return enc

    def get_prior(self) -> torch.Tensor:
        """Prior embedding for the case of no observed events. Shape: ``(n_embd,)``."""
        return self.prior

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
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(n_embd=config.n_embd, num_event_types=config.num_event_types)
        self.config = config
        self.transformer = Transformer(config)

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
