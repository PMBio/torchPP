"""Direct unit tests for the intensity/parameter heads in ``transformer.py``.

Each head is constructed with a tiny :class:`TransformerConfig` and fed a
context tensor produced by running a small :class:`TransformerEncoder` over a
real ``BatchedMVEventData``. We assert:

- forward output SHAPE matches the documented contract;
- POSITIVITY where the head guarantees it (Poisson rates, spline/piecewise
  heights, gamma decay, ScaledSoftplus output).

There is no learned per-head prior (``_full_params`` column 0 is a zeros
column), so the heads expose no ``get_prior()``.

The ``LogNormalMixture`` head is covered separately in
``test_lognormal_mixture.py`` and is not duplicated here.
"""

from __future__ import annotations

import torch

from helpers import DEVICE, D, build_test_data
from torchpp.modules.nn.transformer import (
    TransformerConfig,
    TransformerEncoder,
    PoissonHead,
    LogLinearHead,
    SplineHead,
    THPHead,
    SAHPHead,
    MLPIntensityHead,
    ScaledSoftplus,
)


N_EMBD = 16


def _cfg(d=D):
    return TransformerConfig(num_event_types=d, n_layer=1, n_head=2, n_embd=N_EMBD, dropout=0.0)


def _context():
    """A real (B, L, n_embd) context tensor from a small transformer encoder."""
    cfg = _cfg()
    enc = TransformerEncoder(cfg).to(DEVICE).eval()
    ts = build_test_data(DEVICE)["batch_multi"]
    with torch.no_grad():
        x = enc(ts)
    B, L = ts.time_points.shape
    return x, B, L


class TestPoissonHead:
    def test_forward_shape_positive(self):
        """forward -> (B, L, D), strictly positive rates (softplus)."""
        x, B, L = _context()
        head = PoissonHead(_cfg()).to(DEVICE)
        out = head(x)
        assert out.shape == (B, L, D)
        assert (out > 0).all()
        assert torch.isfinite(out).all()

    def test_no_overflow_on_large_logits(self):
        """Softplus has linear tails: a pre-activation that would overflow the
        old exp head (exp(1e4) = inf) yields a finite rate that grows only
        linearly (softplus(z) ≈ z for large z)."""
        head = PoissonHead(_cfg()).to(DEVICE)
        with torch.no_grad():
            head.linear.weight.zero_()
            head.linear.bias.fill_(1e4)  # exp(1e4) = inf; softplus(1e4) = 1e4
        x = torch.zeros(2, 4, N_EMBD, device=DEVICE)
        out = head(x)
        assert torch.isfinite(out).all()
        assert torch.allclose(out, torch.full_like(out, 1e4))

    def test_equals_softplus_of_linear(self):
        """The head is exactly ``softplus(linear(x))`` on well-behaved logits."""
        head = PoissonHead(_cfg()).to(DEVICE)
        torch.manual_seed(0)
        x = torch.randn(2, 4, N_EMBD, device=DEVICE) * 0.5
        ref = torch.nn.functional.softplus(head.linear(x))
        assert torch.equal(head(x), ref)


class TestLogLinearHead:
    def test_forward_shape(self):
        """forward -> (B, L, 2, D) stacked (b, s); raw (unconstrained)."""
        x, B, L = _context()
        head = LogLinearHead(_cfg()).to(DEVICE)
        out = head(x)
        assert out.shape == (B, L, 2, D)
        assert torch.isfinite(out).all()


class TestSplineHead:
    def test_forward_shape_positive(self):
        """forward -> (B, L, D, K), positive heights (exp)."""
        x, B, L = _context()
        K = 5
        head = SplineHead(_cfg(), num_spline_bins=K).to(DEVICE)
        out = head(x)
        assert out.shape == (B, L, D, K)
        assert (out > 0).all()


class TestTHPHead:
    def test_forward_shape(self):
        """forward -> (B, L, D) projection; base/decay are global params."""
        x, B, L = _context()
        head = THPHead(_cfg()).to(DEVICE)
        out = head(x)
        assert out.shape == (B, L, D)
        assert torch.isfinite(out).all()
        assert head.factor_intensity_base.shape == (1, D)
        assert head.factor_intensity_decay.shape == (1, D)


class TestSAHPHead:
    def test_forward_shape_gamma_positive(self):
        """forward -> (B, L, 3, D); gamma channel (index 2) is softplus -> >= 0."""
        x, B, L = _context()
        head = SAHPHead(_cfg()).to(DEVICE)
        out = head(x)
        assert out.shape == (B, L, 3, D)
        assert torch.isfinite(out).all()
        assert (out[:, :, 2] >= 0).all()  # gamma via Softplus


class TestMLPIntensityHead:
    def test_forward_shape(self):
        """forward(h, t_rel) -> (..., D) raw intensity."""
        x, B, L = _context()
        head = MLPIntensityHead(_cfg(), n_hidden=8, n_mlp_layers=1, time_embd_dim=8).to(DEVICE)
        t_rel = torch.full((B, L), 0.3, dtype=x.dtype, device=DEVICE)
        out = head(x, t_rel)
        assert out.shape == (B, L, D)
        assert torch.isfinite(out).all()


class TestScaledSoftplus:
    def test_positive_and_shape(self):
        """ScaledSoftplus maps any input to strictly-positive output, shape preserved."""
        act = ScaledSoftplus(num_marks=D).to(DEVICE)
        x = torch.tensor([[-50.0, 0.0, 50.0]], device=DEVICE)
        out = act(x)
        assert out.shape == x.shape
        assert (out > 0).all()
        assert torch.isfinite(out).all()

    def test_default_beta_matches_plain_softplus(self):
        """With log_beta=0 (beta=1) it reduces to standard softplus on the safe branch."""
        act = ScaledSoftplus(num_marks=D).to(DEVICE)
        x = torch.tensor([[-2.0, -0.5, 1.0]], device=DEVICE)
        ref = torch.nn.functional.softplus(x)
        torch.testing.assert_close(act(x), ref, rtol=1e-5, atol=1e-6)

    def test_no_dead_zone_between_clamp_and_threshold(self):
        """Pre-activations in (log(1e5)=11.51, threshold=20] must not saturate (audit N1): with
        beta=1 the output is the exact softplus (~x), not the old ~11.51 clamp value."""
        act = ScaledSoftplus(num_marks=D).to(DEVICE)  # log_beta=0 -> beta=1 -> beta_x == x
        x = torch.tensor([[12.0, 15.0, 19.0]], device=DEVICE)  # the old dead zone
        ref = torch.nn.functional.softplus(x)  # ≈ x here (and ≫ 11.51)
        torch.testing.assert_close(act(x), ref, rtol=1e-5, atol=1e-4)

    def test_finite_when_log_beta_driven_negative(self):
        """Fix 3: log_beta is lower-bounded so 1/beta cannot overflow.  With
        log_beta -> -inf (beta -> 0) the pre-fix ``1/beta`` was inf; post-fix the
        output stays finite."""
        act = ScaledSoftplus(num_marks=D).to(DEVICE)
        with torch.no_grad():
            act.log_beta.fill_(-1000.0)  # beta = exp(-1000) underflows to 0 pre-clamp
        x = torch.randn(4, D, device=DEVICE)
        out = act(x)
        assert torch.isfinite(out).all()
        assert (out > 0).all()
