"""LogNormalMixture head/distribution unit tests.

Model-specific checks the parameterised suite cannot express for IntensityFree:
the log-normal-mixture density matches a torch.distributions reference; the
hazard relation ``lambda = d/dt Lambda`` holds via autograd (the generic
finite-difference check is skipped near tau->0); and the joint factorises as
``p(t, d | H) = p(t | H) * p(d | H)`` with a time-independent mark.
"""

from __future__ import annotations

import pytest
import torch

from torchpp.modules.nn.encoder import GRUEncoder
from torchpp.modules.nn.lognormal_mixture import LogNormalMixtureDistribution
from torchpp.tpp.neural import IntensityFreeProcess
from torchpp.utils.event_data import BatchedMVEventData


def _iftpp(d=3, n_mix=4, encoder="gru", seed=0):
    torch.manual_seed(seed)
    if encoder == "gru":
        enc = GRUEncoder(num_event_types=d, n_embd=16, num_layers=1)
    else:
        from torchpp.modules.nn.transformer import TransformerConfig, TransformerEncoder

        enc = TransformerEncoder(TransformerConfig(num_event_types=d, n_layer=1, n_head=2, n_embd=16, dropout=0.0))
    return IntensityFreeProcess(enc, num_mix_components=n_mix, seed=seed)


def _batch():
    return BatchedMVEventData(
        [torch.tensor([0.5, 1.2, 2.0, 3.1], dtype=torch.float64), torch.tensor([0.3, 0.9, 1.5], dtype=torch.float64)],
        [torch.tensor([0, 2, 1, 0]), torch.tensor([1, 1, 2])],
    )


class TestIntensityFreeMixture:
    def test_lognormal_mixture_log_prob_matches_reference(self):
        """Our ported mixture log_prob equals a hand-built torch.distributions one."""
        torch.manual_seed(0)
        B, K = 5, 4
        locs = torch.randn(B, K, dtype=torch.float64)
        log_scales = torch.randn(B, K, dtype=torch.float64).clamp(-2, 2)
        log_weights = torch.log_softmax(torch.randn(B, K, dtype=torch.float64), dim=-1)
        dist = LogNormalMixtureDistribution(locs, log_scales, log_weights, 0.3, 1.7)

        # Reference: mixture over component LogNormals with the same affine+exp transform.
        # log p(t) = logsumexp_k [ log w_k + LogNormal(mu + s*loc_k, (s*scale_k)).log_prob(t) ]
        x = torch.tensor([0.2, 0.7, 1.5, 3.0, 8.0], dtype=torch.float64)
        mu, s = 0.3, 1.7
        comp = torch.distributions.LogNormal(
            loc=mu + s * locs, scale=(s * log_scales.exp())
        )  # (B, K)
        ref = torch.logsumexp(log_weights + comp.log_prob(x.unsqueeze(-1)), dim=-1)  # (B,)
        got = dist.log_prob(x)
        assert torch.allclose(got, ref, atol=1e-8), (got, ref)

    def test_hazard_equals_ci_derivative_autograd(self):
        """lambda(tau) = f(tau)/S(tau) = d/dtau (-log S(tau)) — checked with autograd."""
        torch.manual_seed(0)
        K = 4
        locs = torch.randn(6, K, dtype=torch.float64)
        log_scales = torch.randn(6, K, dtype=torch.float64).clamp(-2, 2)
        log_weights = torch.log_softmax(torch.randn(6, K, dtype=torch.float64), dim=-1)
        dist = LogNormalMixtureDistribution(locs, log_scales, log_weights, 0.0, 1.0)

        tau = torch.linspace(0.2, 5.0, 6, dtype=torch.float64).requires_grad_(True)
        neg_log_S = -dist.log_survival_function(tau)  # = cumulative intensity (per unit p(d))
        (grad,) = torch.autograd.grad(neg_log_S.sum(), tau)
        hazard = torch.exp(dist.log_prob(tau) - dist.log_survival_function(tau)).detach()
        assert torch.allclose(grad, hazard, rtol=1e-4, atol=1e-6), (grad, hazard)

    @pytest.mark.parametrize("encoder", ["gru", "transformer"])
    def test_joint_factorises_time_times_mark(self, encoder):
        """PDF marginal-over-types == f(tau); p(d|t) == p(d|H) (time-independent)."""
        m = _iftpp(encoder=encoder)
        ts = _batch()
        with m.fix_context(ts):
            t = torch.tensor([3.5, 2.0], dtype=torch.float64)  # >= max_time per seq
            pdf = m.PDF(t, ts)  # (B, D), joint p(t, d)
            params = m._get_params_at_time(t, ts)
            dist, mark_lp = m._unpack(params, dtype=t.dtype)
            tau = t - ts.event_time_before(t)
            f = dist.log_prob(tau).exp()
            # marginal over types == mixture density
            assert torch.allclose(pdf.sum(1), f, rtol=1e-5, atol=1e-7)
            # conditional type == history-only mark probs (independent of t)
            cond = pdf / pdf.sum(1, keepdim=True)
            assert torch.allclose(cond, mark_lp.exp(), atol=1e-6)

    def test_loglik_finite_and_decomposition_identity(self):
        m = _iftpp()
        ts = _batch()
        with m.fix_context(ts):
            ll = m.likelihood(ts)
            assert torch.isfinite(ll).all()
            mean_pred = m.analytical_mean_next_event_time(ts)
            assert torch.isfinite(mean_pred).all()
            assert (mean_pred >= ts.max_time).all()
