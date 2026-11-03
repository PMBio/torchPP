# Adapted from EasyTPP (https://github.com/ant-research/EasyTemporalPointProcess), Copyright 2022 The EasyTPP Authors,
# Apache License 2.0; modified for TorchPP. See THIRD_PARTY_NOTICES.md.
"""Log-normal mixture distribution for the IntensityFree TPP.

Port of the distribution used by Shchur et al., *Intensity-Free Learning of
Temporal Point Processes* (ICLR 2020); the implementation it is adapted from is
named in the header above.

The model places a mixture of log-normals directly on the inter-event time
``tau``.  We expose ``log_prob`` (density), ``log_cdf`` and
``log_survival_function`` so the surrounding TPP can derive a hazard intensity
``lambda(tau) = f(tau) / S(tau)`` and a closed-form cumulative intensity
``Lambda(tau) = -log S(tau)``.

The only numerically delicate piece is ``log_cdf`` / ``log_survival_function``
for the Normal component — torch has no numerically stable version, so we
re-implement them (clamping the CDF away from 0 and 1).
"""

from __future__ import annotations

import torch
import torch.distributions as dist
from torch.distributions import TransformedDistribution
from torch.distributions import MixtureSameFamily as TorchMixtureSameFamily
from torch.distributions import Normal as TorchNormal


def clamp_preserve_gradients(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """Clamp ``x`` to ``[min_val, max_val]`` while keeping gradients in the clamped region."""
    return x + (x.clamp(min_val, max_val) - x).detach()


class Normal(TorchNormal):
    """Normal with numerically guarded ``log_cdf`` / ``log_survival_function``."""

    def log_cdf(self, x: torch.Tensor) -> torch.Tensor:
        cdf = clamp_preserve_gradients(self.cdf(x), 1e-7, 1 - 1e-7)
        return cdf.log()

    def log_survival_function(self, x: torch.Tensor) -> torch.Tensor:
        cdf = clamp_preserve_gradients(self.cdf(x), 1e-7, 1 - 1e-7)
        return torch.log(1.0 - cdf)


class MixtureSameFamily(TorchMixtureSameFamily):
    """MixtureSameFamily with ``log_cdf`` / ``log_survival_function`` over components."""

    def log_cdf(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad(x)
        log_cdf_x = self.component_distribution.log_cdf(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_cdf_x + mix_logits, dim=-1)

    def log_survival_function(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad(x)
        log_sf_x = self.component_distribution.log_survival_function(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_sf_x + mix_logits, dim=-1)


class LogNormalMixtureDistribution(TransformedDistribution):
    """Mixture of log-normals over inter-event times.

    Args:
        locs:        means in log-space, shape ``(..., num_mix_components)``.
        log_scales:  log std-devs in log-space, same shape as ``locs``.
        log_weights: mixture logits (typically already log-softmaxed), same shape.

    The model's ``time_scale`` already normalises inter-event times to O(1), so the
    mixture is fit directly in scaled log-time (``τ = exp(log-normal mixture)``).
    """

    def __init__(self, locs, log_scales, log_weights, validate_args=None):
        # validate_args off: it crashes on legitimate B=0 batches.
        mixture_dist = dist.Categorical(logits=log_weights, validate_args=False)
        component_dist = Normal(loc=locs, scale=log_scales.exp(), validate_args=False)
        GMM = MixtureSameFamily(mixture_dist, component_dist, validate_args=False)
        self.transforms = [dist.ExpTransform()]
        sign = 1
        for transform in self.transforms:
            sign = sign * transform.sign
        self.sign = int(sign)
        super().__init__(GMM, self.transforms, validate_args=validate_args)

    def log_cdf(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms[::-1]:
            x = transform.inv(x)
        if self._validate_args:
            self.base_dist._validate_sample(x)
        if self.sign == 1:
            return self.base_dist.log_cdf(x)
        return self.base_dist.log_survival_function(x)

    def log_survival_function(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms[::-1]:
            x = transform.inv(x)
        if self._validate_args:
            self.base_dist._validate_sample(x)
        if self.sign == 1:
            return self.base_dist.log_survival_function(x)
        return self.base_dist.log_cdf(x)
