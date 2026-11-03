"""Native-unit readouts of fitted baselines and kernels (Poisson and Hawkes families).

Parameters live in internal (scaled) time; these helpers evaluate the baseline and kernel
modules there and convert to native units. ``φ`` below is the kernel term of the intensity:
for additive Hawkes ``λ_d(t) = μ_d(t) + Σ_{t_i<t} φ_{d,m_i}(t − t_i)``; for inhibitive models
the same terms enter before the link, ``λ_d(t) = link(μ_d(t) + Σ φ_{d,m_i}(t − t_i))``.
"""

from typing import Optional

import torch

from torchpp.modules.intensity import ExpKernelModule
from torchpp.utils.event_data import BatchedMVEventData, MVEventData
from torchpp.utils.math import right_limit


def _work_dtype(model) -> torch.dtype:
    p = next(model.parameters(), None)
    return p.dtype if p is not None and p.is_floating_point() else torch.get_default_dtype()


def _device(model) -> torch.device:
    p = next(model.parameters(), None)
    return p.device if p is not None else torch.device("cpu")


def _one_event_per_type(model, dtype) -> BatchedMVEventData:
    """``D`` sequences; sequence ``j`` holds a single event of type ``j`` at time 0."""
    dev = _device(model)
    seqs = [
        MVEventData(torch.zeros(1, dtype=dtype, device=dev), torch.tensor([j], device=dev)) for j in range(model.D)
    ]
    return BatchedMVEventData(mv_events=seqs)


def _as_times(x, model) -> torch.Tensor:
    t = torch.as_tensor(x, device=_device(model))
    if not t.is_floating_point():
        t = t.to(_work_dtype(model))
    if t.dim() != 1:
        raise ValueError(f"expected a 1-D tensor of times, got shape {tuple(t.shape)}")
    return t


def baseline_values(model, t, rate_scale: bool = True) -> torch.Tensor:
    """Baseline ``μ_d(t)`` at native times ``t`` ``(Q,)`` → ``(Q, D)``."""
    t = _as_times(t, model)
    s = model.time_scale.to(t.dtype)
    ts = _one_event_per_type(model, t.dtype)
    q = (t / s).unsqueeze(0).expand(ts.time_points.shape[0], -1)
    vals = model.baseline.intensities_at_queries(q, ts)[0]  # baselines ignore history → row 0
    return vals / s if rate_scale else vals


def kernel_values(model, dt, rate_scale: bool = True) -> torch.Tensor:
    """Kernel ``φ_{d,j}(Δt)`` at native lags ``dt`` ``(Q,)`` → ``(Q, D, D)``, indexed ``[q, target d, source j]``.

    Lags ``≤ 0`` give 0 (causality); a lag of exactly 0 is evaluated just after the event.
    """
    dt = _as_times(dt, model)
    s = model.time_scale.to(dt.dtype)
    ts = _one_event_per_type(model, dt.dtype)
    q = dt / s
    q = torch.where(q == 0, right_limit(q, ts.time_points.dtype), q)
    q = q.unsqueeze(0).expand(model.D, -1)  # (D_source, Q)
    vals = model.kernel.intensities_at_queries(q, ts)  # (D_source, Q, D_target)
    vals = vals.permute(1, 2, 0)  # (Q, D_target, D_source)
    return vals / s if rate_scale else vals


def branching_matrix(model, rate_scale: bool, t_max: Optional[float] = None, num_points: int = 4001) -> torch.Tensor:
    """``∫_0^∞ φ_{d,j}(Δt) dΔt`` in native units, ``(D, D)`` indexed ``[target d, source j]``.

    Exact for exponential kernels and for compactly supported kernels with a closed-form
    integral (spline, piecewise, MLP-spline). Other kernels are integrated with the
    trapezoid rule on ``[0, t_max]`` (native units, required), ``num_points`` nodes.
    """
    kernel = model.kernel
    dtype = _work_dtype(model)
    s = model.time_scale.to(dtype)
    if isinstance(kernel, ExpKernelModule):
        alpha, _ = kernel.transform_params()
        internal = alpha.to(dtype)  # ∫ α β e^{-βΔt} dΔt = α
    else:
        support = kernel.breakpoints()
        try:
            if support is None:
                raise NotImplementedError
            ts = _one_event_per_type(model, dtype)
            T_end = support.max().to(dtype).expand(model.D)
            internal = kernel.cumulative_intensity(T_end, ts).T  # (D_source, D_target) → [target, source]
        except NotImplementedError:
            if t_max is None:
                raise ValueError(
                    f"{type(kernel).__name__} has no closed-form integral; pass t_max (native units) to integrate."
                ) from None
            grid = torch.linspace(0.0, float(t_max), num_points, dtype=dtype, device=_device(model))
            native = kernel_values(model, grid, rate_scale=rate_scale)
            return torch.trapezoid(native, grid, dim=0)
    # Additive kernels are rates (φ_native = φ_internal/s): the integral is unit-free.
    # Pre-link terms are not rates (φ_native = φ_internal): the integral scales with s.
    return internal if rate_scale else internal * s
