from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from .event_data import BatchedMVEventData

# build_until geometric continuation (when the fixed-duration loop exhausts max_chunks without
# reaching target_cdf — a low-rate / heavy-tail process).  Growing the chunk geometrically covers
# any finite-mean horizon in a handful of steps; 2^64 is effectively unbounded.
_BUILD_UNTIL_GEOM_GROWTH = 2.0
_BUILD_UNTIL_GEOM_STEPS = 64


# ==============================================================================
# Function Utilities
# ==============================================================================


def inverse_softplus(x: Tensor):
    # Computes the inverse of softplus in a way that stays finite for large x.
    # For large positive x, softplus(x) ≈ x, so the inverse is well-approximated
    # by x itself. The naive log(expm1(x)) form overflows in float32 once x is
    # much above ~88, which can happen when initialising fast Hawkes decays.
    threshold = torch.tensor(20.0, dtype=x.dtype, device=x.device)
    return torch.where(x > threshold, x + torch.log1p(-torch.exp(-x)), torch.log(torch.expm1(x)))


# ==============================================================================
# Spline knot helpers
# ==============================================================================


def baseline_spline_knot_locs(T: float, K: int, schedule: str = "uniform", power: float = 2.0) -> Tensor:
    """Absolute knot positions for spline-baseline / piecewise-Poisson models.

    Returns a 1-D tensor of shape ``(K,)`` with positions on ``[0, T]``.

    Args:
        T:        Right endpoint of the spline range (in scaled time units).
        K:        Number of knots.
        schedule: ``"uniform"`` (equidistant) or ``"geometric"`` (power-law,
                  denser near 0).
        power:    Exponent *p* for the geometric schedule. ``p=1`` → uniform;
                  ``p>1`` → finer near 0.

    Returns:
        Tensor of shape ``(K,)``.
    """
    if schedule == "uniform":
        return torch.linspace(0.0, T, K)
    elif schedule == "geometric":
        k = torch.arange(K, dtype=torch.float32)
        return T * (k / (K - 1)) ** power
    else:
        raise ValueError(f"schedule must be 'uniform' or 'geometric', got '{schedule}'")


def interaction_spline_knot_gaps(T: float, K: int, schedule: str = "geometric", power: float = 2.0) -> Tensor:
    """Inter-knot gaps for spline-kernel and neural-spline / piecewise-constant models.

    Both Hawkes spline kernels and neural relative-time models use a gap
    parametrisation: the first knot is implicitly at 0 and the remaining
    ``K-1`` gaps tile ``[0, T]``.

    Args:
        T:        Right endpoint of the kernel/relative-time range.
        K:        Number of knots (including the implicit 0).
        schedule: ``"uniform"`` (equal gaps) or ``"geometric"`` (power-law,
                  gaps denser near 0, recommended for kernels).
        power:    Exponent *p* for the geometric schedule.

    Returns:
        Tensor of shape ``(K-1,)`` — the inter-knot gap widths.
    """
    if schedule == "uniform":
        return torch.full((K - 1,), T / (K - 1), dtype=torch.float32)
    elif schedule == "geometric":
        k = torch.arange(K, dtype=torch.float32)
        locs = T * (k / (K - 1)) ** power
        return locs[1:] - locs[:-1]
    else:
        raise ValueError(f"schedule must be 'uniform' or 'geometric', got '{schedule}'")


def loglinear_cumulative_intensity(
    mu: Tensor,
    slope: Tensor,
    T: Tensor,
) -> Tensor:
    """Cumulative intensity for a log-linear intensity function.

    Computes the closed-form integral

    .. math::

        \\int_0^T e^{\\mu_d + b_d \\cdot t}\\, dt

        = e^{\\mu_d} \\cdot \\frac{e^{b_d T} - 1}{b_d}

    for each dimension ``d`` independently.  A Taylor expansion guards against
    numerical instability when ``|b_d| < 1e-8``:

    .. math::

        \\frac{e^{bT} - 1}{b} \\approx T + \\frac{b T^2}{2} + \\frac{b^2 T^3}{6}

    This function is the shared core used by :class:`~torchpp.modules.intensity.LinearBaselineModule`
    (and its covariate-aware variants) to avoid duplicating the Taylor guard.

    Args:
        mu: Log-space intercepts per dimension. Shape: ``(D,)`` or ``(..., D)``
        slope: Log-space slopes per dimension. Shape: ``(D,)`` or ``(..., D)``
        T: Upper integration limits. Shape: ``(B, D)`` or ``(..., D)``

    Returns:
        Cumulative intensities. Shape: broadcast of ``(mu, slope, T)``
    """
    slope_near_zero = torch.abs(slope) < 1e-8

    # Taylor expansion for near-zero slope: ∫₀ᵀ exp(b·t) dt ≈ T + b·T²/2 + b²·T³/6
    # Multiplied by exp(μ) outside.
    taylor_integral = T + slope * T**2 / 2 + slope**2 * T**3 / 6

    # Closed-form for non-zero slope: (exp(b·T) - 1) / b
    # Use a safe denominator to avoid division by zero in the masked-out positions.
    safe_slope = torch.where(slope_near_zero, torch.ones_like(slope), slope)
    closed_form_integral = (torch.exp(safe_slope * T) - 1) / safe_slope

    time_integral = torch.where(slope_near_zero, taylor_integral, closed_form_integral)

    return torch.exp(mu) * time_integral


class LinearSpline(torch.nn.Module):
    """
    Linear spline utility for interpolation, integration, and inverse integration.

    All methods support two input shapes for time points t:
        - Shape (B,): Same time point for all D dimensions (broadcast)
        - Shape (B, D): Different time point for each dimension

    Output is always shape (B, D).
    """

    @staticmethod
    def interpolate(x_knots: torch.Tensor, y_knots: torch.Tensor, t: torch.Tensor):
        """Interpolates the splines at the given positions.

        For ``t`` beyond the last knot the value is extrapolated using the last
        knot height (constant/flat extrapolation). Queries before the first
        knot raise a :class:`ValueError` — the spline is undefined there.

        Args:
            x_knots (torch.Tensor): Position of the splines. Shared between all splines. Shape: (K,)
            y_knots (torch.Tensor): Height of the splines for each knot in each spline. Shape: (D, K)
            t (torch.Tensor): Time-points where to interpolate the splines. Shape: (B,) or (B, D)

        Returns:
            interpolation (torch.Tensor): Interpolated splines at t. Shape: (B, D)
        """
        D, K = y_knots.shape
        t_is_1d = t.dim() == 1
        B = t.shape[0]

        k, delta = LinearSpline._get_knot_info(x_knots, t)  # (B,) or (B, D)

        delta_times = x_knots[1:] - x_knots[:-1]  # (K-1)
        delta_heights = y_knots[:, 1:] - y_knots[:, :-1]  # (D, K-1)
        slopes = delta_heights / delta_times  # (D, K-1)
        segment_slopes = F.pad(slopes, (0, 1), value=0.0)  # (D, K)

        if t_is_1d:
            # k has shape (B,), can use simple indexing (broadcasts across D)
            h_k = y_knots[:, k]  # (D, B)
            slope_k = segment_slopes[:, k]  # (D, B)
            delta_2d = delta.unsqueeze(0)  # (1, B) -> broadcasts to (D, B)
        else:
            # k has shape (B, D), need to use gather
            k_2d = k.T  # (D, B)
            h_k = torch.gather(y_knots, 1, k_2d)  # (D, B)
            slope_k = torch.gather(segment_slopes, 1, k_2d)  # (D, B)
            delta_2d = delta.T  # (D, B)

        interpolation = h_k + slope_k * delta_2d  # (D, B)

        return interpolation.T  # (B, D)

    @staticmethod
    def integrate(x_knots: torch.Tensor, y_knots: torch.Tensor, t: torch.Tensor):
        """Calculates the integral of the splines at the given positions.

        For ``t`` beyond the last knot the integral is extrapolated using the
        last knot height (constant/flat extrapolation), consistent with
        :meth:`interpolate`.

        Args:
            x_knots (torch.Tensor): Position of the splines. Shared between all splines. Shape: (K,)
            y_knots (torch.Tensor): Height of the splines for each spline and each knot. Shape: (D, K)
            t (torch.Tensor): Time-points where up to where to integrate the splines (starting from 0).
                              Shape: (B,) or (B, D)

        Returns:
            integral: (torch.Tensor): Integrated splines up to t. Shape: (B, D)
        """
        D, K = y_knots.shape
        t_is_1d = t.dim() == 1
        B = t.shape[0]

        k, delta = LinearSpline._get_knot_info(x_knots=x_knots, t=t)  # (B,) or (B, D)

        delta_times = x_knots[1:] - x_knots[:-1]  # (K-1)
        delta_heights = y_knots[:, 1:] - y_knots[:, :-1]  # (D, K-1)
        slopes = delta_heights / delta_times  # (D, K-1)

        segment_areas = (
            y_knots[:, :-1] * delta_times + slopes / 2.0 * delta_times.unsqueeze(0) ** 2
        )  # integral_0^t ( h + slope * s)  ds, Shape: (D, K-1)

        prefix_sums = torch.cumsum(segment_areas, dim=1)  # Shape: (D, K-1)
        prefix_sums = F.pad(prefix_sums, (1, 0), value=0.0)  # (D, K)
        segment_slopes = F.pad(slopes, (0, 1), value=0.0)  # (D, K)

        if t_is_1d:
            # k has shape (B,), can use simple indexing (broadcasts across D)
            C_k = prefix_sums[:, k]  # (D, B)
            s_k = segment_slopes[:, k]  # (D, B)
            h_k = y_knots[:, k]  # (D, B)
            delta_2d = delta.unsqueeze(0)  # (1, B) -> broadcasts to (D, B)
        else:
            # k has shape (B, D), need to use gather
            k_2d = k.T  # (D, B)
            C_k = torch.gather(prefix_sums, 1, k_2d)  # (D, B)
            s_k = torch.gather(segment_slopes, 1, k_2d)  # (D, B)
            h_k = torch.gather(y_knots, 1, k_2d)  # (D, B)
            delta_2d = delta.T  # (D, B)

        # Integral of previous segments + square_area + triangular area
        # Done like this to avoid zero multiplications which can lead to NaNs if delta_2d is Infinite
        integral = C_k
        integral = integral + torch.where(h_k != 0.0, h_k * delta_2d, torch.zeros_like(h_k))
        integral = integral + torch.where(s_k != 0.0, s_k / 2 * delta_2d**2, torch.zeros_like(s_k))
        # (D, B)

        return integral.T  # (B, D)

    @staticmethod
    def inverse_integral(x_knots: torch.Tensor, y_knots: torch.Tensor, u: torch.Tensor):
        """Computes the inverse of the integral if defined (y_knots has to be strictly positive or negative).

        Given u, finds t such that u = int_0^t spline(x_knots, y_knots, s) ds.

        Args:
            x_knots (torch.Tensor): Position of the splines. Shared between all splines. Shape: (K,)
            y_knots (torch.Tensor): Height of the splines for each spline and each knot. Shape: (D, K)
            u (torch.Tensor): Target integral values. Shape: (B, D)

        Returns:
            t (torch.Tensor): The time values producing the desired u's. Shape: (B, D)
        """
        assert (y_knots > 0).all() or (y_knots < 0).all(), (
            "y_knots must be strictly positive or negative for invertibility"
        )

        # searchsorted requires ascending prefix sums; for all-negative heights the
        # integral descends, so solve the sign-flipped (positive) problem. The
        # solution t is unchanged: both sides of h_k*delta + s_k/2*delta^2 = u - C_k
        # scale by the same sign.
        sign = 1.0 if bool((y_knots > 0).all()) else -1.0
        y_knots = y_knots * sign
        u = u * sign

        D, K = y_knots.shape
        B = u.shape[0]

        delta_times = x_knots[1:] - x_knots[:-1]  # (K-1,)
        delta_heights = y_knots[:, 1:] - y_knots[:, :-1]  # (D, K-1)
        slopes = delta_heights / delta_times  # (D, K-1)

        segment_areas = y_knots[:, :-1] * delta_times + slopes / 2.0 * delta_times**2  # (D, K-1)

        prefix_sums = torch.cumsum(segment_areas, dim=1)  # (D, K-1)
        prefix_sums = F.pad(prefix_sums, (1, 0), value=0.0)  # (D, K) - integral from 0 to x_knots[k]

        segment_slopes = F.pad(slopes, (0, 1), value=0.0)  # (D, K)

        # For each (batch, dim), find which segment the target u falls into
        # u has shape (B, D), prefix_sums has shape (D, K)
        # Find k such that prefix_sums[d, k] <= u[b, d] < prefix_sums[d, k+1]
        k = torch.searchsorted(prefix_sums.unsqueeze(0).expand((B, D, K)), u.unsqueeze(2), right=True) - 1
        k = k.squeeze(2).T  # (D, B)

        C_k = torch.gather(prefix_sums, 1, k)  # (D, B) - cumulative integral up to knot k
        h_k = torch.gather(y_knots, 1, k)  # (D, B) - height at knot k
        s_k = torch.gather(segment_slopes, 1, k)  # (D, B) - slope in segment k
        x_k = x_knots[k]  # (D, B) - position of knot k

        # We need to solve:  h_k * delta + s_k/2 * delta^2 = u - C_k
        # Rearranging: s_k/2 * delta^2 + h_k * delta - (u - C_k) = 0
        #            = s_k/2 * delta^2 + h_k * delta - c = 0
        # Using quadratic formula: delta = (-h_k + sqrt(h_k^2 + 2*s_k*c)) / s_k

        c_val = u.T - C_k  # (D, B)

        slope_is_zero = torch.abs(s_k) < 1e-10

        # For zero slope: h_k * delta = u - C_k => delta = (u - C_k) / h_k = c_val / h_k
        delta_linear = c_val / h_k  # (D, B)

        discriminant = h_k**2 + 2 * s_k * c_val  # (D, B)
        discriminant = torch.clamp(discriminant, min=0.0)
        sqrt_disc = torch.sqrt(discriminant)

        # Safe division for the quadratic case: s_k_safe avoids NaN propagation
        # through the division even in masked-out positions (where s_k == 0).
        s_k_safe = torch.where(slope_is_zero, torch.ones_like(s_k), s_k)
        delta_quadratic = (-h_k + sqrt_disc) / s_k_safe
        delta = torch.where(slope_is_zero, delta_linear, delta_quadratic)  # (D, B)

        t = x_k + delta  # (D, B)

        return t.T  # (B, D)

    @staticmethod
    def interpolate_flat(x_knots: torch.Tensor, y_knots: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Interpolates N independent splines at N time points (one per spline).

        Convenience wrapper that accepts a batch-first ``(N, K)`` layout instead
        of the dimension-first ``(D, K)`` layout used by :meth:`interpolate`.

        Args:
            x_knots (torch.Tensor): Knot positions shared across all splines. Shape: (K,)
            y_knots (torch.Tensor): Knot heights, one row per spline. Shape: (N, K)
            t (torch.Tensor): Query time for each spline. Shape: (N,)

        Returns:
            torch.Tensor: Interpolated values, one per spline. Shape: (N,)
        """
        result = LinearSpline.interpolate(x_knots, y_knots, t.unsqueeze(0))  # (1, N)
        return result.squeeze(0)  # (N,)

    @staticmethod
    def integrate_flat(x_knots: torch.Tensor, y_knots: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Integrates N independent splines from 0 to their respective t values.

        Convenience wrapper that accepts a batch-first ``(N, K)`` layout instead
        of the dimension-first ``(D, K)`` layout used by :meth:`integrate`.
        Beyond the last knot the integral is extrapolated with constant
        intensity (last knot height), consistent with :meth:`integrate`.

        Args:
            x_knots (torch.Tensor): Knot positions shared across all splines. Shape: (K,)
            y_knots (torch.Tensor): Knot heights, one row per spline. Shape: (N, K)
            t (torch.Tensor): Upper integration limit for each spline. Shape: (N,)

        Returns:
            torch.Tensor: Integral values, one per spline. Shape: (N,)
        """
        result = LinearSpline.integrate(x_knots, y_knots, t.unsqueeze(0))  # (1, N)
        return result.squeeze(0)  # (N,)

    @staticmethod
    def _get_knot_info(x_knots: torch.Tensor, t: torch.Tensor):
        """
        Calculates indices and relative offsets for time points.

        Args:
            x_knots: Location of the knots. Shape: (K,)
            t: Time points. Shape: (B,) or (B, D) - any shape is supported

        Returns:
            k: Knot indices. Shape: same as t
            delta: Offset from knot (t - x_knots[k]). Shape: same as t
        """
        k = torch.searchsorted(x_knots, t, right=True) - 1
        if (k < 0).any():
            # Without this check k = -1 would silently wrap around to the LAST
            # knot via Python negative indexing.
            raise ValueError(
                f"Query time precedes the first knot (min query t = {t.min().item():.6g}, "
                f"first knot at {x_knots[0].item():.6g}). The spline is undefined there — "
                f"the module must place its first knot at the start of its support "
                f"(typically t=0). A params-/tensor-constructed baseline missing the "
                f"leading zero knot is the usual cause."
            )
        delta = t - x_knots[k]
        return k, delta


class PiecewiseConstant(torch.nn.Module):
    """
    Piecewise-constant function utility for interpolation, integration, and
    inverse integration.

    Between consecutive knots the function value equals the *left* knot height
    (zero-order hold).  Beyond the last knot the value is extrapolated using
    the last knot height (constant/flat extrapolation), consistent with
    :class:`LinearSpline`.

    All methods support two input shapes for time points t:
        - Shape (B,): Same time point for all D dimensions (broadcast)
        - Shape (B, D): Different time point for each dimension

    Output is always shape (B, D).
    """

    @staticmethod
    def interpolate(x_knots: torch.Tensor, y_knots: torch.Tensor, t: torch.Tensor):
        """Interpolates the piecewise-constant function at the given positions.

        The value at any point *t* is the height of the left knot of the
        segment containing *t*.  For *t* beyond the last knot the last knot
        height is used; queries before the first knot raise a
        :class:`ValueError` — the function is undefined there.

        Args:
            x_knots (torch.Tensor): Knot positions, shared across all dimensions. Shape: (K,)
            y_knots (torch.Tensor): Knot heights for each dimension and knot. Shape: (D, K)
            t (torch.Tensor): Query time points. Shape: (B,) or (B, D)

        Returns:
            interpolation (torch.Tensor): Function values at *t*. Shape: (B, D)
        """
        t_is_1d = t.dim() == 1

        k, _ = PiecewiseConstant._get_knot_info(x_knots, t)  # (B,) or (B, D)

        if t_is_1d:
            h_k = y_knots[:, k]  # (D, B)
        else:
            k_2d = k.T  # (D, B)
            h_k = torch.gather(y_knots, 1, k_2d)  # (D, B)

        return h_k.T  # (B, D)

    @staticmethod
    def integrate(x_knots: torch.Tensor, y_knots: torch.Tensor, t: torch.Tensor):
        """Calculates the integral of the piecewise-constant function up to *t*.

        Each segment contributes ``h_k * segment_length`` (a rectangle).  For
        *t* beyond the last knot the integral is extrapolated with the last
        knot height, consistent with :meth:`interpolate`.

        Args:
            x_knots (torch.Tensor): Knot positions, shared across all dimensions. Shape: (K,)
            y_knots (torch.Tensor): Knot heights for each dimension and knot. Shape: (D, K)
            t (torch.Tensor): Upper integration limit (starting from 0). Shape: (B,) or (B, D)

        Returns:
            integral (torch.Tensor): Integral from 0 to *t*. Shape: (B, D)
        """
        t_is_1d = t.dim() == 1

        k, delta = PiecewiseConstant._get_knot_info(x_knots, t)  # (B,) or (B, D)

        delta_times = x_knots[1:] - x_knots[:-1]  # (K-1,)

        # Area of each complete segment: h_k * Δx_k  (rectangle)
        segment_areas = y_knots[:, :-1] * delta_times  # (D, K-1)

        prefix_sums = torch.cumsum(segment_areas, dim=1)  # (D, K-1)
        prefix_sums = F.pad(prefix_sums, (1, 0), value=0.0)  # (D, K)

        if t_is_1d:
            C_k = prefix_sums[:, k]  # (D, B)
            h_k = y_knots[:, k]  # (D, B)
            delta_2d = delta.unsqueeze(0)  # (1, B) -> broadcasts to (D, B)
        else:
            k_2d = k.T  # (D, B)
            C_k = torch.gather(prefix_sums, 1, k_2d)  # (D, B)
            h_k = torch.gather(y_knots, 1, k_2d)  # (D, B)
            delta_2d = delta.T  # (D, B)

        # Integral = sum of completed rectangles + partial rectangle in current segment
        integral = C_k + torch.where(h_k != 0.0, h_k * delta_2d, torch.zeros_like(h_k))
        # (D, B)

        return integral.T  # (B, D)

    @staticmethod
    def inverse_integral(x_knots: torch.Tensor, y_knots: torch.Tensor, u: torch.Tensor):
        """Computes the inverse of the integral (y_knots must be strictly positive or negative).

        Given *u*, finds *t* such that ``u = ∫₀ᵗ f(s) ds``.

        Because the function is piecewise constant, each segment's integral is
        linear in *delta*, so the inversion is always a simple linear solve:
        ``delta = (u - C_k) / h_k``.

        Args:
            x_knots (torch.Tensor): Knot positions, shared across all dimensions. Shape: (K,)
            y_knots (torch.Tensor): Knot heights for each dimension and knot. Shape: (D, K)
            u (torch.Tensor): Target integral values. Shape: (B, D)

        Returns:
            t (torch.Tensor): Time values producing the target integrals. Shape: (B, D)
        """
        assert (y_knots > 0).all() or (y_knots < 0).all(), (
            "y_knots must be strictly positive or negative for invertibility"
        )

        # searchsorted requires ascending prefix sums; for all-negative heights the
        # integral descends, so solve the sign-flipped (positive) problem. The linear
        # solve delta = (u - C_k) / h_k is invariant under flipping both sides' sign.
        sign = 1.0 if bool((y_knots > 0).all()) else -1.0
        y_knots = y_knots * sign
        u = u * sign

        D, K = y_knots.shape
        B = u.shape[0]

        delta_times = x_knots[1:] - x_knots[:-1]  # (K-1,)

        # Area of each complete segment: h_k * Δx_k
        segment_areas = y_knots[:, :-1] * delta_times  # (D, K-1)

        prefix_sums = torch.cumsum(segment_areas, dim=1)  # (D, K-1)
        prefix_sums = F.pad(prefix_sums, (1, 0), value=0.0)  # (D, K)

        # Find segment: prefix_sums[d, k] <= u[b, d] < prefix_sums[d, k+1]
        k = torch.searchsorted(prefix_sums.unsqueeze(0).expand((B, D, K)), u.unsqueeze(2), right=True) - 1
        k = k.squeeze(2).T  # (D, B)

        C_k = torch.gather(prefix_sums, 1, k)  # (D, B)
        h_k = torch.gather(y_knots, 1, k)  # (D, B)
        x_k = x_knots[k]  # (D, B)

        # Linear solve: h_k * delta = u - C_k  =>  delta = (u - C_k) / h_k
        c_val = u.T - C_k  # (D, B)
        delta = c_val / h_k  # (D, B)

        t = x_k + delta
        return t.T  # (B, D)

    @staticmethod
    def interpolate_flat(x_knots: torch.Tensor, y_knots: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Interpolates N independent piecewise-constant functions at N time points.

        Convenience wrapper with batch-first ``(N, K)`` layout.

        Args:
            x_knots (torch.Tensor): Knot positions shared across all functions. Shape: (K,)
            y_knots (torch.Tensor): Knot heights, one row per function. Shape: (N, K)
            t (torch.Tensor): Query time for each function. Shape: (N,)

        Returns:
            torch.Tensor: Interpolated values, one per function. Shape: (N,)
        """
        result = PiecewiseConstant.interpolate(x_knots, y_knots, t.unsqueeze(0))  # (1, N)
        return result.squeeze(0)  # (N,)

    @staticmethod
    def integrate_flat(x_knots: torch.Tensor, y_knots: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Integrates N independent piecewise-constant functions from 0 to their respective t values.

        Convenience wrapper with batch-first ``(N, K)`` layout.

        Args:
            x_knots (torch.Tensor): Knot positions shared across all functions. Shape: (K,)
            y_knots (torch.Tensor): Knot heights, one row per function. Shape: (N, K)
            t (torch.Tensor): Upper integration limit for each function. Shape: (N,)

        Returns:
            torch.Tensor: Integral values, one per function. Shape: (N,)
        """
        result = PiecewiseConstant.integrate(x_knots, y_knots, t.unsqueeze(0))  # (1, N)
        return result.squeeze(0)  # (N,)

    @staticmethod
    def _get_knot_info(x_knots: torch.Tensor, t: torch.Tensor):
        """
        Calculates indices and relative offsets for time points.

        Args:
            x_knots: Knot positions. Shape: (K,)
            t: Time points. Shape: (B,) or (B, D)

        Returns:
            k: Knot indices. Shape: same as t
            delta: Offset from knot (t - x_knots[k]). Shape: same as t
        """
        k = torch.searchsorted(x_knots, t, right=True) - 1
        if (k < 0).any():
            # Without this check k = -1 would silently wrap around to the LAST
            # knot via Python negative indexing.
            raise ValueError(
                f"Query time precedes the first knot (min query t = {t.min().item():.6g}, "
                f"first knot at {x_knots[0].item():.6g}). The piecewise-constant function "
                f"is undefined there — the module must place its first knot at the start "
                f"of its support (typically t=0). A params-/tensor-constructed baseline "
                f"missing the leading zero knot is the usual cause."
            )
        delta = t - x_knots[k]
        return k, delta


# ==============================================================================
# Numerical Integration Utilities
# ==============================================================================


def _apply_trapezoidal(
    intensity_func: Callable,
    t_vals: torch.Tensor,
    ts: BatchedMVEventData,
    effective_t_end: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply the trapezoidal rule to a pre-built, sorted time grid.

    When ``effective_t_end`` is provided the function returns per-type integrals
    where each type d is integrated only up to ``effective_t_end[:, d]`` instead
    of the last grid point.  This is achieved via cumulative trapezoidal prefix
    sums: the prefix sum at the grid knot closest to ``effective_t_end[:, d]``
    is read off for each type independently.

    When ``effective_t_end`` is ``None`` the standard scalar trapezoidal rule is
    applied and the full integral up to the last grid point is returned.

    Args:
        intensity_func: Model intensity, called once with the full ``(B, M)`` grid and
            returning ``(B, M, D)`` (single multi-query pass).
        t_vals: Sorted time grid. Shape: ``(B, M)``.
        ts: Batched event data.
        effective_t_end: Per-type upper integration limits. Shape: ``(B, D)``.
            Must be ``<= t_vals[:, -1]`` for meaningful results.
            When ``None``, the standard integral over the full grid is returned.

    Returns:
        Integral values. Shape: ``(B, D)``.
    """
    # One multi-query pass over the grid: ``intensity_func`` accepts ``(B, M)`` and
    # returns ``(B, M, D)`` — numerically identical to the per-point loop.
    intensities = intensity_func(t=t_vals, ts=ts)  # (B, M, D)

    if effective_t_end is None:
        integrals = torch.trapezoid(intensities, t_vals.unsqueeze(2), dim=1)  # (B, D)
        return integrals

    # Per-type integrals via cumulative prefix sums.
    # cumulative_trapezoid returns (B, M-1, D); prepend a zero column for index 0.
    B, _, D = intensities.shape
    device = t_vals.device
    ci_increments = torch.cumulative_trapezoid(intensities, t_vals.unsqueeze(2), dim=1)  # (B, M-1, D)
    zero = torch.zeros(B, 1, D, device=device, dtype=ci_increments.dtype)
    ci_prefix = torch.cat([zero, ci_increments], dim=1)  # (B, M, D)

    # For each type find the grid index closest to effective_t_end[:, d].
    # t_vals: (B, M), effective_t_end: (B, D) → diff: (B, M, D)
    diff = (t_vals.unsqueeze(2) - effective_t_end.unsqueeze(1)).abs()
    gather_idx = diff.argmin(dim=1)  # (B, D)

    integrals = ci_prefix.gather(1, gather_idx.unsqueeze(1)).squeeze(1)  # (B, D)
    return integrals


def trapezoidal_integration(
    intensity_func: Callable,
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    ts: BatchedMVEventData,
    num_points: int = 100,
    extra_knots: Optional[torch.Tensor] = None,
    effective_t_end: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute integral using the trapezoidal rule.

    Args:
        intensity_func: Function ``(t, ts) -> (B, D)``.
        t_start: Start times. Shape: ``(B,)``.
        t_end: End times. Shape: ``(B,)``.
        ts: Batched event data.
        num_points: Number of uniform base grid points.
        extra_knots: Optional additional time points to insert into the grid.
            Shape: ``(B, K)``. Must lie in ``[t_start, t_end]``.  Useful for
            pinning the grid to discontinuities such as termination times so
            that the trapezoidal approximation stays accurate regardless of the
            interval width.
        effective_t_end: Per-type upper integration limits. Shape: ``(B, D)``.
            When provided the function returns per-type integrals up to each
            type's own limit instead of the common ``t_end``.  Requires
            ``extra_knots`` to contain the relevant knot positions so that the
            prefix-sum gather lands on an exact grid point.

    Returns:
        Integral values. Shape: ``(B, D)``.
    """
    device = t_start.device

    alpha = torch.linspace(0, 1, num_points, device=device, dtype=t_start.dtype)
    t_vals = t_start.unsqueeze(1) + (t_end - t_start).unsqueeze(1) * alpha.unsqueeze(0)  # (B, num_points)

    if extra_knots is not None:
        t_vals = torch.cat([t_vals, extra_knots], dim=1)
        t_vals, _ = torch.sort(t_vals, dim=1)

    return _apply_trapezoidal(intensity_func, t_vals, ts, effective_t_end)


def monte_carlo_trapezoidal_integration(
    intensity_func: Callable,
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    ts: BatchedMVEventData,
    num_points: int = 100,
    rng: Optional[torch.Generator] = None,
    extra_knots: Optional[torch.Tensor] = None,
    effective_t_end: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Hybrid Monte Carlo + Trapezoidal integration.

    Places ``num_points`` evaluation points randomly in ``[t_start, t_end]``
    while including the boundaries, then applies the trapezoidal rule on the
    sorted points.

    Args:
        intensity_func: Function ``(t, ts) -> (B, D)``.
        t_start: Start times. Shape: ``(B,)``.
        t_end: End times. Shape: ``(B,)``.
        ts: Batched event data.
        num_points: Total number of integration points (including boundaries).
        rng: Optional generator for reproducible quadrature points. ``None``
            uses the device's global RNG so the points differ on every call —
            a freshly constructed ``torch.Generator`` must NOT be used here:
            it always starts from the same default seed, which freezes the
            quadrature grid and lets models place intensity mass between the
            fixed nodes to evade the survival penalty.
        extra_knots: Optional additional time points to insert into the grid.
            Shape: ``(B, K)``. See :func:`trapezoidal_integration` for details.
        effective_t_end: Per-type upper integration limits. Shape: ``(B, D)``.
            See :func:`trapezoidal_integration` for details.

    Returns:
        Integral values. Shape: ``(B, D)``.
    """
    device = ts.time_points.device

    B = ts.shape[0]

    random_points = (t_end - t_start).unsqueeze(1) * torch.rand(
        (B, num_points - 2), generator=rng, device=device
    ) + t_start.unsqueeze(1)  # (B, num_points-2)

    t_vals = torch.cat([t_start.unsqueeze(1), random_points, t_end.unsqueeze(1)], dim=1)  # (B, num_points)

    if extra_knots is not None:
        t_vals = torch.cat([t_vals, extra_knots], dim=1)

    t_vals, _ = torch.sort(t_vals, dim=1)

    return _apply_trapezoidal(intensity_func, t_vals, ts, effective_t_end)


def monte_carlo_integration(
    intensity_func: Callable,
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    ts: BatchedMVEventData,
    num_points: int = 100,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Pure Monte Carlo integration (unbiased estimator).

    Draws ``num_points`` samples uniformly in ``[t_start, t_end]`` and
    estimates the integral as:

        Λ̂ = (t_end - t_start) * mean_i λ(τ_i),   τ_i ~ Uniform(t_start, t_end)

    Unlike the MC-trapezoidal hybrid this has no deterministic grid points, so
    each call returns a slightly different estimate (unless ``rng`` is fixed).
    Variance decreases as O(1/N); for smooth intensities the trapezoidal rule
    is usually more accurate at the same N, but MC integration is unbiased and
    avoids any aliasing artefacts.

    Args:
        intensity_func: Function accepting (t: Tensor shape (B,), ts) -> (B, D).
        t_start: Start times. Shape: (B,)
        t_end: End times. Shape: (B,)
        ts: Batched event data.
        num_points: Number of random samples.
        rng: Optional generator for reproducible samples. ``None`` uses the
            device's global RNG (fresh samples per call); see the note on
            :func:`monte_carlo_trapezoidal_integration`.

    Returns:
        Integral estimates. Shape: (B, D)
    """
    device = t_start.device

    B = t_start.shape[0]
    # Sample τ_i ~ Uniform(t_start, t_end), shape (B, num_points)
    u = torch.rand((B, num_points), generator=rng, device=device, dtype=t_start.dtype)
    t_vals = t_start.unsqueeze(1) + (t_end - t_start).unsqueeze(1) * u  # (B, num_points)

    # One multi-query pass; average over the sample axis (Monte-Carlo estimator).
    intensities = intensity_func(t=t_vals, ts=ts)  # (B, num_points, D)

    interval = (t_end - t_start).to(dtype=intensities.dtype)  # (B,)
    return interval.unsqueeze(1) * intensities.mean(dim=1)  # (B, D)


def numerical_integration(
    func: Callable,
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    ts: BatchedMVEventData,
    method: str = "trapezoidal",
    num_points: int = 100,
    rng: Optional[torch.Generator] = None,
    extra_knots: Optional[torch.Tensor] = None,
    effective_t_end: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Numerically integrate a function using the specified method.

    Args:
        func: Function to integrate, should accept (t: Tensor, ts: BatchedMVEventData) -> Tensor of shape (B, D)
        t_start: Start times. Shape: (B,)
        t_end: End times. Shape: (B,)
        ts: Batched event data
        method: One of ``"trapezoidal"``, ``"mc_trapezoidal"``, ``"monte_carlo"``
        num_points: Number of integration points
        rng: Random number generator (used by MC methods)
        extra_knots: Optional additional time points inserted into the grid before
            integration. Shape: ``(B, K)``. Ignored by ``"monte_carlo"``.
            Useful for anchoring the grid at discontinuities (e.g. termination
            times) so the approximation stays accurate as ``t_end`` grows.
        effective_t_end: Per-type upper integration limits. Shape: ``(B, D)``.
            When provided, returns per-type integrals up to each type's own
            limit via cumulative prefix-sum gather. Ignored by ``"monte_carlo"``.

    Returns:
        Integral values. Shape: (B, D)
    """
    if method == "trapezoidal":
        return trapezoidal_integration(func, t_start, t_end, ts, num_points, extra_knots, effective_t_end)
    elif method == "mc_trapezoidal":
        return monte_carlo_trapezoidal_integration(
            func, t_start, t_end, ts, num_points, rng, extra_knots, effective_t_end
        )
    elif method == "monte_carlo":
        return monte_carlo_integration(func, t_start, t_end, ts, num_points, rng)
    else:
        raise ValueError(
            f"Unknown integration method: {method!r}. Choose from 'trapezoidal', 'mc_trapezoidal', 'monte_carlo'."
        )


# ==============================================================================
# IntensityGrid: Pre-computed intensity grid with fast CI lookup
# ==============================================================================


@dataclass
class IntensityGrid:
    """
    Pre-computed intensity values on a time grid with cumulative intensity via
    trapezoidal prefix sums.

    This avoids the N² problem where methods like marginal_class_distribution
    call cumulative_intensity at each grid point, and each CI call itself
    evaluates intensity at ~100 internal grid points.

    Instead, we evaluate intensity once on a shared grid and derive CI from
    trapezoidal prefix sums over that grid.

    Attributes:
        t_grid: Time grid points. Shape: (B, num_points)
        intensities: Intensity values at each grid point. Shape: (B, num_points, D)
        ci_prefix: Cumulative intensity prefix sums (trapezoidal rule).
                   ci_prefix[:, k, :] = ∫_{t_grid[:,0]}^{t_grid[:,k]} λ(s) ds
                   Shape: (B, num_points, D)
        t_start: Start times of the grid. Shape: (B,)
    """

    t_grid: torch.Tensor  # (B, num_points)
    intensities: torch.Tensor  # (B, num_points, D)
    ci_prefix: torch.Tensor  # (B, num_points, D)
    t_start: torch.Tensor  # (B,)

    @staticmethod
    def build(
        intensity_func: Callable,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        ts: BatchedMVEventData,
        num_points: int = 200,
        ci_func: Optional[Callable] = None,
    ) -> IntensityGrid:
        """
        Build an IntensityGrid by evaluating the intensity function on a uniform grid.

        The cumulative intensity (CI) prefix sums can be computed in two ways:
        - If ``ci_func`` is provided (analytical CI models), CI values are computed
          exactly at each grid point as ``ci_func(t_k, ts) - ci_func(t_start, ts)``.
        - Otherwise, CI is approximated via trapezoidal prefix sums over the
          evaluated intensities (used by numerical CI models).

        Args:
            intensity_func: Model intensity; called once with the full (B, M) time grid and returning (B, M, D).
            t_start: Start times. Shape: (B,)
            t_end: End times. Shape: (B,)
            ts: Batched event data.
            num_points: Number of grid points.
            ci_func: Optional analytical cumulative intensity function
                (t, ts) -> (B, D) giving Λ(t). When provided, ci_prefix is
                computed as ci_func(t_k) - ci_func(t_start) instead of
                trapezoidal approximation. This gives exact CI values for
                models with closed-form cumulative intensity, but costs more compute.

        Returns:
            IntensityGrid with pre-computed intensities and cumulative intensity prefix sums.
        """
        device = t_start.device
        B = t_start.shape[0]

        alpha = torch.linspace(0, 1, num_points, device=device)  # (num_points,)
        t_grid = t_start.unsqueeze(1) + (t_end - t_start).unsqueeze(1) * alpha.unsqueeze(0)  # (B, num_points)

        # One multi-query pass over the whole grid.  ``intensity_func`` (the model's
        # ``intensity``) accepts ``(B, M)`` and returns ``(B, M, D)`` — numerically
        # identical to the per-point loop, but a single vectorised call.
        intensities = intensity_func(t=t_grid, ts=ts)  # (B, num_points, D)

        if ci_func is not None:
            # Analytical CI: compute exact CI at each grid point relative to t_start
            ci_base = ci_func(t_start, ts)  # (B, D)
            ci_list = [ci_func(t_grid[:, k], ts) - ci_base for k in range(num_points)]  # list of (B, D)
            ci_prefix = torch.stack(ci_list, dim=1)  # (B, num_points, D)
        else:
            # Numerical: trapezoidal prefix sums
            dt = t_grid[:, 1:] - t_grid[:, :-1]  # (B, num_points-1)

            # Trapezoidal increments: 0.5 * (f(t_k) + f(t_{k+1})) * dt
            avg_intensity = 0.5 * (intensities[:, :-1, :] + intensities[:, 1:, :])  # (B, num_points-1, D)
            increments = avg_intensity * dt.unsqueeze(2)  # (B, num_points-1, D)

            # Prefix sums: ci_prefix[:, 0, :] = 0, ci_prefix[:, k, :] = sum of increments up to k-1
            ci_prefix = torch.zeros(B, num_points, intensities.shape[2], device=device, dtype=intensities.dtype)
            ci_prefix[:, 1:, :] = torch.cumsum(increments, dim=1)

        return IntensityGrid(
            t_grid=t_grid,
            intensities=intensities,
            ci_prefix=ci_prefix,
            t_start=t_start,
        )

    @property
    def num_points(self) -> int:
        return self.t_grid.shape[1]

    @property
    def D(self) -> int:
        return self.intensities.shape[2]

    @property
    def B(self) -> int:
        return self.t_grid.shape[0]

    def ci_at_grid(self, k: int) -> torch.Tensor:
        """
        Return the cumulative intensity from t_start to t_grid[:, k].

        Args:
            k: Grid index.

        Returns:
            CI values. Shape: (B, D)
        """
        return self.ci_prefix[:, k, :]

    def intensity_at_grid(self, k: int) -> torch.Tensor:
        """
        Return the intensity at t_grid[:, k].

        Args:
            k: Grid index.

        Returns:
            Intensity values. Shape: (B, D)
        """
        return self.intensities[:, k, :]

    def ci_at_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Interpolate cumulative intensity at arbitrary time points t using the
        pre-computed prefix sums and intensities.

        For each batch element, finds the grid segment containing t and computes:
            CI(t) = ci_prefix[k] + 0.5 * (intensity[k] + intensity_interp(t)) * (t - t_grid[k])

        where intensity_interp is linearly interpolated between grid points.

        Args:
            t: Time points. Shape: (B,)

        Returns:
            Cumulative intensity at t. Shape: (B, D)
        """
        B = t.shape[0]
        device = t.device

        # Find which segment t falls into: k such that t_grid[:, k] <= t < t_grid[:, k+1]
        # searchsorted gives the insertion point, so subtract 1 and clamp
        k = torch.searchsorted(self.t_grid, t.unsqueeze(1), right=True).squeeze(1) - 1  # (B,)
        k = k.clamp(min=0, max=self.num_points - 2)  # (B,)

        k_expanded = k.unsqueeze(1).unsqueeze(2).expand(B, 1, self.D)  # (B, 1, D)
        ci_k = torch.gather(self.ci_prefix, 1, k_expanded).squeeze(1)  # (B, D)
        lam_k = torch.gather(self.intensities, 1, k_expanded).squeeze(1)  # (B, D)

        k1_expanded = (k + 1).unsqueeze(1).unsqueeze(2).expand(B, 1, self.D)
        lam_k1 = torch.gather(self.intensities, 1, k1_expanded).squeeze(1)  # (B, D)

        t_k = torch.gather(self.t_grid, 1, k.unsqueeze(1)).squeeze(1)  # (B,)
        t_k1 = torch.gather(self.t_grid, 1, (k + 1).unsqueeze(1)).squeeze(1)  # (B,)

        dt_segment = t_k1 - t_k  # (B,)
        dt = t - t_k  # (B,)

        frac = torch.where(dt_segment > 0, dt / dt_segment, torch.zeros_like(dt))  # (B,)

        # Linearly interpolate intensity at t
        lam_t = lam_k + frac.unsqueeze(1) * (lam_k1 - lam_k)  # (B, D)

        # Trapezoidal increment from t_k to t
        ci_increment = 0.5 * (lam_k + lam_t) * dt.unsqueeze(1)  # (B, D)

        return ci_k + ci_increment  # (B, D)

    def total_ci(self) -> torch.Tensor:
        """
        Return the total cumulative intensity over the full grid range.

        Returns:
            Total CI. Shape: (B, D)
        """
        return self.ci_prefix[:, -1, :]

    def intensity_at_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Interpolate intensity at arbitrary time points t using linear interpolation
        between grid points.

        Args:
            t: Time points. Shape: (B,)

        Returns:
            Interpolated intensity at t. Shape: (B, D)
        """
        B = t.shape[0]

        k = torch.searchsorted(self.t_grid, t.unsqueeze(1), right=True).squeeze(1) - 1  # (B,)
        k = k.clamp(min=0, max=self.num_points - 2)

        k_exp = k.unsqueeze(1).unsqueeze(2).expand(B, 1, self.D)
        k1_exp = (k + 1).unsqueeze(1).unsqueeze(2).expand(B, 1, self.D)
        lam_k = torch.gather(self.intensities, 1, k_exp).squeeze(1)  # (B, D)
        lam_k1 = torch.gather(self.intensities, 1, k1_exp).squeeze(1)  # (B, D)

        t_k = torch.gather(self.t_grid, 1, k.unsqueeze(1)).squeeze(1)  # (B,)
        t_k1 = torch.gather(self.t_grid, 1, (k + 1).unsqueeze(1)).squeeze(1)  # (B,)
        dt_seg = (t_k1 - t_k).clamp(min=1e-12)
        frac = ((t - t_k) / dt_seg).clamp(0, 1)  # (B,)

        return lam_k + frac.unsqueeze(1) * (lam_k1 - lam_k)  # (B, D)

    def extend(
        self,
        intensity_func: Callable,
        t_new_end: torch.Tensor,
        ts: BatchedMVEventData,
        num_new_points: int,
        ci_func: Optional[Callable] = None,
    ) -> IntensityGrid:
        """
        Extend this grid to cover [t_start, t_new_end] by appending new grid points.

        The new points are uniformly spaced from the current grid's last time
        to ``t_new_end``. Cumulative intensity prefix sums continue seamlessly
        from the existing grid.

        Args:
            intensity_func: Model intensity; called once with the full (B, M) time grid and returning (B, M, D).
            t_new_end: New end times. Shape: (B,)
            ts: Batched event data.
            num_new_points: Number of *new* grid points to append (excluding the
                junction point, which is the last point of the current grid).
            ci_func: Optional analytical CI function, same as in build().

        Returns:
            A new IntensityGrid covering [t_start, t_new_end] with the original
            points plus the new points appended.
        """
        device = self.t_grid.device

        t_cur_end = self.t_grid[:, -1]  # (B,)
        ci_at_cur_end = self.ci_prefix[:, -1, :]  # (B, D)
        lam_at_cur_end = self.intensities[:, -1, :]  # (B, D)

        # Build new uniform grid from t_cur_end to t_new_end (exclusive of t_cur_end itself)
        alpha = torch.linspace(0, 1, num_new_points + 1, device=device)[1:]  # skip 0 (= cur end)
        new_t_grid = t_cur_end.unsqueeze(1) + (t_new_end - t_cur_end).unsqueeze(1) * alpha.unsqueeze(
            0
        )  # (B, num_new_points)

        new_intensities = intensity_func(t=new_t_grid, ts=ts)  # (B, num_new_points, D) — one multi-query pass

        if ci_func is not None:
            # Analytical CI relative to original t_start
            ci_base = ci_func(self.t_start, ts)  # (B, D)
            new_ci_list = [ci_func(new_t_grid[:, k], ts) - ci_base for k in range(num_new_points)]
            new_ci_prefix = torch.stack(new_ci_list, dim=1)  # (B, num_new_points, D)
        else:
            # Trapezoidal: first increment bridges from current last point to first new point
            # Prepend the last intensity from the current grid for computing increments
            all_new_lam = torch.cat([lam_at_cur_end.unsqueeze(1), new_intensities], dim=1)  # (B, num_new_points+1, D)
            all_new_t = torch.cat([t_cur_end.unsqueeze(1), new_t_grid], dim=1)  # (B, num_new_points+1)

            dt = all_new_t[:, 1:] - all_new_t[:, :-1]  # (B, num_new_points)
            avg_lam = 0.5 * (all_new_lam[:, :-1, :] + all_new_lam[:, 1:, :])  # (B, num_new_points, D)
            increments = avg_lam * dt.unsqueeze(2)  # (B, num_new_points, D)

            # Continue from ci_at_cur_end
            new_ci_prefix = ci_at_cur_end.unsqueeze(1) + torch.cumsum(increments, dim=1)  # (B, num_new_points, D)

        return IntensityGrid(
            t_grid=torch.cat([self.t_grid, new_t_grid], dim=1),
            intensities=torch.cat([self.intensities, new_intensities], dim=1),
            ci_prefix=torch.cat([self.ci_prefix, new_ci_prefix], dim=1),
            t_start=self.t_start,
        )

    @staticmethod
    def build_until(
        intensity_func: Callable,
        t_start: torch.Tensor,
        ts: BatchedMVEventData,
        points_per_unit_time: float = 200.0,
        target_cdf: Optional[float] = 0.999,
        T_max: Optional[torch.Tensor] = None,
        ci_func: Optional[Callable] = None,
        chunk_duration: float = 0.1,
        max_chunks: int = 100,
    ) -> IntensityGrid:
        """
        Build an IntensityGrid incrementally until a stopping condition is met.

        Grows the grid in chunks of fixed duration, stopping when:
          1. CDF ≥ target_cdf for ALL batch elements (if target_cdf is set), OR
          2. The grid reaches T_max for ALL batch elements (if T_max is set), OR
          3. max_chunks is reached (safety bound).

        If both target_cdf and T_max are set, the grid extends until the CDF
        condition is satisfied OR T_max is reached, whichever comes first per
        batch element.

        Grid density is controlled by ``points_per_unit_time``, giving uniform
        resolution regardless of the total time range.

        Args:
            intensity_func: Model intensity; called once with the full (B, M) time grid and returning (B, M, D).
            t_start: Start times. Shape: (B,)
            ts: Batched event data.
            points_per_unit_time: Number of grid points per unit of scaled time.
                Higher = more accurate integration but more intensity evaluations.
            target_cdf: Stop when CDF ≥ this value for all batch elements.
                CDF(t) = 1 - exp(-Σ_d ΔΛ_d(t)). Set to None to disable.
            T_max: Hard upper time limit. Shape: (B,) or None.
                When set, the grid will not extend past these times.
            ci_func: Optional analytical CI function, same as in build().
            chunk_duration: Duration of each extension chunk in scaled time units.
            max_chunks: Maximum number of chunks (safety limit).

        Returns:
            IntensityGrid covering [t_start, t_stop] where t_stop is determined
            by the stopping conditions.
        """
        device = t_start.device
        B = t_start.shape[0]

        # Compute target total CI from target_cdf: Σ_d ΔΛ_d = -log(1 - target_cdf)
        target_total_ci = None
        if target_cdf is not None:
            target_total_ci = -torch.log1p(torch.tensor(-target_cdf, device=device))  # scalar

        points_per_chunk = max(int(round(points_per_unit_time * chunk_duration)), 5)

        t_end = t_start + chunk_duration  # (B,)
        if T_max is not None:
            t_end = torch.min(t_end, T_max)

        grid = IntensityGrid.build(
            intensity_func=intensity_func,
            t_start=t_start,
            t_end=t_end,
            ts=ts,
            num_points=points_per_chunk,
            ci_func=ci_func,
        )

        for _ in range(max_chunks - 1):
            total_ci = grid.ci_prefix[:, -1, :].sum(dim=1)  # (B,)

            done = torch.ones(B, dtype=torch.bool, device=device)

            if target_total_ci is not None:
                done = done & (total_ci >= target_total_ci)

            if T_max is not None:
                at_tmax = grid.t_grid[:, -1] >= T_max
                done = done | at_tmax  # reaching T_max also counts as done

            if done.all():
                break

            cur_end = grid.t_grid[:, -1]  # (B,)
            new_end = cur_end + chunk_duration  # (B,)
            if T_max is not None:
                new_end = torch.min(new_end, T_max)

            grid = grid.extend(
                intensity_func=intensity_func,
                t_new_end=new_end,
                ts=ts,
                num_new_points=points_per_chunk,
                ci_func=ci_func,
            )

        # Geometric continuation: the fixed-duration loop above caps the horizon at
        # ``t_start + chunk_duration * max_chunks``.  For a low-rate / heavy-tail process the CDF
        # target is not met by then and the grid was silently truncated (corrupting mean/median
        # time and the marginal type distribution).  Keep extending with a geometrically growing
        # chunk until the target is reached, then warn if it still is not.  This is a no-op when
        # the loop above already converged (the first check breaks immediately), so any sequence
        # that met ``target_cdf`` within ``max_chunks`` keeps its exact grid.  Skipped when a hard
        # ``T_max`` bounds the horizon (reaching it is a legitimate stop, not a shortfall).
        if target_total_ci is not None and T_max is None:
            chunk = chunk_duration
            for _ in range(_BUILD_UNTIL_GEOM_STEPS):
                if bool((grid.ci_prefix[:, -1, :].sum(dim=1) >= target_total_ci).all()):
                    break
                chunk = chunk * _BUILD_UNTIL_GEOM_GROWTH
                grid = grid.extend(
                    intensity_func=intensity_func,
                    t_new_end=grid.t_grid[:, -1] + chunk,
                    ts=ts,
                    num_new_points=points_per_chunk,
                    ci_func=ci_func,
                )
            n_short = int((grid.ci_prefix[:, -1, :].sum(dim=1) < target_total_ci).sum())
            if n_short:
                warnings.warn(
                    f"IntensityGrid.build_until did not reach target_cdf for {n_short}/{B} sequences "
                    "even after geometric extension; grid-based quantities (mean/median next-event "
                    "time, marginal type distribution) may be truncated. The intensity is likely "
                    "near-zero over a very long horizon — check the data's time scaling.",
                    stacklevel=2,
                )

        return grid


# ==============================================================================
# Monotone Function Inversion Utilities
# ==============================================================================


def bracket_monotone(
    monot_func,
    low: torch.Tensor,
    target: torch.Tensor,
    initial_high: torch.Tensor | None = None,
    expand_factor: float = 2.0,
    max_expands: int = 60,
    func_kwargs: dict = dict(),
):
    """
    Find a bracket [low, high] such that monot_func(high) >= target assuming monot_func is monotonically increasing.

    Supports batched operations where low, target, and monot_func output are tensors of shape (T,).

    Args:
        monot_func: callable that takes tensor of shape (T,) and returns tensor of shape (T,)
        low: starting lower bounds. Shape: (T,)
        target: target cumulative values. Shape: (T,)
        initial_high: optional initial high to try; if None, uses low + 1.0. Shape: (T,) or None
        expand_factor: multiplicative expansion factor for high
        max_expands: maximum number of expansions

    Returns:
        (low, high, ci_low, ci_high) - all tensors of shape (T,)
    """
    if initial_high is None:
        high = low + 1.0
    else:
        high = initial_high.clone()

    ci_low = monot_func(low, **func_kwargs)
    ci_high = monot_func(high, **func_kwargs)

    needs_expand = ci_high < target

    expands = 0
    while needs_expand.any() and expands < max_expands:
        # Exponentially expand the span for elements that need it
        span = high - low
        new_high = low + span * expand_factor

        high = torch.where(needs_expand, new_high, high)

        ci_high = monot_func(high, **func_kwargs)
        needs_expand = ci_high < target
        expands += 1

    return low, high, ci_low, ci_high


def invert_monotone_newton(
    monot_func,
    d_func,
    target: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    tol: float = 1e-6,
    max_iters: int = 50,
    mono_kwargs: dict = dict(),
    d_kwargs: dict = dict(),
) -> Tensor:
    """
    Invert a monotone cumulative function ci_func(T) to find T such that ci_func(T) = target.

    Uses Newton's method with derivative provided by d_func.
    Falls back to bisection when Newton proposes values outside the [low, high] bracket or derivative is too small.

    Supports batched operations where all inputs and outputs are tensors of shape (T,).

    Args:
        monot_func: callable that takes tensor of shape (T,) and returns tensor of shape (T,)
        d_func: callable that takes tensor of shape (T,) and returns tensor of shape (T,) (derivative)
        target: target cumulative values. Shape: (T,)
        low, high: bracket with ci(low) <= target <= ci(high). Shape: (T,)
        tol: tolerance on |ci(T)-target|
        max_iters: maximum iterations

    Returns:
        T: inverted values. Shape: (T,)
    """
    T = 0.5 * (low + high)

    converged = torch.zeros_like(T, dtype=torch.bool)

    for _ in range(max_iters):
        ci_val = monot_func(T, **mono_kwargs)
        resid = ci_val - target

        newly_converged = torch.abs(resid) <= tol
        converged = converged | newly_converged

        if converged.all():
            return T

        lam = d_func(T, **d_kwargs)

        # Identify where we need bisection (tiny or invalid derivative)
        use_bisection = (lam <= 1e-12) | torch.isnan(lam)

        T_newton = T - resid / torch.where(use_bisection, torch.ones_like(lam), lam)

        T_bisection = 0.5 * (low + high)

        # Use bisection if Newton step leaves bracket or is nan/inf
        out_of_bracket = (T_newton <= low) | (T_newton >= high) | torch.isnan(T_newton) | torch.isinf(T_newton)
        T_new = torch.where(use_bisection | out_of_bracket, T_bisection, T_newton)

        # Don't update already converged elements
        T_new = torch.where(converged, T, T_new)

        ci_new = monot_func(T_new, **mono_kwargs)
        low = torch.where((ci_new < target) & ~converged, T_new, low)
        high = torch.where((ci_new >= target) & ~converged, T_new, high)

        T = T_new

    return T


def invert_CI(
    cumulative_intensity_func: Callable[[torch.Tensor], torch.Tensor],
    intensity_func: Callable[[torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    bracket_init: Tuple[torch.Tensor, torch.Tensor],
    tol: float = 1e-6,
    max_newton_iters: int = 50,
    expand_factor: float = 2.0,
    max_expansions: int = 100,
) -> torch.Tensor:
    """
    Invert cumulative intensity using Newton's method with bracketing.

    Args:
        cumulative_intensity_func: Callable that returns cumulative intensity at time t (thats absolute time)
        intensity_func: Callable that returns intensity (derivative of CDF) at time t (thats absolute time)
        target: Target cumulative intensity value to invert
        bracket_init: Initial bracket interval
        tol: Tolerance for convergence
        max_newton_iters: Maximum Newton iterations
        expand_factor: Factor to expand bracket by
        max_expansions: Maximum bracket expansion iterations

    Returns:
        Time t such that cumulative_intensity_func(t) ≈ target
    """
    low, high, ci_low, ci_high = bracket_monotone(
        cumulative_intensity_func,
        bracket_init[0],
        target,
        initial_high=bracket_init[1],
        expand_factor=expand_factor,
        max_expands=max_expansions,
    )

    t_star = invert_monotone_newton(
        cumulative_intensity_func,
        intensity_func,
        target,
        low,
        high,
        tol=tol,
        max_iters=max_newton_iters,
    )

    return t_star
