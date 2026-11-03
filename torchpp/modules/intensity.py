"""Intensity modules for Hawkes processes — baseline and kernel components."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.event_data import BatchedMVEventData, DatasetStats
from ..utils.init_record import records_init
from ..utils.seeding import init_generator
from ..utils.math import inverse_softplus, LinearSpline, PiecewiseConstant, loglinear_cumulative_intensity


def _check_knots_start_at_zero(knot_locs: torch.Tensor, cls_name: str) -> torch.Tensor:
    """Validate that a knot-location vector starts at t=0.

    Baseline intensities are queried from t=0 (sequence start), and the spline /
    piecewise interpolators raise for queries before the first knot. Tensor
    ``delta_ts`` are turned into knot locations via ``cumsum``, so the first
    entry must be 0 — failing here at construction is far clearer than failing
    on the first intensity query.
    """
    if knot_locs.numel() == 0 or knot_locs[0].item() != 0.0:
        first = knot_locs[0].item() if knot_locs.numel() else "<empty>"
        raise ValueError(
            f"{cls_name}: first knot must lie at t=0, got {first}. Knot locations are "
            f"cumsum(delta_ts), so the first delta_ts entry must be 0.0 — e.g. "
            f"delta_ts = [0, g1, g2, ...] for gaps g_i between consecutive knots."
        )
    return knot_locs


class BaselineIntensityModule(nn.Module, ABC):
    """
    Abstract base class for baseline intensity modules in Hawkes processes.

    A baseline module represents the exogenous intensity (spontaneous event rate)
    in the Hawkes process. It is independent of historical events.

    Termination Handling:
        Baseline modules do NOT know about termination directly. The parent
        HawkesProcess computes termination_times and integrates them into T = torch.minimum(T.unsqueeze(1).expand_as(termination_times), termination_times).

        For intensity(), termination masking is handled at the HawkesProcess level
        (via TemporalPointProcess._get_termination_mask), so modules just return
        raw intensity values.
    """

    _tpp_recorded = True  # constructor arguments are recorded for checkpoints

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        records_init(cls)


    def __init__(self, D: int):
        """
        Args:
            D: Number of event types (dimensions)
        """
        super().__init__()
        self.D = D

    def breakpoints(self) -> Optional[torch.Tensor]:
        """Absolute times where the baseline's slope or value changes (knots), or ``None``.

        Between consecutive breakpoints the baseline is linear or constant, so its maximum over
        a window lies at a breakpoint or a window edge (used to bound the thinning sampler).
        """
        return None

    @abstractmethod
    def transform_params(self) -> Tuple[torch.Tensor, ...]:
        """
        Return baseline parameters in their constrained form.

        Returns:
            Baseline intensity parameters. Shape: (D,)
        """
        pass

    @abstractmethod
    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute baseline intensity at time t.

        Args:
            t: Time points. Shape: (B,)
            ts: Historical events (for context, may not be used by all baselines). Shape: (B,N)

        Returns:
            Baseline intensity for each event type. Shape: (B,D)
        """
        pass

    def intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Baseline intensity at Q query times per sequence. Shape ``(B, Q)`` -> ``(B, Q, D)``.

        Default loops the single-time ``intensity`` over the query axis (identical to
        per-point evaluation).  History-free baselines override with a vectorised form.
        """
        Q = query_times.shape[1]
        return torch.stack([self.intensity(query_times[:, k], ts) for k in range(Q)], dim=1)  # (B, Q, D)

    def cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute the integral of baseline intensity from 0 to T.

        Normalises ``T`` to shape ``(B, D)`` (broadcasting ``(B, 1)`` or expanding
        ``(B,)`` across all D dimensions) before delegating to
        ``_cumulative_intensity``.  Subclasses should override ``_cumulative_intensity``
        and can assume ``T`` is always ``(B, D)``.

        Args:
            T: End times. Shape: ``(B,)``, ``(B, 1)``, or ``(B, D)``
            ts: Historical events. Shape: (B, N)

        Returns:
            Cumulative baseline intensity for each event type. Shape: (B, D)
        """
        B = T.shape[0]
        if T.dim() == 1:
            T = T.unsqueeze(1).expand(B, self.D)  # (B,) -> (B, D)
        elif T.shape[1] == 1:
            T = T.expand(B, self.D)  # (B, 1) -> (B, D)
        return self._cumulative_intensity(T, ts)

    def _cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Internal implementation of ``cumulative_intensity``.  ``T`` is guaranteed
        to have shape ``(B, D)`` when this is called.

        Subclasses used with analytical CI must override this method.
        Subclasses used exclusively with numerical integration may leave it
        unimplemented.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _cumulative_intensity(). "
            "This is only required for baselines used with analytical CI."
        )

    def event_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute the positive part of the likelihood for the baseline intensities: the baseline intensities at event times for the correct class.

        This corresponds to the "data fit" term in the likelihood, measuring how well
        the model predicts the observed events.

        Can be overwritten to provide a far more efficient implementation that only computes the intensities of the "correct" classes at each time-point.

        Args:
            ts: Observed event sequences (batched, padded). Shape: (batch_size, seq_len)

        Returns:
            Baseline intensities for each event in batch. Shape: (batch_size, seq_len)

        """

        B = ts.time_points.shape[0]
        L = ts.time_points.shape[1]
        valid_events = ts.event_mask  # (B, L)

        intensities = torch.stack([self.intensity(ts.time_points[:, i], ts=ts) for i in range(L)], dim=1)  # (B, L, D)
        intensities = torch.gather(intensities, dim=2, index=ts.event_types_idx_safe.unsqueeze(-1)).squeeze(
            -1
        )  # (B, L)

        intensities[~valid_events] = 0.0

        return intensities  # (B, L)


class KernelIntensityModule(nn.Module, ABC):
    """
    Abstract base class for kernel functions in Hawkes processes.

    A kernel module represents the endogenous component (self-exciting/self-inhibiting),
    capturing how past events influence future event intensity.

    Termination Handling:
        Kernel modules do NOT know about termination directly. The parent
        HawkesProcess computes termination_times and passes them to cumulative_intensity().

        For intensity(), termination masking is handled at the HawkesProcess level
        (via TemporalPointProcess._get_termination_mask), so modules just return
        raw intensity values, which are masked later.
    """

    _tpp_recorded = True  # constructor arguments are recorded for checkpoints

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        records_init(cls)


    def __init__(self, D: int):
        """
        Args:
            D: Number of event types (dimensions)
        """
        super().__init__()
        self.D = D

    def breakpoints(self) -> Optional[torch.Tensor]:
        """Lags Δt after an event where the kernel's slope or value changes (knots), or ``None``.

        Between consecutive breakpoints the kernel is linear or constant in Δt, so its maximum
        lies at a breakpoint or a window edge (used to bound the thinning sampler).
        """
        return None

    @abstractmethod
    def transform_params(self) -> Tuple[torch.Tensor, ...]:
        """
        Return kernel parameters in their constrained form.

        Returns:
            Tuple of kernel parameters (structure depends on specific kernel)
        """
        pass

    @abstractmethod
    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute kernel contribution to intensity at time t.

        Args:
            t: Time points. Shape: (B,)
            ts: Historical events. Shape: (B,N)

        Returns:
            Kernel intensity contribution for each event type. Shape: (B,D)
        """
        pass

    def intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Kernel intensity at Q query times per sequence. Shape ``(B, Q)`` -> ``(B, Q, D)``.

        Default loops the single-time ``intensity`` over the query axis (identical to
        per-point evaluation).  Kernels override with a vectorised causal sum.
        """
        Q = query_times.shape[1]
        return torch.stack([self.intensity(query_times[:, k], ts) for k in range(Q)], dim=1)  # (B, Q, D)

    def cumulative_intensity(
        self,
        T: torch.Tensor,
        ts: BatchedMVEventData,
    ) -> torch.Tensor:
        """
        Compute the integral of kernel intensity from 0 to T.

        Normalises ``T`` to shape ``(B, D)`` (broadcasting ``(B, 1)`` or expanding
        ``(B,)`` across all D dimensions) before delegating to
        ``_cumulative_intensity``.  Subclasses should override ``_cumulative_intensity``
        and can assume ``T`` is always ``(B, D)``.

        Args:
            T: End times. Shape: ``(B,)``, ``(B, 1)``, or ``(B, D)``
            ts: Historical events. Shape: (B, N)

        Returns:
            Cumulative kernel intensity for each event type. Shape: (B, D)
        """
        B = T.shape[0]
        if T.dim() == 1:
            T = T.unsqueeze(1).expand(B, self.D)  # (B,) -> (B, D)
        elif T.shape[1] == 1:
            T = T.expand(B, self.D)  # (B, 1) -> (B, D)
        return self._cumulative_intensity(T, ts)

    def _cumulative_intensity(
        self,
        T: torch.Tensor,
        ts: BatchedMVEventData,
    ) -> torch.Tensor:
        """
        Internal implementation of ``cumulative_intensity``.  ``T`` is guaranteed
        to have shape ``(B, D)`` when this is called.

        Subclasses used exclusively with numerical integration may leave this
        unimplemented.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _cumulative_intensity(). "
            "This is only required for kernels used with analytical CI."
        )

    def event_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute the positive part of the likelihood for the kernel intensities: the kernel intensities at event times for the correct class.

        This corresponds to the "data fit" term in the likelihood, measuring how well
        the model predicts the observed events.

        Can be overwritten to provide a far more efficient implementation that only computes the intensities of the "correct" classes at each time-point.

        Args:
            ts: Observed event sequences (batched, padded). Shape: (batch_size, seq_len)

        Returns:
            Kernel intensities for each event in batch. Shape: (batch_size, seq_len)

        """

        B = ts.time_points.shape[0]
        L = ts.time_points.shape[1]
        valid_events = ts.event_mask  # (B, L)

        intensities = torch.stack([self.intensity(ts.time_points[:, l], ts=ts) for l in range(L)], dim=1)  # (B, L, D)
        event_types = ts.event_types_idx_safe.unsqueeze(-1)  # (B, L, 1); padding mapped to a valid index
        intensities = torch.gather(intensities, dim=2, index=event_types).squeeze(-1)  # (B, L)

        intensities[~valid_events] = 0.0

        return intensities  # (B, L)


# ============================================================================
# Concrete implementations
# ============================================================================


@dataclass
class ConstantBaselineParams:
    mu: torch.Tensor  # Shape: (D,)


class ConstantBaselineModule(BaselineIntensityModule):
    """Constant (time-independent) baseline intensity."""

    def __init__(
        self,
        D: int,
        params: Optional[ConstantBaselineParams] = None,
        seed: Optional[int] = None,
        constrained: bool = True,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Args:
            D: Number of event types
            params: Initial baseline intensity values. If None, initialized randomly.
            seed: Random seed for reproducible initialization.
            constrained: If True, apply softplus transformation to ensure positivity.
            dataset_stats: Optional empirical dataset statistics.  When provided, the
                random initialisation is centred on the per-type observed event rate
                rather than the fixed ``[1/12, 10]`` range.
        """
        super().__init__(D)
        self.constrained = constrained

        generator = init_generator(seed)

        if params is None:
            mu = self._init_random_params(D, generator, dataset_stats=dataset_stats)
        else:
            mu = params.mu
            assert mu.shape[0] == D

        self._init_parameters(mu)

    def _init_random_params(
        self,
        D: int,
        generator: torch.Generator,
        dataset_stats: Optional[DatasetStats] = None,
    ) -> torch.Tensor:
        """Initialize random positive mu values.

        When ``dataset_stats`` is provided, draws each type's rate from
        ``[mean_rate_d / 2, mean_rate_d * 2]``, which centres the initial
        baseline on the empirically observed rate for that type.
        Falls back to the fixed ``[1/12, 10]`` range otherwise.
        """
        if dataset_stats is not None:
            # Per-type range: a factor of 2 either side of the observed rate.
            center = dataset_stats.mean_rate.clamp(min=1e-6)  # (D,)
            lo = center / 2.0
            hi = center * 2.0
            noise = torch.rand(D, generator=generator)  # (D,) in [0, 1)
            mu = lo + noise * (hi - lo)
        else:
            mu = torch.rand(D, generator=generator) * (10 - 1 / 12) + 1 / 12

        return mu

    def _init_parameters(self, mu: torch.Tensor):
        """Initialize the learnable parameters."""
        if self.constrained:
            # Store via inverse_softplus so softplus recovers the original positive value.
            self.mu = nn.Parameter(inverse_softplus(mu))
        else:
            self.mu = nn.Parameter(mu.clone())

    def transform_params(self) -> Tuple[torch.Tensor]:
        """Return constrained baseline parameters."""
        return (F.softplus(self.mu),) if self.constrained else (self.mu,)

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Baseline intensity is constant regardless of time or history."""
        (mu,) = self.transform_params()  # Shape: (D,)
        batch_size = t.shape[0]
        return mu.unsqueeze(0).expand(batch_size, self.D)  # Shape: (B,D)

    def intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Constant baseline: μ broadcast over all queries. ``(B, Q)`` -> ``(B, Q, D)``."""
        (mu,) = self.transform_params()  # (D,)
        B, Q = query_times.shape
        return mu.view(1, 1, self.D).expand(B, Q, self.D)  # (B, Q, D)

    def _cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Cumulative intensity: mu * T.  T is guaranteed (B, D) by the base class.
        """
        (mu,) = self.transform_params()  # Shape: (D,)
        return mu.unsqueeze(0) * T  # Shape: (B, D)

    def event_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute the positive part of the likelihood for the baseline intensities: the baseline intensities at event times for the correct class.

        This corresponds to the "data fit" term in the likelihood, measuring how well
        the model predicts the observed events.

        For the poisson process, the intensity per type is always constant.

        Args:
            ts: Observed event sequences (batched, padded). Shape: (batch_size, seq_len)

        Returns:
            Baseline intensities for each event in batch. Shape: (batch_size, seq_len)

        """

        B = ts.time_points.shape[0]
        L = ts.time_points.shape[1]
        valid_events = ts.event_mask  # (B, L)

        mu = self.transform_params()[0]  # (D,)

        safe_types = ts.event_types_idx_safe  # (B, L)
        intensities = mu[safe_types]  # (B, L)
        intensities[~valid_events] = 0.0

        return intensities  # (B, L)


@dataclass
class LogLinearBaselineParams:
    mu: torch.Tensor  # Shape: (D,)
    slope: torch.Tensor  # Shape: (D,)


class LogLinearBaselineModule(BaselineIntensityModule):
    def __init__(
        self,
        params: Optional[LogLinearBaselineParams] = None,
        D: Optional[int] = None,
        seed: Optional[int] = None,
        constrained: bool = True,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """Initialize the log-linear baseline intensity module.

        Args:
            params: Baseline parameters or None to initialize randomly.
            D: Event dimension (required if params is None).
            seed: Random seed for initialization.
            constrained: Must be True; the rate ``exp(mu + slope * t)`` is positive for
                any ``mu`` and ``slope`` (a negative slope gives a decaying rate).
            dataset_stats: Optional empirical statistics.  When provided, mu is
                initialised near ``log(mean_rate_d)`` so the starting intensity
                matches the empirically observed rate for each type.
        """
        super().__init__(D)

        self.constrained = constrained
        if not constrained:
            raise ValueError("Unconstrained is not supported for LogLinearBaselineModule, due to how CI is implemented.")

        generator = init_generator(seed)

        if params is None and D is None:
            raise ValueError("Either params or D (number of dimensions) must be provided.")

        if params is None:
            if dataset_stats is not None:
                # Initialise mu in log-space centred on the per-type observed rate.
                # The linear baseline uses exp(mu + slope*t), so log(mean_rate) is
                # the natural starting point.  Add small noise to break symmetry.
                center = dataset_stats.mean_rate.clamp(min=1e-6)  # (D,)
                lo = center / 2
                hi = center * 2
                center = lo + torch.rand(size=(center.shape), generator=generator) * (hi - lo)  # (D,)
                mu = torch.log(center)
            else:
                mu = torch.log(torch.rand(size=(D,), generator=generator) * (10 - 1 / 12) + 1 / 12)
            slope = torch.zeros(size=(D,))
            self.mu = torch.nn.Parameter(mu)
            self.slope = torch.nn.Parameter(slope)
            self.D = D

        else:
            if D is None:
                D = params.mu.shape[0]

            assert params.mu.shape[0] == D, "Dimension mismatch in mu"
            self.mu = torch.nn.Parameter(params.mu)
            self.slope = torch.nn.Parameter(params.slope)
            self.D = D

    def transform_params(self):
        # mu and slope are unconstrained; exp() in intensity ensures positivity.
        return self.mu, self.slope

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData):
        # lambda_d(t) = exp(mu_d + slope_d * t) (inhomogeneous Poisson with log-linear intensity)
        mu, slope = self.transform_params()

        intensity = torch.exp(mu.unsqueeze(0) + slope.unsqueeze(0) * t.unsqueeze(1))  # Shape: (B,D)
        return intensity

    def _cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData):
        """
        Computes the cumulative intensity function from t=0 up to T.
        T is guaranteed (B, D) by the base class.

        Delegates to :func:`~torchpp.utils.math.loglinear_cumulative_intensity`
        which implements the closed-form integral with a Taylor-expansion guard
        for near-zero slope.
        """
        mu, slope = self.transform_params()  # (D,), (D,)
        return loglinear_cumulative_intensity(mu, slope, T)  # (B, D)

    def event_intensities(
        self,
        ts: BatchedMVEventData,
    ):
        valid_events = ts.event_mask

        mu, slope = self.transform_params()  # (D,)

        # Clamp event_types to avoid negative indexing for padded entries (-1)
        clamped_types = ts.event_types_idx_safe
        intensities = torch.exp(mu[clamped_types] + slope[clamped_types] * ts.time_points)  # Shape: (B,N,)
        masked_intensities = intensities * valid_events.float()
        return masked_intensities


@dataclass
class SplineBaselineParams:
    h_knots: torch.Tensor  # Shape: (D, num_knots)
    # Shape: (num_knots,). Knot locations are cumsum(delta_ts), so the first
    # entry must be 0 (knots start at t=0); entry i>0 is the gap to knot i-1.
    delta_ts: torch.Tensor


class SplineBaselineModule(BaselineIntensityModule):
    """Spline-based baseline intensity module."""

    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        params: Optional[SplineBaselineParams] = None,
        seed: Optional[int] = None,
        constrained=True,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """Initialize the spline baseline module.

        Args:
            D: Number of event types.
            num_knots: Number of spline knots.
            delta_t: Knot spacing (scalar float for uniform, tensor for variable).
            params: Spline parameters. If None, initialized randomly.
            seed: Random seed.
            constrained: If True, knot heights are constrained positive via softplus.
            dataset_stats: Optional empirical statistics.  When provided, knot heights
                are initialised near ``mean_rate_d`` for each type so the spline
                starts close to the empirically observed event rate.
        """
        super().__init__(D)

        self.constrained = constrained

        if params is None:
            self.D = D
            self.num_knots = num_knots

            generator = init_generator(seed)

            h_knots = self._init_random_parameters(D, num_knots, generator, dataset_stats=dataset_stats)
            self._init_parameters(h_knots)

            if isinstance(delta_t, float) or delta_t.numel() == 1:
                knot_locs = torch.arange(num_knots, dtype=torch.float32) * delta_t
            else:
                knot_locs = _check_knots_start_at_zero(torch.cumsum(delta_t, dim=0), type(self).__name__)

            # Register as buffer so it moves with .to(device) and is saved/loaded
            self.register_buffer("knot_locs", knot_locs)

            assert self.knot_locs.shape[0] == self.h_knots.shape[1]

        else:
            self._init_parameters(params.h_knots)
            knot_locs = _check_knots_start_at_zero(torch.cumsum(params.delta_ts, dim=0), type(self).__name__)
            self.register_buffer("knot_locs", knot_locs)
            self.D = params.h_knots.shape[0]
            self.num_knots = params.h_knots.shape[1]

        assert self.num_knots == self.h_knots.shape[1], "Number of knots mismatch"

    def breakpoints(self) -> torch.Tensor:
        return self.knot_locs

    def _init_random_parameters(
        self,
        D: int,
        num_knots: int,
        generator: torch.Generator,
        dataset_stats: Optional[DatasetStats] = None,
    ) -> torch.Tensor:
        """Initialize random knot heights for the spline baseline.

        When ``dataset_stats`` is provided, heights are initialised near the
        per-type observed rate (``mean_rate_d``) with small multiplicative noise,
        giving a flat starting spline that approximates the empirical base rate.
        Falls back to uniform random in [0.1, 1.1] otherwise.
        """
        if dataset_stats is not None:
            center = dataset_stats.mean_rate.clamp(min=1e-6)  # (D,)
            log_scale = torch.log(torch.tensor(1.2)).item()  # log(1.2)
            noise = torch.exp(
                (torch.rand(D, num_knots, generator=generator) * 2 - 1) * log_scale
            )  # uniform in log-space: [1/1.2, 1.2]
            initial_h = center.unsqueeze(1) * noise  # (D, K)
        else:
            initial_h = torch.rand(D, num_knots, generator=generator) + 0.1
        return initial_h

    def _init_parameters(self, h_knots: torch.Tensor):
        """Initialize the learnable parameters."""
        if self.constrained:
            self.h_knots = torch.nn.Parameter(inverse_softplus(h_knots))
        else:
            self.h_knots = torch.nn.Parameter(h_knots.clone())

    def transform_params(self):
        "Return the parameters in their actual/constrained form."
        return (F.softplus(self.h_knots),) if self.constrained else (self.h_knots,)

    def get_heights(self):
        """Returns non-negative knot heights: Shape (D, num_knots)

        Note: Returns a tuple containing the tensor for consistency with transform_params().
        Use (h,) = self.get_heights() to unpack.
        """
        return self.transform_params()

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData):
        """
        lambda_d(t) = h_k + m_k * delta
        Returns: (batch_size, D)
        """

        (h,) = self.get_heights()  # (D, K)

        intensity = LinearSpline.interpolate(self.knot_locs, h, t)  # (B,D)

        return intensity

    def _cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData):
        """
        Calculates the integral from 0 to T for each dimension.
        T is guaranteed (B, D) by the base class.
        """
        (h,) = self.get_heights()  # (D, K)
        return LinearSpline.integrate(x_knots=self.knot_locs, y_knots=h, t=T)  # (B, D)

    def event_intensities(self, ts: BatchedMVEventData):
        valid_events = ts.event_mask  # (B, N)

        flat_times = ts.time_points.flatten()  # (B*N)
        flat_events = ts.event_types.flatten()  # (B*N)

        # Evaluate intensity at all event times. This is valid because the spline
        # baseline is history-independent, so ts is only used for shape context.
        all_intensities = self.intensity(flat_times, ts)  # (B*N, D)
        all_intensities = all_intensities.view(ts.time_points.shape[0], ts.time_points.shape[1], self.D)  # (B, N, D)

        # event_types_idx_safe replaces padding (-1) with 0; masked out below.
        event_intensities = torch.gather(all_intensities, dim=2, index=ts.event_types_idx_safe.unsqueeze(2)).squeeze(2)

        event_intensities = event_intensities.masked_fill(~valid_events, 0.0)

        return event_intensities


@dataclass
class PiecewiseConstantBaselineParams:
    h_knots: torch.Tensor  # Shape: (D, num_knots) — bin heights
    # Shape: (num_knots,). Left bin edges are cumsum(delta_ts), so the first
    # entry must be 0 (bins start at t=0); entry i>0 is the width of bin i-1.
    delta_ts: torch.Tensor


class PiecewiseConstantBaselineModule(BaselineIntensityModule):
    """Piecewise-constant baseline intensity module for Hawkes processes.

    Between consecutive knots the intensity is constant (zero-order hold).
    More flexible than a constant baseline but simpler than a spline.

    Intensity:          λ_d(t) = h_k   for x_k ≤ t < x_{k+1}
    Cumulative intens.: Λ_d(T) = Σ_{j<k} h_j Δx_j  +  h_k (T - x_k)
    """

    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        params: Optional[PiecewiseConstantBaselineParams] = None,
        seed: Optional[int] = None,
        constrained: bool = True,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """Initialize the piecewise-constant baseline module.

        Args:
            D: Number of event types.
            num_knots: Number of constant bins.
            delta_t: Bin width (float for uniform, tensor for variable).
            params: Piecewise-constant parameters. If None, initialized randomly.
            seed: Random seed.
            constrained: If True, bin heights are constrained positive via softplus.
            dataset_stats: Optional empirical statistics.  When provided, bin heights
                are initialised near ``mean_rate_d`` for each type.
        """
        super().__init__(D)

        self.constrained = constrained

        generator = init_generator(seed)

        if params is None:
            self.num_knots = num_knots

            if dataset_stats is not None:
                center = dataset_stats.mean_rate.clamp(min=1e-6)  # (D,)
                log_scale = torch.log(torch.tensor(1.2)).item()  # log(1.2)
                noise = torch.exp(
                    (torch.rand(D, num_knots, generator=generator) * 2 - 1) * log_scale
                )  # uniform in log-space: [1/1.2, 1.2]
                initial_h = center.unsqueeze(1) * noise  # (D, K)
            else:
                initial_h = torch.rand(D, num_knots, generator=generator) + 0.1

            self._init_parameters(initial_h)

            if isinstance(delta_t, float) or (isinstance(delta_t, torch.Tensor) and delta_t.numel() == 1):
                knot_locs = torch.arange(num_knots, dtype=torch.float32) * delta_t
            else:
                knot_locs = _check_knots_start_at_zero(torch.cumsum(delta_t, dim=0), type(self).__name__)
            self.register_buffer("knot_locs", knot_locs)

        else:
            # params takes precedence: knot count and locations come from params.
            self._init_parameters(params.h_knots)
            knot_locs = _check_knots_start_at_zero(torch.cumsum(params.delta_ts, dim=0), type(self).__name__)
            self.register_buffer("knot_locs", knot_locs)
            self.D = params.h_knots.shape[0]
            self.num_knots = params.h_knots.shape[1]

        assert self.knot_locs.shape[0] == self.h_knots.shape[1]

    def breakpoints(self) -> torch.Tensor:
        return self.knot_locs

    def _init_parameters(self, h_knots: torch.Tensor):
        if self.constrained:
            self.h_knots = nn.Parameter(inverse_softplus(h_knots))
        else:
            self.h_knots = nn.Parameter(h_knots.clone())

    def transform_params(self):
        return (F.softplus(self.h_knots),) if self.constrained else (self.h_knots,)

    def get_heights(self):
        return self.transform_params()

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData):

        (h,) = self.get_heights()  # (D, K)
        return PiecewiseConstant.interpolate(self.knot_locs, h, t)  # (B, D)

    def _cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData):
        """
        T is guaranteed (B, D) by the base class.
        """
        (h,) = self.get_heights()  # (D, K)
        return PiecewiseConstant.integrate(x_knots=self.knot_locs, y_knots=h, t=T)

    def event_intensities(self, ts: BatchedMVEventData):
        valid_events = ts.event_mask  # (B, N)
        flat_times = ts.time_points.flatten()  # (B*N)
        all_intensities = self.intensity(flat_times, ts)  # (B*N, D)
        all_intensities = all_intensities.view(ts.time_points.shape[0], ts.time_points.shape[1], self.D)  # (B, N, D)
        event_intensities = torch.gather(all_intensities, dim=2, index=ts.event_types_idx_safe.unsqueeze(2)).squeeze(
            2
        )  # (B, N)
        event_intensities = event_intensities.masked_fill(~valid_events, 0.0)
        return event_intensities


@dataclass
class ExpKernelParams:
    alpha: torch.Tensor  # Shape: (D,D)
    beta: torch.Tensor  # Shape: (D,D)


class ExpKernelModule(KernelIntensityModule):
    """Exponential kernel module for Hawkes processes.

    Supports both constrained (positive alpha, for excitation-only processes)
    and unconstrained alpha (for inhibitive processes where alpha can be negative).

    When ``constrained=True`` (default): alpha is stored via inverse_softplus so it
    is always positive. Use for standard self-exciting Hawkes processes.

    When ``constrained=False``: alpha is stored directly (no positivity constraint),
    allowing negative values for inhibition. Beta remains positive in both modes.
    Initialization always uses a positive starting point for better optimization.
    """

    def __init__(
        self,
        D: int,
        params: Optional[ExpKernelParams] = None,
        seed: Optional[int] = None,
        constrained: bool = True,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Args:
            D: Number of event types
            params: Kernel parameters. If None, initialized randomly.
            seed: Random seed
            constrained: If True, alpha is constrained positive via softplus.
                If False, alpha is unconstrained (can be negative for inhibition).
            dataset_stats: Optional empirical dataset statistics.  When provided:
                - ``beta[i,j]`` is initialised to ``1 / mean_inter_event_time``
                  (global marginal fallback — same for all pairs).
                - ``alpha[i,j]`` starts near-zero (``1e-4`` for terminating data,
                  ``1e-3`` for recurring), with small noise, then spectral-normalised
                  to a branching ratio of 0.5.
        """
        super().__init__(D)
        self.constrained = constrained

        generator = init_generator(seed)

        if params is None:
            alpha, beta = self._init_random_params(D, generator, dataset_stats=dataset_stats)
        else:
            alpha = params.alpha
            beta = params.beta

        self._init_parameters(alpha, beta)

    def _init_random_params(
        self,
        D: int,
        generator: torch.Generator,
        dataset_stats: Optional[DatasetStats] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize random positive alpha and beta values.

        Always returns positive values regardless of ``constrained`` mode —
        unconstrained kernels also benefit from a positive starting point.

        When ``dataset_stats`` is provided:
        - ``beta[i,j]`` is set uniformly to ``1 / mean_inter_event_time`` (global
          marginal fallback) for all pairs.
        - ``alpha[i,j]`` starts near-zero (``1e-4`` for terminating, ``1e-3`` for
          recurring) with small noise, then spectral-normalised to target branching
          ratio 0.5.
        """
        targeted_branching_ratio = 0.5

        if dataset_stats is not None:
            fallback_beta = 1.0 / max(dataset_stats.mean_inter_event_time, 1e-9)
            beta = torch.full((D, D), fallback_beta)
            # alpha: small uniform near-zero values, spectral-normalised
            unseen_alpha = 1e-4 if dataset_stats.is_terminating else 1e-3
            alpha_base = torch.full((D, D), unseen_alpha) + torch.rand(size=(D, D), generator=generator) * 0.1
            max_eig = torch.max(torch.abs(torch.linalg.eigvals(alpha_base)))
            if max_eig > 0:
                alpha = (targeted_branching_ratio / max_eig) * alpha_base
            else:
                alpha = alpha_base
        else:
            # Without dataset statistics: random decay rates and a branching ratio of 0.5.
            decay_lengths = torch.rand(size=(D, D), generator=generator) * (10 - 1 / 12) + 1 / 12
            beta = 1 / decay_lengths

            alphas = torch.rand(size=(D, D), generator=generator) * 100 + 1
            max_eig = torch.max(torch.abs(torch.linalg.eigvals(alphas)))
            alpha = (1 / max_eig) * targeted_branching_ratio * alphas

        return alpha, beta

    def _init_parameters(self, alpha: torch.Tensor, beta: torch.Tensor):
        """Initialize the learnable parameters."""
        if self.constrained:
            # Store via inverse_softplus so softplus recovers the original positive value
            self.log_alpha = nn.Parameter(inverse_softplus(alpha))
        else:
            # Scale by sigmoid(inverse_softplus(alpha)) — the Jacobian of softplus at the
            # init point — so that grad(stored) = grad(alpha) * sigmoid(x_init), matching
            # the constrained path's grad(log_alpha) = grad(alpha) * sigmoid(log_alpha_init).
            # This equalises gradient magnitudes at initialisation; Adam handles the rest.
            x_init = inverse_softplus(alpha.abs().clamp(min=1e-12))
            scale = torch.sigmoid(x_init).mean().clamp(min=1e-12)
            self.register_buffer("_alpha_scale", scale)
            self.alpha = nn.Parameter(alpha / scale)
        self.log_beta = nn.Parameter(inverse_softplus(beta))

    def transform_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (alpha, beta) with beta always positive."""
        if self.constrained:
            return F.softplus(self.log_alpha), F.softplus(self.log_beta)
        else:
            return self.alpha * self._alpha_scale, F.softplus(self.log_beta)

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute kernel contribution to intensity.

        λ_kernel_d(t) = Σ_{t_i < t} α_{d,m_i} β_{d,m_i} exp(-β_{d,m_i} (t - t_i))
        """
        alpha, beta = self.transform_params()

        valid_mask = ts.event_mask  # Shape: (B,N)
        past_event_mask = ts.time_points < t.unsqueeze(1)  # Shape: (B,N)
        valid_mask = valid_mask & past_event_mask

        # Clamp event_types to avoid negative indexing for padded entries (-1)
        clamped_types = ts.event_types_idx_safe
        relevant_alphas = alpha[:, clamped_types]  # Shape: (D,B,N)
        relevant_betas = beta[:, clamped_types]  # Shape: (D,B,N)

        delta_t = t.unsqueeze(0).unsqueeze(2) - ts.time_points.unsqueeze(0)  # Shape: (1,B,N)
        delta_t = torch.where(valid_mask.unsqueeze(0), delta_t, torch.zeros_like(delta_t))

        contributions = relevant_alphas * relevant_betas * torch.exp(-relevant_betas * delta_t)  # Shape: (D,B,N)
        contributions = contributions.masked_fill(~valid_mask.unsqueeze(0), 0)
        contributions = torch.sum(contributions, dim=2)  # Shape: (D,B)

        return contributions.T  # Shape: (B,D)

    def intensities_at_queries(self, query_times: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Vectorised kernel intensity at Q query times. ``(B, Q)`` -> ``(B, Q, D)``.

        ``λ_d(q) = Σ_{t_i < q} α_{d,type_i} β_{d,type_i} exp(-β_{d,type_i} (q - t_i))``, the
        multi-query form of :meth:`intensity` (Q=1 slice is identical).
        """
        alpha, beta = self.transform_params()  # (D, D)
        B, Q = query_times.shape
        N = ts.time_points.shape[1] if len(ts) > 0 else 0
        if N == 0:
            return torch.zeros(B, Q, self.D, device=query_times.device, dtype=query_times.dtype)

        clamped_types = ts.event_types_idx_safe  # (B, N)
        alpha_bnd = alpha[:, clamped_types].permute(1, 2, 0)  # (B, N, D)
        beta_bnd = beta[:, clamped_types].permute(1, 2, 0)  # (B, N, D)

        tp = ts.time_points  # (B, N)
        trigger_ok = ts.event_mask.unsqueeze(1)  # (B, 1, N)
        valid = trigger_ok & (tp.unsqueeze(1) < query_times.unsqueeze(2))  # (B, Q, N)
        delta = query_times.unsqueeze(2) - tp.unsqueeze(1)  # (B, Q, N)
        delta = torch.where(valid, delta, torch.zeros_like(delta))

        beta_q = beta_bnd.unsqueeze(1)  # (B, 1, N, D)
        contrib = alpha_bnd.unsqueeze(1) * beta_q * torch.exp(-beta_q * delta.unsqueeze(3))  # (B, Q, N, D)
        contrib = contrib.masked_fill(~valid.unsqueeze(3), 0)
        return contrib.sum(dim=2)  # (B, Q, D)

    def _cumulative_intensity(
        self,
        T: torch.Tensor,
        ts: BatchedMVEventData,
    ) -> torch.Tensor:
        """
        Compute cumulative kernel intensity from 0 to T.
        T is guaranteed (B, D) by the base class.

            Λ_kernel_d(T_d) = Σ_{t_i < T_d} α_{d,m_i} (1 - exp(-β_{d,m_i} (T_d - t_i)))
        """
        B = T.shape[0]
        N = ts.time_points.shape[1] if len(ts) > 0 else 0

        alpha, beta = self.transform_params()

        integral = torch.zeros(self.D, B, device=T.device, dtype=T.dtype)

        if N > 0:
            valid_events = ts.event_mask  # Shape: (B, N)

            # Clamp event_types to avoid negative indexing for padded entries (-1)
            clamped_types = ts.event_types_idx_safe  # (B, N)

            effective_T_expanded = T.T.unsqueeze(2)  # (D, B, 1) -> will broadcast to (D, B, N)
            time_points_expanded = ts.time_points.unsqueeze(0)  # (1, B, N)

            # Per-dimension causality: events before effective_T for each receiver dim
            past_event_mask = time_points_expanded < effective_T_expanded  # (D, B, N)
            valid_mask = valid_events.unsqueeze(0) & past_event_mask  # (D, B, N)

            delta_t = effective_T_expanded - time_points_expanded  # (D, B, N)
            delta_t = torch.where(valid_mask, delta_t, torch.zeros_like(delta_t))

            relevant_alpha = alpha[:, clamped_types]  # Shape: (D, B, N)
            relevant_beta = beta[:, clamped_types]  # Shape: (D, B, N)

            contributions = relevant_alpha * (1 - torch.exp(-relevant_beta * delta_t))  # Shape: (D, B, N)
            contributions = contributions.masked_fill(~valid_mask, 0)
            integral = integral + torch.sum(contributions, dim=2)  # Shape: (D, B)

        integral = integral.T  # (B, D)
        return integral

    def event_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Efficiently compute the pairwise interaction terms for all events in ts, using a lower triangular matrix approach.

        Args:
            ts (BatchedMVEventData): Batch of past events

        Returns:
            torch.Tensor: Tensor of positive likelihood intensities
        """
        valid_events = ts.event_mask
        # A pair is valid only if both events are real (non-padded).
        valid_event_mask = valid_events.unsqueeze(2) & valid_events.unsqueeze(1)

        valid_time_mask = ts.time_points.unsqueeze(2) > ts.time_points.unsqueeze(1)  # Shape: (B, N, N)
        valid_event_mask = valid_event_mask & valid_time_mask

        # Time-difference matrix: T_{b,i,j} = t_{b,i} - t_{b,j} if i > j else 0
        time_diffs = ts.time_points.unsqueeze(2) - ts.time_points.unsqueeze(1)  # Shape: (B, N, N)
        # Zero out diffs against padded events to avoid +-inf or NaN propagation.
        time_diffs_safe = torch.where(valid_event_mask, time_diffs, torch.zeros_like(time_diffs))

        # Clamp to avoid negative indexing for padded entries (-1)
        clamped_types = ts.event_types_idx_safe
        receiver_types = clamped_types.unsqueeze(2)  # (B, N, 1)
        trigger_types = clamped_types.unsqueeze(1)  # (B, 1, N)
        alpha, beta = self.transform_params()  # (D, D), (D, D)

        # alpha_matrix[b, i, j] = alpha[receiver_types[b, i], trigger_types[b, j]]
        alpha_matrix = alpha[receiver_types, trigger_types]  # Shape: (B, N, N)
        beta_matrix = beta[receiver_types, trigger_types]  # Shape: (B, N, N)

        # Kernel: alpha * beta * exp(-beta * dt). Clamp exponent for safety against underflow.
        exponent = -beta_matrix * time_diffs_safe
        exponent = torch.clamp(exponent, min=-20, max=0)
        exp_kernel = torch.exp(exponent)
        interaction_terms = alpha_matrix * beta_matrix * exp_kernel  # Shape: (B,N, N)

        # Strict lower-triangular keeps only causal pairs (j < i); also batched-safe.
        interaction_terms = torch.tril(interaction_terms, diagonal=-1)

        interaction_terms[~valid_event_mask] = 0.0

        return interaction_terms.sum(dim=2)  # Shape: (B,N)


# =============================================================================
# Spline Kernel Module
# =============================================================================


@dataclass
class SplineKernelParams:
    h_knots: torch.Tensor  # Shape: (D, D, num_knots) — knot heights per (recv, trig) pair
    delta_ts: torch.Tensor  # Shape: (num_knots,) — knot spacings (cumsum → knot_locs on Δt axis)
    tail_delta: Optional[torch.Tensor] = None  # Scalar — initial spacing to the zero-height tail knot


class SplineKernelModule(KernelIntensityModule):
    """Spline-based kernel module for Hawkes processes.

    The kernel function is a linear spline over elapsed time Δt = t - t_j,
    with one spline per (receiver, trigger) pair:

        φ_{d,j}(Δt) = LinearSpline(Δt; knot_locs_with_tail, h_{d,j} ++ [0])

    A learnable *tail knot* with height 0 is appended beyond the last regular knot.
    Its location is ``knot_locs[-1] + softplus(log_tail_delta)``, which is always
    strictly beyond the last knot.  For Δt beyond this tail knot the spline evaluates
    to 0, making the kernel compactly supported and preventing runaway accumulation.

    When ``constrained=True`` (default), regular knot heights are positive (via softplus),
    so the kernel is always non-negative. Use for standard self-exciting Hawkes.

    When ``constrained=False``, regular knot heights are unconstrained (can be negative
    for inhibition). Initialization always uses a positive decaying profile regardless.

    Cumulative intensity is computed analytically via LinearSpline.integrate.
    """

    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        params: Optional[SplineKernelParams] = None,
        seed: Optional[int] = None,
        constrained: bool = True,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Args:
            D: Number of event types
            num_knots: Number of spline knots on the Δt axis
            delta_t: Knot spacing specification. Two forms are accepted:

                * ``float`` or scalar ``Tensor`` — uniform spacing; knots are placed
                  at ``0, delta_t, 2*delta_t, …, (K-1)*delta_t``.
                * 1-D ``Tensor`` of length K-1 — **inter-knot spacings** (gaps
                  between consecutive knots). The first knot is always at 0 and
                  subsequent knots are placed at ``cumsum(delta_t)``. Use
                  :func:`torchpp.utils.math.interaction_spline_knot_gaps`
                  to obtain a geometric schedule.

            params: Kernel parameters. If None, initialized randomly.
            seed: Random seed
            constrained: If True, heights are constrained positive via softplus.
                If False, heights are unconstrained (can be negative for inhibition).
        """
        super().__init__(D)
        self.constrained = constrained

        generator = init_generator(seed)
        self.num_knots = num_knots

        # Knot locations on the Δt axis: first knot is always at 0.
        # Scalar: uniform spacing 0, dt, 2*dt, …
        # Tensor (K-1 spacings): 0, cumsum(delta_t)
        if isinstance(delta_t, float) or (isinstance(delta_t, torch.Tensor) and delta_t.numel() == 1):
            knot_locs = torch.arange(num_knots, dtype=torch.float32) * float(delta_t)
            default_tail_delta = float(delta_t)
        else:
            spacings = delta_t.float()
            knot_locs = torch.cat([torch.zeros(1, dtype=spacings.dtype), torch.cumsum(spacings, dim=0)])
            default_tail_delta = float(spacings[-1])
        self.register_buffer("knot_locs", knot_locs)

        if params is None:
            h_knots = self._init_random_params(D, num_knots, knot_locs, generator, dataset_stats=dataset_stats)
            tail_delta_init = default_tail_delta
        else:
            h_knots = params.h_knots
            tail_delta_init = float(params.tail_delta) if params.tail_delta is not None else default_tail_delta

        self._init_parameters(h_knots, tail_delta_init)

    def breakpoints(self) -> torch.Tensor:
        return self.knot_locs_with_tail

    def _init_random_params(
        self,
        D: int,
        num_knots: int,
        knot_locs: torch.Tensor,
        generator: torch.Generator,
        dataset_stats: Optional[DatasetStats] = None,
    ) -> torch.Tensor:
        """Initialize knot heights to match a target exponential kernel.

        The spline is initialized as a discretized exponential decay so that its
        integral (= effective branching ratio contribution) matches the same target
        as the exp-kernel init.  Specifically, for each ``(i, j)`` pair:

            h_knots[i, j, k]  =  scale[i,j] * exp(-beta[i,j] * knot_locs[k])

        where ``beta[i,j]`` is drawn from the same distribution as
        ``ExpKernelModule._init_random_params`` and ``scale[i,j]`` is chosen so that

            ∫₀^{T_max} scale * exp(-beta * t) dt  =  branching_ratio * alpha_weight[i,j]

        giving  ``scale = alpha_weight * beta / (1 - exp(-beta * T_max))``.

        This makes the spline-kernel and exp-kernel inits directly equivalent:
        same timescale, same total mass, same pair-occurrence weighting.

        When ``dataset_stats`` is provided:
        - ``beta[i,j]`` is set uniformly to ``1 / mean_inter_event_time`` (global
          marginal fallback) for all pairs.
        - ``alpha_weight[i,j]`` starts near-zero (``1e-4`` for terminating, ``1e-3``
          for recurring) with small noise, then spectral-normalised to target
          branching ratio 0.5.

        Returns:
            h_knots: Shape (D, D, num_knots) — positive knot heights
        """
        targeted_branching_ratio = 0.5
        T_max = float(knot_locs[-1])

        # Same decay-length initialisation as ExpKernelModule.
        if dataset_stats is not None:
            fallback_beta = 1.0 / max(dataset_stats.mean_inter_event_time, 1e-9)
            beta = torch.full((D, D), fallback_beta)
            # alpha_weight: small uniform near-zero, spectral-normalised
            unseen_alpha = 1e-4 if dataset_stats.is_terminating else 1e-3
            alpha_weight = torch.full((D, D), unseen_alpha) + torch.rand(size=(D, D), generator=generator) * 0.1
            max_eig = torch.max(torch.abs(torch.linalg.eigvals(alpha_weight)))
            if max_eig > 0:
                alpha_weight = (targeted_branching_ratio / max_eig) * alpha_weight
        else:
            # Without data stats, choose beta so that the exponential decays over the
            # knot range: sample 1–5 decay-lengths per pair so each pair looks distinct.
            # beta_k = decay_lengths_k / T_max  with T_max = knot_locs[-1].
            # Fall back to beta=1 if T_max=0 (degenerate single-knot case).
            if T_max > 0:
                decay_lengths = torch.rand(size=(D, D), generator=generator) * 4.0 + 1.0  # Uniform[1, 5]
                beta = decay_lengths / T_max
            else:
                beta = torch.ones(D, D)

            alphas = torch.rand(size=(D, D), generator=generator) * 100 + 1
            max_eig = torch.max(torch.abs(torch.linalg.eigvals(alphas)))
            alpha_weight = (1 / max_eig) * targeted_branching_ratio * alphas

        # ---- scale: height at t=0 so that integral matches alpha_weight ----
        # ∫₀^T_max scale * exp(-beta * t) dt = alpha_weight
        # => scale = alpha_weight * beta / (1 - exp(-beta * T_max))
        # Guard against beta*T_max→0 (flat profile) with clamp.
        beta_T = (beta * T_max).clamp(min=1e-6)
        scale = alpha_weight * beta / (1.0 - torch.exp(-beta_T)).clamp(min=1e-9)  # (D, D)

        # ---- build knot heights: scale * exp(-beta * t_k) ----
        # knot_locs: (K,), beta/scale: (D, D) → h_knots: (D, D, K)
        h_knots = scale.unsqueeze(-1) * torch.exp(
            -beta.unsqueeze(-1) * knot_locs.unsqueeze(0).unsqueeze(0)
        )  # (D, D, K)

        return h_knots

    def _init_parameters(self, h_knots: torch.Tensor, tail_delta_init: float):
        """Initialize learnable parameters from knot heights and tail delta."""
        if self.constrained:
            # Store via inverse_softplus so softplus recovers the original positive values
            self.log_h_knots = nn.Parameter(inverse_softplus(h_knots))  # (D, D, K)
        else:
            # Scale by sigmoid(inverse_softplus(h_knots)) — the Jacobian of softplus at the
            # init point — so that grad(stored) = grad(h) * sigmoid(x_init), matching
            # the constrained path's grad(log_h) = grad(h) * sigmoid(log_h_init).
            # This equalises gradient magnitudes at initialisation; Adam handles the rest.
            x_init = inverse_softplus(h_knots.abs().clamp(min=1e-12))
            scale = torch.sigmoid(x_init).mean().clamp(min=1e-12)
            self.register_buffer("_h_knots_scale", scale)

            # Store directly — unconstrained, heights can go negative during training
            self.h_knots = nn.Parameter(h_knots.clone() / scale)  # (D, D, K)

        # Learnable spacing from the last regular knot to the zero-height tail knot.
        # Stored as inverse_softplus(tail_delta_init) so softplus gives a positive distance.
        self.log_tail_delta = nn.Parameter(torch.tensor(inverse_softplus(torch.tensor(tail_delta_init)).item()))

    @property
    def knot_locs_with_tail(self) -> torch.Tensor:
        """Knot locations including the zero-height tail knot. Shape: (K+1,)"""
        tail_loc = self.knot_locs[-1] + F.softplus(self.log_tail_delta)
        return torch.cat([self.knot_locs, tail_loc.unsqueeze(0)])

    def transform_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (h_knots_with_tail, knot_locs_with_tail).

        Returns:
            h_knots: Shape (D, D, K+1) — regular knot heights with a zero tail appended
            knot_locs: Shape (K+1,) — knot locations including the learnable tail knot
        """
        if self.constrained:
            h = F.softplus(self.log_h_knots)  # (D, D, K)
        else:
            h = self.h_knots * self._h_knots_scale  # (D, D, K)

        # Append a zero-height tail knot: shape (D, D, K+1)
        tail = torch.zeros(*h.shape[:2], 1, device=h.device, dtype=h.dtype)
        h_with_tail = torch.cat([h, tail], dim=2)

        return h_with_tail, self.knot_locs_with_tail

    def _get_spline_values_for_trigger(self, delta_t: torch.Tensor, trigger_types: torch.Tensor) -> torch.Tensor:
        """Evaluate splines only for the specified trigger types, returning (M, D_recv).

        Instead of computing all D×D spline values, only computes the D_recv values
        for the actual trigger type of each time point. This is O(M*D) instead of O(M*D*D).

        Fully vectorized — no Python loops over types.

        Args:
            delta_t: Elapsed times. Shape: (M,)
            trigger_types: Trigger type index for each time point. Shape: (M,), values in [0, D)

        Returns:
            Kernel values for specified triggers. Shape: (M, D_recv)
                result[m, d] = φ_{d, trigger_types[m]}(delta_t[m])
        """
        h_knots, knot_locs = self.transform_params()  # (D_recv, D_trig, K+1), (K+1,)
        K = h_knots.shape[2]

        # Gather the knots for each point's trigger type: (D_recv, M, K+1)
        selected_h = h_knots[:, trigger_types, :]  # (D_recv, M, K+1)

        # Inline linear-spline interpolation, vectorized over both D_recv and M.
        k = torch.searchsorted(knot_locs, delta_t, right=True) - 1  # (M,)
        k = k.clamp(0, K - 1)
        offset = delta_t - knot_locs[k]  # (M,)

        delta_times = knot_locs[1:] - knot_locs[:-1]  # (K,)
        delta_heights = selected_h[:, :, 1:] - selected_h[:, :, :-1]  # (D_recv, M, K)
        slopes = delta_heights / delta_times  # (D_recv, M, K)
        slopes = F.pad(slopes, (0, 1), value=0.0)  # (D_recv, M, K+1)

        k_exp = k.unsqueeze(0).unsqueeze(2)  # (1, M, 1)
        h_k = torch.gather(selected_h, 2, k_exp.expand(self.D, -1, -1)).squeeze(2)  # (D_recv, M)
        s_k = torch.gather(slopes, 2, k_exp.expand(self.D, -1, -1)).squeeze(2)  # (D_recv, M)

        result = h_k + s_k * offset.unsqueeze(0)  # (D_recv, M)
        return result.T  # (M, D_recv)

    def _get_spline_values_for_pairs(
        self, delta_t: torch.Tensor, receiver_types: torch.Tensor, trigger_types: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate splines only for specific (receiver, trigger) pairs, returning scalars.

        Instead of computing all D×D spline values, only computes the single scalar
        for each (receiver_type, trigger_type) pair. This is O(M) instead of O(M*D*D).

        Fully vectorized — no Python loops over pairs.

        Args:
            delta_t: Elapsed times. Shape: (M,)
            receiver_types: Receiver type for each point. Shape: (M,), values in [0, D)
            trigger_types: Trigger type for each point. Shape: (M,), values in [0, D)

        Returns:
            Scalar kernel values for each pair. Shape: (M,)
                result[m] = φ_{receiver_types[m], trigger_types[m]}(delta_t[m])
        """
        h_knots, knot_locs = self.transform_params()  # (D_recv, D_trig, K+1), (K+1,)

        # Gather the knot heights for each point's (recv, trig) pair: (M, K+1)
        selected_h = h_knots[receiver_types, trigger_types, :]  # (M, K+1)

        # Delegate to LinearSpline.interpolate_flat: (N, K) knots × (N,) times → (N,)
        return LinearSpline.interpolate_flat(knot_locs, selected_h, delta_t)  # (M,)

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute kernel contribution to intensity at time t.

        λ_kernel_d(t) = Σ_{t_i < t} φ_{d, type(i)}(t - t_i)

        where φ_{d,j}(Δt) is the spline kernel for receiver d, trigger type j.

        Memory-efficient: only evaluates the D splines for each event's trigger type,
        avoiding the full (B*N, D, D) tensor.

        Args:
            t: Query times. Shape: (B,)
            ts: Historical events. Shape: (B, N)

        Returns:
            Kernel intensity for each event type. Shape: (B, D)
        """
        B = t.shape[0]
        N = ts.time_points.shape[1] if len(ts) > 0 else 0

        if N == 0:
            return torch.zeros(B, self.D, device=t.device, dtype=t.dtype)

        valid_mask = ts.event_mask  # (B, N)
        past_event_mask = ts.time_points < t.unsqueeze(1)  # (B, N)
        valid_mask = valid_mask & past_event_mask
        delta_t = t.unsqueeze(1) - ts.time_points  # (B, N)
        delta_t = torch.where(valid_mask, delta_t, torch.zeros_like(delta_t))

        # Only evaluate the D_recv splines for each event's actual trigger type
        # to avoid materializing (B*N, D, D).
        clamped_types = ts.event_types_idx_safe  # (B, N)
        flat_dt = delta_t.reshape(-1)  # (B*N,)
        flat_types = clamped_types.reshape(-1)  # (B*N,)

        contributions = self._get_spline_values_for_trigger(flat_dt, flat_types)  # (B*N, D_recv)
        contributions = contributions.reshape(B, N, self.D)  # (B, N, D_recv)

        contributions = contributions.masked_fill(~valid_mask.unsqueeze(2), 0.0)

        return contributions.sum(dim=1)  # (B, D)

    def _cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute cumulative kernel intensity from 0 to T.
        T is guaranteed (B, D) by the base class.

        Λ_kernel_d(T) = Σ_{t_i < T} ∫₀^{T - t_i} φ_{d, type(i)}(s) ds
        """
        B = T.shape[0]
        N = ts.time_points.shape[1] if len(ts) > 0 else 0

        if N == 0:
            return torch.zeros(B, self.D, device=T.device, dtype=T.dtype)

        valid_events = ts.event_mask  # (B, N)
        clamped_types = ts.event_types_idx_safe  # (B, N)

        # Per-dimension elapsed times: (B, N, D) where [b, i, d] = T[b, d] - t_i[b]
        delta_t = T.unsqueeze(1) - ts.time_points.unsqueeze(2)  # (B, N, D)

        past_mask = ts.time_points.unsqueeze(2) < T.unsqueeze(1)  # (B, N, D)
        valid_mask = valid_events.unsqueeze(2) & past_mask  # (B, N, D)
        delta_t = torch.where(valid_mask, delta_t, torch.zeros_like(delta_t))

        # Vectorized cumulative intensity: gather spline knots per trigger type, then
        # inline the integration math to avoid Python loops.
        h_knots, knot_locs = self.transform_params()  # (D_recv, D_trig, K+1), (K+1,)
        K = h_knots.shape[2]
        D = self.D

        flat_types = clamped_types.reshape(-1)  # (B*N,)
        selected_h = h_knots[:, flat_types, :]  # (D_recv, B*N, K+1)

        dt_perm = delta_t.permute(2, 0, 1).reshape(D, B * N)  # (D_recv, B*N)

        # Inline LinearSpline.integrate for (D_recv, B*N, K+1) knots at (D_recv, B*N) times
        dt_flat_for_search = dt_perm.reshape(-1)  # (D*B*N,)
        k_flat = torch.searchsorted(knot_locs, dt_flat_for_search, right=True) - 1
        k_flat = k_flat.clamp(0, K - 1)
        k_all = k_flat.reshape(D, B * N)  # (D, B*N)
        offset_all = dt_perm - knot_locs[k_all]  # (D, B*N)

        delta_times = knot_locs[1:] - knot_locs[:-1]  # (K,)
        delta_heights = selected_h[:, :, 1:] - selected_h[:, :, :-1]  # (D, B*N, K)
        slopes = delta_heights / delta_times  # (D, B*N, K)

        # Segment areas: h_k * dt_k + slope_k/2 * dt_k^2 for each complete segment
        segment_areas = selected_h[:, :, :-1] * delta_times + slopes / 2.0 * delta_times**2  # (D, B*N, K)

        prefix_sums = torch.cumsum(segment_areas, dim=2)  # (D, B*N, K)
        prefix_sums = F.pad(prefix_sums, (1, 0), value=0.0)  # (D, B*N, K+1)

        slopes_padded = F.pad(slopes, (0, 1), value=0.0)  # (D, B*N, K+1)

        k_exp = k_all.unsqueeze(2)  # (D, B*N, 1)
        C_k = torch.gather(prefix_sums, 2, k_exp).squeeze(2)  # (D, B*N)
        h_k = torch.gather(selected_h, 2, k_exp).squeeze(2)  # (D, B*N)
        s_k = torch.gather(slopes_padded, 2, k_exp).squeeze(2)  # (D, B*N)

        # Integral on the active segment: prefix_sum + h_k*offset + slope/2*offset^2.
        # The torch.where guards skip the multiplication when the factor is exactly
        # zero, which avoids 0 * inf = NaN at the zero-height tail knot.
        integrals = C_k
        integrals = integrals + torch.where(h_k != 0.0, h_k * offset_all, torch.zeros_like(h_k))
        integrals = integrals + torch.where(s_k != 0.0, s_k / 2 * offset_all**2, torch.zeros_like(s_k))

        integrals = integrals.reshape(D, B, N).permute(1, 2, 0)  # (B, N, D)
        integrals = integrals.masked_fill(~valid_mask, 0.0)

        return integrals.sum(dim=1)  # (B, D)

    def event_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Efficiently compute kernel intensities at each event time for the correct event type.

        Memory-efficient in two ways:
        1. Uses _get_spline_values_for_pairs to evaluate scalar kernel values instead of (D,D) tensors.
        2. Only evaluates the spline for valid (causal, non-padded) pairs, skipping the majority
           of entries that would be zeroed out anyway.

        Instead of building dense (B, N, N) tensors and masking, we extract only the M_valid
        valid pair indices, compute M_valid spline evaluations, then scatter-add into the output.
        This reduces compute and peak memory from O(B*N*N) to O(M_valid) where
        M_valid << B*N*N for typical padded batches.

        Args:
            ts: Batch of event sequences.

        Returns:
            Kernel intensity at each event for its type. Shape: (B, N)
        """
        B = ts.time_points.shape[0]
        N = ts.time_points.shape[1]

        valid_events = ts.event_mask  # (B, N)
        valid_event_mask = valid_events.unsqueeze(2) & valid_events.unsqueeze(1)  # (B, N_recv, N_trig)
        valid_time_mask = ts.time_points.unsqueeze(2) > ts.time_points.unsqueeze(1)  # (B, N, N)
        valid_event_mask = valid_event_mask & valid_time_mask  # (B, N_recv, N_trig)
        # Extract only the valid pairs: indices (b, i=receiver, j=trigger)
        valid_idx = valid_event_mask.nonzero(as_tuple=False)  # (M_valid, 3)
        if valid_idx.shape[0] == 0:
            return torch.zeros(B, N, device=ts.time_points.device, dtype=ts.time_points.dtype)

        b_idx = valid_idx[:, 0]  # (M_valid,)
        i_idx = valid_idx[:, 1]  # (M_valid,) - receiver
        j_idx = valid_idx[:, 2]  # (M_valid,) - trigger

        valid_diffs = ts.time_points[b_idx, i_idx] - ts.time_points[b_idx, j_idx]  # (M_valid,)
        clamped_types = ts.event_types.clamp(min=0)  # (B, N)
        valid_recv = clamped_types[b_idx, i_idx]  # (M_valid,)
        valid_trig = clamped_types[b_idx, j_idx]  # (M_valid,)

        # Evaluate spline only for M_valid pairs instead of B*N*N
        kernel_vals = self._get_spline_values_for_pairs(valid_diffs, valid_recv, valid_trig)  # (M_valid,)

        # Scatter-add into (B, N) output indexed by receiver (b, i).  Equivalent to
        # building dense (B, N, N) and summing along the trigger axis, but avoids
        # the O(B*N*N) intermediate.
        out = torch.zeros(B, N, device=ts.time_points.device, dtype=ts.time_points.dtype)
        recv_flat = b_idx * N + i_idx  # flat index into (B*N,)
        out.view(-1).scatter_add_(0, recv_flat, kernel_vals)

        return out  # (B, N)


# =============================================================================
# Piecewise-Constant Kernel Module
# =============================================================================


@dataclass
class PiecewiseConstantKernelParams:
    h_knots: torch.Tensor  # Shape: (D, D, num_knots) — bin heights per (recv, trig) pair
    delta_ts: torch.Tensor  # Shape: (num_knots,) — bin widths (cumsum → bin edges on Δt axis)
    tail_delta: Optional[torch.Tensor] = None  # Width of the tail bin (scalar); defaults to last bin width


class PiecewiseConstantKernelModule(KernelIntensityModule):
    """Piecewise-constant kernel module for Hawkes processes.

    The kernel function is a step function over elapsed time Δt = t - t_j,
    with one step function per (receiver, trigger) pair.  The support consists
    of K regular bins followed by a *tail segment* and then zero:

        φ_{d,j}(Δt) = h_{d,j,k}        for Δt ∈ [knot_locs[k], knot_locs[k+1]),  k = 0…K-2
        φ_{d,j}(Δt) = h_{d,j,K-1}      for Δt ∈ [knot_locs[-1], tail_end)   ← constant extrapolation
        φ_{d,j}(Δt) = 0                 for Δt ≥ tail_end

    where ``tail_end = knot_locs[-1] + softplus(log_tail_delta)``.  The tail segment
    carries the **same height as the last regular bin** (constant extrapolation), and its
    width is a learnable parameter.  This makes the effective reach of the kernel
    learnable while keeping the kernel compactly supported.

    Internally, ``transform_params`` encodes this as a standard K+2 piecewise-constant
    spline (K regular bins + tail bin + zero-height sentinel knot at tail_end), so all
    lookup and integration is delegated directly to ``PiecewiseConstant``.

    When ``constrained=True`` (default), all bin heights are positive (via softplus).
    When ``constrained=False``, heights are unconstrained.

    Cumulative intensity is computed analytically:
        ∫₀^{Δt} φ_{d,j}(s) ds = sum of completed bin areas + partial current bin area.
    """

    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        params: Optional[PiecewiseConstantKernelParams] = None,
        seed: Optional[int] = None,
        constrained: bool = True,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        """
        Args:
            D: Number of event types
            num_knots: Number of regular bins (step-function segments, excluding the tail bin)
            delta_t: Width of each bin. Two forms:

                * ``float`` or scalar ``Tensor`` — uniform bin widths.
                * 1-D ``Tensor`` of length K — individual bin widths.

            params: Kernel parameters. If None, initialized randomly.
            seed: Random seed
            constrained: If True, heights are constrained positive via softplus.
                If False, heights are unconstrained (can be negative for inhibition).
            dataset_stats: Optional dataset statistics for data-driven initialisation.
        """
        super().__init__(D)
        self.constrained = constrained

        generator = init_generator(seed)
        self.num_knots = num_knots

        # Bin left-edges on the Δt axis (K regular bins; tail bin added via log_tail_delta)
        if isinstance(delta_t, float) or (isinstance(delta_t, torch.Tensor) and delta_t.numel() == 1):
            dt_val = float(delta_t) if isinstance(delta_t, float) else delta_t.item()
            knot_locs = torch.arange(num_knots, dtype=torch.float32) * dt_val
            bin_widths = torch.full((num_knots,), dt_val)
            default_tail_delta = dt_val
        else:
            widths = delta_t.float()
            knot_locs = torch.cat([torch.zeros(1, dtype=widths.dtype), torch.cumsum(widths[:-1], dim=0)])
            bin_widths = widths.clone()
            default_tail_delta = float(widths[-1])
        self.register_buffer("knot_locs", knot_locs)  # (K,) left edges of regular bins
        self.register_buffer("bin_widths", bin_widths)  # (K,) widths of regular bins

        if params is None:
            h_knots = self._init_random_params(D, num_knots, knot_locs, bin_widths, generator, dataset_stats)
            tail_delta_init = default_tail_delta
        else:
            h_knots = params.h_knots
            tail_delta_init = float(params.tail_delta) if params.tail_delta is not None else default_tail_delta

        self._init_parameters(h_knots, tail_delta_init)

    def breakpoints(self) -> torch.Tensor:
        return self.knot_locs_with_tail

    def _init_random_params(
        self,
        D: int,
        num_knots: int,
        knot_locs: torch.Tensor,
        bin_widths: torch.Tensor,
        generator: torch.Generator,
        dataset_stats: Optional[DatasetStats] = None,
    ) -> torch.Tensor:
        """Initialize bin heights to approximate a target exponential kernel.

        Each bin height is set to the average of an exponential decay over that bin:

            h_{d,j,k}  =  scale[d,j] * (1/w_k) * ∫_{l_k}^{l_k+w_k} exp(-beta*t) dt
                        =  scale[d,j] * exp(-beta*l_k) * (1 - exp(-beta*w_k)) / (beta*w_k)

        where ``beta[d,j]`` and ``scale[d,j]`` are chosen exactly as in
        ``SplineKernelModule._init_random_params`` so that the total integral
        (branching ratio) matches the target.

        When ``dataset_stats`` is provided:
        - ``beta`` is set uniformly to ``1 / mean_inter_event_time``.
        - ``alpha_weight`` starts near-zero with small noise, spectral-normalised
          to target branching ratio 0.5.

        Returns:
            h_knots: Shape (D, D, num_knots) — positive bin heights
        """
        targeted_branching_ratio = 0.5
        T_max = float(knot_locs[-1]) + float(bin_widths[-1])  # right edge of last regular bin

        if dataset_stats is not None:
            fallback_beta = 1.0 / max(dataset_stats.mean_inter_event_time, 1e-9)
            beta = torch.full((D, D), fallback_beta)
            unseen_alpha = 1e-4 if dataset_stats.is_terminating else 1e-3
            alpha_weight = torch.full((D, D), unseen_alpha) + torch.rand(size=(D, D), generator=generator) * 0.1
            max_eig = torch.max(torch.abs(torch.linalg.eigvals(alpha_weight)))
            if max_eig > 0:
                alpha_weight = (targeted_branching_ratio / max_eig) * alpha_weight
        else:
            # Without data stats, choose beta so that the exponential decays over the
            # knot range: sample 1–5 decay-lengths per pair so each pair looks distinct.
            if T_max > 0:
                decay_lengths = torch.rand(size=(D, D), generator=generator) * 4.0 + 1.0  # Uniform[1, 5]
                beta = decay_lengths / T_max
            else:
                beta = torch.ones(D, D)
            alphas = torch.rand(size=(D, D), generator=generator) * 100 + 1
            max_eig = torch.max(torch.abs(torch.linalg.eigvals(alphas)))
            alpha_weight = (1 / max_eig) * targeted_branching_ratio * alphas

        # scale: height at t=0 so that ∫₀^{T_max} scale*exp(-beta*t) dt = alpha_weight
        # => scale = alpha_weight * beta / (1 - exp(-beta * T_max))
        beta_T = (beta * T_max).clamp(min=1e-6)
        scale = alpha_weight * beta / (1.0 - torch.exp(-beta_T)).clamp(min=1e-9)  # (D, D)

        # Per-bin average height: mean of exp decay over each bin [l_k, l_k + w_k]
        # = scale * exp(-beta*l_k) * (1 - exp(-beta*w_k)) / (beta*w_k)
        # (D, D, 1) broadcasted with (1, 1, K)
        l_k = knot_locs.unsqueeze(0).unsqueeze(0)  # (1, 1, K)
        w_k = bin_widths.unsqueeze(0).unsqueeze(0)  # (1, 1, K)
        beta_w = (beta.unsqueeze(-1) * w_k).clamp(min=1e-6)  # (D, D, K)
        h_knots = (
            scale.unsqueeze(-1) * torch.exp(-beta.unsqueeze(-1) * l_k) * (1.0 - torch.exp(-beta_w)) / beta_w
        )  # (D, D, K)

        return h_knots

    def _init_parameters(self, h_knots: torch.Tensor, tail_delta_init: float):
        """Initialize learnable parameters from bin heights and tail delta."""
        if self.constrained:
            self.log_h_knots = nn.Parameter(inverse_softplus(h_knots))  # (D, D, K)
        else:
            x_init = inverse_softplus(h_knots.abs().clamp(min=1e-12))
            scale = torch.sigmoid(x_init).mean().clamp(min=1e-12)
            self.register_buffer("_h_knots_scale", scale)
            self.h_knots_param = nn.Parameter(h_knots.clone() / scale)  # (D, D, K)

        # Learnable distance from the last regular bin's left edge to the tail bin's left edge.
        # Stored as inverse_softplus(tail_delta_init) so softplus always gives a positive offset.
        self.log_tail_delta = nn.Parameter(torch.tensor(inverse_softplus(torch.tensor(tail_delta_init)).item()))

    @property
    def tail_start(self) -> torch.Tensor:
        """Left edge of the tail segment = right edge of the last regular bin. Scalar."""
        return self.knot_locs[-1] + self.bin_widths[-1]

    @property
    def tail_end(self) -> torch.Tensor:
        """Right edge of the tail segment (= start of the zero region). Scalar."""
        return self.tail_start + F.softplus(self.log_tail_delta)

    @property
    def knot_locs_with_tail(self) -> torch.Tensor:
        """Knot locations for the full piecewise-constant spline.  Shape: (K+2,).

        Index 0..K-1: regular bin left edges.
        Index K:      start of the constant-extrapolation tail segment
                      (= right edge of the last regular bin).
        Index K+1:    tail_end — the zero-height sentinel knot that terminates
                      the tail; PiecewiseConstant holds the last height (0) for
                      all t beyond this point.
        """
        return torch.cat([self.knot_locs, self.tail_start.unsqueeze(0), self.tail_end.unsqueeze(0)])

    def transform_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return bin heights and knot locations as a standard piecewise-constant spline.

        The tail and zero-beyond region are encoded as ordinary knots so that
        PiecewiseConstant can handle lookup and integration directly:

            knot_locs  (K+2,): regular left-edges + tail_start + tail_end
            h_knots  (D, D, K+2):
                0 … K-1  : regular bin heights
                K        : tail segment height = h[..., K-1]  (constant extrapolation)
                K+1      : beyond-tail height = 0
        """
        if self.constrained:
            h = F.softplus(self.log_h_knots)  # (D, D, K)
        else:
            h = self.h_knots_param * self._h_knots_scale  # (D, D, K)

        # Tail bin: same height as last regular bin — constant extrapolation
        tail = h[..., -1:]  # (D, D, 1)
        # Beyond-tail: zero
        zero = torch.zeros(*h.shape[:2], 1, device=h.device, dtype=h.dtype)
        h_full = torch.cat([h, tail, zero], dim=2)  # (D, D, K+2)
        return h_full, self.knot_locs_with_tail  # (D, D, K+2), (K+2,)

    def _get_values_for_trigger(self, delta_t: torch.Tensor, trigger_types: torch.Tensor) -> torch.Tensor:
        """Evaluate step function for the given trigger types, returning (M, D_recv).

        Args:
            delta_t: Elapsed times. Shape: (M,)
            trigger_types: Trigger type index. Shape: (M,), values in [0, D)

        Returns:
            Kernel values. Shape: (M, D_recv)
                result[m, d] = φ_{d, trigger_types[m]}(delta_t[m])
        """
        h_knots, knot_locs = self.transform_params()  # (D_recv, D_trig, K+2), (K+2,)

        # Gather bin heights for each trigger type: (D_recv, M, K+2) → (D_recv*M, K+2)
        selected_h = h_knots[:, trigger_types, :]  # (D_recv, M, K+2)
        D, M, K2 = selected_h.shape
        h_flat = selected_h.reshape(D * M, K2)  # (D_recv*M, K+2)

        # Repeat delta_t once per receiver dimension, then delegate to PiecewiseConstant
        t_rep = delta_t.unsqueeze(0).expand(D, -1).reshape(-1)  # (D_recv*M,)
        vals = PiecewiseConstant.interpolate_flat(knot_locs, h_flat, t_rep)  # (D_recv*M,)

        return vals.reshape(D, M).T  # (M, D_recv)

    def _get_values_for_pairs(
        self, delta_t: torch.Tensor, receiver_types: torch.Tensor, trigger_types: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate step function for specific (receiver, trigger) pairs, returning scalars.

        Args:
            delta_t: Elapsed times. Shape: (M,)
            receiver_types: Receiver type. Shape: (M,), values in [0, D)
            trigger_types: Trigger type. Shape: (M,), values in [0, D)

        Returns:
            Scalar kernel values. Shape: (M,)
                result[m] = φ_{receiver_types[m], trigger_types[m]}(delta_t[m])
        """
        h_knots, knot_locs = self.transform_params()  # (D_recv, D_trig, K+2), (K+2,)
        selected_h = h_knots[receiver_types, trigger_types, :]  # (M, K+2)
        return PiecewiseConstant.interpolate_flat(knot_locs, selected_h, delta_t)  # (M,)

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute kernel contribution to intensity at time t.

        λ_kernel_d(t) = Σ_{t_i < t} φ_{d, type(i)}(t - t_i)

        Args:
            t: Query times. Shape: (B,)
            ts: Historical events. Shape: (B, N)

        Returns:
            Kernel intensity for each event type. Shape: (B, D)
        """
        B = t.shape[0]
        N = ts.time_points.shape[1] if len(ts) > 0 else 0

        if N == 0:
            return torch.zeros(B, self.D, device=t.device, dtype=t.dtype)

        valid_mask = ts.event_mask & (ts.time_points < t.unsqueeze(1))  # (B, N)
        delta_t = t.unsqueeze(1) - ts.time_points  # (B, N)
        delta_t = torch.where(valid_mask, delta_t, torch.zeros_like(delta_t))

        clamped_types = ts.event_types_idx_safe  # (B, N)
        flat_dt = delta_t.reshape(-1)  # (B*N,)
        flat_types = clamped_types.reshape(-1)  # (B*N,)

        contributions = self._get_values_for_trigger(flat_dt, flat_types)  # (B*N, D_recv)
        contributions = contributions.reshape(B, N, self.D)  # (B, N, D_recv)
        contributions = contributions.masked_fill(~valid_mask.unsqueeze(2), 0.0)

        return contributions.sum(dim=1)  # (B, D)

    def _cumulative_intensity(
        self,
        T: torch.Tensor,
        ts: BatchedMVEventData,
    ) -> torch.Tensor:
        """
        Compute cumulative kernel intensity from 0 to T.

        Λ_kernel_d(T) = Σ_{t_i < T} ∫₀^{T - t_i} φ_{d, type(i)}(s) ds

        ``T`` is normalised to ``(B, D)`` by the public ``cumulative_intensity``.

        Args:
            T: End times. Shape: (B, D)
            ts: Historical events. Shape: (B, N)

        Returns:
            Cumulative kernel intensity. Shape: (B, D)
        """
        B = T.shape[0]
        N = ts.time_points.shape[1] if len(ts) > 0 else 0

        if N == 0:
            return torch.zeros(B, self.D, device=T.device, dtype=T.dtype)

        valid_events = ts.event_mask  # (B, N)
        clamped_types = ts.event_types_idx_safe  # (B, N)

        # Per-dimension elapsed times: (B, N, D)
        delta_t = T.unsqueeze(1) - ts.time_points.unsqueeze(2)  # (B, N, D)
        past_mask = ts.time_points.unsqueeze(2) < T.unsqueeze(1)  # (B, N, D)
        valid_mask = valid_events.unsqueeze(2) & past_mask  # (B, N, D)
        delta_t = torch.where(valid_mask, delta_t, torch.zeros_like(delta_t))

        h_knots, knot_locs = self.transform_params()  # (D_recv, D_trig, K+2), (K+2,)
        D = self.D
        K2 = h_knots.shape[2]

        flat_types = clamped_types.reshape(-1)  # (B*N,)
        # Gather bin heights for each event's trigger type: (D_recv, B*N, K+2)
        selected_h = h_knots[:, flat_types, :]  # (D_recv, B*N, K+2)

        # delta_t: (B, N, D) → permute to (D_recv, B*N)
        dt_perm = delta_t.permute(2, 0, 1).reshape(D, B * N)  # (D_recv, B*N)

        # For each (recv_dim, event) pair, integrate from 0 to dt_perm using the
        # per-pair heights. Reshape to (D_recv*B*N, K+2) for PiecewiseConstant.
        h_2d = selected_h.reshape(D * B * N, K2)  # (D*B*N, K+2)
        dt_flat = dt_perm.reshape(-1)  # (D*B*N,)
        integrals_flat = PiecewiseConstant.integrate_flat(knot_locs, h_2d, dt_flat)  # (D*B*N,)

        integrals = integrals_flat.reshape(D, B, N).permute(1, 2, 0)  # (B, N, D)
        integrals = integrals.masked_fill(~valid_mask, 0.0)

        return integrals.sum(dim=1)  # (B, D)

    def event_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Efficiently compute kernel intensities at each event time for the correct event type.

        Uses sparse pair extraction to avoid the O(B*N*N) dense tensor.

        Args:
            ts: Batch of event sequences.

        Returns:
            Kernel intensity at each event for its type. Shape: (B, N)
        """
        B = ts.time_points.shape[0]
        N = ts.time_points.shape[1]

        valid_events = ts.event_mask  # (B, N)
        valid_event_mask = valid_events.unsqueeze(2) & valid_events.unsqueeze(1)  # (B, N_recv, N_trig)
        valid_time_mask = ts.time_points.unsqueeze(2) > ts.time_points.unsqueeze(1)  # (B, N, N)
        valid_event_mask = valid_event_mask & valid_time_mask
        valid_idx = valid_event_mask.nonzero(as_tuple=False)  # (M_valid, 3)
        if valid_idx.shape[0] == 0:
            return torch.zeros(B, N, device=ts.time_points.device, dtype=ts.time_points.dtype)

        b_idx = valid_idx[:, 0]
        i_idx = valid_idx[:, 1]  # receiver
        j_idx = valid_idx[:, 2]  # trigger

        valid_diffs = ts.time_points[b_idx, i_idx] - ts.time_points[b_idx, j_idx]  # (M_valid,)
        clamped_types = ts.event_types.clamp(min=0)
        valid_recv = clamped_types[b_idx, i_idx]  # (M_valid,)
        valid_trig = clamped_types[b_idx, j_idx]  # (M_valid,)

        kernel_vals = self._get_values_for_pairs(valid_diffs, valid_recv, valid_trig)  # (M_valid,)

        out = torch.zeros(B, N, device=ts.time_points.device, dtype=ts.time_points.dtype)
        recv_flat = b_idx * N + i_idx
        # Piecewise interpolation returns values in the parameter dtype (float32),
        # which need not match time_points' dtype; align before
        # the in-place scatter, which requires self.dtype == src.dtype.
        out.view(-1).scatter_add_(0, recv_flat, kernel_vals.to(out.dtype))

        return out  # (B, N)


# ============================================================================
# Neural Spline Kernel Module
# ============================================================================


class MLPSplineKernelModule(KernelIntensityModule):
    """Kernel whose per-trigger spline heights are predicted by an MLP.

    For each past event of type ``k``, an MLP maps the learned event-type
    embedding to ``D_recv × K`` spline knot heights::

        h_{:, k, :} = MLP(embedding(k)).reshape(D_recv, K)

    The resulting knot heights define a linear-spline kernel over elapsed time
    ``Δt = t - t_k`` for every receiver type.  Knot locations are shared and
    fixed (not learned), exactly as in :class:`SplineKernelModule`.

    **Advantages over SplineKernelModule (D×D×K params):**
    Parameters scale as ``O(D × emb_dim + MLP_size)`` instead of ``O(D² × K)``,
    which is critical for large D (thousands of event types).

    **Advantages over MLPKernelModule (MLP per query):**
    Has analytical cumulative intensity via ``LinearSpline.integrate``, so it
    can be used with :class:`~torchpp.tpp.hawkes.HawkesProcess` (analytical CI)
    or :class:`~torchpp.tpp.hawkes.InhibitiveHawkesProcess` (numerical CI)
    without evaluating the MLP at every integration grid point.  The MLP is
    called once per past event; the spline handles the temporal shape cheaply.

    Knot heights can be positive or negative (unconstrained), allowing both
    excitation and inhibition.  A learnable zero-height tail knot is appended,
    as in :class:`SplineKernelModule`; the kernel is zero beyond it (compact support).

    Args:
        D: Number of event types.
        num_knots: Number of spline knots (excluding the tail knot).
        delta_t: Knot spacing — float for uniform, tensor for variable gaps.
        emb_dim: Dimension of the learned event-type embedding.
        n_hidden: Width of each MLP hidden layer.
        n_mlp_layers: Number of hidden layers (excluding the output projection).
        seed: Random seed for parameter initialisation.
        dataset_stats: Empirical statistics for data-driven knot spacing.
    """

    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        emb_dim: int = 32,
        n_hidden: int = 64,
        n_mlp_layers: int = 2,
        seed: Optional[int] = None,
        dataset_stats: Optional[DatasetStats] = None,
    ):
        super().__init__(D)
        self.num_knots = num_knots
        self.emb_dim = emb_dim
        K = num_knots

        # ── Knot locations (shared, fixed) ────────────────────────────
        if isinstance(delta_t, (int, float)):
            knot_locs = torch.linspace(0, delta_t * (K - 1), K)
            default_tail_delta = delta_t
        else:
            spacings = delta_t.float()
            knot_locs = torch.cat([torch.zeros(1, dtype=spacings.dtype), torch.cumsum(spacings, dim=0)])
            default_tail_delta = float(spacings[-1])
        self.register_buffer("knot_locs", knot_locs)

        # Learnable tail knot distance (same convention as SplineKernelModule)
        self.log_tail_delta = nn.Parameter(torch.tensor(inverse_softplus(torch.tensor(default_tail_delta)).item()))

        # ── Event-type embedding ──────────────────────────────────────
        generator = init_generator(seed)
        self.type_embedding = torch.nn.Embedding(D, emb_dim)
        torch.nn.init.normal_(self.type_embedding.weight, std=0.02, generator=generator)

        # ── MLP: embedding → (D_recv × K) knot heights ───────────────
        input_dim = emb_dim
        output_dim = D * K  # reshaped to (D_recv, K) after forward
        layers: list[torch.nn.Module] = []
        for i in range(n_mlp_layers):
            layers.append(torch.nn.Linear(input_dim if i == 0 else n_hidden, n_hidden))
            layers.append(torch.nn.GELU())
        layers.append(torch.nn.Linear(n_hidden, output_dim))
        self.mlp = torch.nn.Sequential(*layers)

        # Initialise output layer small so kernel starts near zero
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.01)
            if self.mlp[-1].bias is not None:
                self.mlp[-1].bias.zero_()

        # ── Per-batch cache of MLP outputs for unique trigger types ──
        # Filled by ``prepare_batch`` (or lazily by ``_get_h_unique``);
        # released by ``clear_batch``.
        self._cached_unique_types: Optional[torch.Tensor] = None  # (U,)
        self._cached_h_unique: Optional[torch.Tensor] = None  # (U, D, K+1)
        self._cached_key: Optional[tuple] = None

    def breakpoints(self) -> torch.Tensor:
        return self.knot_locs_with_tail

    @property
    def knot_locs_with_tail(self) -> torch.Tensor:
        """Knot locations including the zero-height tail knot. Shape: (K+1,)"""
        tail_loc = self.knot_locs[-1] + F.softplus(self.log_tail_delta)
        return torch.cat([self.knot_locs, tail_loc.unsqueeze(0)])

    def transform_params(self) -> Tuple[torch.Tensor, ...]:
        """No explicit constrained/unconstrained split."""
        return ()

    # ------------------------------------------------------------------
    # Caching: per-batch reuse of MLP outputs for unique trigger types
    # ------------------------------------------------------------------
    #
    # The MLP output ``MLP(embedding(k))`` only depends on the trigger type
    # ``k``, not on the query time.  When this kernel is plugged into an
    # InhibitiveHawkesProcess, the parent calls ``intensity(t, ts)`` at every
    # numerical-CI grid point — so the same MLP would otherwise be recomputed
    # once per quadrature point for the same set of trigger types.
    #
    # ``prepare_batch(ts)`` runs the MLP once for every unique trigger type
    # in the batch and stores ``(U, D, K+1)`` outputs.  Subsequent calls to
    # ``intensity`` / ``_cumulative_intensity`` / ``event_intensities`` reuse
    # those outputs.  ``clear_batch()`` releases them; ``fix_context(ts)``
    # provides a context manager that handles both.
    #
    # When no cache is set (``prepare_batch`` was not called), each forward
    # method falls back to a *single* deduplicated MLP call (still cheaper
    # than per-event evaluation, but does not amortize across grid points).

    @staticmethod
    def _batch_cache_key(ts: BatchedMVEventData) -> tuple:
        """Cheap CONTENT fingerprint to validate cache hits across forward calls.

        Shape plus two time moments and the type sum — deliberately object-identity
        free (mirrors :meth:`NeuralEncoderMixin._batch_cache_key`).  The time-unit
        shell rebuilds a rescaled batch per public call, so priming on
        the SCALED batch and querying the scaled core must hit even though the two are
        distinct objects; an ``id``-keyed fingerprint never matched at ``s != 1``.
        """
        tp = ts.time_points
        return (
            tuple(tp.shape),
            tp.sum().item(),
            (tp * tp).sum().item(),
            ts.event_types.sum().item(),
        )

    def prepare_batch(self, ts: BatchedMVEventData) -> None:
        """Run the MLP once per unique trigger type in ``ts`` and cache outputs.

        Call before a block of ``intensity`` / ``_cumulative_intensity`` /
        ``event_intensities`` calls against the same batch.  Released by
        ``clear_batch()``.  See :meth:`fix_context` for an automatic version.
        """
        flat = ts.event_types_idx_safe.reshape(-1)  # (B*N,) safe indices in [0, D)
        if flat.numel() == 0:
            unique = flat
            h_unique = torch.empty(0, self.D, self.num_knots + 1, device=flat.device, dtype=self.knot_locs.dtype)
        else:
            unique = torch.unique(flat)  # sorted (U,)
            h_unique = self._predict_knots_unique(unique)  # (U, D, K+1)
        self._cached_unique_types = unique
        self._cached_h_unique = h_unique
        self._cached_key = self._batch_cache_key(ts)

    def clear_batch(self) -> None:
        """Release cached MLP outputs."""
        self._cached_unique_types = None
        self._cached_h_unique = None
        self._cached_key = None

    @contextmanager
    def fix_context(self, ts: BatchedMVEventData) -> Iterator[None]:
        """Context manager: cache MLP outputs for ``ts`` for the duration of the block."""
        self.prepare_batch(ts)
        try:
            yield
        finally:
            self.clear_batch()

    def _predict_knots_unique(self, unique_types: torch.Tensor) -> torch.Tensor:
        """Predict spline knot heights for *unique* trigger types only.

        Args:
            unique_types: Sorted unique integer trigger types.  Shape: ``(U,)``.

        Returns:
            Knot heights with tail appended.  Shape: ``(U, D_recv, K+1)``.
        """
        emb = self.type_embedding(unique_types)  # (U, emb_dim)
        raw = self.mlp(emb)  # (U, D*K)
        h = raw.reshape(-1, self.D, self.num_knots)  # (U, D, K)
        tail = torch.zeros(*h.shape[:2], 1, device=h.device, dtype=h.dtype)
        return torch.cat([h, tail], dim=2)  # (U, D, K+1)

    def _get_h_unique(
        self,
        query_types: torch.Tensor,
        ts: Optional[BatchedMVEventData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(h_unique, idx_in_unique)`` for ``query_types``.

        If a per-batch cache is live and ``ts`` matches it, ``h_unique`` is
        the cached ``(U_batch, D, K+1)`` tensor and ``idx_in_unique`` maps
        each query into that cache via ``searchsorted``.

        Otherwise the MLP is run on the unique set of ``query_types`` and the
        result is returned together with its inverse map.

        Args:
            query_types: Integer trigger types to look up.  Shape: ``(M,)``.
            ts: Batch the query came from — used only to validate the cache
                fingerprint.  Pass ``None`` to bypass cache validation.

        Returns:
            h_unique:       ``(U, D, K+1)``
            idx_in_unique:  ``(M,)`` — indices into ``h_unique`` for each query
        """
        if self._cached_h_unique is not None:
            if ts is not None and self._cached_key != self._batch_cache_key(ts):
                raise RuntimeError(
                    "MLPSplineKernelModule: cached MLP outputs were prepared for a "
                    "different batch.  Call prepare_batch(ts) with the current batch "
                    "before reusing the cache, or call clear_batch()."
                )
            # Cached unique types are sorted; map queries into the cache via searchsorted.
            idx = torch.searchsorted(self._cached_unique_types, query_types)
            return self._cached_h_unique, idx

        # No cache: dedup on the fly.  Still saves the per-event MLP call,
        # just not across multiple grid points.
        unique, inverse = torch.unique(query_types, return_inverse=True)
        h_unique = self._predict_knots_unique(unique)
        return h_unique, inverse

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Kernel intensity at query time ``t``.  Shape: ``(B, D)``.

        For each past event j of type k at time t_j, computes the spline
        ``φ_{:,k}(t - t_j)`` and sums contributions.

        Memory layout: never materializes ``(M, D, K+1)``.  The (small)
        ``(U, D, K+1)`` MLP-output tensor is kept (cached or fresh) and the
        per-event spline values are gathered directly into ``(M, D)``.
        """
        B, N = ts.time_points.shape
        device = ts.time_points.device

        dt = t.unsqueeze(1) - ts.time_points  # (B, N)
        mask = (dt > 0) & ts.event_mask  # (B, N)
        b_idx, n_idx = torch.where(mask)  # each (M,)
        if b_idx.numel() == 0:
            return torch.zeros(B, self.D, device=device, dtype=ts.time_points.dtype)

        dt_valid = dt[b_idx, n_idx]  # (M,)
        types_valid = ts.event_types_idx_safe[b_idx, n_idx]  # (M,)

        h_unique, u_idx = self._get_h_unique(types_valid, ts=ts)  # (U, D, K+1), (M,)
        knot_locs = self.knot_locs_with_tail  # (K+1,)
        K1 = h_unique.shape[2]

        k = torch.searchsorted(knot_locs, dt_valid, right=True) - 1  # (M,)
        k = k.clamp(0, K1 - 2)
        offset = (dt_valid - knot_locs[k]).unsqueeze(1)  # (M, 1)

        delta_times = knot_locs[1:] - knot_locs[:-1]  # (K,)
        seg_dt = delta_times[k].unsqueeze(1)  # (M, 1) — knot spacing of the active segment

        # Fancy indexing: h_unique[u_idx[m], :, k[m]] -> (M, D)
        h_k = h_unique[u_idx, :, k]  # (M, D)
        h_k_plus1 = h_unique[u_idx, :, k + 1]  # (M, D)
        slope = (h_k_plus1 - h_k) / seg_dt  # (M, D)

        vals = h_k + slope * offset  # (M, D)
        vals = vals.masked_fill((dt_valid >= knot_locs[-1]).unsqueeze(1), 0.0)  # compact support past the tail knot

        out = torch.zeros(B, self.D, device=device, dtype=vals.dtype)
        out.scatter_add_(0, b_idx.unsqueeze(1).expand_as(vals), vals)
        return out  # (B, D)

    def _cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Analytical cumulative kernel intensity.  T shape: ``(B, D)``.

        For each past event j of type k at time t_j, integrates the spline
        ``∫₀^{min(T_d, T_d - t_j)} φ_{d,k}(s) ds`` and sums.

        Memory layout: never materializes the per-event ``(B*N, D, K+1)``
        knot tensor.  Spline segment-area prefix sums are computed once on
        the much smaller ``(U, D, K+1)`` MLP-output tensor and reused for
        every (b, n) via fancy indexing.
        """
        B, N = ts.time_points.shape
        device = T.device

        if N == 0:
            return torch.zeros(B, self.D, device=device, dtype=T.dtype)

        valid_events = ts.event_mask  # (B, N)
        clamped_types = ts.event_types_idx_safe  # (B, N)

        # Per-dimension elapsed times: (B, N, D)
        delta_t = T.unsqueeze(1) - ts.time_points.unsqueeze(2)  # (B, N, D)
        past_mask = ts.time_points.unsqueeze(2) < T.unsqueeze(1)  # (B, N, D)
        valid_mask = valid_events.unsqueeze(2) & past_mask  # (B, N, D)
        delta_t = torch.where(valid_mask, delta_t, torch.zeros_like(delta_t))

        flat_types = clamped_types.reshape(-1)  # (B*N,)
        h_unique, u_idx_flat = self._get_h_unique(flat_types, ts=ts)  # (U, D, K+1), (B*N,)
        u_idx = u_idx_flat.reshape(B, N)  # (B, N)

        knot_locs = self.knot_locs_with_tail  # (K+1,)
        K1 = h_unique.shape[2]
        D = self.D
        delta_t = torch.minimum(delta_t, knot_locs[-1].to(delta_t.dtype))  # the kernel is zero past the tail knot

        delta_times = knot_locs[1:] - knot_locs[:-1]  # (K,)

        # Precompute segment areas + prefix sums on the small (U, D, K+1) tensor.
        # segment_areas[u, d, j] = ∫_{knot_locs[j]}^{knot_locs[j+1]} φ_{d,k=u}(s) ds
        h_lo = h_unique[..., :-1]  # (U, D, K)
        h_hi = h_unique[..., 1:]  # (U, D, K)
        slopes_unique = (h_hi - h_lo) / delta_times  # (U, D, K)
        segment_areas = h_lo * delta_times + 0.5 * slopes_unique * delta_times**2  # (U, D, K)
        prefix_sums_unique = F.pad(torch.cumsum(segment_areas, dim=-1), (1, 0))  # (U, D, K+1)
        slopes_padded_unique = F.pad(slopes_unique, (0, 1))  # (U, D, K+1)

        # Per-(b, n, d) integration limit -> active segment.
        # delta_t: (B, N, D)
        k_idx = torch.searchsorted(knot_locs, delta_t.reshape(-1), right=True) - 1
        k_idx = k_idx.clamp(0, K1 - 2).reshape(B, N, D)  # (B, N, D)
        offset = delta_t - knot_locs[k_idx]  # (B, N, D)

        # Gather prefix sum / h / slope at the active segment for each (b, n, d).
        # Indexing pattern: prefix_sums_unique[u_idx[b,n], d, k_idx[b,n,d]] -> (B, N, D)
        u_bn = u_idx.unsqueeze(-1).expand(B, N, D)  # (B, N, D)
        d_idx = torch.arange(D, device=device).view(1, 1, D).expand(B, N, D)
        C_k = prefix_sums_unique[u_bn, d_idx, k_idx]  # (B, N, D)
        h_k = h_unique[u_bn, d_idx, k_idx]  # (B, N, D)
        s_k = slopes_padded_unique[u_bn, d_idx, k_idx]  # (B, N, D)

        integrals = C_k + h_k * offset + 0.5 * s_k * offset**2  # (B, N, D)
        integrals = integrals.masked_fill(~valid_mask, 0.0)
        return integrals.sum(dim=1)  # (B, D)

    def event_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Kernel intensity of the observed event type at each position.  Shape: ``(B, L)``.

        For event i, sum kernel contributions from all events j whose time
        is *strictly* before event i's time, evaluated at the receiver type
        matching event i.  Strict time comparison (not sequence index) so
        simultaneous events do not influence each other — matches the
        convention used in :class:`SplineKernelModule`.

        Memory layout: never materializes ``(M, D, K+1)`` or ``(M, D)``.
        For each (b, i, j) pair we directly gather the single
        ``h_unique[u_idx_j, type_i, k]`` scalar — D times less memory than
        evaluating all receivers and discarding all but one.
        """
        B, N = ts.time_points.shape
        device = ts.time_points.device
        times = ts.time_points
        mask = ts.event_mask

        # Strict-time causality: j contributes to i iff t_j < t_i (and both real).
        valid = (
            mask.unsqueeze(2)  # (B, N, 1) — receiver i is real
            & mask.unsqueeze(1)  # (B, 1, N) — trigger j is real
            & (times.unsqueeze(2) > times.unsqueeze(1))  # (B, N, N) — t_i > t_j
        )

        b_idx, i_idx, j_idx = torch.where(valid)
        if b_idx.numel() == 0:
            return torch.zeros(B, N, device=device, dtype=times.dtype)

        dt_valid = times[b_idx, i_idx] - times[b_idx, j_idx]  # (M,)
        types_j = ts.event_types_idx_safe[b_idx, j_idx]  # (M,) trigger types
        types_i = ts.event_types_idx_safe[b_idx, i_idx]  # (M,) receiver types

        h_unique, u_idx = self._get_h_unique(types_j, ts=ts)  # (U, D, K+1), (M,)
        knot_locs = self.knot_locs_with_tail
        K1 = h_unique.shape[2]

        k = torch.searchsorted(knot_locs, dt_valid, right=True) - 1  # (M,)
        k = k.clamp(0, K1 - 2)
        offset = dt_valid - knot_locs[k]  # (M,)

        delta_times = knot_locs[1:] - knot_locs[:-1]  # (K,)
        seg_dt = delta_times[k]  # (M,)

        # Direct triple fancy-index — single scalar per (m): no D dimension materialized.
        h_at = h_unique[u_idx, types_i, k]  # (M,)
        h_at_plus1 = h_unique[u_idx, types_i, k + 1]  # (M,)
        slope_recv = (h_at_plus1 - h_at) / seg_dt  # (M,)
        recv_vals = h_at + slope_recv * offset  # (M,)
        recv_vals = recv_vals.masked_fill(dt_valid >= knot_locs[-1], 0.0)  # compact support past the tail knot

        out = torch.zeros(B, N, device=device, dtype=recv_vals.dtype)
        recv_flat = b_idx * N + i_idx
        out.view(-1).scatter_add_(0, recv_flat, recv_vals)
        return out  # (B, N)


# ============================================================================
# Neural Kernel Module
# ============================================================================



class MLPKernelModule(KernelIntensityModule):
    """Kernel whose triggering function is parameterised by an MLP.

    For each past event of type ``k`` that occurred ``Δt`` ago, the kernel
    contribution to the intensity of *every* receiver type is::

        φ(k, Δt) = MLP([ embedding(k) ; sin_enc(Δt) ])   ∈ ℝ^D

    Values can be positive (excitation) or negative (inhibition).  The parent
    process is responsible for applying a positivity constraint (e.g. softplus)
    to the total intensity.

    No analytical cumulative intensity is provided — use with
    :class:`~torchpp.tpp.hawkes.InhibitiveHawkesProcess` (numerical CI).

    Args:
        D: Number of event types.
        emb_dim: Dimension of the learned event-type embedding.
        time_embd_dim: Dimension of the sinusoidal time encoding.
        n_hidden: Width of each MLP hidden layer.
        n_mlp_layers: Number of hidden layers (excluding the output projection).
        seed: Random seed for parameter initialisation.
    """

    def __init__(
        self,
        D: int,
        emb_dim: int = 32,
        time_embd_dim: int = 32,
        n_hidden: int = 64,
        n_mlp_layers: int = 2,
        seed: Optional[int] = None,
    ):
        super().__init__(D)
        if time_embd_dim % 2 != 0:
            raise ValueError(f"time_embd_dim must be even, got {time_embd_dim}.")
        self.emb_dim = emb_dim
        self.time_embd_dim = time_embd_dim

        # Event-type embedding
        self.type_embedding = torch.nn.Embedding(D, emb_dim)

        # Sinusoidal time encoding buffers
        div_term = torch.exp(torch.arange(0, time_embd_dim, 2) * (-math.log(10000.0) / time_embd_dim))
        self.register_buffer("div_term", div_term)

        # MLP: [event_emb ; time_enc] → D
        input_dim = emb_dim + time_embd_dim
        layers: list[torch.nn.Module] = []
        for i in range(n_mlp_layers):
            layers.append(torch.nn.Linear(input_dim if i == 0 else n_hidden, n_hidden))
            layers.append(torch.nn.GELU())
        layers.append(torch.nn.Linear(n_hidden, D))
        self.mlp = torch.nn.Sequential(*layers)

        # Initialise output layer near zero so kernel starts small
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.01)
            if self.mlp[-1].bias is not None:
                self.mlp[-1].bias.zero_()

        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
            torch.nn.init.normal_(self.type_embedding.weight, std=0.02, generator=gen)

    def transform_params(self) -> Tuple[torch.Tensor, ...]:
        """No explicit constrained/unconstrained split — all params are unconstrained."""
        return ()

    def _time_encode(self, dt: torch.Tensor) -> torch.Tensor:
        """Sinusoidal encoding of elapsed time.

        Args:
            dt: Time deltas, arbitrary shape ``(...)``.

        Returns:
            Encoding, shape ``(..., time_embd_dim)``.
        """
        t = dt.unsqueeze(-1)  # (..., 1)
        enc = torch.zeros(*dt.shape, self.time_embd_dim, device=dt.device, dtype=dt.dtype)
        enc[..., 0::2] = torch.sin(t * self.div_term)
        enc[..., 1::2] = torch.cos(t * self.div_term)
        return enc

    def _kernel_values(self, dt: torch.Tensor, trigger_types: torch.Tensor) -> torch.Tensor:
        """Compute kernel values for given deltas and trigger types.

        Args:
            dt: Time deltas since trigger event.  Shape ``(M,)`` (flat).
            trigger_types: Integer trigger-event types.  Shape ``(M,)``.

        Returns:
            Kernel contribution to each receiver type.  Shape ``(M, D)``.
        """
        emb = self.type_embedding(trigger_types)  # (M, emb_dim)
        t_enc = self._time_encode(dt.to(emb.dtype))  # (M, time_embd_dim)
        x = torch.cat([emb, t_enc], dim=-1)  # (M, emb_dim + time_embd_dim)
        return self.mlp(x)  # (M, D)

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Kernel intensity contribution at query time ``t``.  Shape: ``(B, D)``.

        Sums the MLP kernel value over all events strictly before ``t``.
        """
        B, N = ts.time_points.shape
        device = ts.time_points.device

        dt = t.unsqueeze(1) - ts.time_points  # (B, N)

        # Only past events (dt > 0) that are real (not padding)
        mask = (dt > 0) & ts.event_mask  # (B, N)
        b_idx, n_idx = torch.where(mask)  # each (M,)
        if b_idx.numel() == 0:
            return torch.zeros(B, self.D, device=device, dtype=ts.time_points.dtype)

        dt_valid = dt[b_idx, n_idx]  # (M,)
        types_valid = ts.event_types_idx_safe[b_idx, n_idx]  # (M,)

        kernel_vals = self._kernel_values(dt_valid, types_valid)  # (M, D)

        out = torch.zeros(B, self.D, device=device, dtype=kernel_vals.dtype)
        out.scatter_add_(0, b_idx.unsqueeze(1).expand_as(kernel_vals), kernel_vals)
        return out  # (B, D)

    def event_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """Kernel intensity of the observed event type at each position.  Shape: ``(B, L)``.

        Efficient lower-triangular computation: for event i, sum kernel
        contributions from all events j < i, then pick the receiver type
        matching event i.
        """
        B, N = ts.time_points.shape
        device = ts.time_points.device

        times = ts.time_points  # (B, N)
        mask = ts.event_mask  # (B, N)

        # Pairwise time deltas: dt_mat[b, i, j] = t_i - t_j
        dt_mat = times.unsqueeze(2) - times.unsqueeze(1)  # (B, N, N) — [b, i, j]

        # Both events real, and j strictly before i in time (strict-time causality).
        mask_i = mask.unsqueeze(2)  # (B, N, 1)
        mask_j = mask.unsqueeze(1)  # (B, 1, N)
        valid = mask_i & mask_j & (dt_mat > 0)  # (B, N, N) — [b, i, j]

        b_idx, i_idx, j_idx = torch.where(valid)
        if b_idx.numel() == 0:
            return torch.zeros(B, N, device=device, dtype=times.dtype)

        dt_valid = dt_mat[b_idx, i_idx, j_idx]  # (M,)
        types_j = ts.event_types_idx_safe[b_idx, j_idx]  # (M,) trigger types
        types_i = ts.event_types_idx_safe[b_idx, i_idx]  # (M,) receiver types

        kernel_vals = self._kernel_values(dt_valid, types_j)  # (M, D)

        # Pick the receiver-type column for each pair
        recv_vals = kernel_vals.gather(1, types_i.unsqueeze(1)).squeeze(1)  # (M,)

        out = torch.zeros(B, N, device=device, dtype=recv_vals.dtype)
        recv_flat = b_idx * N + i_idx
        out.view(-1).scatter_add_(0, recv_flat, recv_vals)

        return out  # (B, N)
