from __future__ import annotations

from dataclasses import dataclass
from tensordict import TensorDict
from torch import Tensor
import math
import warnings

import torch

import numpy as np
from typing import Optional, Dict, List, Union
from collections.abc import Sequence as ABCSequence


@dataclass(frozen=True)
class TokenRoles:
    """How time 0 and each event type are treated (see ``docs/notation.md``, Token roles).

    Args:
        has_origin: Time 0 is meaningful (e.g. admission or birth): every event is scored,
            including the wait until the first one. If False, the first event (and events tied
            with it) is conditioned on, not scored.
        context_types: Types that are read as history but never predicted or scored.
        ignored_types: Types excluded from the evaluation targets. They stay in the history and
            are still scored by the training likelihood.
    """

    has_origin: bool = False
    context_types: tuple[int, ...] = ()
    ignored_types: tuple[int, ...] = ()

    def __post_init__(self):
        for name in ("context_types", "ignored_types"):
            value = getattr(self, name) or ()
            if isinstance(value, Tensor):
                value = value.tolist()
            types = tuple(sorted({int(v) for v in value}))
            if types and types[0] < 0:
                raise ValueError(f"{name} must be non-negative type indices, got {list(types)}.")
            object.__setattr__(self, name, types)
        object.__setattr__(self, "has_origin", bool(self.has_origin))

    def predictable_mask(self, D: int) -> Tensor:
        """``(D,)`` bool mask, True for types that are predicted and scored."""
        if self.context_types and self.context_types[-1] >= D:
            raise ValueError(f"context_types {list(self.context_types)} out of range for D={D}.")
        mask = torch.ones(D, dtype=torch.bool)
        mask[list(self.context_types)] = False
        return mask

    def replace(self, **changes) -> "TokenRoles":
        """A copy with the given fields changed."""
        from dataclasses import replace

        return replace(self, **changes)


@dataclass
class DatasetStats:
    """Empirical statistics computed from a set of training sequences.

    Attributes:
        D: Number of event types.

        mean_first_occ_time: Shape ``(D,)``.

            *Terminating* — censored survival mean time to first occurrence.
            Under a Poisson assumption, ``mu_d ≈ 1 / mean_first_occ_time[d]``.

            *Recurring* — ``total_observation_time / total_occurrences`` per
            type (= mean inter-arrival time).

        mean_rate: ``1 / mean_first_occ_time``, shape ``(D,)``.
            Use for baseline initialisation.

        mean_inter_event_time: Scalar mean gap between any two consecutive
            events (any type) within a sequence.

        type_distribution: Shape ``(D,)``, normalised occurrence frequency of
            each type (sums to 1).

        mean_events_per_sequence: Average number of events per sequence.

        mean_sequence_time: Average total time-span of a sequence
            (``t_last_event`` or ``T_obs`` if provided).

        sequence_time_p0:   Minimum (0th percentile) of per-sequence observation windows.
        sequence_time_p50:  Median (50th percentile) of per-sequence observation windows.
        sequence_time_p75:  75th percentile of per-sequence observation windows.
        sequence_time_p90:  90th percentile of per-sequence observation windows.
        sequence_time_p95:  95th percentile of per-sequence observation windows.
            This is the recommended anchor for baseline / neural-spline time extent:
            it covers almost all sequences without being dominated by outliers.
        sequence_time_p100: Maximum (100th percentile) of per-sequence observation windows.

        inter_event_time_p95: 95th percentile of positive inter-event gaps within
            sequences.  Recommended anchor for spline-kernel time extent.

        is_terminating: Whether the data is from a terminating process.
    """

    D: int
    mean_first_occ_time: Tensor  # (D,)
    mean_rate: Tensor  # (D,)  = 1 / mean_first_occ_time
    mean_inter_event_time: float  # global scalar
    type_distribution: Tensor  # (D,)  normalised
    mean_events_per_sequence: float
    mean_sequence_time: float
    # Percentiles of per-sequence observation windows (effective censoring times)
    sequence_time_p0: float = 0.0
    sequence_time_p50: float = 0.0
    sequence_time_p75: float = 0.0
    sequence_time_p90: float = 0.0
    sequence_time_p95: float = 0.0
    sequence_time_p100: float = 0.0
    # 95th percentile of positive inter-event gaps across all sequences
    inter_event_time_p95: float = 0.0
    is_terminating: bool = True

    def rescale_time(self, factor: float) -> "DatasetStats":
        """Return a copy with time-dimensioned fields multiplied by ``factor``.

        Used by the model factory to convert raw-unit statistics into
        the model's internal *scaled* units (``factor = 1/s``): time-like fields
        (first-occurrence time, sequence-window percentiles, mean inter-event time)
        scale by ``factor``; rates scale by ``1/factor``; dimensionless fields
        (``type_distribution``, counts) are unchanged.  Data-driven parameter
        initialisation then happens in scaled units, matching the scaled core.
        """
        from dataclasses import replace as _dc_replace

        return _dc_replace(
            self,
            mean_first_occ_time=self.mean_first_occ_time * factor,
            mean_rate=self.mean_rate / factor,
            mean_inter_event_time=self.mean_inter_event_time * factor,
            mean_sequence_time=self.mean_sequence_time * factor,
            sequence_time_p0=self.sequence_time_p0 * factor,
            sequence_time_p50=self.sequence_time_p50 * factor,
            sequence_time_p75=self.sequence_time_p75 * factor,
            sequence_time_p90=self.sequence_time_p90 * factor,
            sequence_time_p95=self.sequence_time_p95 * factor,
            sequence_time_p100=self.sequence_time_p100 * factor,
            inter_event_time_p95=self.inter_event_time_p95 * factor,
        )

    @classmethod
    def from_sequences(
        cls,
        sequences: "Union[List, ABCSequence]",
        D: int,
        T_obs: "Optional[float | List[float]]" = None,
        terminating: bool = False,
        device: "Optional[torch.device | str]" = None,
        context_types: "Optional[List[int]]" = None,
    ) -> "DatasetStats":
        """Compute dataset statistics from a collection of sequences.

        ``sequences`` can be any object that supports ``len()`` and integer
        indexing (e.g. a plain ``list``, a ``torch.utils.data.Dataset``, or a
        :class:`~torchpp.data.base_dataset.TPPDataset`).  Each element must be an
        :class:`MVEventData` instance.

        Args:
            sequences: Collection of :class:`MVEventData` objects.
            D: Number of event types.
            T_obs: Observation end-time(s).  ``None`` uses each sequence's own
                ``T_obs`` attribute.  A single ``float`` is applied to every
                sequence.  A ``List[float]`` gives per-sequence windows.
            terminating: Whether the data comes from a terminating process.
            device: Target device for returned tensors.  Defaults to CPU.
            context_types: Event-type indices that are context/read-only (e.g. a
                start-of-record marker).  Events of these types are excluded from
                every event-based statistic (inter-event gaps → ``mean_inter_event_time``
                → the model's ``time_scale``, type counts, first-occurrence times), so
                adding context events does not perturb the scale.  ``None``/empty
                is the default and computes over all types (numerically inert).
        """
        N = len(sequences)
        dev = torch.device(device) if device is not None else torch.device("cpu")

        # Predictable-type mask: context events must
        # not enter any event-based statistic, else they would shift ``time_scale``.
        predictable_type: "Optional[Tensor]" = None
        if context_types:
            predictable_type = torch.ones(D, dtype=torch.bool, device=dev)
            ctx = [c for c in context_types if 0 <= c < D]
            if ctx:
                predictable_type[ctx] = False

        # Resolve per-sequence T_obs via index access so plain lists and
        # torch.utils.data.Dataset objects work uniformly.
        if T_obs is None:
            T_obs_list: List[Optional[float]] = [getattr(sequences[i], "T_obs", None) for i in range(N)]
        elif isinstance(T_obs, (int, float)):
            T_obs_list = [float(T_obs)] * N
        else:
            T_obs_list = [float(v) for v in T_obs]
            assert len(T_obs_list) == N

        if N == 0:
            z = torch.zeros(D, dtype=torch.float32, device=dev)
            return cls(
                D=D,
                mean_first_occ_time=z.clamp(min=1e-9),
                mean_rate=z.clamp(min=1e-9),
                mean_inter_event_time=1.0,
                type_distribution=z,
                mean_events_per_sequence=0.0,
                mean_sequence_time=0.0,
                sequence_time_p0=0.0,
                sequence_time_p50=0.0,
                sequence_time_p75=0.0,
                sequence_time_p90=0.0,
                sequence_time_p95=0.0,
                sequence_time_p100=0.0,
                inter_event_time_p95=1.0,
                is_terminating=terminating,
            )

        # ------------------------------------------------------------------
        # Single pass over sequences — all marginal statistics
        # ------------------------------------------------------------------
        # Terminating: t_first[n, d] = first occurrence time (inf if never)
        # Recurring:   type_counts[d] = total occurrences across all sequences
        t_first = torch.full((N, D), float("inf"), dtype=torch.float64, device=dev)  # terminating
        type_counts = torch.zeros(D, dtype=torch.float64, device=dev)  # both
        T_obs_vec = torch.zeros(N, dtype=torch.float64, device=dev)
        inter_event_sum = 0.0
        inter_event_count = 0
        total_event_count = 0
        total_seq_time = 0.0
        all_gaps: List[float] = []  # positive inter-event gaps for percentile computation

        for seq_idx in range(N):
            seq = sequences[seq_idx]
            times = seq.time_points
            types = seq.event_types

            seq_T_obs = T_obs_list[seq_idx]
            if len(times) == 0:
                T_obs_vec[seq_idx] = seq_T_obs if seq_T_obs is not None else 0.0
                continue

            last_t = float(times[-1])
            eff_T_obs = max(float(seq_T_obs), last_t) if seq_T_obs is not None else last_t
            T_obs_vec[seq_idx] = eff_T_obs
            total_seq_time += eff_T_obs

            valid = (types >= 0) & (types < D)
            v_types = types[valid].long().to(dev)
            v_times = times[valid].double().to(dev)
            if predictable_type is not None:
                keep = predictable_type[v_types]
                v_types = v_types[keep]
                v_times = v_times[keep]
            L = v_types.shape[0]

            if L > 0:
                total_event_count += L
                type_counts.scatter_add_(0, v_types, torch.ones(L, dtype=torch.float64, device=dev))
                # First-occurrence times (only used in the terminating branch below).
                t_first[seq_idx].scatter_reduce_(0, v_types, v_times, reduce="amin", include_self=True)

            if L > 1:
                t_sorted = v_times.sort().values
                gaps = t_sorted[1:] - t_sorted[:-1]
                pos = gaps[gaps > 0]
                inter_event_sum += pos.sum().item()
                inter_event_count += pos.numel()
                all_gaps.extend(pos.tolist())

        mean_inter_event_time = inter_event_sum / inter_event_count if inter_event_count > 0 else 1.0
        mean_events_per_sequence = total_event_count / N
        mean_sequence_time = total_seq_time / N

        # Percentiles of per-sequence observation windows (effective censoring times).
        seq_times_np = T_obs_vec.cpu().numpy()
        pct_seq = np.percentile(seq_times_np, [0, 50, 75, 90, 95, 100]) if N > 0 else np.zeros(6)
        seq_p0, seq_p50, seq_p75, seq_p90, seq_p95, seq_p100 = (float(v) for v in pct_seq)

        if all_gaps:
            inter_event_time_p95 = float(np.percentile(all_gaps, 95))
        else:
            inter_event_time_p95 = mean_inter_event_time

        type_distribution = (type_counts / type_counts.sum().clamp(min=1.0)).float()

        # ------------------------------------------------------------------
        # mean_first_occ_time / mean_rate
        # ------------------------------------------------------------------
        if terminating:
            # Censored survival mean: types that never occurred contribute T_obs.
            occurred = t_first.isfinite()  # (N, D)
            t_cens = torch.where(occurred, t_first, T_obs_vec.unsqueeze(1))  # (N, D)
            mean_first_occ_time = (t_cens.sum(dim=0) / N).float().clamp(min=1e-9)
        else:
            # Mean inter-arrival per type = total_obs_time / count.
            total_obs_time = T_obs_vec.sum()  # scalar
            mean_first_occ_time = (
                torch.where(
                    type_counts > 0,
                    (total_obs_time / type_counts.clamp(min=1.0)),
                    torch.tensor(total_obs_time.item() / max(type_counts.sum().item(), 1), device=dev),
                )
                .float()
                .clamp(min=1e-9)
            )

        mean_rate = 1.0 / mean_first_occ_time

        return cls(
            D=D,
            mean_first_occ_time=mean_first_occ_time,
            mean_rate=mean_rate,
            mean_inter_event_time=float(mean_inter_event_time),
            type_distribution=type_distribution,
            mean_events_per_sequence=float(mean_events_per_sequence),
            mean_sequence_time=float(mean_sequence_time),
            sequence_time_p0=seq_p0,
            sequence_time_p50=seq_p50,
            sequence_time_p75=seq_p75,
            sequence_time_p90=seq_p90,
            sequence_time_p95=seq_p95,
            sequence_time_p100=seq_p100,
            inter_event_time_p95=inter_event_time_p95,
            is_terminating=terminating,
        )


def _as_times(x, name: str = "time_points") -> Tensor:
    """Convert event times to a 1-D floating tensor, or raise a clear error."""
    if not isinstance(x, Tensor):
        x = torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.get_default_dtype())
    if x.dim() != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(x.shape)}.")
    if len(x) > 0 and not x.dtype.is_floating_point:
        raise TypeError(f"{name} must be a floating-point tensor, got {x.dtype}; use .float() or .double().")
    return x


def _as_types(x, name: str = "event_types") -> Tensor:
    """Convert event types to a 1-D integer tensor, or raise a clear error."""
    if not isinstance(x, Tensor):
        arr = np.asarray(x)
        if arr.size and not np.issubdtype(arr.dtype, np.integer):
            raise TypeError(f"{name} must be integers, got {arr.dtype}.")
        x = torch.as_tensor(arr.astype(np.int64) if arr.size else np.zeros(0, dtype=np.int64))
    if x.dim() != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(x.shape)}.")
    if len(x) > 0 and (x.dtype.is_floating_point or x.dtype == torch.bool):
        raise TypeError(f"{name} must be an integer tensor, got {x.dtype}; use .long().")
    return x


def _check_sequence(
    time_points: Tensor, event_types: Tensor, sort: bool, allow_padding: bool = False
) -> tuple[Tensor, Tensor]:
    """Validate one sequence (length match, finite times, non-negative types, non-decreasing times).

    ``allow_padding`` accepts trailing padding entries (batch rows only).
    """
    if time_points.shape != event_types.shape:
        raise ValueError(
            f"time_points and event_types must have the same length, got {len(time_points)} and {len(event_types)}."
        )
    if len(time_points) > 0 and not bool(torch.isfinite(time_points).all()):
        raise ValueError("time_points must be finite (got NaN or inf).")
    if len(event_types) > 0 and not allow_padding and bool((event_types < 0).any()):
        raise ValueError("event_types must be non-negative; padding exists only inside BatchedMVEventData.")
    if len(event_types) > 0 and bool((event_types < 0).any()):
        pad = event_types == EventSequence.EVENT_PADDING_VALUE
        n_real = int((~pad).sum())
        if bool((event_types < EventSequence.EVENT_PADDING_VALUE).any()) or bool(pad[:n_real].any()):
            raise ValueError(
                f"event_types must be non-negative (only trailing padding {EventSequence.EVENT_PADDING_VALUE} "
                "is allowed)."
            )
    if sort and len(time_points) > 1:
        order = torch.argsort(time_points, stable=True)
        time_points, event_types = time_points[order], event_types[order]
    elif len(time_points) > 1 and bool((time_points[1:] < time_points[:-1]).any()):
        raise ValueError("time_points must be non-decreasing; pass sort=True to sort the events.")
    return time_points, event_types


def extend_T_obs(T_obs: Optional[float], time_points: Tensor) -> Optional[float]:
    """``T_obs`` for a sequence extended by new events: ``max(T_obs, last event time)``."""
    if T_obs is None or len(time_points) == 0:
        return None if T_obs is None else float(T_obs)
    return max(float(T_obs), float(time_points[-1]))


def _clamped_T_obs(T_obs: Optional[float], last_time: Optional[float]) -> Optional[float]:
    """``max(T_obs, last event time)``; warns when the given ``T_obs`` lies before the last event."""
    if T_obs is None or last_time is None:
        return T_obs
    T_obs = float(T_obs)
    if T_obs < last_time and not math.isclose(T_obs, last_time, rel_tol=1e-6, abs_tol=1e-12):
        warnings.warn(
            f"T_obs={T_obs} lies before the last event ({last_time}); using the last event time.", stacklevel=3
        )
    return max(T_obs, last_time)


class EventSequence(TensorDict):
    """Shared base for the single- and batched-sequence wrappers.

    Holds ``{time_points, event_types}`` and every method that is agnostic to the
    batch rank: each reduces over the LAST (event) axis, so the same code is correct
    for a single sequence (shape ``(L,)``) and a padded batch (shape ``(B, L)``).
    :class:`MVEventData` and :class:`BatchedMVEventData` add their rank-specific
    construction and helpers (padding, ``seq_lengths``, query-time methods).
    """

    EVENT_PADDING_VALUE = -1

    @property
    def time_points(self):
        return self["time_points"]

    @property
    def event_types(self):
        return self["event_types"]

    @property
    def event_mask(self):
        """Boolean mask indicating which entries are real events (same shape as ``event_types``)."""
        return self.event_types != self.EVENT_PADDING_VALUE

    @property
    def event_types_idx_safe(self) -> torch.Tensor:
        """Event types with padding (-1) replaced by 0 for safe indexing.

        Padding positions should always be excluded via ``event_mask`` before
        using these indices for any meaningful computation.

        Returns:
            Same shape as ``event_types``, dtype int — values in [0, ...]
        """
        return self.event_types.clamp(min=0)

    @property
    def real_counts(self) -> torch.Tensor:
        """Number of real (non-padding) events, reduced over the event axis.

        Returns:
            Shape: (...,), dtype long — ``(B,)`` for a batch, scalar for a single sequence.
        """
        return self.event_mask.sum(dim=-1).long()

    @property
    def causal_indices(self) -> torch.Tensor:
        """For each event position l, the number of real events strictly before it in time.

        This can be used as a column index into a ``(..., L+1, ...)`` parameter
        tensor where column 0 is the no-history slot (a zeros dummy) and column k
        holds the parameters after k real events have been observed.  Simultaneous
        events share the same column (identical causal context).

        Returns:
            Same shape as ``event_types``, dtype long — values in [0, L]
        """
        tp = self.time_points
        # strictly_before[..., q, k] = True iff tp[..., k] < tp[..., q] AND event_mask[..., k]
        strictly_before = (tp.unsqueeze(-2) < tp.unsqueeze(-1)) & self.event_mask.unsqueeze(-2)
        return strictly_before.sum(dim=-1).long()

    def one_hot_event_types(self, num_classes: int) -> torch.Tensor:
        """One-hot encoding of event types, with padding positions zeroed out.

        Args:
            num_classes: Number of event type classes D.

        Returns:
            Shape: (..., D), dtype bool — padding positions are all-False
        """
        clamped = self.event_types_idx_safe  # padding replaced by 0
        one_hot = torch.nn.functional.one_hot(clamped, num_classes=num_classes).bool()  # (..., D)
        return one_hot & self.event_mask.unsqueeze(-1)  # zero out padding rows

    def causal_mask(self, mask_padding=True):
        """Return a strict causal mask (..., L, L): True if time[q] > time[k].

        Used for intensity queries: position q may only attend to events
        strictly *before* it in time.
        """
        causal_matrix = self.time_points.unsqueeze(-1) > self.time_points.unsqueeze(-2)  # (..., L, L)

        if mask_padding:
            event_mask = self.event_mask  # (..., L)
            padding_mask = event_mask.unsqueeze(-1) & event_mask.unsqueeze(-2)  # (..., L, L)
            causal_matrix = causal_matrix & padding_mask

        return causal_matrix

    def attention_mask(self, mask_padding=True):
        """Return an inclusive causal mask (..., L, L): True if time[q] >= time[k].

        Used by the Transformer during training/encoding: position q may attend
        to all events at the *same* time or earlier.  Multiple simultaneous
        events therefore see each other, but not future events.
        """
        causal_matrix = self.time_points.unsqueeze(-1) >= self.time_points.unsqueeze(-2)  # (..., L, L)

        if mask_padding:
            event_mask = self.event_mask  # (..., L)
            padding_mask = event_mask.unsqueeze(-1) & event_mask.unsqueeze(-2)  # (..., L, L)
            causal_matrix = causal_matrix & padding_mask

        return causal_matrix

    def bos_augmented_times_and_mask(self):
        """Times + inclusive attention mask for a ``<bos>``-prefixed ``(L+1)`` sequence.

        Returns ``(aug_times, aug_mask)``:
          - ``aug_times`` ``(..., L+1)`` — position 0 is a synthetic ``<bos>`` time placed
            strictly before the first real event (:func:`bos_time`), positions 1..L are
            the real event times.
          - ``aug_mask`` ``(..., L+1, L+1)`` — the standard inclusive causal mask
            (``time[q] >= time[k]``, padding excluded) over the augmented sequence.

        With the strict-before placement, this reuses ``attention_mask``'s logic so
        ``<bos>`` attends only to itself and every real event attends to ``<bos>`` plus its
        own history.  Times only — no ``D+1``-typed batch is built; the encoders supply the
        learned ``<bos>`` embedding.
        """
        tp = self.time_points  # (..., L)
        mask = self.event_mask  # (..., L)
        t_bos = bos_time(tp, mask).unsqueeze(-1).to(tp.dtype)  # (..., 1)
        aug_times = torch.cat([t_bos, tp], dim=-1)  # (..., L+1)
        # <bos> is always a real token — trailing dim 1 even when L == 0.
        bos_real = torch.ones(mask.shape[:-1] + (1,), dtype=torch.bool, device=tp.device)
        aug_real = torch.cat([bos_real, mask], dim=-1)  # (..., L+1)
        causal = aug_times.unsqueeze(-1) >= aug_times.unsqueeze(-2)  # (..., L+1, L+1): time[q] >= time[k]
        padding = aug_real.unsqueeze(-1) & aug_real.unsqueeze(-2)
        return aug_times, causal & padding

    @property
    def last_event_time(self):
        """Time of the last real event, reduced over the event axis.

        Masked max over positions where ``event_mask`` is True; a sequence with no
        real event yields ``0.0``.  Shape ``(...)`` — ``(B,)`` for a batch (empty
        batch → shape ``(0,)``), scalar for a single sequence.
        """
        tp = self.time_points
        # (B, 0) / (0, 0) / empty single sequence: nothing to reduce.
        if tp.shape[-1] == 0:
            return torch.zeros(tp.shape[:-1], dtype=tp.dtype, device=tp.device)

        mask = self.event_mask
        neg_inf = torch.full((), float("-inf"), dtype=tp.dtype, device=tp.device)
        masked = torch.where(mask, tp, neg_inf)
        result = masked.max(dim=-1).values  # -inf where a sequence has no real event
        has_event = mask.any(dim=-1)
        return torch.where(has_event, result, torch.zeros_like(result))

    def rescale_time(self, factor: float) -> "EventSequence":
        """Return a copy with every time-dimensioned field multiplied by ``factor``.

        Clones (preserving every entry, including a batched ``T_obs``/``covariates``)
        and scales ``time_points``; ``T_obs`` is scaled too, whether it is a tensordict
        entry (batched) or an instance attribute (single sequence).  Event types and
        covariates (static features) are shared unchanged.  ``factor == 1.0`` returns
        ``self`` unchanged (zero-copy), the model's default ``time_scale`` fast path.
        """
        if factor == 1.0:
            return self
        result = self.clone()
        result["time_points"] = self["time_points"] * factor
        if "T_obs" in result.keys():
            # Batched: T_obs is a tensordict entry preserved by clone — scale in place.
            result["T_obs"] = result["T_obs"] * factor
        else:
            # Single sequence: T_obs/covariates are instance attributes (dropped by clone).
            # Scaled in the data dtype, like the event times, so T_obs == t_last survives.
            T_obs = getattr(self, "T_obs", None)
            result.T_obs = (
                float(torch.tensor(T_obs, dtype=self["time_points"].dtype) * factor) if T_obs is not None else None
            )
            result.covariates = getattr(self, "covariates", None)
        return result


class MVEventData(EventSequence):
    """One event sequence: event times, their types and the end of observation.

    Args:
        time_points: Event times, shape ``(L,)``, non-decreasing, in native units.
        event_types: Event types, shape ``(L,)``, integers in ``0 … D−1``.
        covariates: Optional static covariates, ``{name: tensor}``. Carried with the sequence;
            the built-in models do not use them.
        sort: Sort the events by time instead of raising on decreasing times.
        T_obs: End of the observation window; ``None`` means the last event time. A later value
            means no event was observed after the last one; an earlier value is raised to the last
            event time with a warning.

    Lists and NumPy arrays are converted to tensors.
    """

    def __init__(
        self,
        time_points: Tensor,
        event_types: Tensor,
        covariates: Optional[Dict[str, Tensor]] = None,
        sort: bool = False,
        T_obs: Optional[float] = None,
    ):
        time_points = _as_times(time_points)
        event_types = _as_types(event_types).to(time_points.device)
        time_points, event_types = _check_sequence(time_points, event_types, sort)

        if covariates is not None:
            if not isinstance(covariates, dict):
                raise TypeError("covariates must be a dict of 1-D tensors.")
            for key, value in covariates.items():
                if not isinstance(value, torch.Tensor) or value.dim() != 1:
                    raise ValueError(f"Covariate '{key}' must be a 1-D tensor.")

        self.covariates = covariates
        # T_obs is at least the last event time.
        self.T_obs = _clamped_T_obs(T_obs, float(time_points[-1]) if len(time_points) > 0 else None)

        super().__init__({"time_points": time_points, "event_types": event_types}, batch_size=len(time_points))

    def _carry_attributes(self, out: "MVEventData") -> "MVEventData":
        # TensorDict builds the result without __init__, so the per-sequence attributes are copied here.
        if out is not self:
            out.T_obs = getattr(self, "T_obs", None)
            cov = getattr(self, "covariates", None)
            out.covariates = None if cov is None else {k: v.to(out.time_points.device) for k, v in cov.items()}
        return out

    def to(self, *args, **kwargs) -> "MVEventData":
        return self._carry_attributes(super().to(*args, **kwargs))

    def clone(self, *args, **kwargs) -> "MVEventData":
        return self._carry_attributes(super().clone(*args, **kwargs))

    def __getitem__(self, idx):
        # Integer indexing returns a (time, type) tuple of Python scalars;
        # slicing falls through to TensorDict and stays an MVEventData.
        if isinstance(idx, int):
            return (self["time_points"][idx], self["event_types"][idx])
        else:
            return super().__getitem__(idx)

    def __iter__(self):
        for i in range(len(self)):
            yield self.time_points[i], self.event_types[i]

    def __repr__(self):
        def abbrev(tensor, max_len=8, float_fmt="{:.2f}"):
            l = tensor.tolist()
            if tensor.dtype.is_floating_point:
                l = [float_fmt.format(v) for v in l]
            else:
                l = [str(v) for v in l]
            if len(l) > max_len:
                return "[" + ", ".join(l[:4]) + ", ..., " + ", ".join(l[-3:]) + "]"
            return "[" + ", ".join(l) + "]"

        if self.time_points.dim() == 1:
            length = self.shape[0]
        else:
            length = 0
        tps = abbrev(self.time_points)
        ets = abbrev(self.event_types)

        result = (
            f"MVEventData(len={length}, [{self.time_points.dtype}, {self.event_types.dtype}])\n"
            f"  time_points: {tps}\n"
            f"  event_types: {ets}"
        )

        if self.covariates is not None:
            cov_str = ", ".join([f"{k}: {abbrev(v)}" for k, v in self.covariates.items()])
            result += f"\n  covariates: {{{cov_str}}}"

        if self.T_obs is not None:
            result += f"\n  T_obs: {self.T_obs:.4f}"

        return result

    def to_batch(self) -> "BatchedMVEventData":
        """Collate this single sequence into a ``batch_size=(1,)`` batch."""
        return BatchedMVEventData.from_sequences([self])


class BatchedMVEventData(EventSequence):
    """A padded batch of event sequences, shape ``(B, L)`` with ``L`` the longest sequence.

    Build it from a list of :class:`MVEventData` (``mv_events=``) or from per-sequence lists of
    ``time_points`` / ``event_types`` (and ``T_obs``). Padding positions have event type ``-1``
    and a time after every real event of the batch; ``real_counts`` holds the true lengths.
    Every model method takes a batch, or a single :class:`MVEventData`.
    """

    PADDING_TIME_OFFSET = 1.0

    def __init__(
        self,
        time_points: Optional[list[Tensor]] = None,
        event_types: Optional[list[Tensor]] = None,
        covariates: Optional[Dict[str, list[Tensor]]] = None,
        mv_events: Optional[list[MVEventData]] = None,
        sort: bool = False,
        T_obs: Optional[list[Optional[float]]] = None,
    ):
        """
        Initialize a batch of multivariate event sequences.

        Args:
            time_points: List of time tensors, one per sequence
            event_types: List of event type tensors, one per sequence
            covariates: Optional dict where keys are covariate names and values are lists of tensors.
                       After batching, becomes dict with tensors of shape (B, feature_dim).
                       Example: {"sex": [tensor([1.0]), tensor([0.0]), ...]} -> {"sex": tensor([[1.0], [0.0], ...])}
            mv_events: Alternative: provide list of MVEventData objects
            sort: Whether to sort sequences by time
            T_obs: Optional per-sequence observation end-times.  ``None`` (default)
                uses the last event time of each sequence as a proxy.
                When constructing from ``mv_events``, the ``T_obs`` stored on
                each :class:`MVEventData` object is used automatically unless
                this argument is explicitly provided (explicit wins).
        """
        if mv_events is None and (time_points is None or event_types is None):
            raise ValueError("Pass either mv_events or both time_points and event_types.")

        if mv_events is not None:
            if not isinstance(mv_events, (list, tuple)) or not all(isinstance(m, MVEventData) for m in mv_events):
                raise TypeError("mv_events must be a list of MVEventData.")
            mv_events = list(mv_events)
            sort = False
            time_points = [ts.time_points for ts in mv_events]
            event_types = [ts.event_types for ts in mv_events]

            # Caller-supplied T_obs takes precedence over the per-sequence value.
            if T_obs is None:
                T_obs = [seq.T_obs for seq in mv_events]  # list of float|None

            # Robust to mixed batches: enable covariate extraction if ANY sequence
            # carries them.  Every covariate-bearing sequence must expose the SAME key
            # set, else a mismatching sequence would silently drop or gain keys.
            has_covariates = any(seq.covariates is not None for seq in mv_events)
            if has_covariates:
                first_with_cov = next(seq for seq in mv_events if seq.covariates is not None)
                cov_keys = list(first_with_cov.covariates.keys())
                key_set = set(cov_keys)

                for seq in mv_events:
                    if seq.covariates is None or set(seq.covariates.keys()) != key_set:
                        raise ValueError(
                            f"All sequences must carry the same covariate keys {sorted(key_set)}; "
                            f"found a sequence with keys "
                            f"{sorted(seq.covariates.keys()) if seq.covariates is not None else None}."
                        )

                covariates = {key: [seq.covariates[key] for seq in mv_events] for key in cov_keys}
            else:
                covariates = None

            batch_size = len(mv_events)

        else:
            if not isinstance(time_points, (list, tuple)) or not isinstance(event_types, (list, tuple)):
                raise TypeError("time_points and event_types must be lists with one entry per sequence.")
            if len(time_points) != len(event_types):
                raise ValueError(
                    f"time_points and event_types must have one entry per sequence, "
                    f"got {len(time_points)} and {len(event_types)}."
                )
            batch_size = len(time_points)
            checked = [
                _check_sequence(_as_times(t), _as_types(e).to(_as_times(t).device), sort, allow_padding=True)
                for t, e in zip(time_points, event_types)
            ]
            time_points = [t for t, _ in checked]
            event_types = [e for _, e in checked]

        lengths = [len(b) for b in time_points]
        length = max(lengths) if lengths else 0

        if batch_size > 0:
            device = time_points[0].device
            dtype_time = time_points[0].dtype
            dtype_event = event_types[0].dtype
        else:
            device = torch.device("cpu")
            dtype_time = torch.float32
            dtype_event = torch.long

        if batch_size == 0:
            padded_time_points = torch.empty((0, 0), dtype=dtype_time, device=device)
            padded_event_types = torch.empty((0, 0), dtype=dtype_event, device=device)
        else:
            # Pad time_points with (max last-event time + PADDING_TIME_OFFSET) so that
            # padded entries always sort after every real event; pad event_types
            # with -1 (EVENT_PADDING_VALUE) which is what event_mask checks for.
            non_empty_last_times = [tp[-1] for tp in time_points if len(tp) > 0]

            if non_empty_last_times:
                pad_time = torch.tensor(non_empty_last_times).max() + self.PADDING_TIME_OFFSET
            else:
                pad_time = torch.tensor(0, dtype=dtype_time, device=device)

            padded_time_points = torch.full((batch_size, length), pad_time, dtype=dtype_time, device=device)
            padded_event_types = torch.full(
                (batch_size, length), self.EVENT_PADDING_VALUE, dtype=dtype_event, device=device
            )

            for i, (t, e) in enumerate(zip(time_points, event_types)):
                n = len(t)
                padded_time_points[i, :n] = t
                padded_event_types[i, :n] = e

        if covariates is not None:
            if not isinstance(covariates, dict):
                raise TypeError("covariates must be a dict of per-sequence tensor lists.")
            stacked_covariates = {}
            for key, cov_list in covariates.items():
                if len(cov_list) != batch_size:
                    raise ValueError(f"Covariate '{key}' needs one entry per sequence.")
                if batch_size > 0:
                    stacked_covariates[key] = torch.stack(cov_list, dim=0)  # (B, feature_dim)
                else:
                    stacked_covariates[key] = torch.empty((0, cov_list[0].shape[0]) if len(cov_list) > 0 else (0, 0))
        else:
            stacked_covariates = None

        # batch_size=(B,): the (B, L) tensors are valid (L is a trailing feature dim), so
        # T_obs / covariates live as tensordict entries that clone / slice / index / .to
        # carry automatically.  seq_lengths is derived from the padding mask (see property).
        super().__init__(
            {"time_points": padded_time_points, "event_types": padded_event_types}, batch_size=(batch_size,)
        )

        # Build per-sequence T_obs tensor (B,), stored as an entry.
        # Falls back to self.last_event_time (last event time per sequence) for any None entry.
        last_event_times = self.last_event_time  # (B,), 0.0 for empty sequences
        if T_obs is None:
            self["T_obs"] = last_event_times
        else:
            if len(T_obs) != batch_size:
                raise ValueError(f"T_obs must have one entry per sequence, got {len(T_obs)} for {batch_size}.")
            last = last_event_times.tolist()
            T_obs = [
                v if v is None or len(time_points[i]) == 0 else _clamped_T_obs(v, float(last[i]))
                for i, v in enumerate(T_obs)
            ]
            T_obs_tensor = torch.tensor(
                [v if v is not None else 0.0 for v in T_obs],
                dtype=padded_time_points.dtype,
                device=padded_time_points.device,
            )
            # None entries fall back to last_event_times; provided entries are
            # clamped to be at least the last observed event time.
            none_mask = torch.tensor([v is None for v in T_obs], dtype=torch.bool, device=padded_time_points.device)
            self["T_obs"] = torch.where(none_mask, last_event_times, torch.maximum(T_obs_tensor, last_event_times))

        if stacked_covariates is not None:
            self["covariates"] = TensorDict(stacked_covariates, batch_size=(batch_size,))

    @property
    def T_obs(self) -> torch.Tensor:
        """Per-sequence observation end-time, shape ``(B,)`` (a tensordict entry)."""
        return self["T_obs"]

    @T_obs.setter
    def T_obs(self, value: torch.Tensor):
        self["T_obs"] = value

    @property
    def covariates(self):
        """Static per-sequence features as a nested tensordict (``(B, feat)`` per key),
        or ``None`` when the batch carries none."""
        return self["covariates"] if "covariates" in self.keys() else None

    @covariates.setter
    def covariates(self, value):
        if value is None:
            if "covariates" in self.keys():
                del self["covariates"]
        elif isinstance(value, TensorDict):
            self["covariates"] = value
        else:
            self["covariates"] = TensorDict(dict(value), batch_size=self.batch_size)

    @property
    def seq_lengths(self) -> list:
        """Per-sequence real-event counts, derived from the padding mask (read-only)."""
        return self.real_counts.tolist()

    def time_points_before(self, t: torch.Tensor) -> torch.Tensor:
        """Boolean mask of real events whose time is strictly before ``t``.

        Args:
            t: Query times. Shape: (B,)

        Returns:
            Shape: (B, L) — True where event is real AND time_point < t[b]
        """
        past = self.time_points < t.unsqueeze(1)  # (B, L)
        return self.event_mask & past

    def count_events_before(self, t: torch.Tensor) -> torch.Tensor:
        """Count of real events strictly before query time ``t`` for each batch.

        Args:
            t: Query times. Shape: (B,)

        Returns:
            Shape: (B,), dtype long
        """
        return self.time_points_before(t).sum(dim=1).long()

    def event_time_before(self, t: torch.Tensor) -> torch.Tensor:
        """Start time of the intensity interval containing each query time ``t``.

        The time of the last real event strictly before ``t[b]`` (masked max over
        the event axis), or 0.0 if no real event precedes ``t[b]``.

        Args:
            t: Query times. Shape: (B,)

        Returns:
            Shape: (B,)
        """
        dtype = self.time_points.dtype
        tp = self.time_points.to(dtype)  # (B, L)
        B, L = tp.shape
        if L == 0:
            return torch.zeros(B, dtype=dtype, device=tp.device)
        before = self.time_points_before(t.to(dtype))  # (B, L): real event AND time < t
        neg_inf = torch.full((), float("-inf"), dtype=dtype, device=tp.device)
        masked = torch.where(before, tp, neg_inf)  # (B, L)
        mx = masked.max(dim=1).values  # (B,)
        any_before = before.any(dim=1)  # (B,)
        return torch.where(any_before, mx, torch.zeros_like(mx))

    def __iter__(self):
        """Decollate lazily into single :class:`MVEventData` sequences.

        Yields one sequence per batch row with only its real events, each cloned so a
        decollated sequence never aliases the batch tensors.
        """
        for i in range(len(self)):
            n = int(self.real_counts[i]) if self.real_counts.ndim else int(self.real_counts)
            cov = (
                {k: self.covariates[k][i].clone() for k in self.covariates.keys()}
                if self.covariates is not None
                else None
            )
            T_obs = float(self.T_obs[i]) if self.T_obs.numel() > 0 else None
            yield MVEventData(
                self.time_points[i, :n].clone(),
                self.event_types[i, :n].clone(),
                covariates=cov,
                T_obs=T_obs,
            )

    def to_sequences(self) -> list[MVEventData]:
        """Decollate into a list of unpadded, cloned single sequences."""
        return list(self)

    @classmethod
    def from_sequences(cls, sequences: list[MVEventData]) -> "BatchedMVEventData":
        """Collate a list of single sequences into one padded batch."""
        return cls(mv_events=list(sequences))


def bos_time(time_points: Tensor, event_mask: Tensor) -> Tensor:
    """Synthetic ``<bos>`` time placed strictly before the first real event, per sequence.

    Reduces over the last (event) axis, so it works for a single sequence ``(L,)`` and
    a padded batch ``(B, L)``.  Returns ``0`` when the first real event is ``> 0``
    (robust — no float cancellation), else ``t_first - eps``.  A sequence with no real
    event returns ``0``.  Used by the ``<bos>``-augmenting encoders so the standard
    inclusive attention mask lets ``<bos>`` attend only to itself while every real event
    attends to it.
    """
    dtype, device = time_points.dtype, time_points.device
    if time_points.shape[-1] == 0:
        return torch.zeros(time_points.shape[:-1], dtype=dtype, device=device)
    inf = torch.full((), float("inf"), dtype=dtype, device=device)
    t_first = torch.where(event_mask, time_points, inf).min(dim=-1).values  # inf where no real event
    has_event = torch.isfinite(t_first)
    t_first = torch.where(has_event, t_first, torch.zeros_like(t_first))
    eps = 1e-6
    result = torch.where(t_first > 0, torch.zeros_like(t_first), t_first - eps)
    # No real event → 0 (the docstring contract; guards the -eps that t_first=0 would give).
    return torch.where(has_event, result, torch.zeros_like(result))


# =============================================================================
# Termination helpers (data-level)
# =============================================================================
#
# Termination — each event type occurs at most once per sequence — is a universal
# TPP concept, but its *computation* is a pure function of the event data (+ the
# type count D and the ``terminating`` flag).  These live here so any model family
# (intensity- or density-based) can reuse them; the model classes carry the config
# and delegate.  ``terminating=False`` short-circuits to the trivial answer.


def termination_times(ts: "BatchedMVEventData", D: int, terminating: bool) -> Tensor:
    """First-occurrence time per event type (``inf`` where a type never occurred).

    Shape ``(B, D)``.  For non-terminating processes all entries are ``inf``.
    """
    B = ts.time_points.shape[0]
    L = ts.time_points.shape[1]
    device = ts.time_points.device
    dtype = ts.time_points.dtype

    if not terminating or L == 0:
        return torch.full((B, D), float("inf"), dtype=dtype, device=device)

    result = torch.full((B, D), float("inf"), dtype=dtype, device=device)
    # Mask padding times to inf so they don't affect the min.
    times = torch.where(
        ts.event_mask, ts.time_points, torch.tensor(float("inf"), dtype=dtype, device=device)
    )  # (B, L)
    result.scatter_reduce_(
        dim=1, index=ts.event_types_idx_safe, src=times, reduce="amin", include_self=True
    )
    return result


def termination_mask(
    ts: "BatchedMVEventData", D: int, terminating: bool, t: Optional[Tensor] = None, full: bool = False
) -> Tensor:
    """Boolean ``(B, D)``: True where an event type has NOT occurred yet.

    ``t`` restricts to events strictly before those times; non-terminating
    processes return all-True.
    """
    if full and t is not None:
        raise ValueError("Cannot use full=True and provide t simultaneously.")

    if not terminating:
        B = ts.time_points.shape[0]
        return torch.ones(B, D, dtype=torch.bool, device=ts.time_points.device)

    valid_events = ts.time_points_before(t) if t is not None else ts.event_mask  # (B, L)
    one_hot = torch.nn.functional.one_hot(ts.event_types_idx_safe, num_classes=D).bool()  # (B, L, D)
    one_hot = one_hot & valid_events.unsqueeze(2)
    type_occurred = one_hot.any(dim=1)  # (B, D)
    return ~type_occurred


def terminated_query_mask(query_times: Tensor, ts: "BatchedMVEventData", D: int, terminating: bool) -> Tensor:
    """Per-query termination mask ``(B, Q, D)``: True where type d is still active at
    ``query_times[:, q]`` (first-occurrence time ``>=`` the query time)."""
    B, Q = query_times.shape
    if not terminating:
        return torch.ones(B, Q, D, dtype=torch.bool, device=query_times.device)
    t_term = termination_times(ts, D, terminating)  # (B, D), inf if never occurred
    return t_term.unsqueeze(1) >= query_times.unsqueeze(2)  # (B, Q, D)


def has_repeated_event_types(
    ts: "BatchedMVEventData", D: int, predictable_mask: Optional[Tensor] = None
) -> Tensor:
    """Boolean ``(B,)``: True where a sequence repeats a *predictable* event type.

    Under terminating (first-occurrence) semantics a repeated event type is an
    impossible sequence.  ``predictable_mask`` (``(D,)`` bool) scopes the check to the
    competing-risks types: a repeated context/``<bos>`` type is outside that set
    and must not count as a repeat.  ``None`` (default) checks all types — numerically
    inert when every type is predictable.
    """
    B = ts.time_points.shape[0]
    L = ts.time_points.shape[1]
    if L == 0:
        return torch.zeros(B, dtype=torch.bool, device=ts.time_points.device)
    type_counts = ts.one_hot_event_types(D).float().sum(dim=1)  # (B, D)
    repeated = type_counts > 1  # (B, D)
    if predictable_mask is not None:
        repeated = repeated & predictable_mask
    return repeated.any(dim=1)


def scored_event_mask(ts: "BatchedMVEventData", predictable_mask: Tensor, has_origin: bool) -> Tensor:
    """Boolean ``(B, L)``: the events whose density enters the likelihood.

    An event is scored if it is a real (non-padding) event of a predictable type.
    Without an origin (``has_origin=False``) the first event and every event tied with
    it are the conditioned-on start of the sequence and are not scored. Training,
    validation and evaluation all use this set.
    """
    scored = ts.event_mask & predictable_mask[ts.event_types_idx_safe]
    if scored.shape[1] > 0 and not has_origin:
        tp = ts.time_points
        scored = scored & (tp != tp[:, :1])
    return scored
