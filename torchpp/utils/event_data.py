from __future__ import annotations

from dataclasses import dataclass
from tensordict import TensorDict
from torch import Tensor
import torch

import numpy as np
from typing import Optional, Dict, List, Union
from collections.abc import Sequence as ABCSequence


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

        mean_log_inter_time: Mean of ``log`` of positive inter-event gaps.
        std_log_inter_time: Std (unbiased) of ``log`` of positive inter-event gaps.
            Together these standardise the inter-event time for the IntensityFree
            log-normal mixture (Shchur et al. 2020).  Default ``(0.0, 1.0)`` when
            the gaps are empty/degenerate (no affine transform applied).

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
    # Mean / unbiased std of log positive inter-event gaps (IntensityFree standardisation)
    mean_log_inter_time: float = 0.0
    std_log_inter_time: float = 1.0
    is_terminating: bool = True

    @classmethod
    def from_sequences(
        cls,
        sequences: "Union[List, ABCSequence]",
        D: int,
        T_obs: "Optional[float | List[float]]" = None,
        terminating: bool = True,
        device: "Optional[torch.device | str]" = None,
    ) -> "DatasetStats":
        """Compute dataset statistics from a collection of sequences.

        ``sequences`` can be any object that supports ``len()`` and integer
        indexing (e.g. a plain ``list``, a ``torch.utils.data.Dataset``, or a
        :class:`~torchpp.ukb_loading.UKBDataset`).  Each element must be an
        :class:`MVEventData` instance.

        Args:
            sequences: Collection of :class:`MVEventData` objects.
            D: Number of event types.
            T_obs: Observation end-time(s).  ``None`` uses each sequence's own
                ``T_obs`` attribute.  A single ``float`` is applied to every
                sequence.  A ``List[float]`` gives per-sequence windows.
            terminating: Whether the data comes from a terminating process.
            device: Target device for returned tensors.  Defaults to CPU.
        """
        N = len(sequences)
        dev = torch.device(device) if device is not None else torch.device("cpu")

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
                mean_log_inter_time=0.0,
                std_log_inter_time=1.0,
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

        # Standardisation stats for the IntensityFree log-normal mixture: mean/std
        # of log positive inter-event gaps (on the already-scaled gaps the model
        # sees).  Degenerate cases (no gaps, single gap → NaN std, or std <= 0)
        # fall back to the identity transform (0.0, 1.0).
        mean_log_inter_time, std_log_inter_time = 0.0, 1.0
        if all_gaps:
            logs = torch.log(torch.tensor(all_gaps, dtype=torch.float64))
            m = logs.mean().item()
            s = logs.std(unbiased=True).item()
            if np.isfinite(m) and np.isfinite(s) and s > 0:
                mean_log_inter_time, std_log_inter_time = float(m), float(s)

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
            mean_log_inter_time=mean_log_inter_time,
            std_log_inter_time=std_log_inter_time,
            is_terminating=terminating,
        )


class MVEventData(TensorDict):
    def __init__(
        self,
        time_points: Tensor,
        event_types: Tensor,
        covariates: Optional[Dict[str, Tensor]] = None,
        sort: bool = False,
        T_obs: Optional[float] = None,
    ):
        """
        Initialize a multivariate event sequence.

        Args:
            time_points: Event times, shape (L,)
            event_types: Event types, shape (L,)
            covariates: Optional dict of covariate tensors. Each value should be a 1-D tensor
                       representing static features for this sequence.
                       Example: {"sex": tensor([1.0]), "age": tensor([45.2])}
            sort: Whether to sort events by time
            T_obs: Observation end-time for this sequence.  ``None`` (default)
                means the last event time is used as a proxy.
        """
        assert time_points.dim() == 1
        assert event_types.dim() == 1
        assert time_points.shape == event_types.shape

        if len(time_points) > 0:
            assert time_points.dtype.is_floating_point
            assert not event_types.dtype.is_floating_point

        if sort:
            sorted_indices = torch.argsort(time_points)
            time_points = time_points[sorted_indices]
            event_types = event_types[sorted_indices]

        if covariates is not None:
            assert isinstance(covariates, dict), "covariates must be a dictionary"
            for key, value in covariates.items():
                assert isinstance(value, torch.Tensor), f"Covariate '{key}' must be a tensor"
                assert value.dim() == 1, f"Covariate '{key}' must be 1-D, got shape {value.shape}"

        self.covariates = covariates
        # Clamp: T_obs must be at least as large as the last event time.
        if T_obs is not None and len(time_points) > 0:
            T_obs = max(float(T_obs), float(time_points[-1]))
        self.T_obs = T_obs

        super().__init__({"time_points": time_points, "event_types": event_types}, batch_size=len(time_points))

    @property
    def time_points(self):
        return self["time_points"]

    @property
    def event_types(self):
        return self["event_types"]

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


class BatchedMVEventData(TensorDict):
    MAX_TIME_OFFSET = 1.0
    EVENT_PADDING_VALUE = -1

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
        assert (time_points is not None and event_types is not None) or (mv_events is not None)

        if mv_events is not None:
            assert isinstance(mv_events, list)
            sort = False
            time_points = [ts.time_points for ts in mv_events]
            event_types = [ts.event_types for ts in mv_events]

            # Caller-supplied T_obs takes precedence over the per-sequence value.
            if T_obs is None:
                T_obs = [seq.T_obs for seq in mv_events]  # list of float|None

            # Robust to mixed batches: enable covariate extraction if ANY sequence
            # carries them (missing keys on others raise below).
            has_covariates = any(seq.covariates is not None for seq in mv_events)
            if has_covariates:
                first_with_cov = next(seq for seq in mv_events if seq.covariates is not None)
                cov_keys = list(first_with_cov.covariates.keys())

                covariates = {}
                for key in cov_keys:
                    cov_list = []
                    for seq in mv_events:
                        if seq.covariates is None or key not in seq.covariates:
                            raise ValueError(
                                f"All sequences in batch must have the same covariates. "
                                f"Missing covariate '{key}' in one or more sequences."
                            )
                        cov_list.append(seq.covariates[key])
                    covariates[key] = cov_list
            else:
                covariates = None

            batch_size = len(mv_events)

        else:
            assert isinstance(time_points, list)
            assert isinstance(event_types, list)
            assert len(time_points) == len(event_types)

            batch_size = len(time_points)

            for t, e in zip(time_points, event_types):
                assert t.dim() == 1
                assert e.dim() == 1
                assert t.shape == e.shape

            if batch_size > 0:
                assert time_points[0].dtype.is_floating_point
                assert not event_types[0].dtype.is_floating_point

        if sort:
            for i in range(len(time_points)):
                sorted_indices = torch.argsort(time_points[i])
                time_points[i] = time_points[i][sorted_indices]
                event_types[i] = event_types[i][sorted_indices]

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
            # Pad time_points with (max last-event time + MAX_TIME_OFFSET) so that
            # padded entries always sort after every real event; pad event_types
            # with -1 (EVENT_PADDING_VALUE) which is what event_mask checks for.
            non_empty_last_times = [tp[-1] for tp in time_points if len(tp) > 0]

            if non_empty_last_times:
                max_time = torch.tensor(non_empty_last_times).max() + self.MAX_TIME_OFFSET
            else:
                max_time = torch.tensor(0, dtype=dtype_time, device=device)

            padded_time_points = torch.full((batch_size, length), max_time, dtype=dtype_time, device=device)
            padded_event_types = torch.full(
                (batch_size, length), self.EVENT_PADDING_VALUE, dtype=dtype_event, device=device
            )

            for i, (t, e) in enumerate(zip(time_points, event_types)):
                n = len(t)
                padded_time_points[i, :n] = t
                padded_event_types[i, :n] = e

        if covariates is not None:
            assert isinstance(covariates, dict), "covariates must be a dictionary"
            stacked_covariates = {}
            for key, cov_list in covariates.items():
                assert len(cov_list) == batch_size, f"Covariate '{key}' list length must match batch size"
                if batch_size > 0:
                    stacked_covariates[key] = torch.stack(cov_list, dim=0)  # (B, feature_dim)
                else:
                    stacked_covariates[key] = torch.empty((0, cov_list[0].shape[0]) if len(cov_list) > 0 else (0, 0))
            self.covariates = stacked_covariates
        else:
            self.covariates = None

        self.seq_lengths = lengths

        super().__init__(
            {"time_points": padded_time_points, "event_types": padded_event_types}, batch_size=(batch_size, length)
        )

        # Build per-sequence T_obs tensor (B,).
        # Falls back to self.max_time (last event time per sequence) for any None entry.
        last_event_times = self.max_time  # (B,), 0.0 for empty sequences
        if T_obs is None:
            self.T_obs = last_event_times
        else:
            T_obs_tensor = torch.tensor(
                [v if v is not None else 0.0 for v in T_obs],
                dtype=padded_time_points.dtype,
                device=padded_time_points.device,
            )
            # None entries fall back to last_event_times; provided entries are
            # clamped to be at least the last observed event time.
            none_mask = torch.tensor([v is None for v in T_obs], device=padded_time_points.device)
            self.T_obs = torch.where(none_mask, last_event_times, torch.maximum(T_obs_tensor, last_event_times))

    @property
    def time_points(self):
        return self["time_points"]

    @property
    def event_types(self):
        return self["event_types"]

    @property
    def event_mask(self):
        """Returns a boolean mask indicating which entries are real events."""
        return self.event_types != self.EVENT_PADDING_VALUE

    @property
    def event_types_idx_safe(self) -> torch.Tensor:
        """Event types with padding (-1) replaced by 0 for safe indexing.

        Padding positions should always be excluded via ``event_mask`` before
        using these indices for any meaningful computation.

        Returns:
            Shape: (B, L), dtype int — values in [0, ...]
        """
        return self.event_types.clamp(min=0)

    @property
    def real_counts(self) -> torch.Tensor:
        """Number of real (non-padding) events per sequence.

        Returns:
            Shape: (B,), dtype long
        """
        return self.event_mask.sum(dim=1).long()

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

        This is the time of the last real event strictly before ``t[b]``,
        or 0.0 if no real event precedes ``t[b]``.

        Args:
            t: Query times. Shape: (B,)

        Returns:
            Shape: (B,)
        """
        dtype = self.time_points.dtype
        tp = self.time_points.to(dtype)  # (B, L)
        B = tp.shape[0]
        n_before = self.count_events_before(t.to(dtype))  # (B,)
        tp_with_zero = torch.cat([torch.zeros(B, 1, device=tp.device, dtype=dtype), tp], dim=1)  # (B, L+1)
        return tp_with_zero.gather(1, n_before.unsqueeze(1)).squeeze(1)  # (B,)

    @property
    def causal_indices(self) -> torch.Tensor:
        """For each event position l, the number of real events strictly before it in time.

        This can be used as a column index into a ``(B, L+1, ...)`` parameter
        tensor where column 0 holds the prior and column k holds the parameters
        after k real events have been observed.  Simultaneous events share the
        same column (identical causal context).

        Returns:
            Shape: (B, L), dtype long — values in [0, L]
        """
        tp = self.time_points  # (B, L)
        # strictly_before[b, q, k] = True iff tp[b, k] < tp[b, q] AND event_mask[b, k]
        strictly_before = (tp.unsqueeze(1) < tp.unsqueeze(2)) & self.event_mask.unsqueeze(1)  # (B, L, L)
        return strictly_before.sum(dim=2).long()  # (B, L)

    def one_hot_event_types(self, num_classes: int) -> torch.Tensor:
        """One-hot encoding of event types, with padding positions zeroed out.

        Args:
            num_classes: Number of event type classes D.

        Returns:
            Shape: (B, L, D), dtype bool — padding positions are all-False
        """
        clamped = self.event_types_idx_safe  # (B, L) — padding replaced by 0
        one_hot = torch.nn.functional.one_hot(clamped, num_classes=num_classes).bool()  # (B, L, D)
        return one_hot & self.event_mask.unsqueeze(2)  # zero out padding rows

    def causal_mask(self, mask_padding=True):
        """Return a strict causal mask (B, L, L): True if time[q] > time[k].

        Used for intensity queries: position q may only attend to events
        strictly *before* it in time.
        """
        causal_matrix = self.time_points.unsqueeze(2) > self.time_points.unsqueeze(1)  # (B, L, L)

        if mask_padding:
            event_mask = self.event_mask  # (B, L)
            padding_mask = event_mask.unsqueeze(2) & event_mask.unsqueeze(1)  # (B, L, L)
            causal_matrix = causal_matrix & padding_mask

        return causal_matrix

    def attention_mask(self, mask_padding=True):
        """Return an inclusive causal mask (B, L, L): True if time[q] >= time[k].

        Used by the Transformer during training/encoding: position q may attend
        to all events at the *same* time or earlier.  Multiple simultaneous
        events therefore see each other, but not future events.
        """
        causal_matrix = self.time_points.unsqueeze(2) >= self.time_points.unsqueeze(1)  # (B, L, L)

        if mask_padding:
            event_mask = self.event_mask  # (B, L)
            padding_mask = event_mask.unsqueeze(2) & event_mask.unsqueeze(1)  # (B, L, L)
            causal_matrix = causal_matrix & padding_mask

        return causal_matrix

    @property
    def max_time(self):
        """Returns the time of the last event per sequence.

        For empty sequences (length 0), returns 0.0.
        For empty batch (0 sequences), returns empty tensor of shape (0,).
        """
        if len(self.seq_lengths) == 0:
            return torch.empty(0, dtype=self.time_points.dtype, device=self.time_points.device)

        seq_lengths_tensor = torch.tensor(self.seq_lengths, device=self.time_points.device)

        # When every sequence is empty, time_points has shape (B, 0) and cannot
        # be indexed; short-circuit with zeros.
        if self.time_points.shape[1] == 0:
            return torch.zeros(len(self.seq_lengths), dtype=self.time_points.dtype, device=self.time_points.device)

        batch_indices = torch.arange(len(self.seq_lengths), device=self.time_points.device)

        # Empty sequences use index 0 here and are overwritten with 0.0 below.
        last_indices = torch.where(seq_lengths_tensor > 0, seq_lengths_tensor - 1, 0)
        max_times = self.time_points[batch_indices, last_indices]

        empty_mask = seq_lengths_tensor == 0
        max_times = torch.where(empty_mask, torch.zeros_like(max_times), max_times)

        return max_times

    def to(self, device):
        """Override to() to preserve seq_lengths, T_obs and covariates attributes when moving to device."""
        result = super().to(device)
        result.seq_lengths = self.seq_lengths
        result.T_obs = self.T_obs.to(device)
        if self.covariates is not None:
            result.covariates = {k: v.to(device) for k, v in self.covariates.items()}
        else:
            result.covariates = None
        return result

    def cuda(self, device=None):
        """Override cuda() to preserve seq_lengths, T_obs and covariates attributes."""
        result = super().cuda(device)
        result.seq_lengths = self.seq_lengths
        result.T_obs = self.T_obs.cuda(device)
        if self.covariates is not None:
            result.covariates = {k: v.cuda(device) for k, v in self.covariates.items()}
        else:
            result.covariates = None
        return result

    def cpu(self):
        """Override cpu() to preserve seq_lengths, T_obs and covariates attributes."""
        result = super().cpu()
        result.seq_lengths = self.seq_lengths
        result.T_obs = self.T_obs.cpu()
        if self.covariates is not None:
            result.covariates = {k: v.cpu() for k, v in self.covariates.items()}
        else:
            result.covariates = None
        return result

    def get_unpadded_sequences(self) -> list[MVEventData]:
        """
        Returns a list of unpadded MVEventData sequences.

        Returns:
            List of MVEventData objects containing only the actual events (no padding).
        """
        unpadded = []
        for i in range(len(self.seq_lengths)):
            seq_len = self.seq_lengths[i]

            seq_covariates = None
            if self.covariates is not None:
                seq_covariates = {key: value[i].clone() for key, value in self.covariates.items()}

            seq_T_obs = float(self.T_obs[i]) if self.T_obs.numel() > 0 else None
            unpadded.append(
                MVEventData(
                    time_points=self.time_points[i, :seq_len].clone(),
                    event_types=self.event_types[i, :seq_len].clone(),
                    covariates=seq_covariates,
                    T_obs=seq_T_obs,
                )
            )
        return unpadded

    def __iter__(self):
        for i in range(len(self)):
            seq_covariates = None
            if self.covariates is not None:
                seq_covariates = {key: value[i] for key, value in self.covariates.items()}

            seq_T_obs = float(self.T_obs[i]) if self.T_obs.numel() > 0 else None
            yield MVEventData(
                self.time_points[i, :], self.event_types[i, :], covariates=seq_covariates, T_obs=seq_T_obs
            )

## ------------------------------------------------
## Utils to generate random sequences for debugging
## ------------------------------------------------


def generate_data(D, max_len, num_seq, max_T):
    sequences = []

    for _ in range(num_seq):
        l = np.random.randint(0, max_len)
        T = np.random.rand() * max_T

        time_points = np.random.rand(l) * T
        time_points = np.sort(time_points)
        T += 1

        event_types = np.random.randint(0, D, l)

        sequences.append((torch.tensor(time_points), torch.tensor(event_types, dtype=torch.long)))

    return sequences


def prepare_data(sequences, batch_size):
    mv_events = [MVEventData(t, e) for t, e in sequences]

    batched_mv_events = []
    for start in range(0, len(sequences), batch_size):
        end = min(start + batch_size, len(sequences))
        subset = sequences[start:end]
        ts = [s[0] for s in subset]
        es = [s[1] for s in subset]
        batch = BatchedMVEventData(ts, es)
        batched_mv_events.append(batch)

    return mv_events, batched_mv_events


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


def has_repeated_event_types(ts: "BatchedMVEventData", D: int) -> Tensor:
    """Boolean ``(B,)``: True where a sequence contains a repeated event type."""
    B = ts.time_points.shape[0]
    L = ts.time_points.shape[1]
    if L == 0:
        return torch.zeros(B, dtype=torch.bool, device=ts.time_points.device)
    type_counts = ts.one_hot_event_types(D).float().sum(dim=1)  # (B, D)
    return (type_counts > 1).any(dim=1)
