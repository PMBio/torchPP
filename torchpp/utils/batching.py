"""
Efficient batching strategies for temporal point processes with variable-length sequences.

Addresses CUDA OOM issues by minimizing padding waste through:
1. Length-based bucketing to group similar-length sequences
2. Optional sequence truncation with multiple strategies
3. Dynamic batch sizing based on sequence length
"""

from typing import List, Optional
import numpy as np
from torch.utils.data import Sampler, Dataset

from .event_data import MVEventData, BatchedMVEventData


class BucketBatchSampler(Sampler[List[int]]):
    """
    Groups sequences into buckets by length, then samples batches within buckets.

    This minimizes padding waste by ensuring sequences in the same batch have similar lengths.

    Args:
        data_source: Dataset containing sequences
        batch_size: Target batch size for reference bucket (shortest sequences)
        shuffle: Whether to shuffle sequences within each bucket
        drop_last: Whether to drop the last incomplete batch in each bucket
        num_buckets: Number of length buckets (more = less padding, more shuffling constraints)
        dynamic_batch_size: If True, adjust batch size per bucket to maintain constant
            total number of events per batch (batch_size × seq_length ≈ target_total_events).
            This prevents OOM by keeping memory usage predictable.
        target_total_events: Total events per batch when dynamic_batch_size=True. If None,
            computed as batch_size × median_length. Increase this value until you hit OOM,
            then back off ~20% for safety.
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        data_source: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        num_buckets: int = 10,
        dynamic_batch_size: bool = False,
        target_total_events: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_buckets = num_buckets
        self.dynamic_batch_size = dynamic_batch_size
        self.seed = seed
        self.epoch = 0

        self.lengths = [len(data_source[i]) for i in range(len(data_source))]

        if self.dynamic_batch_size:
            if target_total_events is not None:
                self.target_total_events = target_total_events
            else:
                # Default: batch_size × median_length as a reasonable tuning starting point
                median_length = np.median(self.lengths) if len(self.lengths) > 0 else 1
                self.target_total_events = int(self.batch_size * median_length)
                print(
                    f"  Dynamic batching: target_total_events = {self.target_total_events} "
                    f"(batch_size={batch_size} × median_length={median_length:.0f})"
                )
                print("  Tip: Increase target_total_events to use more memory (until OOM), then reduce by ~20%")
        else:
            self.target_total_events = None

        self._create_buckets()

    def _create_buckets(self):
        """Partition sequences into buckets based on length."""
        lengths_array = np.array(self.lengths)

        # Quantile-based boundaries ensure each bucket has roughly equal occupancy.
        if len(lengths_array) == 0:
            self.bucket_boundaries = []
            self.buckets = []
            return

        quantiles = np.linspace(0, 100, self.num_buckets + 1)
        self.bucket_boundaries = np.percentile(lengths_array, quantiles)
        # Ensure boundaries are strictly increasing
        self.bucket_boundaries = np.unique(self.bucket_boundaries)

        self.buckets = [[] for _ in range(len(self.bucket_boundaries) - 1)]
        for idx, length in enumerate(self.lengths):
            bucket_idx = np.searchsorted(self.bucket_boundaries[1:], length, side="right")
            bucket_idx = min(bucket_idx, len(self.buckets) - 1)
            self.buckets[bucket_idx].append(idx)

    def __iter__(self):
        if self.seed is not None:
            rng = np.random.default_rng(self.seed + self.epoch)
        else:
            rng = np.random.default_rng()

        all_batches = []
        for bucket_idx, bucket in enumerate(self.buckets):
            if len(bucket) == 0:
                continue

            bucket_indices = np.array(bucket)
            if self.shuffle:
                rng.shuffle(bucket_indices)

            if self.dynamic_batch_size:
                # Use the bucket's upper boundary as the representative length so
                # target_total_events ≈ adjusted_batch_size × bucket_max_len.
                bucket_max_len = self.bucket_boundaries[bucket_idx + 1]
                adjusted_batch_size = max(1, int(self.target_total_events / max(bucket_max_len, 1)))

                # Cap at the baseline batch_size so dynamic batches never exceed it.
                adjusted_batch_size = min(adjusted_batch_size, self.batch_size)
            else:
                adjusted_batch_size = self.batch_size

            for i in range(0, len(bucket_indices), adjusted_batch_size):
                batch = bucket_indices[i : i + adjusted_batch_size].tolist()

                if self.drop_last and len(batch) < adjusted_batch_size:
                    continue

                all_batches.append(batch)

        # Shuffle batches across buckets while keeping bucket grouping within a batch.
        if self.shuffle:
            rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        """Estimate number of batches (may not be exact with dynamic_batch_size)."""
        total = 0
        for bucket_idx, bucket in enumerate(self.buckets):
            if len(bucket) == 0:
                continue

            if self.dynamic_batch_size:
                bucket_max_len = self.bucket_boundaries[bucket_idx + 1]
                adjusted_batch_size = max(1, int(self.target_total_events / max(bucket_max_len, 1)))
                adjusted_batch_size = min(adjusted_batch_size, self.batch_size)
            else:
                adjusted_batch_size = self.batch_size

            num_batches = len(bucket) // adjusted_batch_size
            if not self.drop_last and len(bucket) % adjusted_batch_size != 0:
                num_batches += 1
            total += num_batches

        return total

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling across epochs."""
        self.epoch = epoch


class TruncationStrategy:
    """Strategies for truncating sequences that exceed max length."""

    @staticmethod
    def right_truncate(seq: MVEventData, max_length: int) -> MVEventData:
        """
        Keep earliest events (left side), drop most recent events.

        Preserves historical context for TPPs - most appropriate strategy.
        The intensity at any time depends on past events, so keeping early
        events maintains the causal structure.
        """
        if len(seq) <= max_length:
            return seq
        return MVEventData(
            time_points=seq.time_points[:max_length],
            event_types=seq.event_types[:max_length],
            covariates=seq.covariates,  # Preserve covariates
        )

    @staticmethod
    def left_truncate(seq: MVEventData, max_length: int) -> MVEventData:
        """
        Keep most recent events (right side), drop earliest events.

        WARNING: This breaks the causal structure for TPPs!
        The intensity function loses information about early events.
        Only use if recent events are more important than history.
        """
        if len(seq) <= max_length:
            return seq
        return MVEventData(
            time_points=seq.time_points[-max_length:],
            event_types=seq.event_types[-max_length:],
            covariates=seq.covariates,  # Preserve covariates
        )

    @staticmethod
    def random_sample(seq: MVEventData, max_length: int, seed: Optional[int] = None) -> MVEventData:
        """
        Randomly select a subset of events while maintaining temporal order.

        Randomly drops (n - max_length) events from the sequence, keeping the
        remaining events in their original temporal order. This preserves the
        relative ordering of selected events but creates gaps in the history.

        WARNING: Creates gaps in the temporal history!
        While temporal order is preserved, the intensity function loses information
        about dropped events. Better than left-truncation but worse than right-truncation.
        Only use when you need to sample throughout the entire time range.
        """
        if len(seq) <= max_length:
            return seq

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        indices = rng.choice(len(seq), size=max_length, replace=False)
        indices = np.sort(indices)  # Keep selected events in temporal order

        return MVEventData(
            time_points=seq.time_points[indices],
            event_types=seq.event_types[indices],
            covariates=seq.covariates,  # Preserve covariates
        )


def collate_batch_with_truncation(
    mv_l: List[MVEventData],
    max_length: Optional[int] = None,
    truncation_strategy: str = "right",
    seed: Optional[int] = None,
) -> BatchedMVEventData:
    """
    Collate function that optionally truncates sequences before batching.

    Args:
        mv_l: List of MVEventData sequences
        max_length: Maximum sequence length (None = no truncation)
        truncation_strategy: Strategy for truncation if max_length is exceeded:
            - "right": Keep earliest events (RECOMMENDED for TPPs)
            - "left": Keep most recent events (breaks history)
            - "random": Random sampling (creates gaps in history, but averaged over many draws could be best)
        seed: Random seed for reproducible truncation

    Returns:
        Batched and padded event data
    """
    if max_length is not None:
        if truncation_strategy == "right":
            truncate_fn = TruncationStrategy.right_truncate
        elif truncation_strategy == "left":
            truncate_fn = TruncationStrategy.left_truncate
        elif truncation_strategy == "random":
            truncate_fn = lambda seq, max_len: TruncationStrategy.random_sample(seq, max_len, seed)
        else:
            raise ValueError(f"Unknown truncation strategy: {truncation_strategy}")

        mv_l = [truncate_fn(seq, max_length) for seq in mv_l]

    if not mv_l:
        return BatchedMVEventData([], [])
    return BatchedMVEventData(mv_events=mv_l)


def analyze_length_distribution(sequences: List[MVEventData]) -> dict:
    """
    Analyze the length distribution of sequences to help choose batching parameters.

    Returns statistics that can guide the choice of:
    - num_buckets: More buckets if high variance
    - max_length: Based on percentiles
    - dynamic_batch_size: Recommended if max/median ratio is high
    """
    if len(sequences) == 0:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "std": 0,
            "percentiles": {},
            "recommendation": "No sequences to analyze",
        }

    lengths = np.array([len(seq) for seq in sequences])

    percentiles = {
        "p10": np.percentile(lengths, 10),
        "p25": np.percentile(lengths, 25),
        "p50": np.percentile(lengths, 50),
        "p75": np.percentile(lengths, 75),
        "p90": np.percentile(lengths, 90),
        "p95": np.percentile(lengths, 95),
        "p99": np.percentile(lengths, 99),
    }

    max_to_median_ratio = percentiles["p50"] / max(percentiles["p50"], 1)
    if percentiles["p50"] > 0:
        max_to_median_ratio = lengths.max() / percentiles["p50"]

    recommendations = []

    if max_to_median_ratio > 3:
        recommendations.append("High length variance detected (max/median > 3)")
        recommendations.append("→ STRONGLY recommend using BucketBatchSampler with num_buckets=10-20")
        recommendations.append("→ STRONGLY recommend dynamic_batch_size=True")

    if percentiles["p99"] > 2 * percentiles["p75"]:
        recommendations.append("Long tail detected (p99 > 2*p75)")
        recommendations.append(
            f"→ Consider setting max_length to p95 or p99 ({percentiles['p95']:.0f} or {percentiles['p99']:.0f})"
        )

    if lengths.max() > 1000:
        recommendations.append(f"Very long sequences detected (max={lengths.max()})")
        recommendations.append("→ Consider truncation to reduce memory usage")

    if len(recommendations) == 0:
        recommendations.append("Length distribution is relatively uniform")
        recommendations.append("→ Standard batching should work well")

    return {
        "count": len(lengths),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(percentiles["p50"]),
        "std": float(lengths.std()),
        "max_to_median_ratio": float(max_to_median_ratio),
        "percentiles": {k: float(v) for k, v in percentiles.items()},
        "recommendations": recommendations,
    }


def print_length_analysis(sequences: List[MVEventData]):
    """Pretty-print length distribution analysis."""
    stats = analyze_length_distribution(sequences)

    print("\n" + "=" * 60)
    print("Sequence Length Analysis")
    print("=" * 60)
    print(f"  Count: {stats['count']}")
    print(f"  Min: {stats['min']}")
    print(f"  Max: {stats['max']}")
    print(f"  Mean: {stats['mean']:.1f}")
    print(f"  Median: {stats['median']:.1f}")
    print(f"  Std Dev: {stats['std']:.1f}")
    print(f"  Max/Median Ratio: {stats['max_to_median_ratio']:.2f}")
    print()
    print("Percentiles:")
    for k, v in stats["percentiles"].items():
        print(f"  {k}: {v:.1f}")
    print()
    print("Recommendations:")
    for rec in stats["recommendations"]:
        print(f"  {rec}")
    print("=" * 60 + "\n")
