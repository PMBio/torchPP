"""Utility modules: event data containers, math helpers, batching, and evaluation."""

from torchpp.utils.event_data import MVEventData, BatchedMVEventData, DatasetStats, TokenRoles
from torchpp.utils.math import (
    LinearSpline,
    PiecewiseConstant,
    inverse_softplus,
    loglinear_cumulative_intensity,
    invert_CI,
    numerical_integration,
    trapezoidal_integration,
)
from torchpp.utils.batching import (
    BucketBatchSampler,
    TruncationStrategy,
    collate_batch_with_truncation,
    analyze_length_distribution,
    print_length_analysis,
)
from torchpp.utils.evaluation import evaluate_tpp, format_metrics

__all__ = [
    # Event data
    "MVEventData",
    "TokenRoles",
    "BatchedMVEventData",
    "DatasetStats",
    # Math utilities
    "LinearSpline",
    "PiecewiseConstant",
    "inverse_softplus",
    "loglinear_cumulative_intensity",
    "invert_CI",
    "numerical_integration",
    "trapezoidal_integration",
    # Batching
    "BucketBatchSampler",
    "TruncationStrategy",
    "collate_batch_with_truncation",
    "analyze_length_distribution",
    "print_length_analysis",
    # Evaluation
    "evaluate_tpp",
    "format_metrics",
]
