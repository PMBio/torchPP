"""torchpp — Multivariate Temporal Point Processes in PyTorch.

Top-level re-exports for the most commonly used symbols. Full subpackage
APIs are available under ``torchpp.tpp``, ``torchpp.modules``,
``torchpp.utils``, and ``torchpp.data``.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("TorchPP")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"

from torchpp.utils.event_data import MVEventData, BatchedMVEventData, DatasetStats, TokenRoles
from torchpp.utils.batching import (
    BucketBatchSampler,
    TruncationStrategy,
    collate_batch_with_truncation,
    analyze_length_distribution,
    print_length_analysis,
)

from torchpp.tpp import *  # noqa: F401,F403  (every model, base class and config)
from torchpp.tpp import __all__ as _tpp_all
from torchpp.training import (
    MODEL_ALIASES,
    MODEL_REGISTRY,
    VALID_MODEL_TYPES,
    FitConfig,
    FitResult,
    ArchConfig,
    build_arch,
    create_model,
    fit,
    load_model,
    register_model,
    save_model,
)

from torchpp.data.base_dataset import TPPDataset
from torchpp.data.benchmark_data import BenchmarkDataset
from torchpp.utils.evaluation import evaluate_tpp

__all__ = [
    "__version__",
    "MVEventData",
    "TokenRoles",
    "BatchedMVEventData",
    "DatasetStats",
    "BucketBatchSampler",
    "TruncationStrategy",
    "collate_batch_with_truncation",
    "analyze_length_distribution",
    "print_length_analysis",
    *_tpp_all,
    "create_model",
    "register_model",
    "fit",
    "FitResult",
    "FitConfig",
    "ArchConfig",
    "build_arch",
    "evaluate_tpp",
    "save_model",
    "load_model",
    "MODEL_REGISTRY",
    "MODEL_ALIASES",
    "VALID_MODEL_TYPES",
    "TPPDataset",
    "BenchmarkDataset",
]
