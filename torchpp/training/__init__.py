"""Model construction, training and checkpoints.

- :func:`create_model` builds any registered model from a ``model_type`` and its architecture
  config (:mod:`torchpp.training.arch`; :func:`build_arch` validates one);
  :func:`register_model` adds your own model to the registry.
- :func:`fit` trains a model and returns a :class:`FitResult`.
- :func:`save_model` / :func:`load_model` write and read self-describing checkpoints.
"""

from torchpp.training.arch import ArchConfig, IntensityArchConfig, build_arch
from torchpp.training.checkpoint import load_model, read_checkpoint, save_model
from torchpp.training.config import FitConfig, StandaloneEvalConfig
from torchpp.training.model_factory import create_model
from torchpp.training.registry import MODEL_ALIASES, MODEL_REGISTRY, VALID_MODEL_TYPES, register_model
from torchpp.training.trainer import FitResult, default_fit_config, fit

__all__ = [
    "create_model",
    "register_model",
    "fit",
    "FitResult",
    "default_fit_config",
    "save_model",
    "load_model",
    "read_checkpoint",
    "FitConfig",
    "ArchConfig",
    "IntensityArchConfig",
    "build_arch",
    "StandaloneEvalConfig",
    "MODEL_REGISTRY",
    "MODEL_ALIASES",
    "VALID_MODEL_TYPES",
]
