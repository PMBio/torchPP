"""
Structured configuration for TorchPP training and evaluation.

Uses Python dataclasses with OmegaConf for composable, type-safe, and
validated configuration management.

Dataset configuration
---------------------
Dataset-intrinsic properties (``terminating`` and ``default_token_roles``) are
**class-level attributes on each** :class:`~torchpp.data.base_dataset.TPPDataset`
**subclass** and are not part of the config.  The ``dataset:`` key in YAML / CLI carries
only *operational* parameters — file paths, size limits, and optional ``context_types`` /
``ignored_types`` overrides of the token roles.  ``dataset.token_roles`` is passed to
``create_model``, like ``terminating``.

Both :class:`FitConfig` (training) and :class:`StandaloneEvalConfig`
(evaluation) carry the same ``dataset_type`` / ``dataset`` pair, so they
load splits through the same registry path::

    from torchpp.training.dataset_registry import load_split
    dataset = load_split(config.dataset_type, config.dataset, root, split="train")
    D = dataset.num_event_types
    terminating = dataset.terminating    # class-level attribute
    roles = dataset.token_roles          # default roles plus per-run overrides

Registering a new dataset requires only:

1. Implementing a :class:`~torchpp.data.base_dataset.TPPDataset` subclass
   with the ``@register_dataset("name")`` decorator and a ``load()``
   classmethod.
2. Ensuring the module is imported at startup (the built-ins are imported
   via ``torchpp/data/__init__.py``).
3. Adding a YAML preset in ``configs/data/`` with the operational parameters.

No changes to :class:`FitConfig` or :class:`StandaloneEvalConfig` are needed.

Other config groups
-------------------
    BatchingConfig      — batching strategy (bucketing, dynamic sizing, truncation)
    TrainingConfig      — optimiser, LR scheduler, gradient accumulation
    WandbConfig         — Weights & Biases logging
    SaveConfig          — model checkpointing
    ProfilingConfig     — PyTorch profiler
    CompileConfig       — torch.compile()
    EvalConfig          — evaluation settings shared between fit and standalone eval

Top-level configs
-----------------
    FitConfig            — all of the above, used by train.py
    StandaloneEvalConfig — slimmer config used by evaluate.py

MISSING fields (``omegaconf.MISSING``) are required — they have no default
and must be supplied via YAML or CLI.  Any run that omits them will fail early
with a clear error message rather than silently using a wrong default value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import MISSING

# Registry helpers re-exported for convenience.
from torchpp.training.dataset_registry import (  # noqa: F401
    DATASET_REGISTRY,
    register_dataset,
)


# Valid ``model_type`` strings come from the model registry (single source of truth).
from torchpp.training.registry import MODEL_ALIASES, VALID_MODEL_TYPES  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Config groups
# ---------------------------------------------------------------------------


@dataclass
class BatchingConfig:
    """Batching-strategy settings for handling variable-length sequences."""

    batch_size: int = MISSING
    """Number of sequences per batch. Must be set in a YAML preset."""

    use_bucket_batching: bool = False
    num_buckets: int = 10
    dynamic_batch_size: bool = False
    # None → auto (batch_size × median_length)
    target_total_events: Optional[int] = None
    # None → no truncation
    max_sequence_length: Optional[int] = None
    # right / left / random
    truncation_strategy: str = "right"
    analyze_lengths: bool = True
    accumulate_gradients: int = 1


@dataclass
class TrainingConfig:
    """Optimiser and training loop settings."""

    num_steps: int = MISSING
    """Number of gradient steps. Must be set in a YAML preset."""

    learning_rate: float = MISSING
    """Optimiser learning rate. Must be set in a YAML preset."""

    weight_decay: float = 0.0
    eval_freq: int = MISSING
    """Validation frequency (steps). Must be set in a YAML preset."""

    # LR scheduler: step | cosine | linear_warmup_cosine | none
    scheduler_type: str = "step"
    scheduler_step_size: int = MISSING
    """LR step size (steps). Must be set in a YAML preset."""

    scheduler_gamma: float = MISSING
    """LR decay factor. Must be set in a YAML preset."""

    scheduler_warmup_steps: int = 0
    """Linear warmup steps (used by linear_warmup_cosine). 0 = no warmup."""

    scheduler_eta_min: float = 0.0
    """Minimum LR at the end of cosine decay (used by cosine / linear_warmup_cosine)."""

    # Gradient clipping — max L2 norm of gradient vector. 0.0 = disabled.
    grad_clip: float = 1.0

    # Early stopping — monitor val loss (higher is better, i.e. log-likelihood).
    # Set early_stopping_patience <= 0 to disable.
    early_stopping_patience: int = 0
    # Minimum relative improvement required to count as a new best.
    # E.g. 0.001 means val_ll must improve by at least 0.1% of |best_val_ll|.
    early_stopping_min_delta: float = 0.001


@dataclass
class EvalConfig:
    """Settings for the evaluation pass (used both during training and standalone eval).

    ``limit_test_size`` has no in-code default — it must be set in a YAML preset
    (use ``null`` to evaluate on the full test set, or an integer to cap it).
    """

    run_test_eval: bool = True
    limit_test_size: Optional[int] = MISSING
    """Number of test sequences to evaluate on. null = all. Must be set via YAML."""

    compute_marginal_type: bool = True
    eval_batch_size: Optional[int] = 32

    integrate_marginal_via_grid: bool = True
    """When True, use trapezoidal prefix sums over the intensity grid for
    cumulative intensity during marginal computation (mean/median time,
    marginal type distribution).  This avoids calling the (expensive)
    closed-form cumulative_intensity at every grid point and roughly halves
    grid-building cost for closed-form-CI models (e.g. Hawkes).  The
    trapezoidal approximation with ≥200 grid points is very accurate.
    Set to False to use the exact closed-form CI on the grid (slower)."""

    grid_points_per_unit_time: float = 200.0
    """Number of intensity grid points per unit of scaled time used when
    building the integration grid for marginal quantities (mean/median time,
    marginal type distribution).  Reducing this (e.g. to 50) directly reduces
    the number of model forward passes during evaluation at the cost of
    integration accuracy."""


@dataclass
class WandbConfig:
    """Weights & Biases logging settings."""

    enable: bool = False
    project: str = MISSING
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: Optional[List[str]] = None
    # online | offline | disabled
    mode: str = "online"


@dataclass
class SaveConfig:
    """Model checkpoint settings."""

    # Relative paths are resolved against the working directory. None → auto-generate under
    # $TORCHPPMODELS/<dataset_type>/.
    # If TORCHPPMODELS is unset, defaults to <repo>/models/<dataset_type>/.
    path: Optional[str] = None
    overwrite: bool = False
    auto_increment: bool = True
    # Steps between resumable checkpoints (model+optim+scheduler+step). 0 = off.
    # Written atomically to ``<save_path>.ckpt``; enables wall-clock-safe requeue.
    checkpoint_freq: int = 0
    # If True, resume from ``<save_path>.ckpt`` when it exists (else start fresh).
    resume: bool = False


@dataclass
class DatasetStatsConfig:
    """Controls empirical dataset statistics used for model initialisation."""

    enable: bool = True
    """If False, skip dataset statistics entirely and use model fallbacks."""

    cache: bool = True
    """If True, reuse computed statistics across runs via an on-disk cache."""

    cache_dir: str = "tmp/dataset_stats"
    """Directory for cached statistics. Relative paths are resolved from repo root."""

    force_recompute: bool = False
    """If True, ignore any cached entry and recompute statistics."""

    max_sequences: Optional[int] = None
    """Optional cap on the number of training sequences used to estimate stats."""

    subsample_seed: int = 0
    """Deterministic seed used when subsampling training sequences for stats."""


@dataclass
class ProfilingConfig:
    """PyTorch profiler settings."""

    enable: bool = False
    steps: int = 20
    output: str = "profile_trace.json"


@dataclass
class CompileConfig:
    """torch.compile() settings (PyTorch 2.x)."""

    enable: bool = False
    # default | reduce-overhead | max-autotune | max-autotune-no-cudagraphs
    mode: str = "default"
    # inductor | cudagraphs | eager | …
    backend: str = "inductor"


# ---------------------------------------------------------------------------
# Top-level configs
# ---------------------------------------------------------------------------


@dataclass
class FitConfig:
    """Top-level config for train.py.

    Fields marked ``MISSING`` are required — they must be provided through a
    YAML file or a ``key=value`` CLI override.  Attempting to start training
    without them raises an error immediately.

    Example CLI usage::

        python train.py -c default_fit -c model/spline_exp_hawkes \\
            -c data/taxi wandb.enable=true
    """

    # Required — no default; must be supplied via YAML or CLI
    model_type: str = MISSING
    """Model identifier string, e.g. ``spline_exp_hawkes``.  See VALID_MODEL_TYPES."""

    device: str = "auto"
    """Compute device. ``auto`` resolves to ``cuda`` if available, else ``cpu``."""

    # Selects which dataset config class to use for validation and loading.
    # The registered types are defined in torchpp.training.dataset_registry.
    # Built-in value: ``"benchmark"``.
    dataset_type: str = "benchmark"

    # Raw dataset parameters — merged over the dataset class's ClassVar defaults at
    # load time.  YAML presets write into this dict using the ``dataset:`` key; a split
    # is loaded via ``dataset_registry.load_split(dataset_type, dataset, root, split)``.
    dataset: dict = field(default_factory=dict)

    dataset_module: Optional[str] = None
    """Python module to import before the dataset is resolved (e.g. ``my_pkg.my_data``),
    so a dataset registered with ``@register_dataset`` there can be used from the CLI."""

    model_module: Optional[str] = None
    """Python module to import before ``model_type`` is resolved (e.g. ``my_pkg.my_model``),
    so a model registered with ``register_model`` there can be used from the CLI."""

    batching: BatchingConfig = field(default_factory=BatchingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    arch: Any = field(default_factory=dict)
    """Architecture of ``model_type``: a nested mapping in YAML, the model's arch config
    (:mod:`torchpp.training.arch`) after loading. Only that model's fields are accepted."""
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    save: SaveConfig = field(default_factory=SaveConfig)
    stats: DatasetStatsConfig = field(default_factory=DatasetStatsConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)


@dataclass
class StandaloneEvalConfig:
    """Top-level config for ``evaluate.py``.

    Mirrors :class:`FitConfig` for dataset selection: ``dataset_type`` chooses
    which loader the dataset registry dispatches to and ``dataset`` carries
    its operational parameters.  Any registered dataset (a built-in benchmark
    or a user-registered one) can therefore be evaluated through the same script, and the
    ``-c data/<preset>`` YAML files used for training apply unchanged.

    Example CLI usage::

        python evaluate.py -c default_eval model_path=models/my_model.pth
    """

    # Required — no default; must be supplied via YAML or CLI
    model_path: str = MISSING
    """Path to the saved model checkpoint (``.pth`` file)."""

    model_type: Optional[str] = None
    """Model identifier. Read from the checkpoint; required only for a bare ``state_dict``."""

    output_path: Optional[str] = None
    """Where to write the metrics JSON (default: ``<model_path>.eval.json`` beside the model)."""

    device: str = "auto"

    # Selects which dataset loader the registry dispatches to.  Mirrors
    # ``FitConfig.dataset_type`` so ``-c data/<preset>`` YAML files used for
    # training apply unchanged here.
    dataset_type: str = "benchmark"

    # Raw dataset parameters — validated at runtime by the registry.
    dataset: dict = field(default_factory=dict)

    dataset_module: Optional[str] = None
    model_module: Optional[str] = None
    """Python module to import before the dataset is resolved (e.g. ``my_pkg.my_data``),
    so a dataset registered with ``@register_dataset`` there can be used from the CLI."""

    arch: Any = field(default_factory=dict)
    """Architecture fields to change on the checkpoint's model (nested mapping), or the full
    architecture for a bare ``state_dict``."""
    eval: EvalConfig = field(default_factory=EvalConfig)
