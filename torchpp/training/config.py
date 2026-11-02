"""
Structured configuration for TorchPP training and evaluation.

Uses Python dataclasses with OmegaConf for composable, type-safe, and
validated configuration management.

Dataset configuration
---------------------
Dataset-intrinsic properties (``terminating``, ``default_ignore_indices``)
are now **class-level attributes on each** :class:`~torchpp.data.base_dataset.TPPDataset`
**subclass** and are no longer part of the config.  The ``dataset:`` key in
YAML / CLI carries only *operational* parameters — file paths, size limits,
and an optional ``ignore_indices`` override.

Both :class:`FitConfig` (training) and :class:`StandaloneEvalConfig`
(evaluation) carry the same ``dataset_type`` / ``dataset`` pair, so they
load splits through the same registry path::

    from torchpp.training.dataset_registry import load_split
    dataset = load_split(config.dataset_type, config.dataset, root, split="train")
    D = dataset.num_event_types
    terminating = dataset.terminating        # class-level attribute
    ignore_indices = dataset.ignore_indices  # default or per-run override

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
    ModelArchConfig     — architecture hyper-params (splines, CI integration)
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
from typing import List, Optional

from omegaconf import MISSING

# Registry helpers re-exported for convenience.
from torchpp.training.dataset_registry import (  # noqa: F401
    DATASET_REGISTRY,
    register_dataset,
)


# ---------------------------------------------------------------------------
# Single source of truth for valid model_type strings.
# Both model_factory.py and tests/conftest.py import from here.
# ---------------------------------------------------------------------------

VALID_MODEL_TYPES: List[str] = [
    # Poisson family
    "poisson",
    "inhomogeneous_poisson",
    "spline_poisson",
    "piecewise_constant",
    # Hawkes family
    "hawkes",
    "linear_exp_hawkes",
    "spline_exp_hawkes",
    "piecewise_exp_hawkes",
    "numerical_spline_exp_hawkes",
    "const_spline_kernel",
    "const_piecewise_kernel",
    "spline_spline_kernel",
    "spline_ns_spline_kernel",
    "ns_exp_hawkes",
    "spline_ns_exp_hawkes",
    "inhibitive_const_exp_hawkes",
    "inhibitive_spline_exp_hawkes",
    "inhibitive_spline_spline_kernel",
    # Hybrid Hawkes + neural kernel
    "neural_hawkes",
    "neural_spline_hawkes",
    # Neural TPPs
    "neural_poisson",
    "neural_log_linear",
    "neural_spline",
    "neural_mlp",
    "neural_piecewise_constant",
    "neural_thp",
    "neural_sahp",
    # EasyTPP-faithful replicas (numerical CI)
    "easytpp_thp",
    "easytpp_sahp",
    # Intensity-free / continuous-time-RNN neural TPPs (ported from EasyTPP)
    "intensity_free",
    "nhp",
    "attnhp",
]


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
    # right / left / random / stratified
    truncation_strategy: str = "right"
    analyze_lengths: bool = True
    accumulate_gradients: int = 1
    auto_accumulate_gradients: bool = False


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
class ModelArchConfig:
    """Architecture hyper-params shared across models.

    ``seed`` is auto-randomised at runtime when set to ``None``; this is the
    recommended value so that different runs get different initialisations.
    """

    seed: Optional[int] = None
    """Random seed for parameter initialisation. None → randomised at startup."""

    num_ci_integration_steps: int = 100
    spline_K: int = 5
    spline_kernel_K: int = 4

    # Number of knots on the ABSOLUTE-TIME axis for non-stationary spline kernels
    # (spline_ns_spline_kernel). Placed uniformly over [0, T_baseline] using the
    # spline-baseline time anchor (spline_T_factor). The Δt axis reuses spline_kernel_K.
    spline_ns_K: int = 5

    # Dimensionless multiplier applied to the data-derived spline time anchor.
    # The effective baseline/neural-spline extent is:
    #   T_baseline = spline_T_factor × DatasetStats.sequence_time_p95
    # This ensures the spline covers essentially the full observation window.
    # When dataset_stats is unavailable, spline_T_factor itself is used as a
    # fallback absolute value (i.e. treated as if sequence_time_p95 == 1.0).
    spline_T_factor: float = 1.0

    # Dimensionless multiplier for the spline *kernel* (and neural splines
    # models, which are relative-time based).  The effective kernel extent is:
    #   T_kernel = spline_kernel_T_factor × DatasetStats.inter_event_time_p95
    # Same fallback convention when dataset_stats is unavailable.
    spline_kernel_T_factor: float = 5.0

    # Knot schedule for the spline *baseline* / piecewise-Poisson models.
    # "uniform"   – equidistant knots over [0, T]
    # "geometric" – power-law, denser near 0: x_k = T * (k/(K-1))^p
    spline_knot_schedule: str = "uniform"
    # Exponent p for the baseline geometric schedule. p=1 → uniform; p>1 → finer near 0.
    spline_knot_power: float = 2.0

    # Knot schedule for the spline *kernel*.
    # "uniform"   – equidistant knots over [0, T_kernel]
    # "geometric" – power-law, denser near 0 (recommended for kernels)
    spline_kernel_knot_schedule: str = "geometric"
    # Exponent p for the kernel geometric schedule. p=1 → uniform; p>1 → finer near 0.
    spline_kernel_knot_power: float = 2.0

    # Second-derivative smoothness regulariser for piecewise-linear neural splines.
    # Penalty: lambda * mean_{batch,event,type,k} ((s_{k+1} - s_k))^2 where
    # s_k is the slope of the spline on knot interval k. This is a discrete proxy for
    # ∫ |f''(t)|^2 dt and controls overfitting when the knot count is high.
    # 0.0 disables. Only applied when the model exposes ``spline_smoothness_penalty``.
    spline_smoothness_reg: float = 0.0

    # Clamping function for inhibitive Hawkes processes: softplus | relu | exp
    inhibitive_clamping_fn: str = "softplus"
    # If True, kernel alpha is constrained positive (excitation only) in inhibitive models
    constrain_kernel: bool = False

    # Transformer / neural model fields
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.1
    bias: bool = False

    # MLP intensity head (neural_mlp model)
    mlp_n_hidden: int = 128
    mlp_n_layers: int = 2
    mlp_time_embd_dim: Optional[int] = None

    # NeuralPoisson (NeuralConstTPP) rate link: "exp" (default; log(1e6) logit clamp) or
    # "softplus" (bounded gradient, linear tails; used on MIMIC-IV in the paper).
    neural_poisson_link: str = "exp"

    # Neural kernel (neural_hawkes model)
    neural_kernel_emb_dim: int = 32
    neural_kernel_time_embd_dim: int = 32
    neural_kernel_n_hidden: int = 64
    neural_kernel_n_mlp_layers: int = 2

    # IntensityFree (intensity_free model)
    num_mix_components: int = 8
    """Number of log-normal mixture components for the inter-event time density."""
    iftpp_encoder: str = "gru"
    """History encoder for IntensityFree: 'gru' (faithful Shchur/EasyTPP) or
    'transformer' (shared torchPP encoder, for matched-encoder comparisons)."""
    iftpp_standardize: bool = False
    """Apply data-driven log-inter-time standardisation (mean/std from DatasetStats).
    Off by default: times are already scaled by time_scale_factor, and the extra
    standardisation lowered the test log-likelihood in our experiments."""
    mean_log_inter_time: float = 0.0
    """Dataset mean of log inter-event times (IntensityFree normalisation)."""
    std_log_inter_time: float = 1.0
    """Dataset std of log inter-event times (IntensityFree normalisation)."""

    # AttNHP (attnhp model): reuses n_embd (d_model), n_head, n_layer, dropout.
    attnhp_time_emb_size: int = 16
    """Dimensionality of the AttNHP sinusoidal temporal embedding."""
    attnhp_use_ln: bool = False
    """Whether AttNHP applies LayerNorm after each attention layer."""
    attnhp_quadrature_chunk_size: int = 10
    """AttNHP survival-quadrature nodes per batch-folded pass (0 = all P at once)."""


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

    # Standalone-eval sequence truncation. ``evaluate.py`` has no BatchingConfig, so to
    # replay the training truncation policy at eval time (e.g. MIMIC's max 316 events) —
    # required for standalone RMSE/Acc/p(d) to match the in-training test eval — set
    # ``max_sequence_length`` here (mirrors ``BatchingConfig``; ``None`` = no truncation).
    max_sequence_length: Optional[int] = None
    truncation_strategy: str = "right"

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

    # None → auto-generate from model_type under $TORCHPPMODELS/<dataset_type>/.
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
            -c data/ukb_real_full wandb.enable=true
    """

    # Required — no default; must be supplied via YAML or CLI
    model_type: str = MISSING
    """Model identifier string, e.g. ``spline_exp_hawkes``.  See VALID_MODEL_TYPES."""

    device: str = "auto"
    """Compute device. ``auto`` resolves to ``cuda`` if available, else ``cpu``."""

    # Selects which dataset config class to use for validation and loading.
    # The registered types are defined in torchpp.training.dataset_registry.
    # Built-in values: ``"ukb"`` | ``"easytpp"``.
    dataset_type: str = "ukb"

    # Raw dataset parameters — validated at runtime by resolve_dataset_config().
    # YAML presets write into this dict using the ``dataset:`` key.  The actual
    # typed config is obtained via:
    #   dataset_cfg = resolve_dataset_config(config.dataset_type, config.dataset)
    dataset: dict = field(default_factory=dict)

    batching: BatchingConfig = field(default_factory=BatchingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    arch: ModelArchConfig = field(default_factory=ModelArchConfig)
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
    its operational parameters.  Any registered dataset (UKB, EasyTPP, MIMIC-IV,
    …) can therefore be evaluated through the same script, and the
    ``-c data/<preset>`` YAML files used for training apply unchanged.

    Example CLI usage::

        python evaluate.py \\
            -c default_eval -c data/ukb_real_full \\
            model_path=models/my_model.pth model_type=spline_exp_hawkes
    """

    # Required — no default; must be supplied via YAML or CLI
    model_path: str = MISSING
    """Path to the saved model checkpoint (``.pth`` file)."""

    model_type: str = MISSING
    """Model identifier string matching the saved checkpoint."""

    device: str = "auto"

    # Selects which dataset loader the registry dispatches to.  Mirrors
    # ``FitConfig.dataset_type`` so ``-c data/<preset>`` YAML files used for
    # training apply unchanged here.
    dataset_type: str = "ukb"

    # Raw dataset parameters — validated at runtime by the registry.
    dataset: dict = field(default_factory=dict)

    arch: ModelArchConfig = field(default_factory=ModelArchConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
