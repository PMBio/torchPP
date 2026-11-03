"""Model registry: the single place that defines every ``model_type``.

Each entry maps a ``model_type`` string to its class, a one-line description, its
architecture config class (:mod:`torchpp.training.arch`), a builder (arch config →
model) and the run-name rule used for default save paths. ``VALID_MODEL_TYPES``,
:func:`~torchpp.training.model_factory.create_model` and the test matrix are all
derived from :data:`MODEL_REGISTRY`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from torchpp.modules.nn.attnhp import AttNHPEncoder
from torchpp.modules.nn.encoder import ContTimeLSTMEncoder, GRUEncoder
from torchpp.modules.nn.transformer import TransformerConfig, TransformerEncoder
from torchpp.tpp.baseline import (
    LogLinearPoissonProcess,
    PiecewisePoissonProcess,
    PoissonProcess,
    SplinePoissonProcess,
)
from torchpp.tpp.hawkes import (
    ConstBaselineExpKernelHawkesProcess,
    ConstBaselinePiecewiseKernelHawkesProcess,
    ConstBaselineSplineKernelHawkesProcess,
    InhibitiveConstBaselineExpKernelHawkesProcess,
    InhibitiveSplineBaselineExpKernelHawkesProcess,
    InhibitiveSplineBaselineMLPKernelHawkesProcess,
    InhibitiveSplineBaselineMLPSplineKernelHawkesProcess,
    InhibitiveSplineBaselineSplineKernelHawkesProcess,
    LogLinearBaselineExpKernelHawkesProcess,
    PiecewiseBaselineExpKernelHawkesProcess,
    SplineBaselineExpKernelHawkesProcess,
    SplineBaselineSplineKernelHawkesProcess,
)
from torchpp.tpp.neural import (
    AttNHPProcess,
    IntensityFreeProcess,
    NeuralLogLinearProcess,
    NeuralMLPProcess,
    NeuralPiecewiseConstantProcess,
    NeuralPoissonProcess,
    NeuralSplineProcess,
    NHPProcess,
    SAHPProcess,
    THPProcess,
)
from torchpp.training import arch as A
from torchpp.utils.math import interaction_spline_knot_gaps

if TYPE_CHECKING:
    from torchpp.utils.event_data import DatasetStats


@dataclass
class BuildContext:
    """Everything a builder needs besides the architecture config.

    ``stats`` are the dataset statistics in scaled time units (``None`` without data).
    ``kw`` holds the arguments shared by every model (terminating, seed, time_scale,
    token_roles).
    """

    D: int
    stats: Optional["DatasetStats"]
    kw: dict = field(default_factory=dict)

    @property
    def has_origin(self) -> bool:
        roles = self.kw.get("token_roles")
        return bool(roles is not None and roles.has_origin)

    def baseline_spacing(self, baseline: A.BaselineKnotsConfig) -> float:
        """Uniform baseline knot spacing: extent ``T_factor × sequence_time_p95`` over ``num_knots - 1`` gaps."""
        stats = self.stats
        T = baseline.T_factor * stats.sequence_time_p95 if stats is not None and stats.sequence_time_p95 > 0 else baseline.T_factor
        return T / (baseline.num_knots - 1)

    def knot_gaps(self, section, num_knots: int):
        """Knot gaps over the time since an event: extent ``T_factor × inter_event_time_p95``."""
        stats = self.stats
        T = section.T_factor * stats.inter_event_time_p95 if stats is not None and stats.inter_event_time_p95 > 0 else section.T_factor
        return interaction_spline_knot_gaps(T, num_knots, schedule=section.knot_schedule, power=section.knot_power)


@dataclass(frozen=True)
class ModelSpec:
    """Registry entry for one ``model_type``."""

    cls: type
    description: str
    config_cls: type
    build: Callable[[Any, BuildContext], Any]
    run_name: Callable[[Any], str]
    experimental: bool = False


def _transformer_config(enc: A.TransformerEncoderConfig, D: int) -> TransformerConfig:
    return TransformerConfig(
        num_event_types=D, n_layer=enc.n_layers, n_head=enc.n_heads, n_embd=enc.n_embd, dropout=enc.dropout
    )


def _neural(cls, extra: Optional[Callable] = None):
    """Builder for a transformer-encoder model; ``extra(arch, ctx)`` returns head-specific kwargs."""

    def build(arch, c):
        kwargs = extra(arch, c) if extra is not None else {}
        return cls(c.D, config=_transformer_config(arch.encoder, c.D), **kwargs, **c.kw)

    return build


def _neural_name(head: str, with_bins: bool = False):
    def name(arch):
        tag = f"neural_{arch.encoder.n_layers}l_{arch.encoder.n_embd}d_{head}"
        return f"{tag}_kK{arch.head.num_bins}" if with_bins else tag

    return name


def _intensity_free(arch, c):
    enc = arch.encoder
    if enc.type == "transformer":
        encoder = TransformerEncoder(_transformer_config(enc, c.D), has_origin=c.has_origin)
    elif enc.type == "gru":
        encoder = GRUEncoder(
            num_event_types=c.D, n_embd=enc.n_embd, num_layers=enc.n_layers, dropout=enc.dropout, has_origin=c.has_origin
        )
    else:
        raise ValueError(f"Unknown arch.encoder.type '{enc.type}' for intensity_free (expected 'gru' or 'transformer').")
    return IntensityFreeProcess(encoder, num_mix_components=arch.head.num_mix_components, **c.kw)


def _nhp(arch, c):
    encoder = ContTimeLSTMEncoder(num_event_types=c.D, hidden_dim=arch.encoder.hidden_dim, has_origin=c.has_origin)
    return NHPProcess(encoder, **c.kw)


def _attnhp(arch, c):
    enc = arch.encoder
    encoder = AttNHPEncoder(
        num_event_types=c.D,
        d_model=enc.n_embd,
        d_time=enc.time_emb_dim,
        n_layers=enc.n_layers,
        n_head=enc.n_heads,
        dropout=enc.dropout,
        use_norm=enc.use_layer_norm,
        has_origin=c.has_origin,
    )
    return AttNHPProcess(encoder, **c.kw)


def _baseline_kw(arch, c) -> dict:
    return dict(baseline_num_knots=arch.baseline.num_knots, baseline_delta_t=c.baseline_spacing(arch.baseline))


def _kernel_kw(arch, c) -> dict:
    return dict(kernel_num_knots=arch.kernel.num_knots, kernel_delta_t=c.knot_gaps(arch.kernel, arch.kernel.num_knots))


def _spline_spline(cls, inhibitive: bool):
    def build(arch, c):
        extra = dict(clamping_fn=arch.clamping_fn) if inhibitive else {}
        return cls(D=c.D, **_baseline_kw(arch, c), **_kernel_kw(arch, c), dataset_stats=c.stats, **extra, **c.kw)

    return build


def _baseline_model(cls):
    return lambda arch, c: cls(D=c.D, **_baseline_kw(arch, c), dataset_stats=c.stats, **c.kw)


def _kernel_model(cls):
    return lambda arch, c: cls(D=c.D, **_kernel_kw(arch, c), dataset_stats=c.stats, **c.kw)


def _bins(arch, c) -> dict:
    return dict(num_bins=arch.head.num_bins, knot_gaps=c.knot_gaps(arch.head, arch.head.num_bins), dataset_stats=c.stats)


def _spec(cls, description, config_cls, build, run_name, experimental=False) -> ModelSpec:
    return ModelSpec(cls, description, config_cls, build, run_name, experimental)


def _K(arch) -> int:
    return arch.baseline.num_knots


def _kK(arch) -> int:
    return arch.kernel.num_knots


MODEL_REGISTRY: dict[str, ModelSpec] = {
    # --- Poisson processes ---
    "poisson": _spec(
        PoissonProcess, "Homogeneous Poisson process", A.IntensityArchConfig,
        lambda arch, c: PoissonProcess(D=c.D, dataset_stats=c.stats, **c.kw),
        lambda arch: "const_poisson",
    ),
    "loglinear_poisson": _spec(
        LogLinearPoissonProcess, "Log-linear Poisson process", A.IntensityArchConfig,
        lambda arch, c: LogLinearPoissonProcess(D=c.D, dataset_stats=c.stats, **c.kw),
        lambda arch: "loglinear_poisson",
    ),
    "spline_poisson": _spec(
        SplinePoissonProcess, "Spline Poisson process", A.SplineBaselineArchConfig,
        _baseline_model(SplinePoissonProcess),
        lambda arch: f"spline_K{_K(arch)}_poisson",
    ),
    "piecewise_poisson": _spec(
        PiecewisePoissonProcess, "Piecewise-constant Poisson process", A.SplineBaselineArchConfig,
        _baseline_model(PiecewisePoissonProcess),
        lambda arch: f"piecewise_K{_K(arch)}_poisson",
    ),
    # --- Hawkes processes (additive, positive kernels) ---
    "const_exp_hawkes": _spec(
        ConstBaselineExpKernelHawkesProcess, "Hawkes process, constant baseline + exponential kernel",
        A.IntensityArchConfig,
        lambda arch, c: ConstBaselineExpKernelHawkesProcess(D=c.D, dataset_stats=c.stats, **c.kw),
        lambda arch: "const_exp_hawkes",
    ),
    "loglinear_exp_hawkes": _spec(
        LogLinearBaselineExpKernelHawkesProcess, "Hawkes process, log-linear baseline + exponential kernel",
        A.IntensityArchConfig,
        lambda arch, c: LogLinearBaselineExpKernelHawkesProcess(D=c.D, dataset_stats=c.stats, **c.kw),
        lambda arch: "loglinear_exp_hawkes",
    ),
    "spline_exp_hawkes": _spec(
        SplineBaselineExpKernelHawkesProcess, "Hawkes process, spline baseline + exponential kernel",
        A.SplineBaselineArchConfig,
        _baseline_model(SplineBaselineExpKernelHawkesProcess),
        lambda arch: f"spline_K{_K(arch)}_exp_hawkes",
    ),
    "piecewise_exp_hawkes": _spec(
        PiecewiseBaselineExpKernelHawkesProcess, "Hawkes process, piecewise-constant baseline + exponential kernel",
        A.SplineBaselineArchConfig,
        _baseline_model(PiecewiseBaselineExpKernelHawkesProcess),
        lambda arch: f"piecewise_K{_K(arch)}_exp_hawkes",
    ),
    "const_spline_hawkes": _spec(
        ConstBaselineSplineKernelHawkesProcess, "Hawkes process, constant baseline + spline kernel",
        A.SplineKernelArchConfig,
        _kernel_model(ConstBaselineSplineKernelHawkesProcess),
        lambda arch: f"const_spline_kK{_kK(arch)}_hawkes",
    ),
    "const_piecewise_hawkes": _spec(
        ConstBaselinePiecewiseKernelHawkesProcess, "Hawkes process, constant baseline + piecewise-constant kernel",
        A.SplineKernelArchConfig,
        _kernel_model(ConstBaselinePiecewiseKernelHawkesProcess),
        lambda arch: f"const_piecewise_kK{_kK(arch)}_hawkes",
    ),
    "spline_spline_hawkes": _spec(
        SplineBaselineSplineKernelHawkesProcess, "Hawkes process, spline baseline + spline kernel",
        A.SplineBaselineSplineKernelArchConfig,
        _spline_spline(SplineBaselineSplineKernelHawkesProcess, inhibitive=False),
        lambda arch: f"spline_K{_K(arch)}_spline_kK{_kK(arch)}_hawkes",
    ),
    # --- Inhibitive Hawkes processes (signed kernels, positivity applied to the sum) ---
    "inhibitive_const_exp_hawkes": _spec(
        InhibitiveConstBaselineExpKernelHawkesProcess, "Inhibitive Hawkes process, constant baseline + exponential kernel",
        A.InhibitiveArchConfig,
        lambda arch, c: InhibitiveConstBaselineExpKernelHawkesProcess(
            D=c.D, clamping_fn=arch.clamping_fn, dataset_stats=c.stats, **c.kw
        ),
        lambda arch: f"inhibitive_{arch.clamping_fn}_const_exp_hawkes",
    ),
    "inhibitive_spline_exp_hawkes": _spec(
        InhibitiveSplineBaselineExpKernelHawkesProcess, "Inhibitive Hawkes process, spline baseline + exponential kernel",
        A.InhibitiveSplineBaselineArchConfig,
        lambda arch, c: InhibitiveSplineBaselineExpKernelHawkesProcess(
            D=c.D, **_baseline_kw(arch, c), clamping_fn=arch.clamping_fn, dataset_stats=c.stats, **c.kw
        ),
        lambda arch: f"inhibitive_{arch.clamping_fn}_spline_K{_K(arch)}_exp_hawkes",
    ),
    "inhibitive_spline_spline_hawkes": _spec(
        InhibitiveSplineBaselineSplineKernelHawkesProcess, "Inhibitive Hawkes process, spline baseline + spline kernel",
        A.InhibitiveSplineSplineArchConfig,
        _spline_spline(InhibitiveSplineBaselineSplineKernelHawkesProcess, inhibitive=True),
        lambda arch: f"inhibitive_{arch.clamping_fn}_spline_K{_K(arch)}_spline_kK{_kK(arch)}_hawkes",
    ),
    "inhibitive_spline_mlp_hawkes": _spec(
        InhibitiveSplineBaselineMLPKernelHawkesProcess,
        "Inhibitive Hawkes process, spline baseline + MLP kernel (experimental)",
        A.InhibitiveMLPKernelArchConfig,
        lambda arch, c: InhibitiveSplineBaselineMLPKernelHawkesProcess(
            D=c.D,
            **_baseline_kw(arch, c),
            clamping_fn=arch.clamping_fn,
            dataset_stats=c.stats,
            kernel_emb_dim=arch.kernel.emb_dim,
            kernel_time_emb_dim=arch.kernel.time_emb_dim,
            kernel_n_hidden=arch.kernel.n_hidden,
            kernel_n_layers=arch.kernel.n_layers,
            **c.kw,
        ),
        lambda arch: f"inhibitive_{arch.clamping_fn}_spline_K{_K(arch)}_mlp_hawkes",
        experimental=True,
    ),
    "inhibitive_spline_mlpspline_hawkes": _spec(
        InhibitiveSplineBaselineMLPSplineKernelHawkesProcess,
        "Inhibitive Hawkes process, spline baseline + MLP-predicted spline kernel (experimental)",
        A.InhibitiveMLPSplineKernelArchConfig,
        lambda arch, c: InhibitiveSplineBaselineMLPSplineKernelHawkesProcess(
            D=c.D,
            **_baseline_kw(arch, c),
            **_kernel_kw(arch, c),
            clamping_fn=arch.clamping_fn,
            dataset_stats=c.stats,
            kernel_emb_dim=arch.kernel.emb_dim,
            kernel_n_hidden=arch.kernel.n_hidden,
            kernel_n_layers=arch.kernel.n_layers,
            **c.kw,
        ),
        lambda arch: f"inhibitive_{arch.clamping_fn}_spline_K{_K(arch)}_mlpspline_kK{_kK(arch)}_hawkes",
        experimental=True,
    ),
    # --- Neural TPPs: shared transformer encoder + intensity head ---
    "neural_poisson": _spec(
        NeuralPoissonProcess, "Neural TPP, constant intensity per interval", A.NeuralArchConfig,
        _neural(NeuralPoissonProcess), _neural_name("poisson"),
    ),
    "neural_log_linear": _spec(
        NeuralLogLinearProcess, "Neural TPP, log-linear intensity per interval", A.NeuralArchConfig,
        _neural(NeuralLogLinearProcess), _neural_name("log_linear"),
    ),
    "neural_spline": _spec(
        NeuralSplineProcess, "Neural TPP, linear-spline intensity per interval", A.NeuralSplineArchConfig,
        _neural(NeuralSplineProcess, lambda arch, c: dict(**_bins(arch, c), spline_smoothness_reg=arch.head.smoothness_reg)),
        _neural_name("spline", with_bins=True),
    ),
    "neural_mlp": _spec(
        NeuralMLPProcess, "Neural TPP, MLP intensity of the elapsed time", A.NeuralMLPArchConfig,
        _neural(
            NeuralMLPProcess,
            lambda arch, c: dict(
                n_hidden=arch.head.n_hidden, n_layers=arch.head.n_layers, time_emb_dim=arch.head.time_emb_dim
            ),
        ),
        _neural_name("mlp"),
    ),
    "neural_piecewise_constant": _spec(
        NeuralPiecewiseConstantProcess, "Neural TPP, piecewise-constant intensity per interval",
        A.NeuralPiecewiseArchConfig,
        _neural(NeuralPiecewiseConstantProcess, _bins),
        _neural_name("piecewise_constant"),
    ),
    # --- Published neural architectures ---
    "thp": _spec(
        THPProcess, "Transformer Hawkes Process (Zuo et al., 2020)", A.NeuralArchConfig,
        _neural(THPProcess), lambda arch: f"transformer_{arch.encoder.n_layers}l_{arch.encoder.n_embd}d_thp",
    ),
    "sahp": _spec(
        SAHPProcess, "Self-Attentive Hawkes Process (Zhang et al., 2020)", A.NeuralArchConfig,
        _neural(SAHPProcess), lambda arch: f"transformer_{arch.encoder.n_layers}l_{arch.encoder.n_embd}d_sahp",
    ),
    "intensity_free": _spec(
        IntensityFreeProcess, "IntensityFree, log-normal mixture (Shchur et al., 2020)", A.IntensityFreeArchConfig,
        _intensity_free, lambda arch: "intensity_free",
    ),
    "nhp": _spec(
        NHPProcess, "Neural Hawkes Process, continuous-time LSTM (Mei & Eisner, 2017)", A.NHPArchConfig,
        _nhp, lambda arch: "nhp",
    ),
    "attnhp": _spec(
        AttNHPProcess, "Attentive Neural Hawkes Process (Yang et al., 2022)", A.AttNHPArchConfig,
        _attnhp, lambda arch: "attnhp",
    ),
}

MODEL_ALIASES: dict[str, str] = {"hawkes": "const_exp_hawkes"}
"""Short names accepted wherever a ``model_type`` is expected."""

VALID_MODEL_TYPES: list[str] = list(MODEL_REGISTRY)


def resolve_model_type(model_type: str) -> str:
    """Canonical ``model_type`` for a registered name or alias; raises ``ValueError`` otherwise."""
    name = MODEL_ALIASES.get(model_type, model_type)
    if name not in MODEL_REGISTRY:
        valid = ", ".join(VALID_MODEL_TYPES + list(MODEL_ALIASES))
        raise ValueError(f"Unknown model_type '{model_type}'. Valid choices: {valid}")
    return name


def register_model(
    name: str,
    cls: Optional[type] = None,
    *,
    build: Optional[Callable[[Any, BuildContext], Any]] = None,
    config_cls: Optional[type] = None,
    description: str = "",
    run_name: Optional[Callable[[Any], str]] = None,
    experimental: bool = False,
    aliases: tuple = (),
):
    """Register a model class under ``model_type`` ``name``; usable as a class decorator.

    Once registered, the model works with :func:`~torchpp.training.model_factory.create_model`,
    the command line (``model_type=name``; import the defining module with
    ``model_module=...``) and checkpoints.

    Args:
        name: The new ``model_type``.
        cls: A :class:`~torchpp.tpp.base_tpp.TemporalPointProcess` subclass. Omit it to use
            ``register_model`` as a decorator.
        build: ``build(arch, ctx) -> model``. Default: ``cls(ctx.D, **ctx.kw)``, where
            ``ctx.kw`` holds ``terminating``, ``seed``, ``time_scale`` and ``token_roles``.
        config_cls: The model's architecture dataclass (fields read by ``build``); it should
            subclass :class:`~torchpp.training.arch.IntensityArchConfig` (intensity-based models)
            or :class:`~torchpp.training.arch.ArchConfig`. Default: the matching base, i.e. only
            the shared fields.
        description: One-line description.
        run_name: ``run_name(arch) -> str`` for default save paths. Default: ``name``.
        experimental: Mark the model as experimental.
        aliases: Further names accepted for this ``model_type``.

    Example:
        >>> @register_model("my_poisson", description="Poisson with a custom prior")
        ... class MyPoisson(PoissonProcess):
        ...     pass
    """
    from torchpp.tpp.base_tpp import TemporalPointProcess

    def register(model_cls):
        if not (isinstance(model_cls, type) and issubclass(model_cls, TemporalPointProcess)):
            raise TypeError(f"register_model expects a TemporalPointProcess subclass, got {model_cls!r}.")
        for key in (name, *aliases):
            if key in MODEL_REGISTRY or key in MODEL_ALIASES:
                raise ValueError(f"model_type '{key}' is already registered.")
        from torchpp.tpp.base_tpp import IntensityBasedTPP

        builder = build if build is not None else (lambda arch, c: model_cls(c.D, **c.kw))
        default_cfg = A.IntensityArchConfig if issubclass(model_cls, IntensityBasedTPP) else A.ArchConfig
        MODEL_REGISTRY[name] = ModelSpec(
            model_cls,
            description or model_cls.__name__,
            config_cls if config_cls is not None else default_cfg,
            builder,
            run_name if run_name is not None else (lambda arch: name),
            experimental=experimental,
        )
        VALID_MODEL_TYPES.append(name)
        for alias in aliases:
            MODEL_ALIASES[alias] = name
        return model_cls

    return register(cls) if cls is not None else register
