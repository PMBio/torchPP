"""Per-model architecture configs.

Every ``model_type`` declares its own architecture dataclass in the registry
(:attr:`~torchpp.training.registry.ModelSpec.config_cls`). The fields shared by all
models sit at the top level (``seed``, ``sampling`` and, for intensity-based models,
``ci``); the components a model is built from have their own sections::

    arch:
      seed: 0
      baseline: {num_knots: 20, T_factor: 1.0}          # spline / piecewise baselines
      kernel:   {num_knots: 8, T_factor: 5.0}           # spline / piecewise / MLP kernels
      encoder:  {n_layers: 2, n_heads: 2, n_embd: 32}   # neural models
      head:     {num_bins: 10}                           # neural intensity heads

A field the selected model does not have raises a ``ValueError`` that lists the valid ones.
"""

from __future__ import annotations

import dataclasses
import difflib
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


# ---------------------------------------------------------------------------
# Component sections
# ---------------------------------------------------------------------------


@dataclass
class BaselineKnotsConfig:
    """Spline or piecewise-constant baseline."""

    num_knots: int = 5
    T_factor: float = 1.0
    """Extent = ``T_factor × sequence_time_p95`` of the training data (absolute extent
    without dataset statistics). Knots are uniform."""


@dataclass
class KernelKnotsConfig:
    """Spline or piecewise-constant kernel over the time since the triggering event."""

    num_knots: int = 4
    T_factor: float = 5.0
    """Extent = ``T_factor × inter_event_time_p95`` of the training data (absolute extent
    without dataset statistics)."""
    knot_schedule: str = "geometric"
    """``uniform`` (equidistant over [0, T]) or ``geometric`` (x_k = T·(k/(K-1))^p, denser near 0)."""
    knot_power: float = 2.0
    """Exponent p of the geometric schedule."""


@dataclass
class MLPKernelConfig:
    """MLP kernel over [type embedding; time embedding of the lag]."""

    emb_dim: int = 32
    time_emb_dim: int = 32
    n_hidden: int = 64
    n_layers: int = 2


@dataclass
class MLPSplineKernelConfig(KernelKnotsConfig):
    """Spline kernel whose knot heights an MLP predicts from the type pair."""

    emb_dim: int = 32
    n_hidden: int = 64
    n_layers: int = 2


@dataclass
class TransformerEncoderConfig:
    """The shared causal transformer encoder."""

    n_layers: int = 4
    n_heads: int = 4
    n_embd: int = 64
    dropout: float = 0.1


@dataclass
class IntensityFreeEncoderConfig(TransformerEncoderConfig):
    """History encoder of IntensityFree; ``n_heads`` applies to the transformer only."""

    type: str = "gru"
    """``gru`` (as in Shchur et al.) or ``transformer`` (the shared TorchPP encoder)."""


@dataclass
class CTLSTMEncoderConfig:
    """Continuous-time LSTM encoder of NHP."""

    hidden_dim: int = 64


@dataclass
class AttNHPEncoderConfig(TransformerEncoderConfig):
    """Attention encoder of AttNHP; ``n_embd`` is its model width."""

    time_emb_dim: int = 16
    """Size of the sinusoidal temporal embedding."""
    use_layer_norm: bool = False
    """Apply LayerNorm after each attention layer."""


@dataclass
class BinsHeadConfig:
    """Knots (or bins) of a neural head over the time since the last event."""

    num_bins: int = 4
    T_factor: float = 5.0
    """Extent = ``T_factor × inter_event_time_p95`` of the training data."""
    knot_schedule: str = "geometric"
    knot_power: float = 2.0


@dataclass
class SplineHeadConfig(BinsHeadConfig):
    """Linear-spline head of ``neural_spline``."""

    smoothness_reg: float = 0.0
    """Weight of the penalty on second differences of the knot heights; 0 disables it."""


@dataclass
class MLPHeadConfig:
    """MLP head of ``neural_mlp``."""

    n_hidden: int = 128
    n_layers: int = 2
    time_emb_dim: Optional[int] = None
    """Time-embedding size; None → the encoder's ``n_embd``."""


@dataclass
class MixtureHeadConfig:
    """Log-normal mixture head of IntensityFree."""

    num_mix_components: int = 8


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------


@dataclass
class ArchConfig:
    """Fields every model has. Also the config of models without further settings."""

    seed: Optional[int] = None
    """Seed for parameter initialisation. None → randomised at startup by train.py."""
    sampling: Dict[str, Any] = field(default_factory=dict)
    """Overrides of the model's ``SamplingConfig`` fields, e.g. ``{mode: thinning}``."""


@dataclass
class IntensityArchConfig(ArchConfig):
    """Fields of every intensity-based model."""

    ci: Dict[str, Any] = field(default_factory=dict)
    """Overrides of the model's ``CIConfig`` fields, e.g. ``{n_points: 200, rule: trapezoidal}``.
    Models with a numerical cumulative intensity use ``n_points: 100`` unless set here."""


@dataclass
class SplineBaselineArchConfig(IntensityArchConfig):
    baseline: BaselineKnotsConfig = field(default_factory=BaselineKnotsConfig)


@dataclass
class SplineKernelArchConfig(IntensityArchConfig):
    kernel: KernelKnotsConfig = field(default_factory=KernelKnotsConfig)


@dataclass
class SplineBaselineSplineKernelArchConfig(IntensityArchConfig):
    baseline: BaselineKnotsConfig = field(default_factory=BaselineKnotsConfig)
    kernel: KernelKnotsConfig = field(default_factory=KernelKnotsConfig)


@dataclass
class InhibitiveArchConfig(IntensityArchConfig):
    clamping_fn: str = "softplus"
    """Positivity map applied to baseline + kernel sum: ``softplus``, ``relu`` or ``exp``."""


@dataclass
class InhibitiveSplineBaselineArchConfig(InhibitiveArchConfig):
    baseline: BaselineKnotsConfig = field(default_factory=BaselineKnotsConfig)


@dataclass
class InhibitiveSplineSplineArchConfig(InhibitiveArchConfig):
    baseline: BaselineKnotsConfig = field(default_factory=BaselineKnotsConfig)
    kernel: KernelKnotsConfig = field(default_factory=KernelKnotsConfig)


@dataclass
class InhibitiveMLPKernelArchConfig(InhibitiveArchConfig):
    baseline: BaselineKnotsConfig = field(default_factory=BaselineKnotsConfig)
    kernel: MLPKernelConfig = field(default_factory=MLPKernelConfig)


@dataclass
class InhibitiveMLPSplineKernelArchConfig(InhibitiveArchConfig):
    baseline: BaselineKnotsConfig = field(default_factory=BaselineKnotsConfig)
    kernel: MLPSplineKernelConfig = field(default_factory=MLPSplineKernelConfig)


@dataclass
class NeuralArchConfig(IntensityArchConfig):
    """Transformer encoder with a head that has no settings."""

    encoder: TransformerEncoderConfig = field(default_factory=TransformerEncoderConfig)


@dataclass
class NeuralSplineArchConfig(NeuralArchConfig):
    head: SplineHeadConfig = field(default_factory=SplineHeadConfig)


@dataclass
class NeuralPiecewiseArchConfig(NeuralArchConfig):
    head: BinsHeadConfig = field(default_factory=BinsHeadConfig)


@dataclass
class NeuralMLPArchConfig(NeuralArchConfig):
    head: MLPHeadConfig = field(default_factory=MLPHeadConfig)


@dataclass
class IntensityFreeArchConfig(ArchConfig):
    encoder: IntensityFreeEncoderConfig = field(default_factory=IntensityFreeEncoderConfig)
    head: MixtureHeadConfig = field(default_factory=MixtureHeadConfig)


@dataclass
class NHPArchConfig(IntensityArchConfig):
    encoder: CTLSTMEncoderConfig = field(default_factory=CTLSTMEncoderConfig)


@dataclass
class AttNHPArchConfig(IntensityArchConfig):
    encoder: AttNHPEncoderConfig = field(default_factory=AttNHPEncoderConfig)


# ---------------------------------------------------------------------------
# Building and validating
# ---------------------------------------------------------------------------


def field_names(cls: type, prefix: str = "") -> list[str]:
    """Dotted names of every leaf field of an arch config class."""
    hints = typing.get_type_hints(cls)
    names = []
    for f in dataclasses.fields(cls):
        if dataclasses.is_dataclass(hints[f.name]):
            names.extend(field_names(hints[f.name], f"{prefix}{f.name}."))
        else:
            names.append(f"{prefix}{f.name}")
    return names


def _section_build(cls: type, values: dict, path: str, model_type: str) -> Any:
    from torchpp.training.config_utils import ConfigError, _check_type

    hints = typing.get_type_hints(cls)
    known = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in values.items():
        if key not in known:
            dotted = f"{path}{key}"
            valid = field_names(cls, path)
            close = difflib.get_close_matches(dotted, valid, n=1, cutoff=0.75)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise ValueError(
                f"model_type '{model_type}' has no architecture field '{dotted}'.{hint} "
                f"Valid fields: {', '.join(valid)}."
            )
        ann = hints[key]
        if dataclasses.is_dataclass(ann):
            if dataclasses.is_dataclass(value):
                value = dataclasses.asdict(value)
            if not isinstance(value, dict):
                raise ValueError(f"'{path}{key}' must be a mapping of {ann.__name__} fields, got {value!r}.")
            kwargs[key] = _section_build(ann, value, f"{path}{key}.", model_type)
        else:
            try:
                _check_type(value, ann, f"{path}{key}")
            except ConfigError as exc:
                raise ValueError(str(exc)) from exc
            kwargs[key] = dict(value) if isinstance(value, dict) else value
    return cls(**kwargs)


def build_arch(model_type: str, values: Union[None, dict, ArchConfig] = None, prefix: str = "arch.") -> ArchConfig:
    """The architecture config of ``model_type`` from a (nested) dict, a config, or None (defaults).

    Raises:
        ValueError: A field the model does not have, or a value of the wrong type.
    """
    from torchpp.training.registry import MODEL_REGISTRY, resolve_model_type

    model_type = resolve_model_type(model_type)
    cls = MODEL_REGISTRY[model_type].config_cls
    if values is None:
        return cls()
    if dataclasses.is_dataclass(values):
        if type(values) is cls:
            return values
        values = dataclasses.asdict(values)
    return _section_build(cls, dict(values), prefix, model_type)


def merge_arch(model_type: str, arch: ArchConfig, overrides: dict) -> ArchConfig:
    """``arch`` with (nested) ``overrides`` applied, validated like :func:`build_arch`."""

    def merge(base: dict, new: dict) -> dict:
        out = dict(base)
        for key, value in new.items():
            out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
        return out

    return build_arch(model_type, merge(dataclasses.asdict(arch), overrides))


def arch_dict(arch: ArchConfig) -> dict:
    """Plain nested dict of an arch config (for checkpoints and printing)."""
    return dataclasses.asdict(arch)
