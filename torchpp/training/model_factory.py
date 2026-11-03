"""Single authoritative entry point for instantiating any supported TPP model
from a ``model_type`` string and an architecture config."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch

from torchpp.tpp.base_tpp import CIConfig, IntensityBasedTPP, SamplingConfig, TemporalPointProcess
from torchpp.training.arch import ArchConfig, build_arch
from torchpp.training.registry import MODEL_REGISTRY, BuildContext, resolve_model_type
from torchpp.utils.event_data import DatasetStats, TokenRoles
from torchpp.utils.seeding import seeded_init

NUMERICAL_CI_POINTS = 100
"""Quadrature points per interval of models with a numerical cumulative intensity, unless
``arch.ci.n_points`` sets them."""


def create_model(
    model_type: str,
    D: int,
    *,
    arch: Optional[ArchConfig | dict] = None,
    terminating: bool = False,
    seed: Optional[int] = None,
    load_path: Optional[Path] = None,
    dataset_stats: Optional[DatasetStats] = None,
    token_roles: Optional[TokenRoles] = None,
    verbose: bool = False,
) -> TemporalPointProcess:
    """Instantiate a TPP model by name.

    Args:
        model_type:     One of the strings in ``VALID_MODEL_TYPES``.
        D:              Number of event types (dimension).
        arch:           Architecture of this model type (see :mod:`torchpp.training.arch`):
                        its config dataclass or a nested dict of its fields, e.g.
                        ``{"baseline": {"num_knots": 20}}``. Fields the model does not have
                        raise. ``None`` uses the defaults.
        terminating:    Whether each event type can occur at most once.
        seed:           Seed for parameter initialisation; the global RNG is left untouched.
                        ``None`` draws from the global RNG (set it with ``torch.manual_seed``).
                        The value is irrelevant when loading weights from a checkpoint.
        load_path:      If given, load the weights from this file after construction:
                        a :func:`~torchpp.training.checkpoint.save_model` checkpoint or a
                        bare ``state_dict``.
        dataset_stats:  Optional empirical statistics computed from the training
                        sequences.  When provided, Hawkes kernel/baseline modules
                        use data-driven initialisation (beta calibrated to observed
                        inter-event times, alpha seeded from pair-occurrence counts).
                        Pass ``None`` to use the default ranges.
        token_roles:    :class:`~torchpp.utils.event_data.TokenRoles` of the data
                        (origin regime, context and ignored types), e.g.
                        ``dataset.token_roles``. ``None``: no origin, every type predictable.
        verbose:        Print what is built and loaded.

    Returns:
        The constructed model. ``model.run_name`` is a descriptive name encoding the
        model type and architecture (e.g. ``"spline_K5_exp_hawkes"``), used as the
        default save filename and W&B run name.

    Raises:
        ValueError: For unknown ``model_type``.
    """
    model_type = resolve_model_type(model_type)
    arch = build_arch(model_type, arch)
    spec = MODEL_REGISTRY[model_type]

    run_name = spec.run_name(arch)
    if verbose:
        print(f"Creating {spec.description} (run_name={run_name})...")

    # ── Time-unit scale ──────────────────────────────
    # The model computes in scaled units t/s, where s = the training-set mean inter-event
    # time (native units). Data-driven initialisation statistics are rescaled into the same
    # units, so knots and baselines land where the scaled core evaluates them. Without
    # statistics s = 1 and the T_factor fields are absolute extents.
    time_scale = float(dataset_stats.mean_inter_event_time) if dataset_stats is not None else 1.0
    scaled_stats = dataset_stats.rescale_time(1.0 / time_scale) if dataset_stats is not None else None

    kw: dict = dict(terminating=terminating)
    if seed is not None:
        kw["seed"] = seed
    kw["time_scale"] = time_scale
    kw["token_roles"] = token_roles if token_roles is not None else TokenRoles()

    # Encoders built outside the model class draw from the same seeded RNG; the global RNG is untouched.
    with seeded_init(seed):
        model = spec.build(arch, BuildContext(D=D, stats=scaled_stats, kw=kw))
    _apply_scheme_overrides(model, arch, model_type)

    # The time-unit scale and the token roles are set at construction; a checkpoint carrying
    # its own ``time_scale`` / ``predictable_mask`` overrides them on load.

    if load_path is not None:
        from torchpp.training.checkpoint import read_checkpoint

        if verbose:
            print(f"Loading weights from {load_path}...")
        ckpt = read_checkpoint(load_path)  # accepts save_model() files and bare state_dicts
        if "model_type" in ckpt and resolve_model_type(ckpt["model_type"]) != model_type:
            raise ValueError(f"{load_path} holds a '{ckpt['model_type']}' model, not '{model_type}'.")
        state = ckpt["state_dict"]
        if "time_scale" not in state:
            raise ValueError(
                f"{load_path} stores no time_scale, so the time units of the saved model are unknown. "
                "It was saved by an incompatible TorchPP version."
            )
        if "predictable_mask" not in state:
            # Checkpoint without a stored mask: keep the mask stamped above (all-True unless overridden)
            # so the load does not fail on the missing buffer.
            state["predictable_mask"] = model.predictable_mask
        model.load_state_dict(state)

    model.run_name = run_name
    model.model_type = model_type  # canonical name and architecture, read by save_model
    model.arch_config = arch
    return model


def _config_overrides(cfg_cls, values: dict, where: str) -> dict:
    """Validate a dict of ``cfg_cls`` field overrides; unknown keys raise with a suggestion."""
    import dataclasses
    import difflib

    known = [f.name for f in dataclasses.fields(cfg_cls)]
    for key in values:
        if key not in known:
            close = difflib.get_close_matches(str(key), known, n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise ValueError(f"Unknown {where} field '{key}'.{hint}")
    return dict(values)


def _apply_scheme_overrides(model: TemporalPointProcess, arch: ArchConfig, model_type: str) -> None:
    """Apply ``arch.ci`` and ``arch.sampling`` to a built model."""
    ci = _config_overrides(CIConfig, getattr(arch, "ci", None) or {}, "arch.ci")
    if isinstance(model, IntensityBasedTPP):
        updates = {"n_points": NUMERICAL_CI_POINTS} if model.ci_config.mode == "numerical" else {}
        updates.update(ci)
        if updates:
            model.ci_config = replace(model.ci_config, **updates)
    elif ci:
        raise ValueError(f"model_type '{model_type}' has no cumulative intensity; arch.ci does not apply.")
    sampling = _config_overrides(SamplingConfig, arch.sampling or {}, "arch.sampling")
    if sampling:
        model.sampling_config = replace(model.sampling_config, **sampling)
