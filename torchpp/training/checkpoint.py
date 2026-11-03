"""Self-describing model checkpoints.

A checkpoint written by :func:`save_model` stores the weights, the model class with the
arguments it was constructed with, its integration / sampling configs and, when built by
:func:`~torchpp.training.model_factory.create_model`, the ``model_type`` and architecture.
:func:`load_model` rebuilds the model from the file alone.

Files that contain only a ``state_dict`` (``torch.save(model.state_dict(), path)``) are
still accepted by ``create_model(load_path=...)``; they need ``model_type`` and the
architecture to be supplied separately.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Optional, Union

import torch

from torchpp.training.arch import ArchConfig, build_arch, merge_arch
from torchpp.utils.init_record import resolve_ref, decode_init, encode_init

CHECKPOINT_FORMAT = 2


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def _torchpp_version() -> str:
    import torchpp

    return torchpp.__version__


def _registered_type_of(model: torch.nn.Module) -> Optional[str]:
    """The registered ``model_type`` whose class is exactly ``type(model)``, if unique."""
    from torchpp.training.registry import MODEL_REGISTRY

    matches = [name for name, spec in MODEL_REGISTRY.items() if spec.cls is type(model)]
    return matches[0] if len(matches) == 1 else None


def save_model(
    path: Union[str, Path],
    model: torch.nn.Module,
    *,
    model_type: Optional[str] = None,
    D: Optional[int] = None,
    arch: Optional[ArchConfig | dict] = None,
    seed: Optional[int] = None,
    dataset_type: Optional[str] = None,
    dataset: Optional[dict] = None,
) -> Path:
    """Write ``model`` and everything needed to rebuild it to ``path``; return the path.

    The model's recorded constructor arguments and its integration / sampling configs are
    stored, so :func:`load_model` rebuilds any model exactly, however it was constructed.
    ``model_type``, ``D`` and ``arch`` default to what :func:`create_model` recorded (``D`` to
    ``model.D``); given explicitly, they are checked against the model.
    """
    from torchpp.training.registry import resolve_model_type

    model = _unwrap(model)
    if getattr(model, "_init_args", None) is None:
        raise ValueError(f"{type(model).__name__} does not record its constructor arguments and cannot be saved.")
    inferred = getattr(model, "model_type", None) or _registered_type_of(model)
    if model_type is None:
        model_type = inferred
    elif inferred is not None and resolve_model_type(model_type) != resolve_model_type(inferred):
        raise ValueError(f"model_type={model_type!r} does not match the model ({inferred!r}).")
    if D is None:
        D = model.D
    elif int(D) != int(model.D):
        raise ValueError(f"D={D} does not match the model (D={model.D}).")
    recorded_arch = getattr(model, "arch_config", None)
    if isinstance(arch, dict):
        if model_type is None:
            raise ValueError("arch given as a dict needs a model_type.")
        arch = build_arch(model_type, arch)
    if arch is None:
        arch = recorded_arch
    elif recorded_arch is not None and dataclasses.asdict(arch) != dataclasses.asdict(recorded_arch):
        raise ValueError("arch does not match the architecture the model was built with.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "state_dict": model.state_dict(),
        "class": f"{type(model).__module__}:{type(model).__qualname__}",
        "init_args": encode_init(model._init_args),
        "ci_config": encode_init(getattr(model, "_ci_config_override", None)),
        "sampling_config": encode_init(getattr(model, "_sampling_config_override", None)),
        "model_type": model_type,
        "arch": dataclasses.asdict(arch) if arch is not None else None,
        "run_name": getattr(model, "run_name", None),
        "D": int(D),
        "terminating": bool(getattr(model, "terminating", False)),
        "token_roles": encode_init(model.token_roles),
        "time_scale": float(getattr(model, "time_scale", 1.0)),
        "seed": seed,
        "dataset_type": dataset_type,
        "dataset": dict(dataset) if dataset is not None else None,
        "torchpp_version": _torchpp_version(),
    }
    torch.save(payload, str(path))
    return path


def _token_roles(ckpt: dict[str, Any]):
    """The checkpoint's :class:`TokenRoles`."""
    from torchpp.utils.event_data import TokenRoles

    if ckpt.get("token_roles") is not None:
        return decode_init(ckpt["token_roles"])
    return TokenRoles(has_origin=bool(ckpt.get("has_origin", False)))


def read_checkpoint(path: Union[str, Path]) -> dict[str, Any]:
    """Load a checkpoint file. A bare ``state_dict`` is returned as ``{"state_dict": ...}``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj and ("class" in obj or "model_type" in obj):
        return obj
    return {"state_dict": obj}


def arch_from_dict(model_type: str, values: Optional[dict]) -> ArchConfig:
    """The arch config of ``model_type`` from a stored (nested) dict."""
    try:
        return build_arch(model_type, values or None)
    except ValueError as exc:
        raise ValueError(f"The checkpoint's architecture does not match this TorchPP version: {exc}") from exc


def _rebuild(ckpt: dict[str, Any]):
    """Construct the model class of ``ckpt`` from its recorded arguments and configs."""
    from torchpp.tpp.base_tpp import TemporalPointProcess
    from torchpp.training.registry import MODEL_REGISTRY

    registered = [spec.cls for spec in MODEL_REGISTRY.values() if f"{spec.cls.__module__}:{spec.cls.__qualname__}" == ckpt["class"]]
    try:
        cls = registered[0] if registered else resolve_ref(ckpt["class"])
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Cannot rebuild the saved model: {exc}") from exc
    if not (isinstance(cls, type) and issubclass(cls, TemporalPointProcess)):
        raise ValueError(f"{ckpt['class']} is not a TorchPP model class.")
    model = cls(**decode_init(ckpt["init_args"]))
    if ckpt.get("ci_config") is not None:
        model.ci_config = decode_init(ckpt["ci_config"])
    if ckpt.get("sampling_config") is not None:
        model.sampling_config = decode_init(ckpt["sampling_config"])
    return model


def load_model(path: Union[str, Path], device: Union[str, torch.device] = "cpu", **arch_overrides: Any):
    """Rebuild the model saved by :func:`save_model` and load its weights.

    Args:
        path: Checkpoint file.
        device: Device to move the model to.
        **arch_overrides: Architecture fields to change at load time, as top-level fields
            or nested sections, e.g. ``ci={"n_points": 500}`` for a finer evaluation
            quadrature. The model is then rebuilt through :func:`create_model`, which needs
            a registered ``model_type`` and a recorded architecture.

    Returns:
        The model in eval mode.
    """
    from torchpp.training.model_factory import create_model

    ckpt = read_checkpoint(path)
    if "class" not in ckpt and "model_type" not in ckpt:
        raise ValueError(
            f"{path} holds only a state_dict; rebuild the model with "
            "create_model(model_type, D, arch=..., load_path=...) instead."
        )
    if "class" in ckpt and not arch_overrides:
        model = _rebuild(ckpt)
        model.load_state_dict(ckpt["state_dict"])
        for name in ("model_type", "run_name"):
            if ckpt.get(name) is not None:
                setattr(model, name, ckpt[name])
        if ckpt.get("arch") is not None and ckpt.get("model_type") is not None:
            model.arch_config = arch_from_dict(ckpt["model_type"], ckpt["arch"])
        return model.to(device).eval()
    if ckpt.get("model_type") is None or ckpt.get("arch") is None:
        raise ValueError(
            f"{path} was saved from a model built without create_model; arch overrides "
            "need a registered model_type and a recorded architecture."
        )
    arch = merge_arch(ckpt["model_type"], arch_from_dict(ckpt["model_type"], ckpt["arch"]), arch_overrides)
    model = create_model(
        ckpt["model_type"],
        ckpt["D"],
        arch=arch,
        terminating=ckpt["terminating"],
        token_roles=_token_roles(ckpt),
        load_path=path,
    )
    return model.to(device).eval()
