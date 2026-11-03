"""Constructor-argument recording, so that a model can be rebuilt from a checkpoint.

Models, encoders and intensity modules record the arguments passed to their outermost
``__init__`` (``self._init_args``). :func:`encode_init` turns them into data that
``torch.load(weights_only=True)`` accepts — primitives, tensors, dataclasses and
references to recorded components — and :func:`decode_init` rebuilds them.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
import sys
from typing import Any, Callable

import torch

_TAG = "__torchpp__"


def record_init(init: Callable) -> Callable:
    """Wrap ``__init__`` so the outermost call stores its explicit arguments on ``self``."""
    if getattr(init, "_tpp_records_init", False):
        return init
    sig = inspect.signature(init)

    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        outermost = "_init_args" not in self.__dict__
        if outermost:
            self.__dict__["_init_args"] = None  # claim; inner super().__init__ calls do not record
        init(self, *args, **kwargs)
        if outermost:
            bound = sig.bind(self, *args, **kwargs).arguments
            bound.pop(next(iter(sig.parameters)))  # self
            for p in sig.parameters.values():
                if p.kind is p.VAR_KEYWORD and p.name in bound:
                    bound.update(bound.pop(p.name))
                elif p.kind is p.VAR_POSITIONAL and bound.get(p.name):
                    raise TypeError(f"{type(self).__name__}.__init__ takes *{p.name}; it cannot be recorded.")
            self.__dict__["_init_args"] = dict(bound)

    wrapper._tpp_records_init = True
    return wrapper


def records_init(cls) -> None:
    """Make ``cls`` (and, via ``__init_subclass__``, its subclasses) record constructor args."""
    init = cls.__dict__.get("__init__")
    if init is not None:
        cls.__init__ = record_init(init)
    cls._tpp_recorded = True


def _ref(obj) -> str:
    return f"{obj.__module__}:{obj.__qualname__}"


def resolve_ref(ref: str):
    """Look up a ``module:qualname`` reference without importing modules outside TorchPP."""
    module, _, qualname = ref.partition(":")
    if module in sys.modules:
        mod = sys.modules[module]
    elif module == "torchpp" or module.startswith("torchpp."):
        mod = importlib.import_module(module)
    else:
        raise ValueError(f"The checkpoint refers to {ref}; import {module} before loading it.")
    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def encode_init(value: Any) -> Any:
    """Recorded constructor arguments → ``weights_only``-safe data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (list, tuple)):
        return {_TAG: "tuple" if isinstance(value, tuple) else "list", "items": [encode_init(v) for v in value]}
    if isinstance(value, dict):
        return {_TAG: "dict", "items": {str(k): encode_init(v) for k, v in value.items()}}
    if isinstance(value, torch.nn.Module) and getattr(value, "_tpp_recorded", False):
        return {_TAG: "component", "cls": _ref(type(value)), "args": encode_init(value._init_args or {})}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {f.name: encode_init(getattr(value, f.name)) for f in dataclasses.fields(value)}
        return {_TAG: "dataclass", "cls": _ref(type(value)), "fields": fields}
    if callable(value) and getattr(value, "__module__", "").split(".")[0] in ("torch", "torchpp"):
        return {_TAG: "callable", "ref": _ref(value)}
    raise TypeError(f"Cannot store a constructor argument of type {type(value).__name__} in a checkpoint.")


def decode_init(value: Any) -> Any:
    """Inverse of :func:`encode_init`."""
    if not isinstance(value, dict) or _TAG not in value:
        return value
    kind = value[_TAG]
    if kind in ("list", "tuple"):
        items = [decode_init(v) for v in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "dict":
        return {k: decode_init(v) for k, v in value["items"].items()}
    if kind == "component":
        cls = resolve_ref(value["cls"])
        if not getattr(cls, "_tpp_recorded", False):
            raise ValueError(f"{value['cls']} is not a TorchPP component that can be rebuilt.")
        return cls(**decode_init(value["args"]))
    if kind == "dataclass":
        cls = resolve_ref(value["cls"])
        if not dataclasses.is_dataclass(cls):
            raise ValueError(f"{value['cls']} is not a dataclass.")
        return cls(**{k: decode_init(v) for k, v in value["fields"].items()})
    if kind == "callable":
        return resolve_ref(value["ref"])
    raise ValueError(f"Unknown checkpoint entry {kind!r}.")
