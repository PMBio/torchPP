"""Seeded parameter initialisation that leaves the global RNG untouched."""

import threading
from contextlib import contextmanager
from typing import Iterator, Optional

import torch

_STATE = threading.local()


@contextmanager
def seeded_init(seed: Optional[int]) -> Iterator[None]:
    """Seed the CPU RNG (and the current CUDA device's RNG) inside the block, then restore them.

    Covers parameters created on the CPU and, under ``torch.set_default_device("cuda")``,
    on the GPU. ``seed=None`` uses the global RNG as is. Nested blocks keep the outermost
    seed, so a model built inside :func:`create_model` is seeded once.
    """
    if seed is None or getattr(_STATE, "active", False):
        yield
        return
    on_cuda = torch.cuda.is_available() and (torch.cuda.is_initialized() or torch.get_default_device().type == "cuda")
    devices = [torch.cuda.current_device()] if on_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.default_generator.manual_seed(int(seed))
        for d in devices:
            torch.cuda.default_generators[d].manual_seed(int(seed))
        _STATE.active = True
        try:
            yield
        finally:
            _STATE.active = False


def init_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """A CPU generator seeded with ``seed``, or ``None`` (the global RNG)."""
    return None if seed is None else torch.Generator().manual_seed(int(seed))
