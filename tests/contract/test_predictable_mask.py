"""Sequence-representation role mask: context types are excluded from the
training likelihood, and the default all-True mask is numerically inert.

The mask is a ``(D,)`` bool buffer on every model (``True`` = predictable: scored
and integrated; ``False`` = context/read-only).  It enters three sites of
the training objective — the positive log-intensity term, the survival Σ_d Λ_d, and
the empty-sequence gate — all no-ops when the mask is all-True (see the Gate A
bit-identity harness).  These tests exercise the *active* (non-trivial) mask.
"""

import pytest
import torch
from torchpp.utils.event_data import TokenRoles

from helpers import create_model, D, SEED, DEVICE
from torchpp.utils.event_data import BatchedMVEventData
from torchpp.training.model_factory import create_model as factory_create_model

# Intensity-family models that route through IntensityBasedTPP.likelihood — cheap,
# closed-form CI, deterministic (no dropout).
_INTENSITY_MODELS = ["poisson", "const_exp_hawkes"]

_CTX = 1  # the context/read-only type used throughout


def _mask_off(model, k):
    """Mark type ``k`` as context (non-predictable) on ``model``."""
    mask = torch.ones(model.D, dtype=torch.bool, device=model.predictable_mask.device)
    mask[k] = False
    model.predictable_mask.copy_(mask)


def _mixed_batch():
    """One sequence with type-``_CTX`` events interleaved with predictable types."""
    times = torch.tensor([0.2, 0.7, 1.1, 1.6, 2.2], dtype=torch.float64, device=DEVICE)
    types = torch.tensor([0, _CTX, 2, _CTX, 0], dtype=torch.long, device=DEVICE)
    return BatchedMVEventData([times], [types], T_obs=[3.0]).to(DEVICE)


def _paired_models(model_type):
    """A (full, context-masked) pair sharing identical parameters."""
    full = create_model(model_type, seed=SEED)
    masked = create_model(model_type, seed=SEED)
    masked.load_state_dict(full.state_dict())
    _mask_off(masked, _CTX)
    full.eval()
    masked.eval()
    return full, masked


# The clean positive/survival decomposition below assumes masking type K leaves every
# other term unchanged.  That holds only for kernel-free intensities (poisson): the kernel exclusion
# makes the Hawkes kernel role-aware, so masking a context type ALSO strips its
# excitation from the *predictable* events' intensities and survival columns — the
# extra term the decomposition does not model.
_DECOMPOSABLE_MODELS = ["poisson"]


@pytest.mark.parametrize("model_type", _DECOMPOSABLE_MODELS)
def test_context_type_excluded_from_positive_and_survival(model_type):
    """(a) type-K events add 0 to the positive term; (b) type K drops out of Σ_d Λ_d.

    With identical parameters, masking type K must lower the likelihood by exactly the
    type-K positive contribution and raise it by exactly the type-K survival column:
        L_masked = L_full − Σ_{i: m_i=K} log λ_K(t_i) + Λ_K(T).

    Restricted to kernel-free intensities — see the note above ``_DECOMPOSABLE_MODELS``.
    """
    full, masked = _paired_models(model_type)
    batch = _mixed_batch()
    T = batch.T_obs

    with torch.no_grad():
        L_full = full.likelihood(batch, T=T, log=True)  # (1,)
        L_masked = masked.likelihood(batch, T=T, log=True)  # (1,)

        # (a) positive-term contribution of type-K events, via the shared per-event hook
        lam = full._event_intensities(batch)  # (1, L)
        log_lam = torch.log(lam.clamp(min=torch.finfo(lam.dtype).tiny))
        is_k = (batch.event_types == _CTX) & batch.event_mask  # (1, L)
        type_k_pos = log_lam.masked_fill(~is_k, 0.0).sum(dim=1)  # (1,)

        # (b) survival contribution of type-K, via the per-type cumulative intensity.
        # the survival window is [t₁, T] (no origin → condition-on-first), so the
        # type-K column must be integrated over the same window as the likelihood uses.
        T_low = batch.time_points[:, 0]  # first event t₁
        lambda_k = full.cumulative_intensity(T, batch, T_low=T_low)[:, _CTX]  # (1,)

    assert torch.allclose(L_masked, L_full - type_k_pos + lambda_k, atol=1e-6, rtol=1e-6)
    # guard the test is non-trivial: type K genuinely occurred and integrated to > 0
    assert (type_k_pos.abs() > 0).all() and (lambda_k > 0).all()


@pytest.mark.parametrize("model_type", _INTENSITY_MODELS)
def test_context_only_sequence_empty_gate(model_type):
    """(c) a sequence of only context events has no scorable events.

    The positive term is exactly 0 (empty-seq gate ignores type K) and the full
    likelihood reduces to the survival over predictable types only.
    """
    _, masked = _paired_models(model_type)
    times = torch.tensor([0.3, 0.9, 1.4], dtype=torch.float64, device=DEVICE)
    types = torch.tensor([_CTX, _CTX, _CTX], dtype=torch.long, device=DEVICE)
    batch = BatchedMVEventData([times], [types], T_obs=[2.0]).to(DEVICE)
    T = batch.T_obs

    with torch.no_grad():
        pos = masked._event_log_intensity_sum(batch)  # (1,)
        L = masked.likelihood(batch, T=T, log=True)  # (1,)
        # survival window [t₁, T] (no origin → condition on the first event, even a
        # context one); integrate the predictable columns over that same window.
        T_low = batch.time_points[:, 0]  # first event t₁
        lam_all = masked.cumulative_intensity(T, batch, T_low=T_low)  # (1, D)
        predictable = [d for d in range(D) if d != _CTX]
        expected = -lam_all[:, predictable].sum(dim=1)  # (1,)

    assert torch.equal(pos, torch.zeros_like(pos))  # gate fires: no scorable events
    assert torch.isfinite(L).all()
    assert torch.allclose(L, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("model_type", ["poisson", "const_exp_hawkes", "neural_poisson", "intensity_free"])
def test_empty_context_reproduces_full_likelihood(model_type):
    """Positive control: an explicit all-True mask reproduces the full-D likelihood."""
    m1 = create_model(model_type, seed=SEED)  # default all-True mask
    m2 = create_model(model_type, seed=SEED)
    m2.load_state_dict(m1.state_dict())
    m2.predictable_mask.copy_(torch.ones(m2.D, dtype=torch.bool, device=m2.predictable_mask.device))
    m1.eval()
    m2.eval()
    batch = _mixed_batch()
    with torch.no_grad():
        assert torch.equal(m1.likelihood(batch, log=True), m2.likelihood(batch, log=True))


def test_context_types_derive_predictable_mask():
    """create_model derives ``predictable_mask`` from ``context_types``: the masked-off
    (non-predictable) set equals the declared context types; empty context ⇒ all-True."""
    arch = None

    model = factory_create_model("poisson", D=3, arch=arch, terminating=False, seed=0, token_roles=TokenRoles(context_types=[1]))
    non_predictable = torch.nonzero(~model.predictable_mask, as_tuple=False).flatten().tolist()
    assert non_predictable == [1]

    # Empty / None context ⇒ all-True mask (inert).
    m0 = factory_create_model("poisson", D=3, arch=arch, terminating=False, seed=0)
    assert bool(m0.predictable_mask.all())


def _role_stub(cls, D):
    """A data-free instance of dataset class ``cls`` for exercising the role-partition
    properties.  ``token_roles`` reads only the class-level ``default_token_roles``, so
    bypassing ``__init__`` (which would need on-disk data) is sufficient and faithful."""
    stub = object.__new__(cls)
    stub._num_event_types = D
    return stub


def test_role_partition_invariant_across_registered_datasets():
    """The role-partition invariant, checked on every *registered* dataset class.

    ``<bos>`` is no longer a data-level token (it is an encoder-internal device), so the
    role lists contain only real event-type indices.  The single-source-of-truth
    invariant is ``context_types ⊆ ignored_types`` — context is always eval-ignored.

    The invariant is NOT ``context_types == ignored_types``: a dataset may eval-ignore
    tokens that are still modelled (e.g. covariates), an intended strict subset.
    """
    import torchpp.data  # noqa: F401 — importing registers the built-in datasets
    from torchpp.training.dataset_registry import DATASET_REGISTRY

    assert DATASET_REGISTRY, "no datasets registered"
    Dtypes = 32  # arbitrary vocab size for the role stub

    for name, spec in sorted(DATASET_REGISTRY.items()):
        cls = spec.loader_class
        stub = _role_stub(cls, Dtypes)
        roles = stub.token_roles
        ctx, ign = set(roles.context_types), set(roles.ignored_types)
        assert ctx <= ign, f"{name}: context_types {ctx} must be a subset of ignored_types {ign}"
