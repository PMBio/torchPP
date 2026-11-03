"""Abstract base class for TPP datasets."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from torch.utils.data import Dataset

from ..utils.event_data import MVEventData, TokenRoles


class TPPDataset(Dataset, ABC):
    """Abstract base class for temporal point process datasets.

    Subclasses implement :meth:`__len__`, :meth:`__getitem__` (returning
    :class:`MVEventData` with raw native-unit times), :meth:`load`, and the
    ``num_event_types`` property (the number of event types ``D``).

    Class attributes describe the data and are passed to the model:

    * ``terminating`` — whether each event type occurs at most once per sequence.
    * ``default_token_roles`` — the data's :class:`~torchpp.utils.event_data.TokenRoles`:
      whether time 0 is a meaningful origin, which types are context (read, never
      predicted) and which are ignored by the evaluation.
    * ``cfg_keys`` — accepted ``dataset.*`` config keys (empty accepts any key).

    ``dataset.context_types`` and ``dataset.ignored_types`` override the default roles per
    run; the effective roles are the ``token_roles`` property. See ``docs/notation.md``
    (Token roles) for the definitions.
    """

    # ------------------------------------------------------------------ #
    # Class-level dataset-intrinsic attributes (override in subclasses)   #
    # ------------------------------------------------------------------ #

    terminating: ClassVar[bool] = False
    """Whether each event type can occur at most once per sequence."""

    default_token_roles: ClassVar[TokenRoles] = TokenRoles()
    """Origin regime, context types and ignored types of the data."""

    cfg_keys: ClassVar[frozenset] = frozenset()
    """Accepted ``dataset.*`` config keys; unknown keys raise. Empty accepts any key."""

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def num_event_types(self) -> int:
        """Number of distinct event types ``D`` in this dataset."""
        ...

    @property
    def time_scale_factor(self) -> float:
        """Always ``1.0``: datasets emit native-unit times and the model owns time
        normalisation (``model.time_scale``)."""
        return 1.0

    @classmethod
    @abstractmethod
    def load(cls, cfg: dict, root: Path, split: str) -> "TPPDataset":
        """Construct a dataset instance from a config dict and a split name.

        Called by :func:`~torchpp.training.dataset_registry.load_split`.
        Subclasses receive the ``dataset:`` config block without the shared keys
        ``context_types`` and ``ignored_types``, which the registry handles.

        Args:
            cfg:   Flat dict of dataset-specific parameters (e.g. ``data_dir``,
                   ``dataset_name``).  Shared keys have been stripped.
            root:  Data root (``$TORCHPPDATA``); resolve relative paths in *cfg* against it.
            split: One of ``"train"``, ``"val"``, ``"test"``.

        Returns:
            A dataset instance for the requested split.
        """
        ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> MVEventData: ...

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # ------------------------------------------------------------------ #
    # Token roles — class default plus per-run overrides                  #
    # ------------------------------------------------------------------ #

    @property
    def token_roles(self) -> TokenRoles:
        """Effective :class:`TokenRoles`: :attr:`default_token_roles` with any per-run
        ``dataset.context_types`` / ``dataset.ignored_types`` override applied by
        :func:`~torchpp.training.dataset_registry.load_split`."""
        return getattr(self, "_token_roles_override", None) or self.__class__.default_token_roles

    @token_roles.setter
    def token_roles(self, roles: TokenRoles) -> None:
        self._token_roles_override = roles


class LimitedDataset(TPPDataset):
    """The first ``n`` sequences of another dataset; every other attribute is forwarded.

    Used by :func:`~torchpp.training.dataset_registry.load_split` for the ``limit_*_size``
    keys of datasets whose loader does not apply them itself.
    """

    def __init__(self, inner: TPPDataset, n: int) -> None:
        self._inner = inner
        self._n = min(int(n), len(inner))
        self.terminating = inner.terminating
        self.token_roles = inner.token_roles

    @property
    def num_event_types(self) -> int:
        return self._inner.num_event_types

    @classmethod
    def load(cls, cfg: dict, root: Path, split: str) -> "TPPDataset":
        raise TypeError("LimitedDataset wraps a loaded dataset; it is not registered.")

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> MVEventData:
        if not -self._n <= idx < self._n:
            raise IndexError(idx)
        return self._inner[idx % self._n]

    def __getattr__(self, name: str):
        if name.startswith("__") or name in ("_inner", "_n"):
            raise AttributeError(name)
        return getattr(self._inner, name)
