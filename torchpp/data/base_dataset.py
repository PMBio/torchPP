"""Abstract base class for TPP datasets."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, List

from torch.utils.data import Dataset

from ..utils.event_data import MVEventData


class TPPDataset(Dataset, ABC):
    """Abstract base class for temporal point process datasets.

    Subclasses must implement :meth:`__len__`, :meth:`__getitem__`, and
    :meth:`load`, and expose two dataset-level properties:

    * ``num_event_types`` — the number of distinct event types ``D``.
    * ``time_scale_factor`` — converts the model's internal scaled time back
      to the natural unit (e.g. days): ``t_natural = t_scaled * factor``.

    Two class-level attributes encode dataset-intrinsic behaviour that would
    otherwise be duplicated in per-dataset config classes:

    * ``terminating`` — whether each event type can occur at most once.
      Override in subclasses (e.g. ``terminating = True`` for UKB).
    * ``default_ignore_indices`` — event-type indices excluded from evaluation
      metrics by default.  Override in subclasses where some indices are not
      valid prediction targets (e.g. demographic tokens in UKB).

    Both can be overridden per-run via the ``dataset.ignore_indices`` YAML/CLI
    key; the effective value is accessible through the ``ignore_indices``
    property.
    """

    # ------------------------------------------------------------------ #
    # Class-level dataset-intrinsic attributes (override in subclasses)   #
    # ------------------------------------------------------------------ #

    terminating: ClassVar[bool] = False
    """Whether each event type can occur at most once per sequence."""

    default_ignore_indices: ClassVar[List[int]] = []
    """Event-type indices excluded from evaluation by default."""

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def num_event_types(self) -> int:
        """Number of distinct event types ``D`` in this dataset."""
        ...

    @property
    @abstractmethod
    def time_scale_factor(self) -> float:
        """Conversion factor from the model's scaled time unit to the natural
        time unit (e.g. days).

        ``t_natural = t_scaled * time_scale_factor``
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, cfg: dict, root: Path, split: str) -> "TPPDataset":
        """Construct a dataset instance from a config dict and a split name.

        Called by :func:`~torchpp.training.dataset_registry.load_split`.
        Subclasses receive the operational parameters from the ``dataset:``
        YAML block (minus the shared keys ``ignore_indices``,
        ``limit_train_size``, ``limit_val_size`` which are handled by the
        registry before this method is called).

        Args:
            cfg:   Flat dict of dataset-specific parameters (e.g. ``data_dir``,
                   ``dataset_name``).  Shared keys have been stripped.
            root:  Repository root; use to resolve relative paths in *cfg*.
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
    # ignore_indices — default or per-run override                        #
    # ------------------------------------------------------------------ #

    @property
    def ignore_indices(self) -> List[int]:
        """Effective list of event-type indices to exclude from evaluation.

        Returns the per-instance override (set by
        :func:`~torchpp.training.dataset_registry.load_split` when
        ``dataset.ignore_indices`` is present in the config) if one has been
        applied, otherwise falls back to the class-level
        :attr:`default_ignore_indices`.
        """
        return getattr(self, "_ignore_indices_override", self.__class__.default_ignore_indices)
