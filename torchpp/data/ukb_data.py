import numpy as np
from pathlib import Path
from typing import ClassVar, List, Optional
import torch
from torch.utils.data import Dataset
from ..utils.event_data import MVEventData
from .base_dataset import TPPDataset
from ..training.dataset_registry import register_dataset


# Original token IDs in the .bin file (before any remapping):
#   0 = padding (never in file), 1 = no-event (never in file)
#   2 = female, 3 = male
#   4 = BMI_low, 5 = BMI_mid, 6 = BMI_high
#   7 = Smoking_low, 8 = Smoking_mid, 9 = Smoking_high
#   10 = Alcohol_low, 11 = Alcohol_mid, 12 = Alcohol_high
#   13–1268 = disease ICD-10 codes
#   1269 = death
LIFESTYLE_TOKENS_ORIGINAL = set(range(4, 13))  # tokens 4–12 in the raw .bin file
DEATH_ORIGINAL = 1269
SEX_FEMALE_ORIGINAL = 2
SEX_MALE_ORIGINAL = 3


# After -2 remap (what the TPP model sees):
# female: 2 - 2 = 0
# male:   3 - 2 = 1
SEX_FEMALE_REMAPPED = 0  # originally 2
SEX_MALE_REMAPPED = 1    # originally 3
LIFESTYLE_TOKENS_REMAPPED = set(range(2, 11))  # originally 4–12, after -2 → 2–10
DEATH_REMAPPED = 1267  # originally 1269, after -2 remap

# UKB time scaling: raw times in days are converted to a normalised unit where
# 1.0 ≈ 80 years.  This factor converts scaled time back to days.
UKB_TIME_SCALE_FACTOR: float = 365.25 * 80  # ~29 220 days


def load_ukb_sequences(
    data_path: str | Path,
    limit_size: Optional[int] = None,
    keep_sex_token: bool = False,
    keep_lifestyle_covariates: bool = False,
):
    """Load UKB sequences from a ``.bin`` file and return them as a list.

    .. deprecated::
        This function is a thin compatibility wrapper around :class:`UKBDataset`.
        Prefer constructing a :class:`UKBDataset` directly; it provides lazy,
        memory-efficient access without loading all sequences into a Python list.

    Returns:
        sequences: list of :class:`~torchpp.utils.event_data.MVEventData`
        sexes: list of remapped sex tokens (int) for each sequence
        num_event_types: total number of distinct event types
    """
    dataset = UKBDataset(
        data_path,
        limit_size=limit_size,
        keep_sex_token=keep_sex_token,
        keep_lifestyle_covariates=keep_lifestyle_covariates,
    )
    sequences = list(dataset)
    # Derive the sex token for each sequence from the sex covariate: 1.0 → male (0), 0.0 → female (1).
    sexes = [
        torch.tensor([SEX_MALE_REMAPPED if seq.covariates["sex"].item() == 1.0 else SEX_FEMALE_REMAPPED])
        for seq in sequences
    ]
    return sequences, sexes, dataset.num_event_types


@register_dataset("ukb")
class UKBDataset(TPPDataset):
    """A ``torch.utils.data.Dataset`` over a UKB `.bin` event file.

    Each item is one :class:`~torchpp.utils.event_data.MVEventData` sequence,
    identical to what :func:`load_ukb_sequences` would produce for that
    sequence.  The raw ``numpy.memmap`` is kept open for the lifetime of the
    dataset; parsing and tensor construction happen lazily on ``__getitem__``,
    so the full data is never copied into memory as a Python list.

    The number of valid (non-empty, correctly-sexed) sequences is exposed via
    ``__len__`` after a lightweight scan of the file on construction.

    Args:
        data_path: Path to the ``.bin`` file produced by the UKB pipeline.
        limit_size: If given, only the first ``limit_size`` valid sequences
            are retained.
        keep_sex_token: If ``True``, the sex token is left in the event
            sequence instead of being extracted as a covariate.
        keep_lifestyle_covariates: If ``True``, BMI/Smoking/Alcohol tokens
            are kept in the event sequence instead of being extracted as a
            covariate vector.
        time_scale_factor: Conversion factor from raw days to the model's
            internal scaled time unit (``t_scaled = t_days / time_scale_factor``).
            Also used by callers such as :func:`~torchpp.utils.evaluation.evaluate_tpp`
            to convert scaled time back to days for reporting metrics.
            Defaults to :data:`UKB_TIME_SCALE_FACTOR` (``365.25 * 80``).
    """

    # Dataset-intrinsic properties — not user-configurable per run.
    terminating: ClassVar[bool] = True
    """UKB is first-occurrence data: each disease code appears at most once."""

    default_ignore_indices: ClassVar[List[int]] = list(range(11))
    """Tokens 0–10 are sex and lifestyle covariates, not disease codes."""

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        data_path: str | Path,
        limit_size: Optional[int] = None,
        keep_sex_token: bool = False,
        keep_lifestyle_covariates: bool = False,
        time_scale_factor: float = UKB_TIME_SCALE_FACTOR,
    ) -> None:
        data_path = Path(data_path)
        assert data_path.exists(), f"Data file does not exist: {data_path}"

        self._keep_sex_token = keep_sex_token
        self._keep_lifestyle_covariates = keep_lifestyle_covariates
        self._time_scale_factor: float = time_scale_factor

        # Open the file as a memory-mapped array; stays open for random access.
        np_data = np.memmap(str(data_path), mode="r", dtype=np.uint32).reshape(-1, 3)
        self._np_data = np_data  # kept alive on the instance

        self._female_token = SEX_FEMALE_REMAPPED  # 0
        self._male_token = SEX_MALE_REMAPPED  # 1

        self._num_event_types: int = DEATH_REMAPPED + 1  # = 1268

        # ---- Index the file: find per-sequence row slices ---- #
        batch_ids = np_data[:, 0].astype(np.int64)
        change_idcs = np.flatnonzero(batch_ids[1:] != batch_ids[:-1]) + 1
        change_idcs = np.concatenate([[0], change_idcs, [len(batch_ids)]])

        # Build list of (start, stop) row ranges and filter to valid sequences.
        valid_slices: list[tuple[int, int]] = []
        for start, stop in zip(change_idcs[:-1], change_idcs[1:]):
            # Peek at the first event type (after remap) to check sex token
            first_type = int(np_data[start, 2]) - 2  # remap: subtract 2
            if first_type == self._male_token or first_type == self._female_token:
                valid_slices.append((int(start), int(stop)))
            if limit_size is not None and len(valid_slices) >= limit_size:
                break

        self._slices: list[tuple[int, int]] = valid_slices

    # ------------------------------------------------------------------ #
    # Registry factory                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, cfg: dict, root: Path, split: str) -> "UKBDataset":
        """Construct a :class:`UKBDataset` from a flat config dict.

        Called by :func:`~torchpp.training.dataset_registry.load_split`.

        Expected cfg keys (all optional except ``data_dir``):

        * ``data_dir`` — directory containing the split ``.bin`` files.
        * ``train_file`` / ``val_file`` / ``test_file`` — filenames within
          *data_dir*.  Defaults: ``train.bin``, ``val.bin``, ``test.bin``.
        * ``limit_train_size`` / ``limit_val_size`` / ``limit_test_size`` —
          restrict the number of sequences loaded for the respective split.
        * ``keep_sex_token`` — if ``True``, the sex token stays in the event
          sequence rather than being extracted as a covariate.
          Default: ``False``.
        * ``keep_lifestyle_covariates`` — if ``True``, BMI/Smoking/Alcohol
          tokens stay in the event sequence rather than being extracted as a
          covariate vector.  Default: ``False``.
        * ``time_scale_factor`` — override the day-to-scaled-time conversion
          factor.  Default: :data:`UKB_TIME_SCALE_FACTOR` (``365.25 * 80``).
        """
        data_dir = cfg.get("data_dir", "")
        if not data_dir:
            raise ValueError(
                "dataset.data_dir is not set for dataset_type='ukb'.  "
                "Provide it via a data config preset or as a CLI override "
                "(dataset.data_dir=<path>)."
            )

        split_file_map = {
            "train": cfg.get("train_file", "train.bin"),
            "val": cfg.get("val_file", "val.bin"),
            "test": cfg.get("test_file", "test.bin"),
        }
        if split not in split_file_map:
            raise ValueError(f"Unknown split '{split}' for UKBDataset.  Use 'train', 'val', or 'test'.")

        limit_key = "limit_val_size" if split == "val" else f"limit_{split}_size"
        limit_size: Optional[int] = cfg.get(limit_key, None)

        keep_sex_token: bool = cfg.get("keep_sex_token", False)
        keep_lifestyle_covariates: bool = cfg.get("keep_lifestyle_covariates", False)
        time_scale_factor: float = cfg.get("time_scale_factor", UKB_TIME_SCALE_FACTOR)

        data_path = root / data_dir / split_file_map[split]
        return cls(
            data_path=data_path,
            limit_size=limit_size,
            keep_sex_token=keep_sex_token,
            keep_lifestyle_covariates=keep_lifestyle_covariates,
            time_scale_factor=time_scale_factor,
        )

    # ------------------------------------------------------------------ #
    # TPPDataset properties                                                #
    # ------------------------------------------------------------------ #

    @property
    def num_event_types(self) -> int:  # type: ignore[override]
        """Number of distinct event types (D) in this dataset."""
        return self._num_event_types

    @property
    def time_scale_factor(self) -> float:  # type: ignore[override]
        """Conversion factor from raw days to the model's scaled time unit.

        ``t_scaled = t_days / time_scale_factor``
        """
        return self._time_scale_factor

    # ------------------------------------------------------------------ #
    # Dataset protocol                                                     #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._slices)

    def __getitem__(self, idx: int) -> MVEventData:
        """Return the ``idx``-th valid sequence as an :class:`MVEventData`.

        Processing mirrors :func:`load_ukb_sequences` exactly:

        * Token IDs are shifted by ``-2`` so that disease types start at 0.
        * Times are converted to scaled years (``days / 365.25 / 80``).
        * The sex token is extracted as a ``sex`` covariate (1-D float tensor).
                * Lifestyle tokens (BMI / Smoking / Alcohol, remapped indices 2–10)
                    are encoded as a 9-D float covariate vector. The vector is always
                    present so every sequence shares the same covariate schema; when no
                    lifestyle token is observed it is all zeros. Tokens are removed from
                    the event sequence unless ``keep_lifestyle_covariates=True``.
        """
        start, stop = self._slices[idx]
        raw = self._np_data[start:stop]  # shape (L, 3)

        # Column layout: [batch_id, time_days, token_id]
        times_days = raw[:, 1].astype(np.float64)
        event_types = raw[:, 2].astype(np.int64) - 2  # remap

        # Scale times: days → scaled years using the dataset's time_scale_factor.
        # Done in float64 before casting to float32 for numerical precision.
        times_scaled = times_days / self._time_scale_factor

        # ---- Extract sex covariate from the raw first token ---- #
        # The sex token is always the first record in the raw file for a valid
        # sequence (guaranteed by the index scan in __init__).
        sex_token = int(event_types[0])
        sex_value = 1.0 if sex_token == self._male_token else 0.0
        sex_covariate = torch.tensor([sex_value], dtype=torch.float32)

        if self._keep_sex_token:
            remaining_times = torch.tensor(times_scaled, dtype=torch.float32)
            remaining_events = torch.tensor(event_types, dtype=torch.long)
        else:
            remaining_times = torch.tensor(times_scaled[1:], dtype=torch.float32)
            remaining_events = torch.tensor(event_types[1:], dtype=torch.long)

        # ---- Handle lifestyle tokens ---- #
        lifestyle_mask = torch.tensor(
            [int(e) in LIFESTYLE_TOKENS_REMAPPED for e in remaining_events.tolist()],
            dtype=torch.bool,
        )

        lifestyle_vec = torch.zeros(9, dtype=torch.float32)
        if lifestyle_mask.any():
            lifestyle_event_types = remaining_events[lifestyle_mask]
            for lt in lifestyle_event_types.tolist():
                lifestyle_vec[lt - 2] = 1.0  # offset: remapped lifestyle tokens start at 2

        covariates: dict[str, torch.Tensor] = {
            "sex": sex_covariate,
            "lifestyle": lifestyle_vec,
        }

        if self._keep_lifestyle_covariates:
            disease_times = remaining_times
            disease_events = remaining_events
        else:
            disease_mask = ~lifestyle_mask
            disease_times = remaining_times[disease_mask]
            disease_events = remaining_events[disease_mask]

        # Skip empty sequences (can arise after filtering) — return an empty item.
        # Callers that use BucketBatchSampler / collate_batch should handle this
        # gracefully, but in practice the dataset only contains non-empty sequences
        # after the index scan already verified the first token is a sex token.
        return MVEventData(disease_times, disease_events, covariates=covariates, sort=True)
