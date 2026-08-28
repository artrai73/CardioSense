"""PyTorch ``Dataset`` and ``DataLoader`` for PTB-XL waveforms.

The dataset reads from the memory-mapped waveform cache built in
:mod:`cardiosense.ecg.preprocessing`, so ``__getitem__`` is a slice out of a
contiguous file rather than a WFDB parse. That is what makes an epoch take
seconds instead of minutes.

Colab-specific DataLoader notes
-------------------------------

* ``num_workers=2`` — the free tier gives 2 vCPUs. Raising this past the core
  count makes workers contend and *slows* training.
* ``persistent_workers=True`` — worker startup costs a second or two; without
  this it is paid at the start of every epoch.
* ``pin_memory=True`` — only helps with a CUDA device; it is switched off
  automatically on CPU, where it would just waste memory.
* ``worker_init_fn=seed_worker`` — without it every worker inherits the same
  NumPy seed and any NumPy-based augmentation repeats identically across workers.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..common.config import Config
from ..common.logging_utils import get_logger
from ..common.seeding import get_generator, seed_worker

__all__ = ["ECGDataset", "build_dataloader", "build_dataloaders", "compute_pos_weight"]

logger = get_logger("ecg.dataset")


class ECGDataset(Dataset):
    """12-lead ECG waveforms with multi-hot diagnostic superclass labels.

    Args:
        waveforms: Array of shape ``(n_total, n_leads, n_samples)``, normally a
            memory-mapped ``.npy``.
        labels: Multi-hot matrix of shape ``(n_total, n_classes)``.
        indices: Row positions belonging to this split. Keeping the full array and
            indexing into it means the three splits share one memory map rather
            than holding three copies.
        class_names: Names in column order, carried for readability.
        augment: Apply light training-time augmentation. Off for val/test.
        augment_config: Augmentation parameters (see :meth:`_augment`).
    """

    def __init__(
        self,
        waveforms: np.ndarray,
        labels: np.ndarray,
        indices: Sequence[int] | np.ndarray,
        class_names: Sequence[str] = (),
        augment: bool = False,
        augment_config: dict[str, Any] | None = None,
    ) -> None:
        self.waveforms = waveforms
        self.labels = np.asarray(labels, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.class_names = list(class_names)
        self.augment = augment
        self.augment_config = augment_config or {}

        if self.labels.shape[0] != waveforms.shape[0]:
            raise ValueError(
                f"waveforms has {waveforms.shape[0]} rows but labels has "
                f"{self.labels.shape[0]}; they must be aligned row-for-row."
            )
        if self.indices.max(initial=-1) >= waveforms.shape[0]:
            raise IndexError("Split indices point past the end of the waveform array.")

    def __len__(self) -> int:
        return int(len(self.indices))

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Light, physiologically defensible augmentation.

        * **Amplitude scaling** (±10%) simulates electrode-contact and gain
          variation between recordings.
        * **Random time shift** (circular, up to 10% of the record) makes the model
          insensitive to where in the 10-second window a beat happens to fall.

        Deliberately absent: lead permutation (which would imply the leads are
        interchangeable — they are not, each views the heart from a fixed angle)
        and sign inversion (which would turn a normal QRS into a pathological one).
        """
        scale = self.augment_config.get("amplitude_scale", 0.1)
        if scale:
            x = x * np.float32(1.0 + np.random.uniform(-scale, scale))

        shift_fraction = self.augment_config.get("time_shift", 0.1)
        if shift_fraction:
            max_shift = int(shift_fraction * x.shape[-1])
            if max_shift > 0:
                x = np.roll(x, np.random.randint(-max_shift, max_shift + 1), axis=-1)
        return x

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.indices[position])
        # np.array(copy=True) rather than np.asarray: indexing a memory-mapped
        # array returns a READ-ONLY view. torch.from_numpy on a read-only buffer
        # warns and produces a tensor PyTorch considers unsafe to write, and
        # handing a mmap view to a worker process is undefined behaviour. The
        # copy is per-record and cheap (48 KB at 100 Hz).
        waveform = np.array(self.waveforms[row], dtype=np.float32, copy=True)
        if self.augment:
            waveform = self._augment(waveform)
        return (
            torch.from_numpy(np.ascontiguousarray(waveform)),
            torch.from_numpy(np.array(self.labels[row], copy=True)),
        )

    def label_matrix(self) -> np.ndarray:
        """The label matrix for this split only, in split order."""
        return self.labels[self.indices]

    def summary(self) -> dict[str, Any]:
        labels = self.label_matrix()
        return {
            "n_records": len(self),
            "augment": self.augment,
            "support": {name: int(labels[:, i].sum())
                        for i, name in enumerate(self.class_names)},
            "prevalence": {name: round(float(labels[:, i].mean()), 4)
                           for i, name in enumerate(self.class_names)},
        }


def build_dataloader(
    dataset: ECGDataset,
    cfg: Config,
    shuffle: bool,
    batch_size: int | None = None,
) -> DataLoader:
    """Wrap an :class:`ECGDataset` in a DataLoader with Colab-appropriate settings."""
    training = cfg.training
    workers = int(training.get("num_workers", 2))
    use_cuda = torch.cuda.is_available()

    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size or training.batch_size),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": bool(training.get("pin_memory", True)) and use_cuda,
        "drop_last": False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(training.get("persistent_workers", True))
        kwargs["worker_init_fn"] = seed_worker
        kwargs["prefetch_factor"] = 2
    if shuffle:
        kwargs["generator"] = get_generator(int(cfg.seed))

    return DataLoader(dataset, **kwargs)


def build_dataloaders(
    waveforms: np.ndarray,
    labels: np.ndarray,
    splits: dict[str, dict[str, Any]],
    cfg: Config,
) -> dict[str, DataLoader]:
    """Build train / val / test loaders from the fold split.

    Augmentation is enabled for **train only**. Applying random transforms to
    validation makes the monitored metric noisy and the early-stopping decision
    unreliable; applying them to test would make the reported result irreproducible.
    """
    classes = list(cfg.task.classes)
    augment_config = dict(cfg.training.get("augment", {}) or {})
    use_augment = bool(cfg.training.get("augment_enabled", True))

    loaders: dict[str, DataLoader] = {}
    for name in ("train", "val", "test"):
        dataset = ECGDataset(
            waveforms, labels, splits[name]["indices"],
            class_names=classes,
            augment=(name == "train" and use_augment),
            augment_config=augment_config,
        )
        loaders[name] = build_dataloader(dataset, cfg, shuffle=(name == "train"))
        logger.info("%-5s loader: %d records, %d batches, augment=%s",
                    name, len(dataset), len(loaders[name]), dataset.augment)
    return loaders


def compute_pos_weight(labels: np.ndarray, cap: float = 20.0) -> torch.Tensor:
    """Per-class ``pos_weight`` for ``BCEWithLogitsLoss``.

    ``pos_weight[c] = n_negative[c] / n_positive[c]``, which scales up the loss
    contribution of positive examples for rare classes. In PTB-XL, HYP appears in
    about 12% of records against NORM's 44%, so without this the model can
    minimise loss by rarely predicting HYP at all.

    The weight is capped: an unbounded ratio on a very rare class produces huge
    gradients and unstable early training.

    Args:
        labels: Multi-hot label matrix for the **training split only**.
        cap: Maximum weight.

    Returns:
        A ``float32`` tensor of shape ``(n_classes,)``.
    """
    labels = np.asarray(labels, dtype=np.float32)
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    weights = np.where(positives > 0, negatives / np.maximum(positives, 1.0), 1.0)
    weights = np.clip(weights, 0.0, cap)
    logger.info("pos_weight per class: %s", np.round(weights, 3).tolist())
    return torch.tensor(weights, dtype=torch.float32)
