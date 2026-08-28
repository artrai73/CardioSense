"""PyTorch ``Dataset`` and ``DataLoader`` for chest X-ray images.

Unlike the ECG pipeline, images are read from disk on demand rather than cached
into one array: 10,000 PNGs at 1024x1024 would be ~10 GB as float32, which does
not fit in a Colab session. Decoding is the bottleneck instead of I/O, which is
what ``num_workers`` exists for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..common.config import Config
from ..common.logging_utils import get_logger
from ..common.seeding import get_generator, seed_worker

__all__ = ["ChestXrayDataset", "build_dataloader", "build_dataloaders",
           "compute_pos_weight", "build_weighted_sampler"]

logger = get_logger("xray.dataset")


class ChestXrayDataset(Dataset):
    """Chest X-ray images with a binary target.

    Args:
        frame: Split metadata with the image column and a ``target`` column.
        images_dir: Directory holding the flat PNG files.
        transform: Torchvision transform to apply.
        image_column: Column holding the filename.
        return_index: Also return the row position, which the error-analysis and
            Grad-CAM code uses to trace a prediction back to its file.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        images_dir: Path | str,
        transform: Any,
        image_column: str = "Image Index",
        return_index: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.image_column = image_column
        self.return_index = return_index

        if "target" not in self.frame.columns:
            raise KeyError("frame must contain a 'target' column; call build_target first.")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.images_dir}")

        self.filenames = self.frame[image_column].to_numpy()
        self.targets = self.frame["target"].to_numpy().astype(np.float32)

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[Any, ...]:
        from PIL import Image

        path = self.images_dir / str(self.filenames[index])
        try:
            # Convert to L first: a handful of ChestX-ray14 PNGs are stored with
            # 4 channels, which would otherwise reach the transform as RGBA and
            # break the Grayscale step.
            with Image.open(path) as handle:
                image = handle.convert("L")
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Could not read image {path}: {exc}. If the download was interrupted, "
                "re-run the extraction step in 00_colab_setup.ipynb."
            ) from exc

        tensor = self.transform(image)
        target = torch.tensor([self.targets[index]], dtype=torch.float32)

        if self.return_index:
            return tensor, target, index
        return tensor, target

    def summary(self) -> dict[str, Any]:
        return {
            "n_images": len(self),
            "n_positive": int(self.targets.sum()),
            "prevalence": round(float(self.targets.mean()), 5),
        }


def compute_pos_weight(targets: np.ndarray, cap: float = 50.0) -> torch.Tensor:
    """``pos_weight`` for ``BCEWithLogitsLoss``: ``n_negative / n_positive``.

    At ~2.5% prevalence this is around 39, meaning each positive image contributes
    as much to the loss as 39 negatives. Without it, the loss is minimised by
    predicting "no cardiomegaly" for everything.

    The cap prevents an enormous weight on a split that happens to contain very
    few positives, which would produce unstable early gradients.

    Args:
        targets: Binary targets from the **training split only**.
        cap: Maximum weight.
    """
    targets = np.asarray(targets, dtype=np.float32).ravel()
    n_positive = float(targets.sum())
    n_negative = float(len(targets) - n_positive)
    if n_positive == 0:
        logger.warning("No positives in the training split; pos_weight set to 1.")
        return torch.tensor([1.0], dtype=torch.float32)

    weight = min(n_negative / n_positive, cap)
    logger.info("pos_weight = %.2f (%d negative / %d positive)",
                weight, int(n_negative), int(n_positive))
    return torch.tensor([weight], dtype=torch.float32)


def build_weighted_sampler(targets: np.ndarray) -> Any:
    """A ``WeightedRandomSampler`` that balances classes within each batch.

    Implemented but **disabled by default**. It is an alternative to
    ``pos_weight``, not a companion: stacking both applies the imbalance
    correction twice, which produces wildly over-confident probabilities. Since
    Phase 2 fuses on confidence, that would be actively harmful. See
    ``class_imbalance.method`` in the config.
    """
    from torch.utils.data import WeightedRandomSampler

    targets = np.asarray(targets, dtype=np.int64).ravel()
    class_counts = np.bincount(targets, minlength=2).astype(float)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    sample_weights = class_weights[targets]

    logger.info("WeightedRandomSampler: class counts %s -> weights %s",
                class_counts.tolist(), np.round(class_weights, 6).tolist())
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(targets),
        replacement=True,
    )


def build_dataloader(
    dataset: ChestXrayDataset,
    cfg: Config,
    shuffle: bool,
    sampler: Any = None,
    batch_size: int | None = None,
) -> DataLoader:
    """Wrap a dataset in a DataLoader with Colab-appropriate settings."""
    training = cfg.training
    workers = int(training.get("num_workers", 2))
    use_cuda = torch.cuda.is_available()

    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size or training.batch_size),
        "num_workers": workers,
        "pin_memory": bool(training.get("pin_memory", True)) and use_cuda,
        "drop_last": False,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler          # sampler and shuffle are mutually exclusive
    else:
        kwargs["shuffle"] = shuffle
        if shuffle:
            kwargs["generator"] = get_generator(int(cfg.seed))

    if workers > 0:
        kwargs["persistent_workers"] = bool(training.get("persistent_workers", True))
        kwargs["worker_init_fn"] = seed_worker
        kwargs["prefetch_factor"] = 2

    return DataLoader(dataset, **kwargs)


def build_dataloaders(
    splits: dict[str, pd.DataFrame],
    images_dir: Path | str,
    transforms: dict[str, Any],
    cfg: Config,
) -> dict[str, DataLoader]:
    """Build train / val / test loaders.

    Augmentation applies to train only, enforced by passing the deterministic
    transform for val and test.
    """
    image_column = str(cfg.dataset.image_column)
    method = str(cfg.class_imbalance.get("method", "pos_weight")).lower()

    loaders: dict[str, DataLoader] = {}
    for name in ("train", "val", "test"):
        dataset = ChestXrayDataset(
            splits[name], images_dir, transforms[name],
            image_column=image_column, return_index=(name != "train"),
        )
        sampler = None
        if name == "train" and method == "weighted_sampler":
            sampler = build_weighted_sampler(dataset.targets)

        loaders[name] = build_dataloader(dataset, cfg, shuffle=(name == "train"),
                                         sampler=sampler)
        logger.info("%-5s loader: %d images, %d batches, prevalence %.4f",
                    name, len(dataset), len(loaders[name]),
                    dataset.summary()["prevalence"])
    return loaders
