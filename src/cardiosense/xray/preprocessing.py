"""Image preprocessing and augmentation for chest X-rays.

Two transform pipelines, and the difference between them is the point:

* **Training** — resize, light augmentation, normalise.
* **Validation / test** — resize, normalise. **No randomness whatsoever.**

Random augmentation on validation makes the monitored metric noisy, which makes
early stopping pick a checkpoint by luck. Random augmentation on test makes the
reported result irreproducible. Both are silent failures, so the two pipelines are
built by separate functions and the eval one has no random component to disable.

Why these augmentations and not the usual defaults
---------------------------------------------------

**Horizontal flip is OFF, deliberately.** It is the default in almost every
ImageNet recipe, and it is wrong here. Mirroring a chest X-ray moves the cardiac
silhouette to the right side of the thorax. That is *dextrocardia* — a real and
rare congenital condition. Training with random horizontal flips teaches the model
that a right-sided heart is a normal variant, which is both anatomically false and
destroys the left-right asymmetry that a cardiomegaly assessment depends on.

**Vertical flip is OFF.** An upside-down chest film does not occur.

**Rotation is limited to ±7°.** Real films vary by patient positioning and
detector alignment, and that variation is small. Large rotations would generate
images no radiographer would ever produce.

**Translation is limited to 5%.** Same reasoning: framing varies a little.

**Brightness and contrast jitter of ±10%** covers exposure and detector
differences between machines, which is genuine nuisance variation in a
multi-site dataset.

**RandomResizedCrop is limited to a 0.9–1.0 scale range.** The default 0.08–1.0
would routinely crop the heart out of the image entirely and label the result
"cardiomegaly".

Normalisation uses **ImageNet statistics**, not dataset statistics, because the
DenseNet121 weights were fitted under exactly those statistics. Recomputing them
from chest X-rays would shift the input distribution away from what the pretrained
filters expect.
"""

from __future__ import annotations

from typing import Any

from ..common.config import Config
from ..common.logging_utils import get_logger

__all__ = ["build_train_transform", "build_eval_transform", "build_transforms",
           "denormalize"]

logger = get_logger("xray.preprocessing")


def build_eval_transform(cfg: Config) -> Any:
    """Deterministic transform for validation, test and inference.

    Resize -> tensor -> normalise. Nothing random. The same function is used at
    inference time, so what is evaluated is exactly what is deployed.
    """
    from torchvision import transforms

    size = int(cfg.preprocessing.image_size)
    steps: list[Any] = [transforms.Resize((size, size))]
    if bool(cfg.preprocessing.get("to_rgb", True)):
        # Chest X-rays are single-channel; DenseNet121 expects three. Replicating
        # the grey channel is the standard adaptation and keeps the pretrained
        # first-layer filters meaningful.
        steps.append(transforms.Grayscale(num_output_channels=3))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(
        mean=list(cfg.preprocessing.normalize_mean),
        std=list(cfg.preprocessing.normalize_std),
    ))
    return transforms.Compose(steps)


def build_train_transform(cfg: Config) -> Any:
    """Augmented transform for training only.

    See the module docstring for why horizontal flip is absent. If augmentation is
    disabled in config, this returns the deterministic evaluation transform.
    """
    from torchvision import transforms

    augment = cfg.augmentation
    if not bool(augment.get("enabled", True)):
        logger.info("Augmentation disabled; training will use the evaluation transform.")
        return build_eval_transform(cfg)

    size = int(cfg.preprocessing.image_size)
    scale = tuple(augment.get("random_resized_crop_scale", [0.9, 1.0]))
    degrees = float(augment.get("rotation_degrees", 7))
    translate = tuple(augment.get("translate", [0.05, 0.05]))
    brightness = float(augment.get("brightness", 0.10))
    contrast = float(augment.get("contrast", 0.10))

    steps: list[Any] = [
        transforms.RandomResizedCrop(size, scale=scale, ratio=(0.95, 1.05)),
        transforms.RandomAffine(degrees=degrees, translate=translate),
    ]
    if brightness or contrast:
        steps.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))

    # Guard rails: these must stay off for chest radiographs. If someone enables
    # them in config, the code refuses rather than silently doing the wrong thing.
    if bool(augment.get("horizontal_flip", False)):
        raise ValueError(
            "horizontal_flip is enabled in configs/xray_config.yaml. Mirroring a chest "
            "X-ray moves the heart to the right side of the thorax (dextrocardia) and "
            "destroys the left-right asymmetry cardiomegaly assessment depends on. "
            "If you genuinely intend this, remove the guard in "
            "xray/preprocessing.py and document why in the report."
        )
    if bool(augment.get("vertical_flip", False)):
        raise ValueError("vertical_flip is enabled. An upside-down chest film does not occur.")

    if bool(cfg.preprocessing.get("to_rgb", True)):
        steps.append(transforms.Grayscale(num_output_channels=3))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(
        mean=list(cfg.preprocessing.normalize_mean),
        std=list(cfg.preprocessing.normalize_std),
    ))

    logger.info("Train augmentation: crop scale %s, rotation +/-%.0f deg, translate %s, "
                "brightness/contrast +/-%.2f/%.2f, NO horizontal flip",
                scale, degrees, translate, brightness, contrast)
    return transforms.Compose(steps)


def build_transforms(cfg: Config) -> dict[str, Any]:
    """Return ``{"train": ..., "val": ..., "test": ...}``.

    Val and test share the identical deterministic transform object.
    """
    evaluation = build_eval_transform(cfg)
    return {"train": build_train_transform(cfg), "val": evaluation, "test": evaluation}


def denormalize(tensor: Any, cfg: Config) -> Any:
    """Undo ImageNet normalisation, for displaying an image or a Grad-CAM overlay.

    Args:
        tensor: Shape ``(3, H, W)`` or ``(B, 3, H, W)``.
        cfg: X-ray configuration.

    Returns:
        A tensor in ``[0, 1]``, same shape.
    """
    import torch

    mean = torch.tensor(list(cfg.preprocessing.normalize_mean)).view(-1, 1, 1)
    std = torch.tensor(list(cfg.preprocessing.normalize_std)).view(-1, 1, 1)
    if tensor.dim() == 4:
        mean, std = mean.unsqueeze(0), std.unsqueeze(0)
    return torch.clamp(tensor.detach().cpu() * std + mean, 0.0, 1.0)
