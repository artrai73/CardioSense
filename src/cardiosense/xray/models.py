"""DenseNet121 transfer learning for chest X-ray classification.

Why DenseNet121
---------------

It is the standard architecture for ChestX-ray14 (CheXNet and everything after
it), which makes results comparable. Structurally it suits the problem: dense
connectivity means every layer receives the feature maps of all preceding layers,
so fine texture from early layers stays available at the classification head. In a
chest film the relevant evidence — the cardiac border against lung — is a
large-scale shape defined by fine edges, and both need to survive to the end.

It is also small for its accuracy: ~7 M parameters against ResNet-50's ~25 M,
which matters on a Colab GPU.

Transfer learning, explained
----------------------------

**Pretrained weights.** ImageNet (``DenseNet121_Weights.IMAGENET1K_V1``). Chest
X-rays look nothing like ImageNet photographs, and the transfer still works,
because the early layers learn edge, blob and texture detectors that are close to
universal for natural images. Only the later, more semantic layers are
domain-specific — which is exactly the part we retrain.

**Classification head.** The original 1000-way ImageNet classifier is discarded
and replaced with dropout + a single linear unit emitting one logit. One logit,
not two: this is binary, and ``BCEWithLogitsLoss`` on a single logit is equivalent
to and more stable than a 2-way softmax.

**Frozen layers, then staged unfreezing.** Two stages:

1. *Head-only* (``freeze_backbone_epochs``). The entire backbone is frozen and
   only the new head trains. The head starts randomly initialised, so its early
   gradients are large and noisy; letting them propagate into pretrained weights
   at that point destroys the features that make transfer work in the first place.
2. *Partial fine-tune*. Layers from ``unfreeze_from_block`` onward are unfrozen and
   trained at a much lower learning rate (``finetune_learning_rate``, 6x smaller by
   default). Early blocks stay frozen: their edge detectors are already right, and
   retraining them on ~10k images mostly overfits.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..common.config import Config
from ..common.logging_utils import get_logger

__all__ = ["XrayDenseNet121", "build_model", "set_backbone_trainable", "model_summary"]

logger = get_logger("xray.models")

#: DenseNet121 blocks in forward order, used to resolve ``unfreeze_from_block``.
_BLOCK_ORDER = (
    "conv0", "norm0", "relu0", "pool0",
    "denseblock1", "transition1",
    "denseblock2", "transition2",
    "denseblock3", "transition3",
    "denseblock4", "norm5",
)


class XrayDenseNet121(nn.Module):
    """DenseNet121 with a binary classification head.

    The ``features`` submodule is kept intact and exposed under its original name
    so Grad-CAM can hook ``features.denseblock4`` — the last convolutional block,
    whose activations carry the spatial evidence the head actually uses.

    Args:
        pretrained: Load ImageNet weights.
        n_classes: Number of output logits (1 for binary).
        dropout: Dropout before the final linear layer.
    """

    def __init__(self, pretrained: bool = True, n_classes: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        from torchvision import models

        weights = None
        if pretrained:
            try:
                weights = models.DenseNet121_Weights.IMAGENET1K_V1
                backbone = models.densenet121(weights=weights)
                logger.info("Loaded ImageNet pretrained DenseNet121 weights.")
            except Exception as exc:  # noqa: BLE001 - offline runtimes must fail loudly
                logger.warning(
                    "Could not download pretrained weights (%s). Falling back to random "
                    "initialisation. Expect substantially worse results — transfer "
                    "learning is doing most of the work on a dataset this size.", exc
                )
                backbone = models.densenet121(weights=None)
                weights = None
        else:
            backbone = models.densenet121(weights=None)
            logger.info("DenseNet121 initialised randomly (pretrained=False).")

        self.pretrained = weights is not None
        self.n_classes = n_classes

        #: The convolutional trunk. Grad-CAM hooks into this.
        self.features = backbone.features
        self.n_features = int(backbone.classifier.in_features)   # 1024 for DenseNet121

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.n_features, n_classes),
        )
        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.zeros_(self.classifier[1].bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Convolutional feature maps, shape ``(B, 1024, H/32, W/32)``.

        The trailing ReLU is part of the standard DenseNet forward pass and must
        be applied here too, or Grad-CAM would read pre-activation maps.
        """
        return torch.relu(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape ``(B, 3, H, W)``.

        Returns:
            Raw logits, shape ``(B, n_classes)``. Apply the sigmoid at evaluation.
        """
        feature_maps = self.forward_features(x)
        pooled = torch.flatten(nn.functional.adaptive_avg_pool2d(feature_maps, 1), 1)
        return self.classifier(pooled)


def set_backbone_trainable(
    model: XrayDenseNet121,
    trainable: bool,
    from_block: str | None = None,
) -> dict[str, Any]:
    """Freeze or unfreeze the backbone, optionally only from a given block onward.

    Args:
        model: The model.
        trainable: Target ``requires_grad`` value for the selected layers.
        from_block: Unfreeze from this block onward, e.g. ``"denseblock3"``.
            Layers before it stay frozen. ``None`` applies to the whole backbone.

    Returns:
        A report naming which blocks are now trainable.
    """
    if from_block is not None and from_block not in _BLOCK_ORDER:
        raise ValueError(
            f"Unknown block {from_block!r}. DenseNet121 blocks: {list(_BLOCK_ORDER)}"
        )

    start_index = _BLOCK_ORDER.index(from_block) if from_block else 0
    unfrozen: list[str] = []
    frozen: list[str] = []

    for name, module in model.features.named_children():
        position = _BLOCK_ORDER.index(name) if name in _BLOCK_ORDER else len(_BLOCK_ORDER)
        should_train = trainable and position >= start_index
        for parameter in module.parameters():
            parameter.requires_grad = should_train
        (unfrozen if should_train else frozen).append(name)

    # The head is always trainable.
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    report = {
        "trainable_blocks": unfrozen,
        "frozen_blocks": frozen,
        "trainable_parameters": int(n_trainable),
        "total_parameters": int(n_total),
        "trainable_fraction": round(n_trainable / max(n_total, 1), 4),
    }
    logger.info("Backbone %s%s: %d of %d parameters trainable (%.1f%%)",
                "unfrozen" if trainable else "frozen",
                f" from {from_block}" if from_block else "",
                n_trainable, n_total, 100 * report["trainable_fraction"])
    return report


def build_model(cfg: Config) -> XrayDenseNet121:
    """Construct the model from config, with the backbone frozen for stage 1."""
    params = cfg.model
    model = XrayDenseNet121(
        pretrained=bool(params.get("pretrained", True)),
        n_classes=int(params.get("n_classes", 1)),
        dropout=float(params.get("dropout", 0.2)),
    )

    # Stage 1 starts with everything frozen except the new head.
    if int(params.get("freeze_backbone_epochs", 0)) > 0:
        set_backbone_trainable(model, trainable=False)

    return model


def model_summary(model: nn.Module, input_shape: tuple[int, ...] | None = None) -> dict[str, Any]:
    """Parameter counts and, optionally, the output shape for a probe input."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary: dict[str, Any] = {
        "class": type(model).__name__,
        "total": int(total),
        "trainable": int(trainable),
        "size_mb": round(total * 4 / 1024**2, 2),
        "pretrained": bool(getattr(model, "pretrained", False)),
    }
    if input_shape is not None:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            probe = torch.zeros(*input_shape)
            summary["output_shape"] = list(model(probe).shape)
            if hasattr(model, "forward_features"):
                summary["feature_map_shape"] = list(model.forward_features(probe).shape)
        model.train(was_training)
    return summary
