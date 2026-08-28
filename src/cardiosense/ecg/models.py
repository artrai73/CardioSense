"""1D convolutional architectures for 12-lead ECG classification.

Two models, deliberately in this order:

* :class:`ECGCNN1D` — the main model. Four convolutional stages, understandable
  end to end, ~0.5 M parameters.
* :class:`ECGResNet1D` — the optional comparison. Implemented properly with
  pre-activation residual blocks, but only worth running once the plain CNN is
  training stably (see the guidance in ``docs/experiments.md``).

Why a 1D CNN at all
-------------------

An ECG is a set of 12 simultaneous time series. A 1D convolution slides a kernel
along time and across all 12 leads at once, which matches the physiology: leads
are simultaneous views of the same electrical event, so the network should see
them jointly at every layer rather than treating them as independent channels to
be merged at the end.

Kernel size 7 at 100 Hz spans 70 ms — roughly the width of a QRS complex, so a
first-layer filter can respond to a whole complex rather than a fragment. Each
stage halves the time axis by pooling, so the receptive field grows to cover
several beats by the final stage, which is what ST/T and rhythm-adjacent
morphology needs.

Why not deeper
--------------

With ~17,000 training records and no pretraining, a much deeper network overfits
before it generalises. Four stages, batch norm and dropout is the smallest thing
that can plausibly represent the task, and the ResNet variant exists to test
whether extra depth actually buys anything measurable.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from ..common.config import Config
from ..common.logging_utils import get_logger

__all__ = ["ECGCNN1D", "ECGResNet1D", "build_model", "model_summary"]

logger = get_logger("ecg.models")


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> ReLU -> MaxPool -> Dropout.

    Order matters. BatchNorm before the activation is the standard formulation
    and stabilises training given the wide amplitude variation between leads.
    Dropout comes last so it does not interfere with the batch statistics.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        pool_size: int = 2,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2          # 'same' padding, so length is set only by pooling
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=padding, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if pool_size > 1:
            layers.append(nn.MaxPool1d(pool_size))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GlobalAvgMaxPool(nn.Module):
    """Concatenated global average and max pooling over time.

    Average pooling captures sustained morphology (an ST segment depressed for
    the whole record); max pooling captures transient events (a single ectopic
    beat). Concatenating both gives the classifier access to each, at the cost of
    doubling the feature width, which is cheap here.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)


class ECGCNN1D(nn.Module):
    """A compact 1D CNN for multi-label ECG classification.

    Architecture (defaults, 12 leads x 1000 samples in)::

        input          (B, 12, 1000)
        ConvBlock  32  (B,  32,  500)   kernel 7 = 70 ms at 100 Hz
        ConvBlock  64  (B,  64,  250)
        ConvBlock 128  (B, 128,  125)
        ConvBlock 256  (B, 256,   62)
        GlobalAvgMaxPool (B, 512)
        Linear -> 128 -> ReLU -> Dropout
        Linear -> 5 logits

    The head emits **raw logits**, not probabilities: ``BCEWithLogitsLoss`` fuses
    the sigmoid with the loss in a numerically stable way. Call ``torch.sigmoid``
    explicitly at evaluation time.
    """

    def __init__(
        self,
        in_channels: int = 12,
        n_classes: int = 5,
        channels: Sequence[int] = (32, 64, 128, 256),
        kernel_size: int = 7,
        stride: int = 1,
        pool_size: int = 2,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        global_pool: str = "avgmax",
        fc_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.global_pool = global_pool

        blocks: list[nn.Module] = []
        previous = in_channels
        for out_channels in channels:
            blocks.append(ConvBlock(previous, out_channels, kernel_size, stride,
                                    pool_size, dropout, use_batchnorm))
            previous = out_channels
        self.features = nn.Sequential(*blocks)

        if global_pool == "avgmax":
            self.pool: nn.Module = GlobalAvgMaxPool()
            pooled_width = previous * 2
        elif global_pool == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
            pooled_width = previous
        elif global_pool == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
            pooled_width = previous
        else:
            raise ValueError(f"Unknown global_pool {global_pool!r}; use avgmax, avg or max.")

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pooled_width, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """He initialisation for conv layers, which suits ReLU activations."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape ``(batch, n_leads, n_samples)``.

        Returns:
            Raw logits of shape ``(batch, n_classes)``.
        """
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class ResidualBlock1D(nn.Module):
    """Pre-activation residual block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv (+ skip).

    Pre-activation (He et al., 2016) places normalisation and activation *before*
    each convolution, which keeps the skip path a clean identity. That is what
    lets gradients reach the early layers of a deep stack unattenuated.

    When the block changes channel count or stride, the shortcut needs a 1x1
    projection so the shapes match; otherwise it is a true identity with zero
    parameters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                      stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(x))
        identity = self.shortcut(out)      # projection sees the pre-activated input
        out = self.conv1(out)
        out = self.dropout(torch.relu(self.bn2(out)))
        out = self.conv2(out)
        return out + identity


class ECGResNet1D(nn.Module):
    """A ResNet-style 1D network, for the depth comparison (Experiment E-C).

    Four stages; each stage after the first halves the time axis with a strided
    convolution instead of pooling, which lets the network learn its own
    downsampling.

    Only run this after the plain CNN trains stably. If it does not beat
    :class:`ECGCNN1D` by a margin larger than the seed-to-seed variation, the
    honest recommendation is to keep the simpler model — and to say so.
    """

    def __init__(
        self,
        in_channels: int = 12,
        n_classes: int = 5,
        base_channels: int = 64,
        blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
        kernel_size: int = 7,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15,
                      stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        stages: list[nn.Module] = []
        previous = base_channels
        for stage_index, n_blocks in enumerate(blocks_per_stage):
            out_channels = base_channels * (2 ** stage_index)
            for block_index in range(n_blocks):
                stride = 2 if (block_index == 0 and stage_index > 0) else 1
                stages.append(ResidualBlock1D(previous, out_channels, kernel_size,
                                              stride, dropout))
                previous = out_channels
        self.stages = nn.Sequential(*stages)

        self.head_norm = nn.Sequential(nn.BatchNorm1d(previous), nn.ReLU(inplace=True))
        self.pool = GlobalAvgMaxPool()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(previous * 2, n_classes),
        )

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_norm(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(cfg: Config, name: str | None = None) -> nn.Module:
    """Construct the model named by ``model.name`` (or the override).

    Args:
        cfg: ECG configuration.
        name: ``"cnn1d"`` or ``"resnet1d"``; defaults to ``cfg.model.name``.

    Returns:
        An unfitted model on CPU. Move it to the device yourself.
    """
    name = (name or str(cfg.model.name)).lower()

    if name == "cnn1d":
        params = cfg.model.cnn1d
        model: nn.Module = ECGCNN1D(
            in_channels=int(params.get("in_channels", 12)),
            n_classes=int(params.get("n_classes", 5)),
            channels=list(params.get("channels", [32, 64, 128, 256])),
            kernel_size=int(params.get("kernel_size", 7)),
            stride=int(params.get("stride", 1)),
            pool_size=int(params.get("pool_size", 2)),
            dropout=float(params.get("dropout", 0.3)),
            use_batchnorm=bool(params.get("use_batchnorm", True)),
            global_pool=str(params.get("global_pool", "avgmax")),
            fc_hidden=int(params.get("fc_hidden", 128)),
        )
    elif name == "resnet1d":
        params = cfg.model.resnet1d
        model = ECGResNet1D(
            in_channels=int(params.get("in_channels", 12)),
            n_classes=int(params.get("n_classes", 5)),
            base_channels=int(params.get("base_channels", 64)),
            blocks_per_stage=list(params.get("blocks_per_stage", [2, 2, 2, 2])),
            kernel_size=int(params.get("kernel_size", 7)),
            dropout=float(params.get("dropout", 0.2)),
        )
    else:
        raise ValueError(f"Unknown model {name!r}; use cnn1d or resnet1d.")

    logger.info("Built %s: %s parameters", name, f"{model_summary(model)['total']:,}")
    return model


def model_summary(model: nn.Module, input_shape: tuple[int, int, int] | None = None) -> dict[str, Any]:
    """Parameter counts and, optionally, the output shape for a probe input."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary: dict[str, Any] = {
        "class": type(model).__name__,
        "total": int(total),
        "trainable": int(trainable),
        "size_mb": round(total * 4 / 1024**2, 2),
    }
    if input_shape is not None:
        model.eval()
        with torch.no_grad():
            summary["output_shape"] = list(model(torch.zeros(*input_shape)).shape)
    return summary
