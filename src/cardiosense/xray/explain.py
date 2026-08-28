"""Grad-CAM for the chest X-ray model.

Implemented directly with forward and backward hooks rather than pulling in a
Grad-CAM package. It is about thirty lines of real work, and writing it out makes
the method auditable — which matters, because a heatmap is the most persuasive and
most over-read output this project produces.

How it works
------------

1. Forward the image and capture the activations ``A^k`` of the last
   convolutional block (``features.denseblock4``, 1024 channels at 7x7 for a
   224px input).
2. Backpropagate the target logit and capture the gradients ``dY/dA^k``.
3. Weight each channel by its spatially averaged gradient:
   ``w_k = mean_ij(dY/dA^k_ij)`` — how much this feature map raises the score.
4. Combine and rectify: ``CAM = ReLU(sum_k w_k A^k)``. The ReLU keeps only
   evidence *for* the class; negative contributions argue against it and would
   muddy the map.
5. Upsample to the input resolution bilinearly.

Why ``denseblock4``: it is the last layer with spatial structure, so its
activations are the most semantically specific ones that still have a location.
Earlier layers give sharper but less meaningful maps; after global pooling there
is no spatial information left at all.

**What a Grad-CAM heatmap does not establish**

This is the caveat that has to survive into the report, because a red blob over a
heart is extraordinarily convincing and proves far less than it appears to.

1. **It shows where gradient mass concentrated, not why a diagnosis is correct.**
   Highlighting the cardiac silhouette is *consistent with* the model measuring
   heart size. It does not demonstrate it, and it is not evidence of medical
   reasoning.
2. **It is coarse.** A 7x7 grid upsampled to 224x224 means each cell covers about
   32x32 pixels. Apparent precision is an artefact of bilinear interpolation.
3. **A plausible map can accompany a wrong prediction, and vice versa.** Both
   happen in the examples this pipeline exports, which is exactly why false
   negatives are included in the saved figures.
4. **It cannot detect shortcut learning on its own.** If the model keys on a
   pacemaker or a portable-film marker that correlates with cardiomegaly, the map
   may still land near the heart because those things are near the heart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..common.config import Config
from ..common.io_utils import save_json
from ..common.logging_utils import get_logger
from ..common.paths import ensure_dir
from ..common.plots import save_figure

__all__ = ["GradCAM", "compute_gradcam", "plot_gradcam", "run_gradcam_analysis"]

logger = get_logger("xray.explain")

_CAVEAT = (
    "Grad-CAM shows where gradient mass concentrated for this prediction. It does NOT "
    "demonstrate that the model measured heart size, and it is not evidence of medical "
    "reasoning. The map is 7x7 upsampled to the input resolution, so its apparent "
    "precision is an interpolation artefact."
)


class GradCAM:
    """Grad-CAM via forward/backward hooks on a chosen convolutional layer.

    Use as a context manager so the hooks are always removed — a leaked backward
    hook silently slows every later forward pass and can hold tensors alive.

    Args:
        model: The trained model.
        target_layer: Dotted module path, e.g. ``"features.denseblock4"``.
    """

    def __init__(self, model: torch.nn.Module, target_layer: str = "features.denseblock4") -> None:
        self.model = model
        self.target_layer_name = target_layer
        self.layer = self._resolve_layer(model, target_layer)
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles: list[Any] = []

    @staticmethod
    def _resolve_layer(model: torch.nn.Module, path: str) -> torch.nn.Module:
        module: Any = model
        for part in path.split("."):
            if not hasattr(module, part):
                available = [name for name, _ in model.named_modules()][:40]
                raise AttributeError(
                    f"Layer {path!r} not found (failed at {part!r}). "
                    f"Available modules include: {available}"
                )
            module = getattr(module, part)
        return module

    def __enter__(self) -> "GradCAM":
        def forward_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(_module: Any, _grad_input: Any, grad_output: tuple) -> None:
            self.gradients = grad_output[0].detach()

        self._handles.append(self.layer.register_forward_hook(forward_hook))
        # full_backward_hook is the non-deprecated API and fires reliably for
        # modules with multiple inputs, which dense blocks have.
        self._handles.append(self.layer.register_full_backward_hook(backward_hook))
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.activations = None
        self.gradients = None

    def __call__(
        self,
        images: torch.Tensor,
        target_logit: int = 0,
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute CAMs for a batch.

        Args:
            images: Shape ``(B, 3, H, W)``.
            target_logit: Which output logit to explain (0 for binary).
            normalize: Scale each CAM to ``[0, 1]`` independently.

        Returns:
            ``(cams, probabilities)`` with cams shaped ``(B, H, W)``.
        """
        was_training = self.model.training
        self.model.eval()

        images = images.clone().requires_grad_(True)
        logits = self.model(images)
        score = logits[:, target_logit].sum()

        self.model.zero_grad(set_to_none=True)
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                f"No activations captured from {self.target_layer_name}. The layer may "
                "not lie on the forward path."
            )

        # w_k = spatially averaged gradient for channel k.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)

        if normalize:
            flat = cam.flatten(1)
            minimum = flat.min(dim=1)[0].view(-1, 1, 1)
            maximum = flat.max(dim=1)[0].view(-1, 1, 1)
            cam = (cam - minimum) / torch.clamp(maximum - minimum, min=1e-8)

        probabilities = torch.sigmoid(logits[:, target_logit].detach()).cpu().numpy()

        self.model.train(was_training)
        return cam.detach().cpu().numpy(), probabilities


def compute_gradcam(
    model: torch.nn.Module,
    images: torch.Tensor,
    target_layer: str = "features.denseblock4",
    target_logit: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: compute CAMs and always clean up the hooks."""
    with GradCAM(model, target_layer) as explainer:
        return explainer(images, target_logit=target_logit)


def plot_gradcam(
    image_tensor: torch.Tensor,
    cam: np.ndarray,
    path: Path | str,
    probability: float,
    true_label: int,
    threshold: float,
    cfg: Config,
    image_name: str = "",
    alpha: float = 0.4,
    colormap: str = "jet",
) -> Path:
    """Save a three-panel figure: original, heatmap, overlay.

    Three panels rather than just the overlay, because the overlay alone hides how
    much of the heatmap is genuine structure versus interpolation, and hides the
    original film that a reader needs in order to judge the claim.
    """
    import matplotlib.pyplot as plt

    from .preprocessing import denormalize

    grey = denormalize(image_tensor, cfg).numpy()
    if grey.ndim == 3:
        grey = grey.mean(axis=0)      # the three channels are replicas

    prediction = int(probability >= threshold)
    verdict = "correct" if prediction == int(true_label) else "INCORRECT"
    case_type = {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}[
        (int(true_label), prediction)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))

    axes[0].imshow(grey, cmap="gray")
    axes[0].set_title("Original radiograph", fontsize=10)

    heat = axes[1].imshow(cam, cmap=colormap, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM (denseblock4)", fontsize=10)
    fig.colorbar(heat, ax=axes[1], fraction=0.046)

    axes[2].imshow(grey, cmap="gray")
    axes[2].imshow(cam, cmap=colormap, alpha=alpha, vmin=0, vmax=1)
    axes[2].set_title("Overlay", fontsize=10)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{case_type} — p = {probability:.3f} (threshold {threshold:.2f}), "
        f"true = {int(true_label)} [{verdict}]"
        + (f"  |  {image_name}" if image_name else "")
        + "\nHeatmap shows where gradient mass concentrated — NOT proof the model "
          "measured heart size",
        fontsize=10,
    )
    return save_figure(fig, path)


def run_gradcam_analysis(
    model: torch.nn.Module,
    frame: Any,
    loader_dataset: Any,
    selection: dict[str, Sequence[int]],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cfg: Config,
    device: torch.device,
    out_dir: Path | str,
) -> dict[str, Any]:
    """Generate Grad-CAM figures for representative TP / TN / FP / FN cases.

    False negatives are included deliberately. A gallery of successful heatmaps is
    a marketing exercise; the failures are what tell you whether the model is
    looking at the heart.

    Args:
        model: Trained model.
        frame: Split metadata, row-aligned with ``y_true``.
        loader_dataset: The :class:`ChestXrayDataset` for this split, used to fetch
            the preprocessed tensor for a given row.
        selection: ``{"TP": [rows], "FN": [rows], ...}`` from
            :func:`~cardiosense.xray.evaluate.select_error_examples`.
        y_true: Split labels.
        y_prob: Split probabilities.
        threshold: Operating threshold.
        cfg: X-ray configuration.
        device: Device to run on.
        out_dir: ``results/xray/gradcam``.

    Returns:
        A summary dict, also written to ``gradcam_summary.json``.
    """
    out = ensure_dir(out_dir)
    target_layer = str(cfg.explainability.get("target_layer", "features.denseblock4"))
    alpha = float(cfg.explainability.get("overlay_alpha", 0.4))
    colormap = str(cfg.explainability.get("colormap", "jet"))
    image_column = str(cfg.dataset.image_column)

    records: list[dict[str, Any]] = []

    with GradCAM(model, target_layer) as explainer:
        for case_type, rows in selection.items():
            for row in rows:
                row = int(row)
                sample = loader_dataset[row]
                image_tensor = sample[0]
                batch = image_tensor.unsqueeze(0).to(device)

                cams, probabilities = explainer(batch, target_logit=0)
                image_name = str(frame.iloc[row][image_column]) if image_column in frame else ""

                filename = f"gradcam_{case_type}_{row:05d}.png"
                plot_gradcam(
                    image_tensor, cams[0], out / filename,
                    probability=float(y_prob[row]), true_label=int(y_true[row]),
                    threshold=threshold, cfg=cfg, image_name=image_name,
                    alpha=alpha, colormap=colormap,
                )

                # Where is the attention mass? Upper vs lower half is a crude but
                # useful sanity check: the heart sits in the lower-central chest.
                cam = cams[0]
                height, width = cam.shape
                records.append({
                    "row": row,
                    "case_type": case_type,
                    "image": image_name,
                    "probability": round(float(y_prob[row]), 4),
                    "true_label": int(y_true[row]),
                    "mass_lower_half": round(float(cam[height // 2:].sum()
                                                   / max(cam.sum(), 1e-9)), 4),
                    "mass_central_third": round(
                        float(cam[:, width // 3: 2 * width // 3].sum()
                              / max(cam.sum(), 1e-9)), 4),
                    "peak_location_yx": [int(np.unravel_index(cam.argmax(), cam.shape)[0]),
                                         int(np.unravel_index(cam.argmax(), cam.shape)[1])],
                    "figure": filename,
                })

    summary = {
        "method": "grad-cam",
        "target_layer": target_layer,
        "n_figures": len(records),
        "cases": records,
        "caveat": _CAVEAT,
        "limitations": [
            "Shows where gradient mass concentrated, not why a diagnosis is correct.",
            "7x7 feature grid upsampled to the input size; apparent sharpness is "
            "interpolation.",
            "A plausible map can accompany a wrong prediction, and an odd map a correct one.",
            "Cannot by itself rule out shortcut learning on features near the heart.",
        ],
    }
    save_json(summary, out / "gradcam_summary.json")
    logger.info("Wrote %d Grad-CAM figures to %s", len(records), out)

    if records:
        mean_lower = float(np.mean([r["mass_lower_half"] for r in records]))
        logger.info("Mean Grad-CAM mass in the lower half of the image: %.2f "
                    "(the cardiac silhouette sits there; this is a weak sanity check, "
                    "not validation)", mean_lower)
    return summary
