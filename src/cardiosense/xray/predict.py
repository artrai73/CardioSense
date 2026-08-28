"""Inference for the chest X-ray pipeline.

Loads ``xray_model.pth`` plus ``xray_config.json`` and turns a chest radiograph
into a structured prediction. The config sidecar carries the image size,
normalisation statistics and operating threshold, so inference reproduces the
evaluated preprocessing exactly.

Command line::

    python -m cardiosense.xray.predict --image data/xray/nih/images/00000013_005.png
    python -m cardiosense.xray.predict --image <path> --gradcam out/cam.png
    python -m cardiosense.xray.predict --dir some/folder --output predictions.csv

Python::

    from cardiosense.xray.predict import XrayPredictor
    predictor = XrayPredictor.load()
    predictor.predict_image("path/to/image.png")

Output contract (Phase 1 — NOT fused with the other modalities)::

    {
        "prediction": 1,
        "label": "cardiomegaly",
        "probability": 0.78,
        "threshold": 0.41,
        "model_version": "xray-v0.1.0",
        ...
    }

Note there is no ``calibrated_confidence`` field, unlike the clinical pipeline.
No calibrator is fitted for this modality in Phase 1, so the probability is a
ranking score and must not be read as a frequency claim — doubly so because
negative subsampling shifted the prevalence away from the population rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..common.config import Config, load_config
from ..common.io_utils import load_json
from ..common.logging_utils import get_logger
from ..common.paths import PATHS
from .models import XrayDenseNet121

__all__ = ["XrayPredictor", "main"]

logger = get_logger("xray.predict")


class XrayPredictor:
    """Loads a trained DenseNet121 and produces structured binary predictions."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: Mapping[str, Any],
        device: torch.device | None = None,
        cfg: Config | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.config = dict(config)
        self.cfg = cfg

        self.threshold = float(self.config.get("threshold", 0.5))
        self.model_version = str(self.config.get("model_version", "unknown"))
        self.target_label = str(self.config.get("target_label", "target"))
        self.label_mapping = dict(self.config.get("label_mapping",
                                                  {"0": "negative", "1": "positive"}))
        self.transform = self._build_transform()

    def _build_transform(self) -> Any:
        """Rebuild the exact evaluation transform recorded in the config sidecar."""
        from torchvision import transforms

        size = int(self.config.get("image_size", 224))
        steps: list[Any] = [transforms.Resize((size, size))]
        if bool(self.config.get("to_rgb", True)):
            steps.append(transforms.Grayscale(num_output_channels=3))
        steps.append(transforms.ToTensor())
        steps.append(transforms.Normalize(
            mean=list(self.config.get("normalize_mean", [0.485, 0.456, 0.406])),
            std=list(self.config.get("normalize_std", [0.229, 0.224, 0.225])),
        ))
        return transforms.Compose(steps)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(
        cls,
        models_dir: Path | str | None = None,
        cfg: Config | None = None,
        device: torch.device | None = None,
    ) -> "XrayPredictor":
        """Load the model and its config sidecar from ``models/xray/``."""
        cfg = cfg or load_config("xray")
        directory = Path(models_dir) if models_dir else PATHS.root / str(cfg.output.models_dir)

        model_path = directory / cfg.output.model_file
        config_path = directory / cfg.output.config_file
        for path in (model_path, config_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing artifact: {path}\nRun `python -m cardiosense.xray.train` first."
                )

        inference_config = load_json(config_path)
        resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # pretrained=False: the trained weights are about to overwrite everything,
        # so downloading ImageNet weights first would waste bandwidth and would
        # fail on an offline runtime.
        model = XrayDenseNet121(
            pretrained=False,
            n_classes=int(inference_config.get("n_classes", 1)),
            dropout=float(inference_config.get("dropout", 0.2)),
        )
        try:
            payload = torch.load(model_path, map_location=resolved_device, weights_only=False)
        except TypeError:  # torch < 2.4
            payload = torch.load(model_path, map_location=resolved_device)
        model.load_state_dict(payload["model_state"])

        return cls(model, inference_config, device=resolved_device, cfg=cfg)

    # ------------------------------------------------------------- inference
    def _load_tensor(self, path: Path | str) -> torch.Tensor:
        from PIL import Image

        with Image.open(path) as handle:
            image = handle.convert("L")
        return self.transform(image)

    @torch.no_grad()
    def predict_tensors(self, batch: torch.Tensor) -> np.ndarray:
        """Return P(positive) for a preprocessed batch of shape ``(B, 3, H, W)``."""
        logits = self.model(batch.to(self.device))
        return torch.sigmoid(logits.float()).cpu().numpy().ravel()

    def _result(self, probability: float, source: str = "") -> dict[str, Any]:
        prediction = int(probability >= self.threshold)
        return {
            "prediction": prediction,
            "label": self.label_mapping.get(str(prediction), str(prediction)),
            "probability": round(float(probability), 4),
            "threshold": round(float(self.threshold), 4),
            "target": self.target_label,
            "model": "densenet121",
            "model_version": self.model_version,
            "modality": "xray",
            "source": source,
            "notes": {
                "probability": "Sigmoid output. A ranking score, NOT a calibrated "
                               "frequency — no calibrator is fitted for this modality "
                               "in Phase 1.",
                "prevalence": self.config.get("prevalence_warning", ""),
                "view": f"Model trained on {self.config.get('view_filter') or 'all'} views; "
                        "applying it to other views is out of distribution.",
                "disclaimer": "Research artifact. Not for clinical use.",
            },
        }

    def predict_image(self, path: Path | str) -> dict[str, Any]:
        """Predict for a single image file."""
        tensor = self._load_tensor(path).unsqueeze(0)
        probability = float(self.predict_tensors(tensor)[0])
        return self._result(probability, source=str(path))

    def predict_images(
        self,
        paths: Sequence[Path | str],
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        """Predict for many image files, batched."""
        results: list[dict[str, Any]] = []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            batch = torch.stack([self._load_tensor(path) for path in chunk])
            for path, probability in zip(chunk, self.predict_tensors(batch)):
                results.append(self._result(float(probability), source=str(path)))
        return results

    def explain_image(self, path: Path | str, output_path: Path | str) -> dict[str, Any]:
        """Predict and save a Grad-CAM figure for one image."""
        from .explain import compute_gradcam, plot_gradcam

        cfg = self.cfg or load_config("xray")
        tensor = self._load_tensor(path)
        batch = tensor.unsqueeze(0).to(self.device)

        target_layer = str(cfg.explainability.get("target_layer", "features.denseblock4"))
        cams, probabilities = compute_gradcam(self.model, batch, target_layer=target_layer)
        probability = float(probabilities[0])

        plot_gradcam(
            tensor, cams[0], output_path, probability=probability,
            true_label=int(probability >= self.threshold),   # unknown at inference
            threshold=self.threshold, cfg=cfg, image_name=Path(path).name,
            alpha=float(cfg.explainability.get("overlay_alpha", 0.4)),
            colormap=str(cfg.explainability.get("colormap", "jet")),
        )

        result = self._result(probability, source=str(path))
        result["gradcam"] = str(output_path)
        result["notes"]["gradcam"] = (
            "The heatmap shows where gradient mass concentrated. It does not demonstrate "
            "that the model measured heart size."
        )
        return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run chest X-ray inference.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Path to a single PNG.")
    source.add_argument("--dir", help="Directory of PNGs.")
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--gradcam", default=None,
                        help="Also save a Grad-CAM figure to this path (single image only).")
    parser.add_argument("--output", default=None, help="Write results to a CSV/JSON path.")
    args = parser.parse_args(argv)

    predictor = XrayPredictor.load(models_dir=args.models_dir)

    if args.image:
        if args.gradcam:
            results = [predictor.explain_image(args.image, args.gradcam)]
        else:
            results = [predictor.predict_image(args.image)]
    else:
        paths = sorted(Path(args.dir).glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No .png files found in {args.dir}")
        logger.info("Predicting for %d images", len(paths))
        results = predictor.predict_images(paths)

    if args.output:
        import pandas as pd

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".csv":
            pd.DataFrame([{k: v for k, v in r.items() if k != "notes"}
                          for r in results]).to_csv(out_path, index=False)
        else:
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Wrote %d predictions to %s", len(results), out_path)
    else:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
