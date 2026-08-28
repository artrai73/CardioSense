"""Inference for the ECG pipeline.

Loads ``ecg_model.pth`` plus ``ecg_config.json`` and turns a raw 12-lead
waveform into structured multi-label output. The config sidecar carries the
architecture, the class order, the per-class thresholds and the preprocessing
settings, so inference reproduces training exactly without guessing.

Command line::

    python -m cardiosense.ecg.predict --record data/ecg/ptbxl/records100/00000/00001_lr
    python -m cardiosense.ecg.predict --npy waveform.npy
    python -m cardiosense.ecg.predict --record <path> --output prediction.json

Python::

    from cardiosense.ecg.predict import ECGPredictor
    predictor = ECGPredictor.load()
    predictor.predict_record("path/to/00001_lr")

Output contract (Phase 1 — NOT fused with the other modalities)::

    {
        "predictions": {"NORM": 0, "MI": 1, "STTC": 1, "CD": 0, "HYP": 0},
        "probabilities": {"NORM": 0.11, "MI": 0.83, ...},
        "positive_classes": ["MI", "STTC"],
        "thresholds": {...},
        "model_version": "ecg-v0.1.0",
        ...
    }

Multiple classes can be positive at once — that is the point of a multi-label
task, and the output deliberately does not collapse to a single "diagnosis".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..common.config import Config, load_config
from ..common.io_utils import load_json
from ..common.logging_utils import get_logger
from ..common.paths import PATHS
from .models import ECGCNN1D, ECGResNet1D
from .preprocessing import load_raw_record, preprocess_signal

__all__ = ["ECGPredictor", "main"]

logger = get_logger("ecg.predict")


class ECGPredictor:
    """Loads a trained ECG model and produces structured multi-label predictions."""

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

        self.classes: list[str] = list(self.config.get("classes", []))
        self.thresholds = np.array(
            [float(self.config.get("thresholds", {}).get(name, 0.5)) for name in self.classes],
            dtype=float,
        )
        self.model_version = str(self.config.get("model_version", "unknown"))
        self.sampling_rate = int(self.config.get("sampling_rate", 100))
        self.signal_length = int(self.config.get("signal_length", 1000))
        self.n_leads = int(self.config.get("n_leads", 12))
        self.lead_names = list(self.config.get("lead_names", []))

    # ------------------------------------------------------------------ load
    @classmethod
    def load(
        cls,
        models_dir: Path | str | None = None,
        cfg: Config | None = None,
        device: torch.device | None = None,
    ) -> "ECGPredictor":
        """Load the model and its config sidecar from ``models/ecg/``.

        Raises:
            FileNotFoundError: If an artifact is missing, naming the command that
                produces it.
        """
        cfg = cfg or load_config("ecg")
        directory = Path(models_dir) if models_dir else PATHS.root / str(cfg.output.models_dir)

        model_path = directory / cfg.output.model_file
        config_path = directory / cfg.output.config_file

        for path in (model_path, config_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing artifact: {path}\nRun `python -m cardiosense.ecg.train` first."
                )

        inference_config = load_json(config_path)
        resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # weights_only=False: the checkpoint stores class names and thresholds
        # alongside the tensors, and it was written by this codebase.
        try:
            payload = torch.load(model_path, map_location=resolved_device, weights_only=False)
        except TypeError:  # torch < 2.4
            payload = torch.load(model_path, map_location=resolved_device)

        model = cls._build_architecture(inference_config, cfg)
        model.load_state_dict(payload["model_state"])

        return cls(model, inference_config, device=resolved_device, cfg=cfg)

    @staticmethod
    def _build_architecture(inference_config: Mapping[str, Any], cfg: Config) -> torch.nn.Module:
        """Rebuild the exact architecture recorded in the config sidecar."""
        name = str(inference_config.get("model_name", "cnn1d")).lower()
        params = dict(inference_config.get("model_params", {}) or {})
        if not params:
            params = cfg.model.get(name, {}).to_dict()

        if name == "cnn1d":
            return ECGCNN1D(
                in_channels=int(params.get("in_channels", 12)),
                n_classes=int(params.get("n_classes", len(inference_config.get("classes", [])) or 5)),
                channels=list(params.get("channels", [32, 64, 128, 256])),
                kernel_size=int(params.get("kernel_size", 7)),
                stride=int(params.get("stride", 1)),
                pool_size=int(params.get("pool_size", 2)),
                dropout=float(params.get("dropout", 0.3)),
                use_batchnorm=bool(params.get("use_batchnorm", True)),
                global_pool=str(params.get("global_pool", "avgmax")),
                fc_hidden=int(params.get("fc_hidden", 128)),
            )
        if name == "resnet1d":
            return ECGResNet1D(
                in_channels=int(params.get("in_channels", 12)),
                n_classes=int(params.get("n_classes", 5)),
                base_channels=int(params.get("base_channels", 64)),
                blocks_per_stage=list(params.get("blocks_per_stage", [2, 2, 2, 2])),
                kernel_size=int(params.get("kernel_size", 7)),
                dropout=float(params.get("dropout", 0.2)),
            )
        raise ValueError(f"Unknown architecture {name!r} in the saved config.")

    # --------------------------------------------------------------- helpers
    def preprocess(self, waveform: np.ndarray) -> np.ndarray:
        """Apply the same preprocessing chain used during training."""
        if self.cfg is not None:
            return preprocess_signal(waveform, self.cfg)

        # Fall back to the settings recorded in the config sidecar.
        from ..common.config import Config as _Config

        shim = _Config({
            "dataset": {"sampling_rate": self.sampling_rate, "n_leads": self.n_leads,
                        "signal_length": self.signal_length},
            "preprocessing": dict(self.config.get("preprocessing", {})),
        })
        return preprocess_signal(waveform, shim)

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_array(
        self,
        waveforms: np.ndarray,
        already_preprocessed: bool = False,
    ) -> list[dict[str, Any]]:
        """Predict for one or more waveforms.

        Args:
            waveforms: ``(n_leads, n_samples)`` for a single record, or
                ``(batch, n_leads, n_samples)``.
            already_preprocessed: Skip filtering/normalisation. Only set this when
                the input came out of the training cache.

        Returns:
            One structured result per record.
        """
        array = np.asarray(waveforms, dtype=np.float32)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3:
            raise ValueError(f"Expected 2-D or 3-D input, got shape {array.shape}.")

        if not already_preprocessed:
            array = np.stack([self.preprocess(record) for record in array])

        tensor = torch.from_numpy(np.ascontiguousarray(array)).to(self.device)
        probabilities = torch.sigmoid(self.model(tensor).float()).cpu().numpy()

        results: list[dict[str, Any]] = []
        for row in probabilities:
            predictions = (row >= self.thresholds).astype(int)
            positive = [name for name, flag in zip(self.classes, predictions) if flag]
            results.append({
                "predictions": {name: int(flag) for name, flag in zip(self.classes, predictions)},
                "probabilities": {name: round(float(p), 4) for name, p in zip(self.classes, row)},
                "positive_classes": positive,
                "top_class": self.classes[int(np.argmax(row))],
                "top_probability": round(float(row.max()), 4),
                "thresholds": {name: round(float(t), 4)
                               for name, t in zip(self.classes, self.thresholds)},
                "model": str(self.config.get("model_name")),
                "model_version": self.model_version,
                "modality": "ecg",
                "task_type": str(self.config.get("task_type", "multilabel")),
                "notes": {
                    "multilabel": "Several classes can be positive simultaneously; this output "
                                  "is deliberately not collapsed to one diagnosis.",
                    "probabilities": "Sigmoid outputs, one independent binary decision per "
                                     "class. They do not sum to 1 and are NOT calibrated — "
                                     "unlike the clinical pipeline, no calibrator is fitted "
                                     "in Phase 1.",
                    "disclaimer": "Research artifact. Not for clinical use.",
                },
            })
        return results

    def predict_record(self, record_path: Path | str) -> dict[str, Any]:
        """Predict for a WFDB record given its path without an extension."""
        signal, header = load_raw_record(record_path)
        result = self.predict_array(signal)[0]
        result["source"] = {"record": str(record_path),
                            "sampling_rate": header.get("fs"),
                            "leads": header.get("sig_name")}
        return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run ECG inference.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--record", help="WFDB record path without extension.")
    source.add_argument("--npy", help="Path to a .npy waveform, (n_leads, n_samples).")
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--output", default=None, help="Write the result to this JSON path.")
    parser.add_argument("--preprocessed", action="store_true",
                        help="Input is already preprocessed (came from the cache).")
    args = parser.parse_args(argv)

    predictor = ECGPredictor.load(models_dir=args.models_dir)

    if args.record:
        result = predictor.predict_record(args.record)
    else:
        waveform = np.load(args.npy)
        result = predictor.predict_array(waveform, already_preprocessed=args.preprocessed)[0]

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Wrote prediction to %s", out_path)
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
