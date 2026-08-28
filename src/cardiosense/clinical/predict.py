"""Inference for the clinical pipeline.

Loads the three saved artifacts — preprocessor, model, calibrator — and turns raw
patient features into a structured prediction. Nothing is refit and no default
values are invented: a patient row missing a feature is imputed by the *fitted*
imputer inside the preprocessor, using statistics learned from the training split,
which is exactly the behaviour that was evaluated.

Command line::

    python -m cardiosense.clinical.predict --json '{"age": 63, "sex": 1, "cp": 1, ...}'
    python -m cardiosense.clinical.predict --csv patients.csv --output predictions.csv
    python -m cardiosense.clinical.predict --example

Python::

    from cardiosense.clinical.predict import ClinicalPredictor
    predictor = ClinicalPredictor.load()
    predictor.predict_one({"age": 63, "sex": 1, ...})

Output contract (Phase 1 — deliberately NOT fused with the other modalities)::

    {
        "prediction": 1,
        "label": "disease present",
        "probability": 0.81,               # raw model output (a score)
        "calibrated_probability": 0.74,    # frequency-matched
        "calibrated_confidence": 0.74,     # max(p, 1-p): confidence in the DECISION
        "threshold": 0.42,
        "model_version": "clinical-v0.1.0",
        ...
    }

``probability`` versus ``calibrated_confidence`` is not a stylistic distinction:
the first is a ranking score, the second is a reliability statement. Phase 2
fusion must weight by the latter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..common.config import Config, load_config
from ..common.io_utils import load_json, load_pickle
from ..common.logging_utils import get_logger
from ..common.paths import PATHS
from .calibrate import PROBABILITY_CLIP, clip_probabilities

__all__ = ["ClinicalPredictor", "main"]

logger = get_logger("clinical.predict")


class ClinicalPredictor:
    """Loads the saved clinical artifacts and produces structured predictions."""

    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        calibrator: Any | None,
        metadata: Mapping[str, Any],
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.calibrator = calibrator
        self.metadata = dict(metadata)

        self.feature_order: list[str] = list(
            self.metadata.get("raw_features", {}).get("order", [])
        )
        self.threshold: float = float(self.metadata.get("threshold", 0.5))
        self.model_version: str = str(self.metadata.get("model_version", "unknown"))
        self.label_mapping: dict[str, str] = dict(
            self.metadata.get("label_mapping", {"0": "negative", "1": "positive"})
        )

    # ------------------------------------------------------------------ load
    @classmethod
    def load(
        cls,
        models_dir: Path | str | None = None,
        cfg: Config | None = None,
    ) -> "ClinicalPredictor":
        """Load artifacts from ``models/clinical/`` (or a given directory).

        Raises:
            FileNotFoundError: If a required artifact is missing, with a message
                explaining which training step produces it.
        """
        cfg = cfg or load_config("clinical")
        directory = Path(models_dir) if models_dir else PATHS.root / str(cfg.output.models_dir)

        model_path = directory / cfg.output.model_file
        preprocessor_path = directory / cfg.output.preprocessor_file
        calibrator_path = directory / cfg.output.calibrator_file
        metadata_path = directory / cfg.output.metadata_file

        for path, produced_by in (
            (model_path, "cardiosense.clinical.train"),
            (preprocessor_path, "cardiosense.clinical.train"),
            (metadata_path, "cardiosense.clinical.train"),
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing artifact: {path}\nRun `python -m {produced_by}` first."
                )

        calibrator = None
        if calibrator_path.exists():
            calibrator = load_pickle(calibrator_path)
        else:
            logger.warning("No calibrator at %s; calibrated outputs will mirror the raw "
                           "probability and must not be treated as reliability estimates.",
                           calibrator_path)

        return cls(
            model=load_pickle(model_path),
            preprocessor=load_pickle(preprocessor_path),
            calibrator=calibrator,
            metadata=load_json(metadata_path),
        )

    # --------------------------------------------------------------- helpers
    def _frame(self, records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        """Build a DataFrame with the exact column order the preprocessor expects.

        Unknown keys are dropped with a warning; absent expected columns are added
        as NaN so the fitted imputer handles them, rather than crashing or being
        silently filled with a guess.
        """
        frame = pd.DataFrame(list(records))

        if not self.feature_order:
            return frame

        unexpected = [c for c in frame.columns if c not in self.feature_order]
        if unexpected:
            logger.warning("Ignoring columns the model was not trained on: %s", unexpected)

        missing = [c for c in self.feature_order if c not in frame.columns]
        if missing:
            logger.warning("Missing features %s — imputed by the fitted preprocessor.", missing)
            for column in missing:
                frame[column] = np.nan

        return frame[self.feature_order]

    def _probabilities(self, X_transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(raw_probability, calibrated_probability)``."""
        raw = np.asarray(self.model.predict_proba(X_transformed))[:, 1]
        if self.calibrator is None:
            return raw, raw
        calibrated = np.asarray(self.calibrator.predict_proba(X_transformed))[:, 1]
        # Isotonic calibration saturates at exactly 0 and 1 on small calibration
        # sets. Returning "probability 1.0" would be a certainty claim the data
        # cannot support, so calibrated outputs are clipped. See calibrate.py.
        return raw, np.asarray(clip_probabilities(calibrated))

    # ------------------------------------------------------------- inference
    def predict(
        self,
        records: Sequence[Mapping[str, Any]] | pd.DataFrame,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Predict for one or more patients.

        Args:
            records: A sequence of feature dicts, or a DataFrame.
            threshold: Override the stored operating threshold.

        Returns:
            One structured result dict per input row.
        """
        if isinstance(records, pd.DataFrame):
            frame = self._frame(records.to_dict(orient="records"))
        else:
            frame = self._frame(records)

        if frame.empty:
            return []

        threshold = float(threshold if threshold is not None else self.threshold)
        X_transformed = self.preprocessor.transform(frame)
        raw, calibrated = self._probabilities(X_transformed)

        results: list[dict[str, Any]] = []
        for index in range(len(frame)):
            probability = float(raw[index])
            calibrated_probability = float(calibrated[index])
            prediction = int(calibrated_probability >= threshold)
            confidence = float(max(calibrated_probability, 1.0 - calibrated_probability))

            results.append({
                "prediction": prediction,
                "label": self.label_mapping.get(str(prediction), str(prediction)),
                "probability": round(probability, 4),
                "calibrated_probability": round(calibrated_probability, 4),
                "calibrated_confidence": round(confidence, 4),
                "threshold": round(threshold, 4),
                "calibration_method": self.metadata.get("calibration", {}).get("method"),
                "probability_clip": list(PROBABILITY_CLIP),
                "model": self.metadata.get("selected_model"),
                "model_version": self.model_version,
                "modality": "clinical",
                "target": self.metadata.get("dataset", {}).get("target_rule"),
                "notes": {
                    "probability": "Raw model output. A ranking score, not a frequency claim.",
                    "calibrated_confidence": "Reliability-adjusted confidence in the returned "
                                             "decision. Use THIS for Phase 2 fusion weighting.",
                    "disclaimer": "Research artifact. Not for clinical use.",
                },
            })
        return results

    def predict_one(
        self,
        record: Mapping[str, Any],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Predict for a single patient."""
        return self.predict([record], threshold=threshold)[0]

    def example_patient(self) -> dict[str, Any]:
        """A syntactically valid example input, for smoke-testing the interface.

        The values are a plausible feature vector in the right ranges — they are a
        FORMAT example, not a real patient and not a dataset row.
        """
        template = {
            "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 240,
            "fbs": 0, "restecg": 0, "thalach": 140, "exang": 1,
            "oldpeak": 1.8, "slope": 2, "ca": 1, "thal": 7,
        }
        return {k: template.get(k) for k in (self.feature_order or template)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run clinical inference.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--json", help='Single patient as a JSON object string.')
    source.add_argument("--json-file", help="Path to a JSON file (object or array).")
    source.add_argument("--csv", help="Path to a CSV with one patient per row.")
    source.add_argument("--example", action="store_true",
                        help="Run on a built-in example feature vector.")
    parser.add_argument("--models-dir", default=None, help="Override models/clinical.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the stored operating threshold.")
    parser.add_argument("--output", default=None, help="Write results to this CSV/JSON path.")
    args = parser.parse_args(argv)

    predictor = ClinicalPredictor.load(models_dir=args.models_dir)

    if args.example:
        records: list[Mapping[str, Any]] = [predictor.example_patient()]
        logger.info("Example input: %s", records[0])
    elif args.json:
        payload = json.loads(args.json)
        records = payload if isinstance(payload, list) else [payload]
    elif args.json_file:
        payload = load_json(args.json_file)
        records = payload if isinstance(payload, list) else [payload]
    else:
        records = pd.read_csv(args.csv).to_dict(orient="records")

    results = predictor.predict(records, threshold=args.threshold)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".csv":
            pd.DataFrame([{k: v for k, v in r.items() if k != "notes"}
                          for r in results]).to_csv(out_path, index=False)
        else:
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Wrote %d predictions to %s", len(results), out_path)
    else:
        print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
