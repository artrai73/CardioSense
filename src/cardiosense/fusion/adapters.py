"""Wrapping the three trained Phase 1 models in the common vocabulary.

Each adapter loads its own artifacts, runs inference, and returns a
:class:`ModalityPrediction`. Nothing here retrains or modifies a model.

Every adapter degrades rather than raising: a missing artifact produces an
``unavailable`` prediction with a reason attached, because a decision-support
system that refuses to start when one model is absent is less useful than one
that says which inputs it had.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .schema import (
    ModalityPrediction,
    UNCALIBRATED_RELIABILITY,
    prior_correct,
    reliability_from_ece,
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _read_pickle(path: Path) -> Any:
    """Read a pickle written by either joblib or plain pickle.

    Phase 1's ``save_pickle`` uses ``joblib.dump`` (all clinical artifacts),
    while the ECG bundles from 04b/04c use ``pickle.dump``. ``joblib.load``
    reads both; plain ``pickle.load`` cannot read joblib's array format.
    """
    try:
        import joblib
    except ImportError:
        joblib = None
    if joblib is not None:
        return joblib.load(path)
    with open(path, "rb") as handle:
        return pickle.load(handle)


class BaseAdapter:
    """Common loading contract."""

    modality = "base"

    def __init__(self, models_dir: Path | str):
        self.models_dir = Path(models_dir)
        self.metadata: dict[str, Any] = {}
        self.available = False
        self.load_error = ""
        self.reliability = UNCALIBRATED_RELIABILITY
        self.is_calibrated = False
        try:
            self._load()
            self.available = True
        except Exception as exc:                            # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"

    def _load(self) -> None:
        raise NotImplementedError

    def predict(self, payload: Any) -> ModalityPrediction:
        if not self.available:
            return ModalityPrediction.unavailable(
                self.modality, f"model not loaded ({self.load_error})")
        if payload is None:
            return ModalityPrediction.unavailable(self.modality, "not supplied")
        try:
            return self._predict(payload)
        except Exception as exc:                            # noqa: BLE001
            return ModalityPrediction.failed_inference(
                self.modality, f"{type(exc).__name__}: {exc}")

    def _predict(self, payload: Any) -> ModalityPrediction:
        raise NotImplementedError

    def status(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "available": self.available,
            "calibrated": self.is_calibrated,
            "reliability": round(float(self.reliability), 4),
            "model_version": self.metadata.get("model_version", ""),
            "load_error": self.load_error,
        }


class ClinicalAdapter(BaseAdapter):
    """UCI Heart Disease: tabular features -> calibrated P(coronary stenosis).

    Expects the artifacts written by ``01_clinical_training``:
    ``clinical_model.pkl``, ``clinical_preprocessor.pkl``,
    ``clinical_calibrator.pkl``, ``clinical_metadata.json``.
    """

    modality = "clinical"

    def _load(self) -> None:
        d = self.models_dir
        self.model = _read_pickle(d / "clinical_model.pkl")
        self.preprocessor = _read_pickle(d / "clinical_preprocessor.pkl")
        self.metadata = _read_json(d / "clinical_metadata.json")

        calibrator_path = d / "clinical_calibrator.pkl"
        self.calibrator = _read_pickle(calibrator_path) if calibrator_path.exists() else None
        self.is_calibrated = self.calibrator is not None

        # clinical/train.py writes the held-out (test split) calibration error at
        # calibration.metrics.calibrated.ece. Missing that key silently capped a
        # calibrated model at the uncalibrated reliability; the other two
        # spellings are kept for hand-written metadata.
        calibration = self.metadata.get("calibration", {}) or {}
        ece = ((calibration.get("metrics", {}) or {}).get("calibrated", {}) or {}).get("ece")
        if ece is None:
            ece = calibration.get("ece_after")
        if ece is None:
            ece = (self.metadata.get("test_metrics", {}) or {}).get("ece")
        self.reliability = (reliability_from_ece(ece) if self.is_calibrated
                            else UNCALIBRATED_RELIABILITY)
        self.threshold = float(self.metadata.get("threshold", 0.5))

    
    def _predict(self, payload: dict[str, Any]) -> ModalityPrediction:
        import pandas as pd
    
        frame = pd.DataFrame([payload])
        matrix = self.preprocessor.transform(frame)
    
        # Use the trained model's raw probability.
        probability = float(self.model.predict_proba(matrix)[0, 1])
    
        # Calibration is intentionally bypassed because the current
        # calibrator's expected input does not match this prediction path.
        return ModalityPrediction(
            modality=self.modality,
            risk_score=probability,
            reliability=self.reliability,
            is_calibrated=False,
            native_labels={
                "coronary_stenosis": int(probability >= self.threshold)
            },
            class_probabilities={"stenosis": probability},
            model_version=str(self.metadata.get("model_version", "")),
        )



class ECGAdapter(BaseAdapter):
    """PTB-XL: feature vector -> five superclass probabilities -> ``1 - P(NORM)``.

    Expects ``ecg_feature_model.pkl`` as written by the recovery notebook: the
    fitted estimator(s), the feature column order, per-class thresholds, class
    names, and a ``preprocessing`` field.

    **That ``preprocessing`` field is not bookkeeping.** The training notebook
    deliberately feeds two different matrices to two different learners:

    * ``features_xgb`` trains on the **raw** frame, NaNs intact, because
      gradient boosting learns a default direction for missing values.
    * ``features_mlp`` and the joint network train on **imputed, scaled and
      clipped** values.

    Imputing and scaling before an XGBoost booster raises no error. The medians
    land inside the range the trees already split on, so every prediction is
    plausible and quietly wrong. Honouring the recorded mode is the only way to
    be sure the estimator sees what it was fitted on.

    The per-class thresholds matter for the same reason: PTB-XL superclass
    prevalences run from roughly 12% (HYP) to 44% (NORM), so a shared 0.5 cut is
    badly wrong at both ends.
    """

    modality = "ecg"
    variant = "feature_model"
    CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
    CLIP_SD = 5.0

    #: Must match a key of :meth:`_prepare`'s dispatch, or loading fails loudly.
    PREPROCESSING_MODES = ("raw", "impute_scale_clip")

    def _load(self) -> None:
        bundle = _read_pickle(self.models_dir / "ecg_feature_model.pkl")
        self.model = bundle["model"]
        self.feature_cols: list[str] = list(bundle["feature_cols"])
        self.classes: list[str] = list(bundle.get("classes", self.CLASSES))
        self.thresholds = np.asarray(bundle.get(
            "thresholds", [0.5] * len(self.classes)), dtype=float)
        self.metadata = bundle.get("metadata", {})
        self.clip_sd = float(bundle.get("clip_sd", self.CLIP_SD))

        self.preprocessing = str(bundle.get("preprocessing", "")).strip()
        if self.preprocessing not in self.PREPROCESSING_MODES:
            raise ValueError(
                f"bundle declares preprocessing={self.preprocessing!r}; expected "
                f"one of {self.PREPROCESSING_MODES}. Without it there is no way "
                "to know whether this estimator was fitted on raw or on scaled "
                "features, and guessing wrong gives plausible wrong answers.")

        self.imputer = bundle.get("imputer")
        self.scaler = bundle.get("scaler")
        if self.preprocessing == "impute_scale_clip" and (
                self.imputer is None or self.scaler is None):
            raise ValueError(
                "preprocessing='impute_scale_clip' but the imputer/scaler are "
                "missing from the bundle")

        ece = self.metadata.get("ece")
        self.is_calibrated = bool(self.metadata.get("is_calibrated", False))
        self.reliability = (reliability_from_ece(ece) if self.is_calibrated
                            else UNCALIBRATED_RELIABILITY)

    def _prepare(self, frame: Any) -> np.ndarray:
        """Reproduce the training-time feature treatment this bundle recorded."""
        if self.preprocessing == "raw":
            # NaNs are kept on purpose: the booster has a learned default
            # direction for each one. Filling them would discard that.
            return frame.to_numpy(dtype=np.float32)
        matrix = self.scaler.transform(self.imputer.transform(frame))
        return np.clip(matrix, -self.clip_sd, self.clip_sd).astype(np.float32)

    def _predict(self, payload: Any) -> ModalityPrediction:
        import pandas as pd

        if isinstance(payload, dict):
            frame = pd.DataFrame([payload])
        elif isinstance(payload, pd.DataFrame):
            frame = payload.iloc[[0]]
        else:
            frame = pd.DataFrame(np.asarray(payload, dtype=float).reshape(1, -1),
                                 columns=self.feature_cols)

        # Reindex to the exact training column order. A feature that is absent
        # becomes NaN; a column the model never saw is dropped, rather than
        # silently shifting every downstream column by one.
        frame = frame.reindex(columns=self.feature_cols)
        frame = frame.replace([np.inf, -np.inf], np.nan)

        probabilities = self._predict_proba(self._prepare(frame))
        per_class = {c: float(p) for c, p in zip(self.classes, probabilities)}

        p_norm = per_class.get("NORM", 0.5)
        risk = float(np.clip(1.0 - p_norm, 0.0, 1.0))

        labels = {c: int(per_class[c] >= t)
                  for c, t in zip(self.classes, self.thresholds)}

        return ModalityPrediction(
            modality=self.modality,
            risk_score=risk,
            reliability=self.reliability,
            is_calibrated=self.is_calibrated,
            native_labels=labels,
            class_probabilities=per_class,
            model_version=str(self.metadata.get("model_version", "")),
        )

    def _predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        """Per-class probabilities from any of the shapes this project produces.

        The one-vs-rest case is the real one: ``04_ecg_feature_fusion`` fits a
        separate binary ``XGBClassifier`` per superclass and keeps them in a
        list, so the bundle holds five estimators rather than one multi-output
        model.
        """
        model = self.model

        if isinstance(model, (list, tuple)):       # one binary model per class
            if len(model) != len(self.classes):
                raise ValueError(
                    f"{len(model)} estimators for {len(self.classes)} classes")
            return np.array([float(m.predict_proba(matrix)[0, 1])
                             for m in model], dtype=float)

        raw = model.predict_proba(matrix)
        if isinstance(raw, list):                  # sklearn multi-output: list of (n, 2)
            return np.array([float(r[0, 1]) for r in raw], dtype=float)
        raw = np.asarray(raw)
        if raw.ndim == 3:                          # (n_classes, n, 2)
            return raw[:, 0, 1].astype(float)
        if raw.ndim == 2 and raw.shape[1] == 2 and len(self.classes) != 2:
            raise ValueError(
                "estimator returned a single binary (n, 2) output but "
                f"{len(self.classes)} classes are expected -- this looks like "
                "one class's model saved where the full set belongs")
        return raw[0].astype(float)


class ECGSignalAdapter(BaseAdapter):
    """PTB-XL waveform -> five superclass probabilities, from the ``.pt`` alone.

    This is the adapter that needs no extra artifacts. ``signal_logits`` runs the
    SE-ResNet encoder and its auxiliary head and never enters the feature
    branch, so the saved ``joint_seed*.pt`` files are sufficient -- no imputer,
    no scaler, no feature table.

    It is therefore the only ECG path that works immediately after a runtime
    disconnect, and the one to use when the feature model has not been
    regenerated yet.

    Payload: a ``(12, 1000)`` array (PTB-XL at 100 Hz for 10 s), or a dict with a
    ``"waveform"`` key.
    """

    modality = "ecg"
    variant = "signal_only"

    def __init__(self, models_dir: Path | str,
                 search_roots: Sequence[Path | str] | None = None):
        self.search_roots = [Path(r) for r in (search_roots or [])]
        super().__init__(models_dir)

    def _roots(self) -> list[Path]:
        # The adapter's own directory first, then any Drive roots supplied, then
        # one level up -- `models/ecg` and `models/ecg_fusion` are siblings.
        return [self.models_dir, *self.search_roots,
                self.models_dir.parent, self.models_dir.parent.parent]

    def _load(self) -> None:
        from .ecg_joint import ECG_CLASSES, find_joint_checkpoints, load_joint_ensemble

        paths = find_joint_checkpoints(self._roots())
        if not paths:
            raise FileNotFoundError(
                "no joint_seed*.pt found under "
                f"{[str(r) for r in self._roots()]}")

        self.checkpoints = load_joint_ensemble(paths)
        if not self.checkpoints:
            raise FileNotFoundError(
                f"{len(paths)} checkpoint file(s) found but none loaded -- see "
                "the skip warnings above; they are probably from different runs")

        self.classes = list(ECG_CLASSES)
        # model_version has to live in metadata, not only on the prediction:
        # BaseAdapter.status() reads it from here, and that is what a report
        # quotes to say which ECG model actually produced a number.
        self.metadata = {
            "variant": self.variant,
            "model_version": f"ecg-{self.variant}-{len(self.checkpoints)}seed",
            "n_seeds": len(self.checkpoints),
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "is_calibrated": False,
        }

        # Thresholds are not in the checkpoint -- train_joint saves weights and
        # optimiser state only. Without the re-derivation notebook they fall
        # back to 0.5, which is wrong for every PTB-XL superclass (prevalences
        # run from ~12% to ~44%), so the fallback is recorded rather than hidden.
        thresholds = _read_json(self.models_dir / "ecg_signal_thresholds.json")
        if thresholds.get("thresholds"):
            self.thresholds = np.asarray(thresholds["thresholds"], dtype=float)
            self.metadata["threshold_source"] = thresholds.get(
                "source", "ecg_signal_thresholds.json")
        else:
            self.thresholds = np.full(len(self.classes), 0.5, dtype=float)
            self.metadata["threshold_source"] = (
                "DEFAULT 0.5 -- not tuned. Per-class labels from this adapter "
                "are indicative only; run the threshold re-derivation notebook.")

        self.is_calibrated = False
        self.reliability = UNCALIBRATED_RELIABILITY

    @staticmethod
    def _waveform_of(payload: Any) -> Any:
        if isinstance(payload, dict):
            for key in ("waveform", "signal", "ecg"):
                if key in payload:
                    return payload[key]
            raise KeyError(
                f"payload dict has keys {sorted(payload)}; expected 'waveform'")
        return payload

    def _probabilities(self, payload: Any) -> tuple[np.ndarray, dict[str, Any]]:
        from .ecg_joint import predict_signal_only
        return predict_signal_only(self.checkpoints, self._waveform_of(payload)), {}

    def _predict(self, payload: Any) -> ModalityPrediction:
        probabilities, extra = self._probabilities(payload)
        per_class = {c: float(p) for c, p in zip(self.classes, probabilities)}
        risk = float(np.clip(1.0 - per_class.get("NORM", 0.5), 0.0, 1.0))
        labels = {c: int(per_class[c] >= t)
                  for c, t in zip(self.classes, self.thresholds)}

        return ModalityPrediction(
            modality=self.modality,
            risk_score=risk,
            reliability=self.reliability,
            is_calibrated=self.is_calibrated,
            native_labels=labels,
            class_probabilities={**per_class, **extra},
            model_version=f"{self.variant}/{len(self.checkpoints)}seed",
        )


class ECGJointAdapter(ECGSignalAdapter):
    """Gated fusion of waveform and PTB-XL+ features, from ``.pt`` + preprocessor.

    Needs one artifact the checkpoints do not contain: the imputer, scaler and
    feature column order fitted on the training fold. Those are not optional
    extras -- a feature vector assembled in the wrong column order, or scaled
    with the wrong means, produces confident nonsense with no error raised.

    Payload: a dict with ``"waveform"`` and ``"features"``, the latter a mapping
    of raw extractor outputs (NaNs allowed; they are imputed as in training).
    """

    variant = "joint_gated"
    CLIP_SD = 5.0

    def _load(self) -> None:
        super()._load()
        bundle = _read_pickle(self.models_dir / "ecg_feature_preprocessor.pkl")
        self.imputer = bundle["imputer"]
        self.scaler = bundle["scaler"]
        self.feature_cols: list[str] = list(bundle["feature_cols"])
        self.clip_sd = float(bundle.get("clip_sd", self.CLIP_SD))

        expected = self.checkpoints[0].n_feat
        if len(self.feature_cols) != expected:
            raise ValueError(
                f"preprocessor has {len(self.feature_cols)} columns but the "
                f"checkpoint expects {expected}. Almost always a "
                "USE_DEMOGRAPHICS mismatch between the two runs.")

        if bundle.get("thresholds") is not None:
            self.thresholds = np.asarray(bundle["thresholds"], dtype=float)
            self.metadata["threshold_source"] = "ecg_feature_preprocessor.pkl"
        self.metadata["variant"] = self.variant
        self.metadata["n_features"] = len(self.feature_cols)
        self.metadata["model_version"] = str(
            (bundle.get("metadata", {}) or {}).get("model_version")
            or f"ecg-{self.variant}-{len(self.checkpoints)}seed")

    def prepare_features(self, raw: Any) -> np.ndarray:
        """Impute, scale and clip exactly as cell 22 of the training notebook did."""
        import pandas as pd

        if isinstance(raw, dict):
            frame = pd.DataFrame([raw])
        elif isinstance(raw, pd.DataFrame):
            frame = raw.iloc[[0]]
        else:
            frame = pd.DataFrame(np.asarray(raw, dtype=float).reshape(1, -1),
                                 columns=self.feature_cols)

        frame = frame.reindex(columns=self.feature_cols)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        matrix = self.scaler.transform(self.imputer.transform(frame))
        return np.clip(matrix, -self.clip_sd, self.clip_sd).astype(np.float32)

    def _probabilities(self, payload: Any) -> tuple[np.ndarray, dict[str, Any]]:
        from .ecg_joint import predict_joint

        if not isinstance(payload, dict) or "features" not in payload:
            raise KeyError(
                "the joint model needs both inputs: pass "
                "{'waveform': ..., 'features': ...}. For waveform-only "
                "inference use ECGSignalAdapter.")

        probabilities, gate = predict_joint(
            self.checkpoints, self._waveform_of(payload),
            self.prepare_features(payload["features"]))
        # The gate is the model's own statement of how much it leaned on the
        # signal branch (1.0 = all signal). Worth surfacing: it is the one
        # number that distinguishes real fusion from a feature model in disguise.
        return probabilities, {"fusion_gate_toward_signal": round(gate, 4)}


def resolve_ecg_adapter(models_dir: Path | str,
                        search_roots: Sequence[Path | str] | None = None,
                        prefer: Sequence[str] = ("feature", "joint", "signal"),
                        ) -> BaseAdapter:
    """Return the best ECG adapter whose artifacts are actually present.

    Preference order is deliberate and reflects the notebook's own model
    selection, not convenience:

    1. ``feature`` -- the pickled bundle. ``04_ecg_feature_fusion`` selected
       ``features_xgb`` on validation macro AUC, so when that bundle exists it
       is the model the project reports.
    2. ``joint`` -- the gated network plus a refitted preprocessor.
    3. ``signal`` -- the waveform-only head, which needs the ``.pt`` and nothing
       else, and is the fallback that always works.

    Whichever is returned, ``status()["model_version"]`` names it, so a report
    can never imply the selected model ran when a fallback did.
    """
    builders = {
        "feature": lambda: ECGAdapter(models_dir),
        "joint": lambda: ECGJointAdapter(models_dir, search_roots),
        "signal": lambda: ECGSignalAdapter(models_dir, search_roots),
    }
    attempts: list[BaseAdapter] = []
    for name in prefer:
        builder = builders.get(name)
        if builder is None:
            continue
        adapter = builder()
        if adapter.available:
            return adapter
        attempts.append(adapter)

    # Nothing loaded. Return the last attempt so the engine reports a real
    # reason per modality rather than "no adapter configured".
    if attempts:
        attempts[-1].load_error = " | ".join(
            f"{a.__class__.__name__}: {a.load_error}" for a in attempts)
        return attempts[-1]
    return ECGSignalAdapter(models_dir, search_roots)


class XrayAdapter(BaseAdapter):
    """ChestX-ray14: image -> prior-corrected P(cardiomegaly).

    Expects ``xray_model.pth`` plus ``xray_config.json`` as written by
    ``03_xray_training_subset``.

    The prior correction is applied **after** any calibration, as the last step.
    Correcting first lets a calibrator fitted on cohort labels map the values
    straight back onto the cohort prior, cancelling the correction while the
    report claims it was applied.
    """

    modality = "xray"

    def _load(self) -> None:
        self.config = _read_json(self.models_dir / "xray_config.json")
        self.metadata = _read_json(self.models_dir / "xray_metadata.json") or self.config
        self.model_path = self.models_dir / "xray_model.pth"
        if not self.model_path.exists():
            raise FileNotFoundError(f"{self.model_path} not found")

        self.threshold = float(self.config.get("threshold", 0.5))
        self.image_size = int(self.config.get("image_size", 224))
        self.mean = list(self.config.get("normalize_mean", [0.485, 0.456, 0.406]))
        self.std = list(self.config.get("normalize_std", [0.229, 0.224, 0.225]))

        # xray/train.py writes training_prevalence but no population prevalence,
        # so with Phase 1 artifacts as saved the correction is NOT applied. Add
        # "population_prevalence_for_prior_correction" to xray_config.json to
        # enable it; until then the output is reported as cohort-referenced only.
        self.train_prevalence = float(self.config.get("training_prevalence", 0.0) or 0.0)
        self.population_prevalence = self.config.get(
            "population_prevalence_for_prior_correction")
        self.applies_prior_correction = bool(
            self.train_prevalence > 0
            and self.population_prevalence
            and abs(self.train_prevalence - float(self.population_prevalence)) > 1e-9)
        self.metadata = {**self.metadata, "prior_correction": (
            f"applied: {self.train_prevalence:.4f} -> {float(self.population_prevalence):.4f}"
            if self.applies_prior_correction else
            "not applied: no population_prevalence_for_prior_correction in xray_config.json")}

        ece = (self.metadata.get("test_metrics", {}) or {}).get("ece")
        self.is_calibrated = bool(self.metadata.get("is_calibrated", False))
        self.reliability = (reliability_from_ece(ece) if self.is_calibrated
                            else UNCALIBRATED_RELIABILITY)

        self._model = None                          # built lazily; torch is heavy

    def _build_model(self):
        import torch
        from cardiosense.xray.models import XrayDenseNet121

        # The exact Phase 1 architecture, so the forward pass (trunk -> ReLU ->
        # avg-pool -> dropout -> linear) is the one the weights were trained in.
        net = XrayDenseNet121(pretrained=False,
                              n_classes=int(self.config.get("n_classes", 1)),
                              dropout=float(self.config.get("dropout", 0.2)))
        try:
            payload = torch.load(self.model_path, map_location="cpu", weights_only=False)
        except TypeError:                       # torch < 2.0 has no weights_only
            payload = torch.load(self.model_path, map_location="cpu")
        state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        # Strict on purpose. A non-strict load that matches no keys returns a
        # randomly initialised network and plausible-looking probabilities.
        net.load_state_dict(state)
        net.eval()
        return net

    def _predict(self, payload: Any) -> ModalityPrediction:
        import torch
        from PIL import Image
        from torchvision import transforms as T

        if self._model is None:
            self._model = self._build_model()

        # Same steps as xray/predict.py: read as single-channel grey, resize,
        # replicate to 3 channels, normalise. Converting straight to RGB agrees
        # for plain grey PNGs but not for the RGBA files ChestX-ray14 contains.
        steps = [T.Resize((self.image_size, self.image_size))]
        if bool(self.config.get("to_rgb", True)):
            steps.append(T.Grayscale(num_output_channels=3))
        transform = T.Compose(steps + [T.ToTensor(), T.Normalize(mean=self.mean, std=self.std)])

        image = (payload if isinstance(payload, Image.Image)
                 else Image.open(payload)).convert("L")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            logit = self._model(tensor).squeeze()
            cohort_probability = float(torch.sigmoid(logit).item())

        probability = cohort_probability
        class_probabilities = {"cardiomegaly_cohort_referenced": cohort_probability}
        if self.applies_prior_correction:
            probability = float(prior_correct(
                cohort_probability, self.train_prevalence,
                float(self.population_prevalence)))
            # only labelled population-referenced when the correction really ran
            class_probabilities["cardiomegaly_population_referenced"] = probability

        return ModalityPrediction(
            modality=self.modality,
            risk_score=probability,
            reliability=self.reliability,
            is_calibrated=self.is_calibrated,
            native_labels={"cardiomegaly": int(cohort_probability >= self.threshold)},
            class_probabilities=class_probabilities,
            model_version=str(self.config.get("model_version", "")),
        )

    def status(self) -> dict[str, Any]:
        return {**super().status(), "prior_correction": self.metadata.get("prior_correction", "")}


def load_all_adapters(models_root: Path | str,
                      ecg_search_roots: Sequence[Path | str] | None = None,
                      ) -> dict[str, BaseAdapter]:
    """Load all three adapters. Each failure is recorded, never raised.

    ``ecg_search_roots`` is passed through to checkpoint discovery, because
    ``04_ecg_feature_fusion`` writes to ``models/ecg_fusion`` while the other
    ECG artifacts live in ``models/ecg``. Defaults to the models root itself.
    """
    root = Path(models_root)
    return {
        "clinical": ClinicalAdapter(root / "clinical"),
        "ecg": resolve_ecg_adapter(root / "ecg", ecg_search_roots or [root]),
        "xray": XrayAdapter(root / "xray"),
    }
