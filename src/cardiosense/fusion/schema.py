"""The contract every modality must satisfy before it can be fused.

The three CardioSense models predict different things on different cohorts:

* clinical  -> P(coronary stenosis > 50% in at least one vessel)   [UCI]
* ECG       -> five diagnostic superclasses, multi-label           [PTB-XL]
* X-ray     -> P(cardiomegaly), PA views only                      [ChestX-ray14]

Fusing them requires a common scale, and inventing one silently would be the
single most dishonest thing this layer could do. So the mapping is written down
here, printed in every report, and referred to as a *composite cardiovascular
abnormality indicator* rather than a diagnosis of anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: How each modality's native output becomes a risk score in [0, 1].
RISK_DEFINITIONS: dict[str, str] = {
    "clinical": (
        "Calibrated P(coronary stenosis >50% in >=1 major vessel), used directly. "
        "Source: UCI Heart Disease, angiographic ground truth."
    ),
    "ecg": (
        "1 - P(NORM), where NORM is an explicitly trained class. Used rather than "
        "a maximum over abnormal classes because it is the model's own estimate "
        "that the trace is not normal, not an ad-hoc aggregation. "
        "Source: PTB-XL diagnostic superclasses, multi-label."
    ),
    "xray": (
        "Prior-corrected P(cardiomegaly). The training cohort was negative-"
        "subsampled, so the raw output is conditioned on an enriched prior and "
        "must be shifted before it is read as population risk. "
        "Source: NIH ChestX-ray14, PA views."
    ),
}

#: Risk band cut-points. A stated project convention, not a clinical standard.
RISK_THRESHOLDS: dict[str, float] = {"low_max": 0.33, "intermediate_max": 0.66}

#: Reliability band cut-points. Also a project convention.
RELIABILITY_BANDS: dict[str, float] = {"low_max": 0.40, "moderate_max": 0.70}

#: Expected calibration error at which a model is considered wholly unreliable.
#: 0.25 is the ECE of a model that is confidently wrong half the time.
ECE_FLOOR = 0.25

#: Reliability assigned to a modality whose probabilities were never calibrated.
#: Deliberately mid-scale: such a model still ranks, but its numbers are not
#: frequency claims, so it must not be allowed to dominate a weighted average.
UNCALIBRATED_RELIABILITY = 0.50


def band(value: float | None, cuts: dict[str, float], labels: tuple[str, str, str]) -> str:
    """Map a scalar onto three named bands."""
    if value is None or not np.isfinite(value):
        return "UNKNOWN"
    low, mid = cuts[next(iter(cuts))], cuts[list(cuts)[1]]
    if value <= low:
        return labels[0]
    if value <= mid:
        return labels[1]
    return labels[2]


def risk_level(score: float | None) -> str:
    return band(score, RISK_THRESHOLDS, ("LOW", "INTERMEDIATE", "HIGH"))


def reliability_label(value: float | None) -> str:
    return band(value, RELIABILITY_BANDS, ("LOW", "MODERATE", "HIGH"))


def reliability_from_ece(ece: float | None) -> float:
    """Turn a calibration error into a reliability scalar in [0, 1].

    ``1 - ECE / 0.25``, clipped. A model with zero calibration error is fully
    reliable; one at or beyond :data:`ECE_FLOOR` contributes nothing.

    Returns :data:`UNCALIBRATED_RELIABILITY` when no ECE is available, because
    "unknown" is not the same as "perfect" and must not be treated as either.
    """
    if ece is None or not np.isfinite(ece):
        return UNCALIBRATED_RELIABILITY
    return float(np.clip(1.0 - float(ece) / ECE_FLOOR, 0.0, 1.0))


def decisiveness(probability: float) -> float:
    """How far from a coin flip this particular prediction is: ``|2p - 1|``.

    Case-specific, unlike reliability. A model can be trustworthy in general
    (high reliability) and still be unsure about one patient (low decisiveness).
    """
    return float(abs(2.0 * float(probability) - 1.0))


def prior_correct(p: np.ndarray | float, train_prevalence: float,
                  target_prevalence: float,
                  clip: tuple[float, float] = (1e-6, 1 - 1e-6)) -> np.ndarray:
    """Shift probabilities from the training prior to a target prior, in odds space.

    Strictly monotone, so it changes no ranking and no AUC -- only the prior the
    numbers are expressed under.

    The clip is far tighter than the usual (0.001, 0.999) because correcting
    toward a low population prevalence legitimately produces very small values.
    At a prevalence of 2.5% a risk of 0.0003 is meaningful, and clipping it at
    0.001 would collapse a large fraction of the negatives onto the floor,
    destroying the ranking among them.
    """
    p = np.clip(np.asarray(p, dtype=float), *clip)
    odds = p / (1.0 - p)
    ratio = ((target_prevalence / (1.0 - target_prevalence))
             * ((1.0 - train_prevalence) / train_prevalence))
    corrected = odds * ratio
    return np.clip(corrected / (1.0 + corrected), *clip)


@dataclass
class ModalityPrediction:
    """One modality's contribution, in the common vocabulary.

    Attributes:
        modality: ``clinical`` | ``ecg`` | ``xray``.
        risk_score: Mapped onto [0, 1] per :data:`RISK_DEFINITIONS`. ``None``
            when the modality was unavailable -- never substituted.
        reliability: Model-level trust, from held-out calibration error.
        is_calibrated: ``False`` means the probability is a ranking score, not
            a frequency claim. Carried through to every display.
        available: ``False`` for a modality that was not supplied.
        failed: ``True`` when it was supplied but inference raised.
        native_labels: The modality's own output, for the report.
    """

    modality: str
    risk_score: float | None = None
    reliability: float = UNCALIBRATED_RELIABILITY
    is_calibrated: bool = False
    available: bool = True
    failed: bool = False
    reason: str = ""
    native_labels: dict[str, Any] = field(default_factory=dict)
    class_probabilities: dict[str, float] = field(default_factory=dict)
    model_version: str = ""

    @classmethod
    def unavailable(cls, modality: str, reason: str = "not supplied") -> "ModalityPrediction":
        return cls(modality=modality, available=False, reason=reason,
                   reliability=0.0)

    @classmethod
    def failed_inference(cls, modality: str, reason: str) -> "ModalityPrediction":
        return cls(modality=modality, available=False, failed=True,
                   reason=reason, reliability=0.0)

    @property
    def decisiveness(self) -> float:
        if self.risk_score is None:
            return 0.0
        return decisiveness(self.risk_score)

    @property
    def confidence(self) -> float:
        """``reliability x decisiveness`` -- the weight this modality earns.

        Both terms are needed and each blocks a different failure. Without
        reliability an overconfident model dominates, and ``pos_weight`` training
        pushes both neural models toward extreme outputs by construction.
        Without decisiveness the fusion cannot adapt per patient.
        """
        return float(self.reliability * self.decisiveness)

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "available": self.available,
            "failed": self.failed,
            "reason": self.reason,
            "risk_score": None if self.risk_score is None else round(float(self.risk_score), 5),
            "reliability": round(float(self.reliability), 4),
            "is_calibrated": self.is_calibrated,
            "decisiveness": round(self.decisiveness, 4),
            "confidence": round(self.confidence, 4),
            "native_labels": self.native_labels,
            "class_probabilities": {k: round(float(v), 4)
                                    for k, v in self.class_probabilities.items()},
            "model_version": self.model_version,
            "risk_definition": RISK_DEFINITIONS.get(self.modality, ""),
        }
