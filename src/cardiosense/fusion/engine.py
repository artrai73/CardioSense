"""The fusion engine: one call from raw patient inputs to a full assessment."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters import BaseAdapter, load_all_adapters
from .disagreement import (
    compute_agreement,
    compute_reliability_indicator,
    detect_disagreement,
)
from .schema import (
    ModalityPrediction,
    RISK_DEFINITIONS,
    reliability_label,
    risk_level,
)
from .strategies import build_strategy

MODALITIES = ("clinical", "ecg", "xray")

CLINICIAN_NOTICE = (
    "CardioSense is an academic decision-support prototype. It is not a "
    "substitute for clinical judgment, does not diagnose, and has not been "
    "clinically validated. Every output requires review by a qualified "
    "clinician."
)

NOT_FOR = [
    "autonomous diagnosis",
    "treatment or prescribing decisions",
    "use as a clinically validated medical device",
    "use without review by a qualified clinician",
]


@dataclass
class AssessmentResult:
    assessment_id: str
    created_utc: str
    fused: dict[str, Any]
    modalities: list[dict[str, Any]]
    availability: dict[str, Any]
    disagreement: dict[str, Any]
    reliability_basis: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    notice: str = CLINICIAN_NOTICE

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "created_utc": self.created_utc,
            "fused": self.fused,
            "modalities": self.modalities,
            "availability": self.availability,
            "disagreement": self.disagreement,
            "reliability_basis": self.reliability_basis,
            "warnings": self.warnings,
            "notice": self.notice,
            "not_for": NOT_FOR,
            "risk_definitions": RISK_DEFINITIONS,
        }

    def summary_line(self) -> str:
        score = self.fused.get("risk_score")
        shown = "n/a" if score is None else f"{score:.3f}"
        return (f"{self.fused['risk_level']} risk ({shown}) | "
                f"reliability {self.fused['reliability_label']} | "
                f"based on {self.availability['fusion_basis'] or 'no modality'}")


class FusionEngine:
    """Runs the three adapters and combines whatever they produced.

    Three invariants hold regardless of input:

    1. **A missing modality is never imputed.** It is excluded from the weighted
       combination and named in ``availability.missing``.
    2. **A failing modality never crashes the assessment.** It is reported as
       failed with its reason, and the others proceed.
    3. **No modality available is a valid result**, reported as ``UNKNOWN``
       rather than a fabricated number.
    """

    def __init__(self, models_root: Path | str | None = None,
                 strategy: str = "confidence_logodds",
                 adapters: dict[str, BaseAdapter] | None = None,
                 **strategy_kwargs):
        self.adapters = (adapters if adapters is not None
                         else load_all_adapters(models_root or "models"))
        self.strategy = build_strategy(strategy, **strategy_kwargs)

    # -- inference ------------------------------------------------------
    def predict_modalities(self, inputs: dict[str, Any]) -> list[ModalityPrediction]:
        out: list[ModalityPrediction] = []
        for name in MODALITIES:
            adapter = self.adapters.get(name)
            if adapter is None:
                out.append(ModalityPrediction.unavailable(name, "no adapter configured"))
            else:
                out.append(adapter.predict(inputs.get(name)))
        return out

    def fuse_predictions(self, predictions: list[ModalityPrediction]) -> AssessmentResult:
        fused = self.strategy.fuse(predictions)
        agreement = compute_agreement(predictions)
        report = detect_disagreement(predictions)
        reliability, basis = compute_reliability_indicator(
            predictions, agreement, report.severity, n_expected=len(MODALITIES))

        available = [p.modality for p in predictions if p.available and p.risk_score is not None]
        failed = [p.modality for p in predictions if p.failed]
        missing = [p.modality for p in predictions
                   if not p.available and not p.failed]

        warnings: list[str] = []
        uncalibrated = [p.modality for p in predictions
                        if p.available and p.risk_score is not None and not p.is_calibrated]
        if uncalibrated:
            warnings.append(
                f"Uncalibrated modality output ({', '.join(uncalibrated)}): these "
                "are ranking scores, not frequency claims. Do not read them as "
                "population risk.")
        if missing:
            warnings.append(
                f"These modalities were unavailable and contributed nothing: "
                f"{', '.join(missing)}. No value was substituted.")
        for p in predictions:
            if p.failed:
                warnings.append(f"{p.modality} inference failed: {p.reason}")
        if report.detected:
            warnings.append(
                f"{report.severity.title()} disagreement between modalities. "
                "The composite score averages over a real conflict.")
        if not available:
            warnings.append(
                "No modality was available. No risk estimate can be produced, "
                "and none has been invented.")

        # A weight is only reported for a modality that actually contributed.
        # This is asserted rather than assumed, because a weight appearing for a
        # missing modality would mean something was imputed upstream.
        assert all(m in available for m in fused.weights), (
            "a weight was assigned to a modality that did not contribute")

        fused_block = {
            **fused.to_dict(),
            "risk_level": risk_level(fused.risk_score),
            "reliability": round(float(reliability), 4),
            "reliability_label": reliability_label(reliability),
            "agreement": None if agreement is None else round(float(agreement), 4),
            "disagreement_flag": report.detected,
        }

        return AssessmentResult(
            assessment_id=uuid.uuid4().hex,
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            fused=fused_block,
            modalities=[p.to_dict() for p in predictions],
            availability={
                "available": available,
                "missing": missing,
                "failed": failed,
                "n_available": len(available),
                "n_expected": len(MODALITIES),
                "fusion_basis": " + ".join(available),
            },
            disagreement=report.to_dict(),
            reliability_basis=basis,
            warnings=warnings,
        )

    def assess(self, **inputs: Any) -> AssessmentResult:
        """Full pipeline. Pass any subset of ``clinical=``, ``ecg=``, ``xray=``."""
        return self.fuse_predictions(self.predict_modalities(inputs))

    # -- diagnostics ----------------------------------------------------
    def status(self) -> dict[str, Any]:
        """What this engine can actually do right now."""
        per_modality = {name: a.status() for name, a in self.adapters.items()}
        ready = [n for n, s in per_modality.items() if s["available"]]
        return {
            "strategy": self.strategy.name,
            "adapters": per_modality,
            "n_ready": len(ready),
            "ready": ready,
            "degraded": len(ready) < len(MODALITIES),
        }
