"""Detecting when modalities contradict each other.

The system flags a conflict and lowers its own reliability. It does **not**
decide which modality is right, because it has no basis on which it could: a
disagreement may mean one model is wrong, or may mean the patient genuinely has
a normal ECG and an enlarged heart. Both are real clinical pictures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from .schema import ModalityPrediction

#: Conflict score cut-points. Project conventions, not clinical standards.
DISAGREEMENT_THRESHOLDS = {"mild": 0.15, "moderate": 0.30, "severe": 0.50}

#: How much each severity multiplies the reliability indicator by.
RELIABILITY_PENALTY = {"NONE": 1.00, "MILD": 0.85, "MODERATE": 0.70, "SEVERE": 0.50}


@dataclass
class DisagreementReport:
    detected: bool
    severity: str
    conflict_score: float
    pairs: list[dict[str, Any]] = field(default_factory=list)
    interpretation: str = ""
    recommended_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "severity": self.severity,
            "conflict_score": round(float(self.conflict_score), 4),
            "pairs": self.pairs,
            "interpretation": self.interpretation,
            "recommended_action": self.recommended_action,
        }


def detect_disagreement(predictions: Sequence[ModalityPrediction]) -> DisagreementReport:
    """Score pairwise conflict, weighted by how sure each modality is.

    The weighting is the point. Two modalities landing on 0.4 and 0.6 are both
    saying "I don't know", which is not a conflict worth a clinician's
    attention. Two *confident* modalities at 0.05 and 0.95 are making
    incompatible claims about the same patient.

        conflict = |risk_i - risk_j| x min(confidence_i, confidence_j)

    Taking the minimum means an unsure modality cannot manufacture a conflict on
    its own -- both sides have to be committed for the gap to count.
    """
    usable = [p for p in predictions
              if p.available and p.risk_score is not None and np.isfinite(p.risk_score)]

    if len(usable) < 2:
        return DisagreementReport(
            False, "NOT ASSESSABLE", 0.0,
            interpretation=("Fewer than two modalities were available, so the "
                            "modalities cannot be compared. This is not the same "
                            "as them agreeing."),
            recommended_action="Interpret the single available modality on its own terms.")

    pairs: list[dict[str, Any]] = []
    for a, b in combinations(usable, 2):
        gap = abs(a.risk_score - b.risk_score)
        weight = min(a.confidence, b.confidence)
        score = gap * weight
        pairs.append({
            "modalities": [a.modality, b.modality],
            "risk_scores": [round(float(a.risk_score), 4), round(float(b.risk_score), 4)],
            "gap": round(float(gap), 4),
            "min_confidence": round(float(weight), 4),
            "conflict_score": round(float(score), 4),
        })

    worst = max(pairs, key=lambda d: d["conflict_score"])
    score = float(worst["conflict_score"])

    if score >= DISAGREEMENT_THRESHOLDS["severe"]:
        severity = "SEVERE"
    elif score >= DISAGREEMENT_THRESHOLDS["moderate"]:
        severity = "MODERATE"
    elif score >= DISAGREEMENT_THRESHOLDS["mild"]:
        severity = "MILD"
    else:
        severity = "NONE"

    detected = severity != "NONE"
    if detected:
        a, b = worst["modalities"]
        ra, rb = worst["risk_scores"]
        interpretation = (
            f"{severity.title()} disagreement: {a} estimates {ra:.2f} while {b} "
            f"estimates {rb:.2f}, and both are reasonably committed. The system "
            f"does not determine which modality is correct -- a divergence of this "
            f"kind can reflect a model error or a genuine clinical picture in "
            f"which the modalities legitimately differ.")
        action = ("Review the per-modality evidence individually rather than "
                  "relying on the composite score, which averages over a real "
                  "conflict.")
    else:
        interpretation = ("The available modalities are broadly consistent, "
                          "or any gap between them is between modalities that "
                          "are themselves unsure.")
        action = "No conflict-specific review indicated."

    return DisagreementReport(detected, severity, score, pairs,
                              interpretation, action)


def compute_agreement(predictions: Sequence[ModalityPrediction]) -> float | None:
    """A simple 0-1 agreement figure: 1 minus the mean pairwise gap.

    ``None`` when fewer than two modalities are available, because "no
    comparison was possible" must not be reported as "they agreed".
    """
    usable = [p for p in predictions
              if p.available and p.risk_score is not None and np.isfinite(p.risk_score)]
    if len(usable) < 2:
        return None
    gaps = [abs(a.risk_score - b.risk_score) for a, b in combinations(usable, 2)]
    return float(1.0 - np.mean(gaps))


def compute_reliability_indicator(
    predictions: Sequence[ModalityPrediction],
    agreement: float | None,
    severity: str = "NONE",
    n_expected: int = 3,
) -> tuple[float, dict[str, Any]]:
    """How much internal support this assessment has, in [0, 1].

    A weighted geometric mean of four factors:

    =================  ======  ====================================================
    factor             weight  meaning
    =================  ======  ====================================================
    model reliability   0.40   mean calibration-derived trust of what was used
    coverage            0.25   how many of the three modalities were available
    agreement           0.25   how closely they concur
    decisiveness        0.10   how committed they are
    =================  ======  ====================================================

    A geometric mean rather than arithmetic, so a near-zero factor drags the
    whole indicator down instead of being averaged away -- one wholly unreliable
    modality should not be rescued by three good ones.

    **The weights are a stated design choice, not an empirical fit.** No dataset
    exists on which to fit them for this system, and every display of the
    indicator says so. It describes internal support, not correctness.
    """
    usable = [p for p in predictions
              if p.available and p.risk_score is not None and np.isfinite(p.risk_score)]

    if not usable:
        return 0.0, {"reason": "no modality available", "factors": {}}

    model_reliability = float(np.mean([p.reliability for p in usable]))
    coverage = len(usable) / float(max(n_expected, 1))
    decisive = float(np.mean([p.decisiveness for p in usable]))
    # With one modality there is nothing to agree with. Using a neutral 0.5
    # rather than 1.0 avoids rewarding a single-modality assessment for an
    # agreement it never demonstrated.
    agree = 0.5 if agreement is None else float(agreement)

    weights = {"model_reliability": 0.40, "coverage": 0.25,
               "agreement": 0.25, "decisiveness": 0.10}
    factors = {"model_reliability": model_reliability, "coverage": coverage,
               "agreement": agree, "decisiveness": decisive}

    log_sum = sum(weights[k] * np.log(max(factors[k], 1e-6)) for k in weights)
    value = float(np.exp(log_sum))
    penalised = float(np.clip(value * RELIABILITY_PENALTY.get(severity, 1.0), 0.0, 1.0))

    basis = {
        "factors": {k: round(float(v), 4) for k, v in factors.items()},
        "weights": weights,
        "combination": "weighted geometric mean",
        "before_disagreement_penalty": round(value, 4),
        "disagreement_severity": severity,
        "disagreement_penalty": RELIABILITY_PENALTY.get(severity, 1.0),
        "caveat": ("These weights are a stated project convention, not an "
                   "empirical fit. This indicator describes how much internal "
                   "support the assessment has. It is not a validated measure "
                   "of clinical correctness."),
    }
    return penalised, basis
