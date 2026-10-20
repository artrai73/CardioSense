"""Turning an assessment into guideline queries.

**Every query is built only from what a model actually produced.** Each one
records the finding that generated it, so a retrieved passage can always be
traced back to the model output that asked for it.

This is the first place hallucination could enter, and it would enter before
generation even started: if a query could invent "chest pain on exertion"
because the risk score came out high, retrieval would faithfully fetch evidence
for a patient who does not exist, and every downstream step would be grounded in
that fiction. So the query builder reads findings and nothing else.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

#: PTB-XL superclass codes are not words a guideline uses. Translating them is
#: not invention -- it is the dataset's own documented expansion of its codes.
ECG_CLASS_TERMS = {
    "NORM": "normal electrocardiogram",
    "MI": "myocardial infarction",
    "STTC": "ST segment and T wave changes, myocardial ischaemia",
    "CD": "conduction disturbance, bundle branch block",
    "HYP": "left ventricular hypertrophy",
}

XRAY_FINDING_TERMS = {
    "cardiomegaly": "cardiomegaly, enlarged cardiac silhouette, "
                    "cardiothoracic ratio, heart failure assessment",
}

CLINICAL_FEATURE_TERMS = {
    "trestbps": "resting blood pressure, hypertension management, blood pressure targets",
    "chol": "serum cholesterol, lipid management, LDL targets",
    "fbs": "fasting blood glucose, diabetes and cardiovascular risk",
    "thalach": "maximum heart rate achieved, exercise capacity",
    "oldpeak": "ST depression induced by exercise, exercise ECG interpretation",
    "exang": "exercise induced angina, stable angina assessment",
    "cp": "chest pain characterisation, typical and atypical angina",
    "age": "age and cardiovascular risk estimation",
}

RISK_LEVEL_TERMS = {
    "HIGH": "high cardiovascular risk management, intensive risk factor control",
    "INTERMEDIATE": "intermediate cardiovascular risk, risk reclassification",
    "LOW": "low cardiovascular risk, lifestyle advice and reassessment interval",
    "UNKNOWN": "cardiovascular risk assessment when investigations are incomplete",
}


@dataclass
class GuidelineQuery:
    text: str
    topic: str
    source_finding: str
    modality: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_queries(assessment: dict[str, Any], *,
                  max_queries: int = 6) -> list[GuidelineQuery]:
    """Derive guideline queries from a :class:`AssessmentResult` dictionary.

    Args:
        assessment: The dict produced by ``AssessmentResult.to_dict()``.
        max_queries: Cap, so one assessment cannot flood the index.

    Returns:
        One query per finding, each tracing to the finding that produced it.
    """
    queries: list[GuidelineQuery] = []
    fused = assessment.get("fused", {}) or {}

    level = fused.get("risk_level", "UNKNOWN")
    queries.append(GuidelineQuery(
        text=RISK_LEVEL_TERMS.get(level, RISK_LEVEL_TERMS["UNKNOWN"]),
        topic="overall_risk", source_finding=f"fused risk level = {level}",
        modality="fused"))

    for block in assessment.get("modalities", []) or []:
        if not block.get("available") or block.get("risk_score") is None:
            continue
        modality = block["modality"]
        labels = block.get("native_labels", {}) or {}

        if modality == "ecg":
            for code, present in labels.items():
                if present and code != "NORM" and code in ECG_CLASS_TERMS:
                    queries.append(GuidelineQuery(
                        text=ECG_CLASS_TERMS[code], topic=f"ecg_{code.lower()}",
                        source_finding=f"ECG model predicted {code} present",
                        modality="ecg"))

        elif modality == "xray":
            for finding, present in labels.items():
                key = finding.lower()
                if present and key in XRAY_FINDING_TERMS:
                    queries.append(GuidelineQuery(
                        text=XRAY_FINDING_TERMS[key], topic=f"xray_{key}",
                        source_finding=f"X-ray model predicted {finding} present",
                        modality="xray"))

        elif modality == "clinical":
            if labels.get("coronary_stenosis"):
                queries.append(GuidelineQuery(
                    text="obstructive coronary artery disease, chronic coronary "
                         "syndrome, revascularisation assessment",
                    topic="clinical_cad",
                    source_finding="clinical model predicted coronary stenosis present",
                    modality="clinical"))
            # Feature-driven queries come only from values actually supplied.
            for feature, term in CLINICAL_FEATURE_TERMS.items():
                if feature in (block.get("input_features") or {}):
                    queries.append(GuidelineQuery(
                        text=term, topic=f"clinical_{feature}",
                        source_finding=f"clinical input included {feature}",
                        modality="clinical"))

    if (assessment.get("disagreement", {}) or {}).get("detected"):
        queries.append(GuidelineQuery(
            text="integrating discordant non-invasive test results, "
                 "further diagnostic testing when investigations conflict",
            topic="disagreement",
            source_finding="modalities disagreed",
            modality="fused"))

    # De-duplicate on topic, preserving order.
    seen: set[str] = set()
    unique = [q for q in queries
              if not (q.topic in seen or seen.add(q.topic))]
    return unique[:max_queries]
