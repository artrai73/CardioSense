"""Assembling the grounded report.

The default generator is **extractive**: it copies guideline sentences verbatim
with their citations and composes nothing. It therefore *cannot* hallucinate --
not "is unlikely to", cannot, because no text is generated. Every clause either
came from a model output or is a quotation with a page number.

An LLM provider is available behind a flag for comparison, but it is opt-in and
its output is screened.

Three content types are kept visibly separate throughout:

===================  ================================================
``AI MODEL FINDING``  what a CardioSense model estimated
``GUIDELINE EVIDENCE`` text reproduced verbatim, with citation
``GENERATED SUMMARY``  narrative composed by the system
===================  ================================================

Conflating them is the central safety risk of a system like this: a clinician
who cannot tell a model's guess from a guideline's recommendation has lost the
ability to weigh either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .index import SearchHit

MODEL_FINDING = "AI MODEL FINDING"
GUIDELINE_EVIDENCE = "GUIDELINE EVIDENCE"
GENERATED_SUMMARY = "GENERATED SUMMARY"

#: Language a decision-support prototype must never emit. Each pattern is paired
#: with why it is prohibited.
PROHIBITED = [
    (re.compile(r"\bthe patient (has|is suffering from|is diagnosed with)\b", re.I),
     "diagnostic assertion"),
    (re.compile(r"\b(prescribe|administer|titrate)\b", re.I), "prescription"),
    (re.compile(r"\b\d+\s*mg\b(?!\s*/)", re.I), "drug dose"),
    (re.compile(r"\b(definitely|certainly|guaranteed|without doubt)\b", re.I),
     "false certainty"),
    (re.compile(r"\b(no further (testing|investigation) (is )?(needed|required))\b", re.I),
     "discharging a patient"),
]

#: Phrases that legitimately contain a prohibited word while forbidding the act.
_PROHIBITION_CONTEXT = re.compile(
    r"\b(must not|should not|does not|cannot|never|not[_ ]for|is not)\b", re.I)


def screen(text: str, window: int = 200) -> list[dict[str, str]]:
    """Flag prohibited language, excusing it inside an explicit prohibition.

    The window matters. "This report must not be used for prescribing
    decisions" contains the word *prescribing* while forbidding it, and a screen
    that fires on the system's own safety statements is worse than no screen --
    it trains everyone to ignore the alerts.
    """
    violations = []
    for pattern, reason in PROHIBITED:
        for match in pattern.finditer(text):
            lo = max(0, match.start() - window)
            context = text[lo: match.end() + window]
            if _PROHIBITION_CONTEXT.search(context):
                continue
            violations.append({"reason": reason, "match": match.group(0),
                               "context": text[lo: match.end() + 60].strip()})
    return violations


@dataclass
class Section:
    heading: str
    content_type: str
    body: Any
    citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"heading": self.heading, "content_type": self.content_type,
                "body": self.body, "citations": self.citations}


@dataclass
class GroundedReport:
    sections: list[Section]
    provider: str
    can_hallucinate: bool
    citations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "can_hallucinate": self.can_hallucinate,
            "sections": [s.to_dict() for s in self.sections],
            "citations": self.citations,
            "warnings": self.warnings,
        }

    def to_text(self) -> str:
        lines = ["=" * 72, "CARDIOSENSE DECISION-SUPPORT REPORT", "=" * 72, ""]
        for section in self.sections:
            lines += [f"[{section.content_type}]", section.heading,
                      "-" * len(section.heading)]
            body = section.body
            if isinstance(body, str):
                lines.append(body)
            elif isinstance(body, list):
                lines += [f"  - {item}" for item in body]
            elif isinstance(body, dict):
                lines += [f"  {k}: {v}" for k, v in body.items()]
            lines.append("")
        return "\n".join(lines)


class ExtractiveProvider:
    """Copies guideline text verbatim. Cannot hallucinate by construction."""

    name = "extractive"
    can_hallucinate = False

    def generate(self, assessment: dict[str, Any],
                 hits: Sequence[SearchHit],
                 queries: Sequence[Any] = ()) -> GroundedReport:
        fused = assessment.get("fused", {}) or {}
        availability = assessment.get("availability", {}) or {}
        sections: list[Section] = []

        # 1 -- what the models said
        findings = []
        for block in assessment.get("modalities", []) or []:
            name = block["modality"]
            if not block.get("available") or block.get("risk_score") is None:
                findings.append(f"{name}: not available ({block.get('reason','not supplied')}); "
                                f"no value was substituted")
                continue
            calib = "calibrated" if block.get("is_calibrated") else "UNCALIBRATED"
            positives = [k for k, v in (block.get("native_labels") or {}).items() if v]
            findings.append(
                f"{name}: risk {block['risk_score']:.3f} ({calib}, "
                f"reliability {block['reliability']:.2f})"
                + (f"; predicted present: {', '.join(positives)}" if positives else ""))
        sections.append(Section("Model Findings", MODEL_FINDING, findings))

        # 2 -- the composite, with its basis stated
        score = fused.get("risk_score")
        sections.append(Section("Composite Indicator", MODEL_FINDING, {
            "composite_risk": "not computable" if score is None else f"{score:.3f}",
            "risk_level": fused.get("risk_level", "UNKNOWN"),
            "reliability": f"{fused.get('reliability', 0):.3f} "
                           f"({fused.get('reliability_label','UNKNOWN')})",
            "based_on": availability.get("fusion_basis") or "no modality",
            "weights": fused.get("weights", {}),
            "note": ("This is a composite cardiovascular abnormality indicator "
                     "across models trained on different cohorts for different "
                     "targets. It is not a diagnosis of any single condition."),
        }))

        # 3 -- verbatim guideline text, or an explicit statement that there is none
        if hits:
            excerpts = [{"text": h.chunk.text, "citation": h.chunk.citation,
                         "relevance": round(float(h.score), 3)} for h in hits]
            sections.append(Section(
                "Guideline Evidence", GUIDELINE_EVIDENCE, excerpts,
                citations=[h.chunk.citation for h in hits]))
        else:
            sections.append(Section("Guideline Evidence", GUIDELINE_EVIDENCE, [
                "No guideline evidence was retrieved. No statement in this report "
                "is guideline-derived. Build the knowledge base to enable retrieval."]))

        # 4 -- considerations, each hedged and non-prescriptive
        considerations = [
            "This output is for review by a qualified clinician and is not a diagnosis.",
            "The composite indicator reflects model estimates, not a clinical finding.",
        ]
        if fused.get("disagreement_flag"):
            considerations.append(
                "The modalities disagreed. Review each modality's evidence "
                "individually rather than relying on the composite score.")
        if availability.get("missing"):
            considerations.append(
                f"Unavailable modalities ({', '.join(availability['missing'])}) "
                "contributed nothing and were not imputed.")
        sections.append(Section("Decision-Support Considerations",
                                GENERATED_SUMMARY, considerations))

        # 5 -- limitations, always present
        limitations = list(assessment.get("warnings", []) or [])
        limitations += [
            "The constituent models were trained on disjoint public cohorts with "
            "different targets. No paired multimodal evaluation exists for this "
            "system, so no clinical accuracy claim is made.",
            "None of the training cohorts is representative of an Indian population.",
            "CardioSense is an academic prototype and has not been clinically validated.",
        ]
        sections.append(Section("Limitations and Uncertainty",
                                GENERATED_SUMMARY, limitations))

        # 6 -- sources
        citations = sorted({h.chunk.citation for h in hits})
        sections.append(Section("Sources", GUIDELINE_EVIDENCE,
                                citations or ["No guideline documents were retrieved."],
                                citations=citations))

        report = GroundedReport(sections, self.name, self.can_hallucinate, citations)

        violations = screen(report.to_text())
        if violations:
            report.warnings.append(
                f"Safety screen flagged {len(violations)} item(s): "
                f"{[v['reason'] for v in violations]}")
        return report


def build_provider(name: str = "auto") -> ExtractiveProvider:
    """Return a generation provider.

    Only the extractive provider ships enabled. Adding an LLM provider means
    accepting a non-zero hallucination rate, which must then be *measured*
    rather than assumed, so it is a deliberate opt-in rather than a default.
    """
    if name in {"auto", "extractive"}:
        return ExtractiveProvider()
    raise ValueError(
        f"Unknown provider {name!r}. Only 'extractive' is enabled. An LLM "
        "provider can be added, but its outputs must then be evaluated for "
        "groundedness before any result is reported.")
