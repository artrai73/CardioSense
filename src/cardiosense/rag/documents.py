"""The guideline corpus: what is indexed, and where it came from.

**No guideline PDF is redistributed with this project.** Guidelines are free to
read but carry their own redistribution terms, so the repository ships a
manifest of pointers and the build step reads PDFs the user has downloaded
themselves.

Every chunk that reaches a report carries its organisation, year, section and
page, because retrieved text that cannot be traced back to a document is
indistinguishable from a hallucination.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GuidelineDocument:
    doc_id: str
    title: str
    organization: str
    year: int
    url: str
    filename: str
    topics: tuple[str, ...]

    def citation(self, page: int | None = None) -> str:
        base = f"{self.organization}, {self.year}"
        return f"{base}, p. {page}" if page else base


#: Pointers only. Download these yourself into knowledge_base/guidelines/.
REGISTRY: tuple[GuidelineDocument, ...] = (
    GuidelineDocument(
        doc_id="esc_cvd_prevention_2021",
        title="2021 ESC Guidelines on cardiovascular disease prevention in clinical practice",
        organization="European Society of Cardiology",
        year=2021,
        url="https://academic.oup.com/eurheartj/article/42/34/3227/6358713",
        filename="esc_cvd_prevention_2021.pdf",
        topics=("risk assessment", "prevention", "lipids", "blood pressure", "diabetes"),
    ),
    GuidelineDocument(
        doc_id="esc_acs_2023",
        title="2023 ESC Guidelines for the management of acute coronary syndromes",
        organization="European Society of Cardiology",
        year=2023,
        url="https://academic.oup.com/eurheartj/article/44/38/3720/7243210",
        filename="esc_acs_2023.pdf",
        topics=("acute coronary syndrome", "myocardial infarction", "ECG", "troponin"),
    ),
    GuidelineDocument(
        doc_id="esc_hf_2021",
        title="2021 ESC Guidelines for the diagnosis and treatment of acute and chronic heart failure",
        organization="European Society of Cardiology",
        year=2021,
        url="https://academic.oup.com/eurheartj/article/42/36/3599/6358045",
        filename="esc_hf_2021.pdf",
        topics=("heart failure", "cardiomegaly", "ejection fraction", "echocardiography"),
    ),
    GuidelineDocument(
        doc_id="acc_aha_primary_prevention_2019",
        title="2019 ACC/AHA Guideline on the Primary Prevention of Cardiovascular Disease",
        organization="American College of Cardiology / American Heart Association",
        year=2019,
        url="https://www.ahajournals.org/doi/10.1161/CIR.0000000000000678",
        filename="acc_aha_primary_prevention_2019.pdf",
        topics=("primary prevention", "risk estimation", "statins", "lifestyle"),
    ),
    GuidelineDocument(
        doc_id="who_hearts_2020",
        title="HEARTS technical package for cardiovascular disease management in primary health care",
        organization="World Health Organization",
        year=2020,
        url="https://www.who.int/publications/i/item/9789240001367",
        filename="who_hearts_2020.pdf",
        topics=("primary care", "risk-based management", "low-resource settings"),
    ),
)

BY_ID: dict[str, GuidelineDocument] = {d.doc_id: d for d in REGISTRY}


def write_manifest(path: Path | str) -> Path:
    """Write the download manifest beside the (empty) guidelines folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": ("Download each PDF yourself and place it in this folder under "
                 "the given filename. Guideline PDFs are not redistributed with "
                 "CardioSense. One document is enough to start."),
        "verify_before_citing": ("Guidelines are revised. Check the title, year "
                                 "and issuing body against the publisher's page "
                                 "before citing anything retrieved from them."),
        "documents": [asdict(d) for d in REGISTRY],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def check_available(guidelines_dir: Path | str) -> dict[str, Any]:
    """Which registered documents are actually present on disk."""
    directory = Path(guidelines_dir)
    present, missing = [], []
    for doc in REGISTRY:
        (present if (directory / doc.filename).exists() else missing).append(doc.doc_id)
    return {
        "directory": str(directory),
        "present": present,
        "missing": missing,
        "n_present": len(present),
        "ready": bool(present),
    }
