"""The RAG facade: assessment in, grounded report out.

Tolerates a missing index throughout. A system that refuses to produce a report
because the guideline corpus has not been built yet is less useful than one that
produces the report and states plainly that nothing in it is guideline-derived.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .chunking import Chunk, chunk_document, extract_pdf, save_chunks
from .documents import BY_ID, REGISTRY, check_available, write_manifest
from .generate import ExtractiveProvider, GroundedReport, build_provider
from .index import (
    MIN_SCORE,
    SearchHit,
    SentenceTransformerBackend,
    VectorIndex,
    build_index,
)
from .query import GuidelineQuery, build_queries


class RAGService:
    """Retrieval and grounded generation over the guideline corpus."""

    def __init__(self, index_dir: Path | str | None = None,
                 provider: str = "extractive"):
        self.index_dir = Path(index_dir) if index_dir else None
        self.provider: ExtractiveProvider = build_provider(provider)
        self.index: VectorIndex | None = None
        self.load_error = ""
        if self.index_dir and (self.index_dir / "index_meta.json").exists():
            try:
                self.index = VectorIndex.load(self.index_dir)
            except Exception as exc:                        # noqa: BLE001
                self.load_error = f"{type(exc).__name__}: {exc}"

    @property
    def retrieval_available(self) -> bool:
        return self.index is not None and len(self.index) > 0

    def status(self) -> dict[str, Any]:
        return {
            "retrieval_available": self.retrieval_available,
            "n_chunks": len(self.index) if self.index else 0,
            "embedding_model": (self.index.metadata.get("embedding_model")
                                if self.index else None),
            "provider": self.provider.name,
            "can_hallucinate": self.provider.can_hallucinate,
            "load_error": self.load_error,
            "hint": ("" if self.retrieval_available else
                     "No index loaded. Build the knowledge base to enable "
                     "guideline retrieval; reports will otherwise state that "
                     "no content is guideline-derived."),
        }

    def search(self, query: str, k: int = 5,
               min_score: float = MIN_SCORE) -> list[SearchHit]:
        if not self.retrieval_available:
            return []
        return self.index.search(query, k=k, min_score=min_score)

    def retrieve_for_assessment(self, assessment: dict[str, Any], *,
                                per_query: int = 2, max_total: int = 6,
                                min_score: float = MIN_SCORE
                                ) -> tuple[list[SearchHit], list[GuidelineQuery]]:
        """Build queries from the findings, then retrieve for each."""
        queries = build_queries(assessment)
        if not self.retrieval_available:
            return [], queries

        hits: list[SearchHit] = []
        seen: set[str] = set()
        for query in queries:
            for hit in self.search(query.text, k=per_query, min_score=min_score):
                if hit.chunk.chunk_id in seen:
                    continue
                seen.add(hit.chunk.chunk_id)
                hits.append(hit)
        hits.sort(key=lambda h: -h.score)
        return hits[:max_total], queries

    def generate_report(self, assessment: dict[str, Any], **kwargs
                        ) -> tuple[GroundedReport, list[GuidelineQuery]]:
        hits, queries = self.retrieve_for_assessment(assessment, **kwargs)
        report = self.provider.generate(assessment, hits, queries)
        if not self.retrieval_available:
            report.warnings.append(
                "Guideline retrieval was unavailable; no statement in this "
                "report is guideline-derived.")
        return report, queries


def build_knowledge_base(guidelines_dir: Path | str,
                         output_dir: Path | str,
                         model_name: str = "all-MiniLM-L6-v2",
                         verbose: bool = True) -> dict[str, Any]:
    """Extract, chunk, embed and index every guideline PDF present.

    Returns a report of what was indexed, including which registered documents
    were absent -- so a partially built corpus is visible rather than assumed
    complete.
    """
    guidelines_dir, output_dir = Path(guidelines_dir), Path(output_dir)
    availability = check_available(guidelines_dir)

    if not availability["present"]:
        return {
            "built": False,
            "reason": "No guideline PDFs found.",
            "expected_in": str(guidelines_dir),
            "download": [{"doc_id": d.doc_id, "filename": d.filename, "url": d.url}
                         for d in REGISTRY],
        }

    all_chunks: list[Chunk] = []
    per_document = []
    for doc_id in availability["present"]:
        doc = BY_ID[doc_id]
        pages = extract_pdf(guidelines_dir / doc.filename)
        chunks = chunk_document(pages, doc.doc_id, organization=doc.organization,
                                year=doc.year, title=doc.title)
        all_chunks.extend(chunks)
        per_document.append({"doc_id": doc_id, "pages": len(pages),
                             "chunks": len(chunks)})
        if verbose:
            print(f"  {doc_id:<34} {len(pages):>4} pages -> {len(chunks):>4} chunks")

    if verbose:
        print(f"\n  embedding {len(all_chunks)} chunks with {model_name} ...")
    backend = SentenceTransformerBackend(model_name)
    index = build_index(all_chunks, backend,
                        metadata={"source_documents": availability["present"]})
    index.save(output_dir)
    save_chunks(all_chunks, output_dir / "chunks.json")

    return {
        "built": True,
        "n_chunks": len(all_chunks),
        "documents": per_document,
        "missing_documents": availability["missing"],
        "embedding_model": model_name,
        "index_dir": str(output_dir),
    }


__all__ = ["RAGService", "build_knowledge_base", "write_manifest",
           "check_available", "GroundedReport"]
