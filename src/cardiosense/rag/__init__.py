"""Guideline-grounded retrieval for CardioSense."""

from .documents import (BY_ID, REGISTRY, GuidelineDocument, check_available,
                        write_manifest)
from .chunking import (CHUNK_OVERLAP, CHUNK_SIZE, Chunk, chunk_document,
                       chunk_page, clean_text, extract_pdf, load_chunks,
                       remove_repeated_headers, save_chunks)
from .index import (DEFAULT_MODEL, MIN_SCORE, EmbeddingBackend, HashingBackend,
                    SearchHit, SentenceTransformerBackend, VectorIndex, build_index)
from .query import GuidelineQuery, build_queries
from .generate import (GENERATED_SUMMARY, GUIDELINE_EVIDENCE, MODEL_FINDING,
                       ExtractiveProvider, GroundedReport, Section,
                       build_provider, screen)
from .service import RAGService, build_knowledge_base

__all__ = [
    "BY_ID", "REGISTRY", "GuidelineDocument", "check_available", "write_manifest",
    "CHUNK_OVERLAP", "CHUNK_SIZE", "Chunk", "chunk_document", "chunk_page",
    "clean_text", "extract_pdf", "load_chunks", "remove_repeated_headers", "save_chunks",
    "DEFAULT_MODEL", "MIN_SCORE", "EmbeddingBackend", "HashingBackend", "SearchHit",
    "SentenceTransformerBackend", "VectorIndex", "build_index",
    "GuidelineQuery", "build_queries",
    "GENERATED_SUMMARY", "GUIDELINE_EVIDENCE", "MODEL_FINDING",
    "ExtractiveProvider", "GroundedReport", "Section", "build_provider", "screen",
    "RAGService", "build_knowledge_base",
]
