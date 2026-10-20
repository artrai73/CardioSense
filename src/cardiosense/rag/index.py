"""Embedding and exact retrieval over the guideline chunks.

**Exact search, not approximate.** The corpus is a few thousand chunks and an
exact inner-product scan takes under a millisecond. FAISS's approximate indexes
trade recall for speed on millions of vectors; choosing one here would mean a
guideline passage could be *silently missed* in a system whose entire purpose is
grounding claims in retrievable text.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .chunking import Chunk

DEFAULT_MODEL = "all-MiniLM-L6-v2"
#: Below this cosine similarity a "hit" is noise. Returning top-k regardless of
#: score is how a grounded system starts citing irrelevant passages.
MIN_SCORE = 0.25


class EmbeddingBackend:
    name = "base"
    dimension = 0
    is_semantic = False

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerBackend(EmbeddingBackend):
    """The production backend. Downloads ~80 MB of weights on first use."""

    is_semantic = True

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), convert_to_numpy=True,
                                     normalize_embeddings=True,
                                     show_progress_bar=len(texts) > 256)
        return np.asarray(vectors, dtype=np.float32)


class HashingBackend(EmbeddingBackend):
    """A deterministic offline stand-in, for tests only.

    It matches on shared vocabulary, not meaning, so a paraphrase scores near
    zero. :func:`build_index` refuses it unless explicitly forced, because an
    index built on it looks like it works and retrieves badly.
    """

    name = "hashing-offline"
    is_semantic = False

    def __init__(self, dimension: int = 384):
        self.dimension = int(dimension)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in str(text).lower().split():
                out[row, hash(token) % self.dimension] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


@dataclass
class SearchHit:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.chunk.text,
            "citation": self.chunk.citation,
            "doc_id": self.chunk.doc_id,
            "organization": self.chunk.organization,
            "year": self.chunk.year,
            "page_number": self.chunk.page_number,
            "chunk_id": self.chunk.chunk_id,
            "score": round(float(self.score), 4),
        }


class VectorIndex:
    """Exact inner-product search over normalised embeddings."""

    def __init__(self, backend: EmbeddingBackend, chunks: list[Chunk],
                 vectors: np.ndarray, metadata: dict[str, Any] | None = None):
        self.backend = backend
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 5,
               min_score: float = MIN_SCORE) -> list[SearchHit]:
        """Top-k chunks above ``min_score``. Exact scan, no approximation."""
        if not self.chunks:
            return []
        q = self.backend.encode([query])[0]
        scores = self.vectors @ q
        order = np.argsort(-scores)[: max(k * 3, k)]
        hits = [SearchHit(self.chunks[i], float(scores[i])) for i in order
                if scores[i] >= min_score]
        return hits[:k]

    def save(self, directory: Path | str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with open(directory / "chunks.pkl", "wb") as handle:
            pickle.dump(self.chunks, handle)
        (directory / "index_meta.json").write_text(
            json.dumps({**self.metadata,
                        "embedding_model": self.backend.name,
                        "dimension": int(self.backend.dimension),
                        "is_semantic": bool(self.backend.is_semantic),
                        "n_chunks": len(self.chunks),
                        "search": "exact inner product"}, indent=2),
            encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: Path | str,
             backend: EmbeddingBackend | None = None) -> "VectorIndex":
        directory = Path(directory)
        meta = json.loads((directory / "index_meta.json").read_text(encoding="utf-8"))
        backend = backend or SentenceTransformerBackend(meta["embedding_model"])

        # A query embedded by a different model than the index lands in a
        # different space, and every score is then meaningless -- but nothing
        # crashes, so the failure would be silent. Refuse instead.
        if backend.name != meta["embedding_model"]:
            raise ValueError(
                f"Index was built with {meta['embedding_model']!r} but the "
                f"backend supplied is {backend.name!r}. Query and index vectors "
                "must come from the same model.")

        vectors = np.load(directory / "vectors.npy")
        with open(directory / "chunks.pkl", "rb") as handle:
            chunks = pickle.load(handle)
        return cls(backend, chunks, vectors, meta)


def build_index(chunks: list[Chunk], backend: EmbeddingBackend,
                allow_non_semantic: bool = False,
                metadata: dict[str, Any] | None = None) -> VectorIndex:
    if not chunks:
        raise ValueError("No chunks to index. Build the knowledge base first.")
    if not backend.is_semantic and not allow_non_semantic:
        raise ValueError(
            f"{backend.name!r} is not a semantic embedding model. It matches on "
            "shared words, not meaning, so retrieval will look like it works "
            "and be wrong. Use SentenceTransformerBackend, or pass "
            "allow_non_semantic=True if this is a test.")
    vectors = backend.encode([c.text for c in chunks])
    return VectorIndex(backend, chunks, vectors, metadata)
