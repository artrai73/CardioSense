"""Turning guideline PDFs into citable passages.

Chunks are 1000 characters with 200 of overlap, split on paragraph boundaries
where possible and sentence boundaries otherwise.

* **1000 characters** is roughly a paragraph, which is the unit a guideline
  recommendation is actually written in, and it fits inside the embedding
  model's 256-token window so nothing is silently truncated.
* **200 of overlap** stops a recommendation that straddles a boundary from
  being retrievable in neither chunk.
* **Never across a page break**, so every chunk has exactly one page number and
  a citation can be checked by opening that page.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_CHARS = 120          # below this a passage carries no usable content

_WHITESPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page_number: int
    section: str = ""
    organization: str = ""
    year: int = 0
    title: str = ""

    @property
    def citation(self) -> str:
        parts = [p for p in (self.organization, str(self.year) if self.year else "") if p]
        base = ", ".join(parts) if parts else self.doc_id
        return f"{base}, p. {self.page_number}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_text(text: str) -> str:
    """Normalise whitespace and rejoin words broken across a line by a hyphen."""
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # de-hyphenate line breaks
    text = _WHITESPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def remove_repeated_headers(pages: list[str], min_repeats: int | None = None) -> list[str]:
    """Strip lines that appear on most pages -- running headers and footers.

    Left in, they dominate the embedding of every short chunk, so a search for
    anything retrieves the journal name rather than the recommendation.
    """
    if len(pages) < 3:
        return pages
    threshold = min_repeats or max(3, int(0.6 * len(pages)))

    counts: dict[str, int] = {}
    for page in pages:
        for line in {ln.strip() for ln in page.split("\n") if 0 < len(ln.strip()) < 120}:
            counts[line] = counts.get(line, 0) + 1
    boilerplate = {line for line, n in counts.items() if n >= threshold}
    if not boilerplate:
        return pages

    return ["\n".join(ln for ln in page.split("\n")
                      if ln.strip() not in boilerplate) for page in pages]


def _split_long(block: str, size: int) -> list[str]:
    """Break an over-long block on sentence boundaries, hard-cutting only if forced."""
    if len(block) <= size:
        return [block]
    out, current = [], ""
    for sentence in _SENTENCE_END.split(block):
        if current and len(current) + len(sentence) + 1 > size:
            out.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        out.append(current.strip())

    final: list[str] = []
    for piece in out:
        while len(piece) > size:                      # a sentence longer than a chunk
            final.append(piece[:size])
            piece = piece[size:]
        if piece:
            final.append(piece)
    return final


def chunk_page(text: str, size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Chunk one page, preferring paragraph then sentence boundaries."""
    text = clean_text(text)
    if len(text) <= size:
        return [text] if len(text) >= MIN_CHUNK_CHARS else []

    blocks: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph:
            blocks.extend(_split_long(paragraph, size))

    chunks: list[str] = []
    current = ""
    for block in blocks:
        if current and len(current) + len(block) + 2 > size:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            # Resume at a word boundary so the overlap never starts mid-word.
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1:]
            current = f"{tail} {block}".strip()
        else:
            current = f"{current}\n\n{block}".strip() if current else block
    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def chunk_document(pages: Iterable[str], doc_id: str, *, organization: str = "",
                   year: int = 0, title: str = "",
                   size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Chunk a whole document, one page at a time so page numbers stay exact."""
    pages = list(pages)
    pages = remove_repeated_headers(pages)

    out: list[Chunk] = []
    for page_index, page_text in enumerate(pages, start=1):
        for piece_index, piece in enumerate(chunk_page(page_text, size, overlap)):
            out.append(Chunk(
                chunk_id=f"{doc_id}:p{page_index}:c{piece_index}",
                doc_id=doc_id, text=piece, page_number=page_index,
                organization=organization, year=year, title=title,
            ))
    return out


def save_chunks(chunks: list[Chunk], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([c.to_dict() for c in chunks], indent=1),
                    encoding="utf-8")
    return path


def load_chunks(path: Path | str) -> list[Chunk]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Chunk(**item) for item in raw]


def extract_pdf(path: Path | str) -> list[str]:
    """Per-page text from a PDF, preserving page boundaries.

    Tries ``pypdf`` then ``PyPDF2``. Raises with an install hint if neither is
    present, rather than returning empty pages that would silently produce an
    index containing nothing.
    """
    path = Path(path)
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader               # type: ignore
        except ImportError as exc:
            raise ImportError("Install a PDF reader:  pip install pypdf") from exc

    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]
