"""
PDF parsing, chunking, embeddings, and vector-store helpers.
"""

from __future__ import annotations

import logging
import re
import uuid
from functools import lru_cache
from typing import Optional

import fitz
import numpy as np

from schemas import Document, DocumentMetadata, TextChunk

logger = logging.getLogger(__name__)


PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "aadhaar": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "din": re.compile(r"\b\d{8}\b"),
}

REDACTION_LABELS: dict[str, str] = {
    "aadhaar": "[AADHAAR_REDACTED]",
    "pan": "[PAN_REDACTED]",
    "din": "[DIN_REDACTED]",
}


def mask_pii(text: str, pii_types: Optional[list[str]] = None) -> str:
    if pii_types is None:
        pii_types = list(PII_PATTERNS.keys())

    for pii_type in pii_types:
        text = PII_PATTERNS[pii_type].sub(REDACTION_LABELS[pii_type], text)
    return text


def _normalize_page_text(page: fitz.Page) -> str:
    blocks = page.get_text("blocks")
    ordered_blocks = sorted(blocks, key=lambda block: (block[1], block[0]))
    text_parts = [block[4].strip() for block in ordered_blocks if block[4].strip()]
    text = "\n".join(text_parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(filepath: str, apply_pii_mask: bool = True) -> tuple[list[dict], int]:
    doc = fitz.open(filepath)
    pages: list[dict] = []

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        text = _normalize_page_text(page)
        if apply_pii_mask:
            text = mask_pii(text)
        pages.append(
            {
                "page_number": page_index + 1,
                "text": text,
            }
        )

    doc.close()
    return pages, len(pages)


def chunk_text(
    pages: list[dict],
    doc_id: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 250,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_value = text[start:end].strip()
            if chunk_value:
                chunks.append(
                    TextChunk(
                        chunk_id=f"{doc_id}_p{page_number}_c{len(chunks)}",
                        doc_id=doc_id,
                        page_number=page_number,
                        text=chunk_value,
                    )
                )
            if end >= len(text):
                break
            start = max(end - chunk_overlap, start + 1)

    return chunks


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def get_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = _load_embedding_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype("float32")


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class FAISSStore:
    """
    Tries to use FAISS when available and falls back to NumPy search otherwise.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.chunks: list[TextChunk] = []
        self.pages_by_doc: dict[str, list[dict]] = {}
        self.backend = "numpy"
        self.embeddings = np.empty((0, embedding_dim), dtype="float32")
        self.index = None

        try:
            import faiss

            self.index = faiss.IndexFlatIP(embedding_dim)
            self.backend = "faiss"
        except Exception as exc:
            logger.warning("FAISS unavailable, using NumPy similarity fallback: %s", exc)

    def add_document_pages(self, doc_id: str, pages: list[dict]) -> None:
        self.pages_by_doc[doc_id] = pages

    def get_document_pages(self, doc_id: str) -> list[dict]:
        return self.pages_by_doc.get(doc_id, [])

    def add_chunks(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None:
        if len(chunks) == 0:
            return

        embeddings = _normalize_vectors(embeddings.astype("float32"))
        self.chunks.extend(chunks)

        if self.backend == "faiss" and self.index is not None:
            self.index.add(embeddings)
            return

        if self.embeddings.size == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[TextChunk, float]]:
        if not self.chunks:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = _normalize_vectors(query_embedding.astype("float32"))

        if self.backend == "faiss" and self.index is not None:
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
            return [
                (self.chunks[index], float(score))
                for score, index in zip(scores[0], indices[0])
                if index >= 0
            ]

        similarity_scores = np.dot(self.embeddings, query_embedding[0])
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]
        return [(self.chunks[index], float(similarity_scores[index])) for index in top_indices]


def ingest_document(
    filepath: str,
    metadata: DocumentMetadata,
    store: FAISSStore,
) -> Document:
    doc_id = str(uuid.uuid4())[:8]
    pages, num_pages = parse_pdf(filepath, apply_pii_mask=True)
    store.add_document_pages(doc_id, pages)
    chunks = chunk_text(pages, doc_id)

    if not chunks:
        logger.warning("No text extracted from %s", filepath)
        return Document(
            doc_id=doc_id,
            filename=filepath,
            metadata=metadata,
            num_pages=num_pages,
            num_chunks=0,
            pii_masked=True,
        )

    embeddings = get_embeddings([chunk.text for chunk in chunks])
    store.add_chunks(chunks, embeddings)

    return Document(
        doc_id=doc_id,
        filename=filepath,
        metadata=metadata,
        num_pages=num_pages,
        num_chunks=len(chunks),
        pii_masked=True,
    )
