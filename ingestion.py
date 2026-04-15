"""
Document ingestion utilities for PDF parsing, chunking, embeddings, and FAISS indexing.
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


def parse_pdf(filepath: str, apply_pii_mask: bool = True) -> tuple[list[dict], int]:
    doc = fitz.open(filepath)
    pages: list[dict] = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        if apply_pii_mask:
            text = mask_pii(text)

        pages.append(
            {
                "page_number": page_num + 1,
                "text": text.strip(),
            }
        )

    doc.close()
    return pages, len(pages)


def chunk_text(
    pages: list[dict],
    doc_id: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]
        if not text:
            continue

        normalized = re.sub(r"[ \t]+", " ", text).strip()
        start = 0

        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            chunk_value = normalized[start:end].strip()
            if chunk_value:
                chunks.append(
                    TextChunk(
                        chunk_id=f"{doc_id}_p{page_number}_c{len(chunks)}",
                        doc_id=doc_id,
                        page_number=page_number,
                        text=chunk_value,
                    )
                )

            if end >= len(normalized):
                break

            start = max(end - chunk_overlap, start + 1)

    return chunks


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def get_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = _load_embedding_model(model_name)
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


class FAISSStore:
    def __init__(self, embedding_dim: int = 384):
        import faiss

        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: list[TextChunk] = []
        self.embedding_dim = embedding_dim

    def add_chunks(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None:
        import faiss

        if len(chunks) == 0:
            return

        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[TextChunk, float]]:
        import faiss

        if not self.chunks:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype("float32")
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)

        results: list[tuple[TextChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.chunks[idx], float(score)))

        return results


def ingest_document(
    filepath: str,
    metadata: DocumentMetadata,
    store: FAISSStore,
) -> Document:
    doc_id = str(uuid.uuid4())[:8]
    pages, num_pages = parse_pdf(filepath, apply_pii_mask=True)
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

    texts = [chunk.text for chunk in chunks]
    embeddings = get_embeddings(texts)
    store.add_chunks(chunks, embeddings)

    return Document(
        doc_id=doc_id,
        filename=filepath,
        metadata=metadata,
        num_pages=num_pages,
        num_chunks=len(chunks),
        pii_masked=True,
    )
