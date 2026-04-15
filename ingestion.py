"""
ingestion.py — Document Ingestion Layer for Institutional Intelligence Platform

Handles PDF parsing (PyMuPDF), text chunking, PII masking (Aadhaar/PAN/DIN),
Indic-language embedding placeholders, and FAISS vector store indexing.
"""

from __future__ import annotations

import re
import uuid
import logging
from typing import Optional

import fitz  # PyMuPDF
import numpy as np

from schemas import (
    Document,
    DocumentMetadata,
    TextChunk,
    PIIType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII Masking
# ---------------------------------------------------------------------------

# Regex patterns for Indian PII
PII_PATTERNS: dict[PIIType, re.Pattern] = {
    PIIType.AADHAAR: re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),  # 12 digits
    PIIType.PAN: re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),  # ABCDE1234F
    PIIType.DIN: re.compile(r"\b\d{8}\b"),  # 8-digit DIN
}

REDACTION_LABELS: dict[PIIType, str] = {
    PIIType.AADHAAR: "[AADHAAR_REDACTED]",
    PIIType.PAN: "[PAN_REDACTED]",
    PIIType.DIN: "[DIN_REDACTED]",
}

def mask_pii(text: str, pii_types: Optional[list[PIIType]] = None) -> str:
    """
    Redact PII from text before sending to any LLM.

    Args:
        text: Raw text to sanitize.
        pii_types: Subset of PII types to mask. Defaults to all.

    Returns:
        Text with PII patterns replaced by redaction labels.
    """
    if pii_types is None:
        pii_types = list(PIIType)

    for pii_type in pii_types:
        pattern = PII_PATTERNS[pii_type]
        label = REDACTION_LABELS[pii_type]
        text = pattern.sub(label, text)

    return text

# ---------------------------------------------------------------------------
# PDF Parsing
# ---------------------------------------------------------------------------

def parse_pdf(filepath: str, apply_pii_mask: bool = True) -> tuple[list[dict], int]:
    """
    Extract text from a PDF file using PyMuPDF.

    Args:
        filepath: Path to the PDF file.
        apply_pii_mask: Whether to mask PII in extracted text.

    Returns:
        Tuple of (list of page dicts with 'page_number' and 'text', total page count).
    """
    doc = fitz.open(filepath)
    pages: list[dict] = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")

        if apply_pii_mask:
            text = mask_pii(text)

        pages.append({
            "page_number": page_num + 1,  # 1-indexed
            "text": text.strip(),
        })

    doc.close()
    return pages, len(pages)

# ---------------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    pages: list[dict],
    doc_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[TextChunk]:
    """
    Split page texts into overlapping chunks for RAG retrieval.

    Args:
        pages: List of page dicts from parse_pdf.
        doc_id: Parent document identifier.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of TextChunk objects.
    """
    chunks: list[TextChunk] = []

    for page in pages:
        text = page["text"]
        page_num = page["page_number"]

        if not text:
            continue

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text_slice = text[start:end]

            chunks.append(
                TextChunk(
                    chunk_id=f"{doc_id}_p{page_num}_c{len(chunks)}",
                    doc_id=doc_id,
                    page_number=page_num,
                    text=chunk_text_slice,
                )
            )
            start += chunk_size - chunk_overlap

    return chunks

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generate embeddings using sentence-transformers.

    Args:
        texts: List of text strings to embed.
        model_name: HuggingFace model identifier.

    Returns:
        NumPy array of shape (len(texts), embedding_dim).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def get_indic_embeddings(texts: list[str]) -> np.ndarray:
    """
    Placeholder for Indic-specific embeddings (Sarvam AI / Bhashini).

    In production, replace this with an actual Indic embedding model call.
    Currently falls back to the default English model.

    Args:
        texts: List of text strings (may contain Hindi, Hinglish, or regional languages).

    Returns:
        NumPy array of embeddings.
    """
    logger.warning(
        "Using English embeddings as placeholder for Indic model. "
        "Replace with Sarvam AI or Bhashini integration for production."
    )
    # TODO: Replace with actual Indic embedding model
    # Example: sarvam_client.embed(texts, model="sarvam-embed-v1")
    return get_embeddings(texts)

# ---------------------------------------------------------------------------
# FAISS Vector Store
# ---------------------------------------------------------------------------

class FAISSStore:
    """
    Simple FAISS-based vector store for document chunks.
    Stores chunks in-memory with their embeddings for similarity search.
    """

    def __init__(self, embedding_dim: int = 384):
        import faiss

        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner-product (cosine after normalization)
        self.chunks: list[TextChunk] = []
        self.embedding_dim = embedding_dim

    def add_chunks(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None:
        """Add chunks and their embeddings to the index."""
        import faiss

        # L2-normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[TextChunk, float]]:
        """
        Search for the most similar chunks.

        Args:
            query_embedding: 1-D or 2-D query vector.
            top_k: Number of results to return.

        Returns:
            List of (TextChunk, similarity_score) tuples.
        """
        import faiss

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)

        results: list[tuple[TextChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.chunks[idx], float(score)))

        return results

# ---------------------------------------------------------------------------
# High-Level Ingestion Pipeline
# ---------------------------------------------------------------------------

def ingest_document(
    filepath: str,
    metadata: DocumentMetadata,
    store: FAISSStore,
    use_indic: bool = False,
) -> Document:
    """
    Full ingestion pipeline: parse → chunk → embed → index.

    Args:
        filepath: Path to the PDF.
        metadata: Document metadata.
        store: FAISSStore instance to add chunks to.
        use_indic: Whether to use the Indic embedding placeholder.

    Returns:
        Document object with populated metadata.
    """
    doc_id = str(uuid.uuid4())[:8]

    # 1. Parse PDF with PII masking
    pages, num_pages = parse_pdf(filepath, apply_pii_mask=True)

    # 2. Chunk text
    chunks = chunk_text(pages, doc_id)

    if not chunks:
        logger.warning(f"No text extracted from {filepath}")
        return Document(
            doc_id=doc_id,
            filename=filepath,
            metadata=metadata,
            num_pages=num_pages,
            num_chunks=0,
            pii_masked=True,
        )

    # 3. Generate embeddings
    texts = [c.text for c in chunks]
    embed_fn = get_indic_embeddings if use_indic else get_embeddings
    embeddings = embed_fn(texts)

    # 4. Add to FAISS store
    store.add_chunks(chunks, embeddings)

    return Document(
        doc_id=doc_id,
        filename=filepath,
        metadata=metadata,
        num_pages=num_pages,
        num_chunks=len(chunks),
        pii_masked=True,
    )


