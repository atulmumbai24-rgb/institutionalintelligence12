"""
Pure retrieval-based question answering for ingested company documents.
"""

from __future__ import annotations

import logging
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ingestion import FAISSStore, get_embeddings
from schemas import CellStatus, Document, SourceCitation, TaskCell

logger = logging.getLogger(__name__)


def _split_candidate_passages(text: str) -> list[str]:
    raw_passages = re.split(r"\n{2,}|(?<=[.!?])\s+", text)
    passages = [passage.strip() for passage in raw_passages if len(passage.strip()) >= 40]
    return passages[:12] if passages else [text.strip()]


def extract_answer(
    question: str,
    chunks: list[tuple[str, int, float]],
    top_n: int = 4,
) -> tuple[str, list[SourceCitation]]:
    if not chunks:
        return "No relevant information found in the document.", []

    top_chunks = chunks[:top_n]

    try:
        candidate_passages: list[tuple[str, int, float]] = []
        for text, page_number, retrieval_score in top_chunks:
            for passage in _split_candidate_passages(text):
                candidate_passages.append((passage, page_number, retrieval_score))

        if not candidate_passages:
            candidate_passages = top_chunks

        question_embedding = get_embeddings([question])[0]
        passage_texts = [passage[0] for passage in candidate_passages]
        passage_embeddings = get_embeddings(passage_texts)
        passage_scores = cosine_similarity([question_embedding], passage_embeddings)[0]

        best_index = int(np.argmax(passage_scores))
        answer = passage_texts[best_index].strip()

        citations: list[SourceCitation] = []
        for index, (text, page_number, retrieval_score) in enumerate(top_chunks):
            relevance = retrieval_score
            if index < len(passage_scores):
                relevance = max(retrieval_score, float(passage_scores[index]))

            citations.append(
                SourceCitation(
                    document_name="Company Document",
                    page_number=page_number,
                    text_snippet=text[:800],
                    relevance_score=max(0.0, min(1.0, float(relevance))),
                )
            )

        citations.sort(key=lambda citation: citation.relevance_score, reverse=True)
        return answer, citations[:3]

    except Exception as exc:
        logger.error("Error in extract_answer: %s", exc)
        best_chunk = chunks[0]
        return (
            best_chunk[0].strip(),
            [
                SourceCitation(
                    document_name="Company Document",
                    page_number=best_chunk[1],
                    text_snippet=best_chunk[0][:800],
                    relevance_score=max(0.0, min(1.0, float(best_chunk[2]))),
                )
            ],
        )


def answer_question(
    question: str,
    document: Document,
    store: FAISSStore,
    top_k: int = 6,
) -> TaskCell:
    cell = TaskCell(
        doc_id=document.doc_id,
        q_id=question,
        status=CellStatus.PROCESSING,
    )

    try:
        query_embedding = get_embeddings([question])[0]
        results = store.search(query_embedding, top_k=top_k)
        doc_chunks = [
            (chunk.text, chunk.page_number, score)
            for chunk, score in results
            if chunk.doc_id == document.doc_id
        ]

        if not doc_chunks:
            cell.answer = "No relevant information found in this document."
            cell.status = CellStatus.COMPLETED
            return cell

        answer, citations = extract_answer(question, doc_chunks, top_n=4)
        for citation in citations:
            citation.document_name = document.filename
            citation.source_url = document.metadata.source_url

        cell.answer = answer
        cell.citations = citations
        cell.status = CellStatus.COMPLETED

    except Exception as exc:
        logger.error("Error answering question for doc %s: %s", document.doc_id, exc)
        cell.status = CellStatus.ERROR
        cell.error_message = str(exc)

    return cell


def query_document(
    question: str,
    document: Document,
    store: FAISSStore,
) -> dict:
    cell = answer_question(question, document, store)
    return {
        "question": question,
        "answer": cell.answer,
        "citations": [
            {
                "page": citation.page_number,
                "text": citation.text_snippet,
                "score": citation.relevance_score,
                "source_url": citation.source_url,
            }
            for citation in cell.citations
        ],
        "status": cell.status.value,
        "error": cell.error_message,
    }
