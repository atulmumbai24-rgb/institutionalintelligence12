"""
Question answering engine with finance-specific routing and grounded retrieval.
"""

from __future__ import annotations

import re

from financials import answer_metric_question, detect_financial_intent
from ingestion import FAISSStore, get_embeddings
from schemas import Document, SourceCitation


def _shorten_text(text: str, max_chars: int = 650) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    output = ""
    for sentence in sentences:
        if len(output) + len(sentence) + 1 > max_chars:
            break
        output = f"{output} {sentence}".strip()
    return output or text[:max_chars].strip()


def _general_retrieval_answer(
    question: str,
    documents: list[Document],
    store: FAISSStore,
) -> dict:
    if not documents:
        return {
            "question": question,
            "answer": "No documents are loaded for this company yet.",
            "citations": [],
            "status": "error",
            "error": "No documents loaded",
            "confidence": 0.0,
            "intent": {"kind": "general", "metric_name": None},
            "metric_rows": [],
        }

    query_embedding = get_embeddings([question])[0]
    results = store.search(query_embedding, top_k=8)
    document_map = {document.doc_id: document for document in documents}

    citations: list[SourceCitation] = []
    seen_pages: set[tuple[str, int]] = set()
    for chunk, score in results:
        page_key = (chunk.doc_id, chunk.page_number)
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        document = document_map[chunk.doc_id]
        citations.append(
            SourceCitation(
                document_name=document.filename,
                document_type=document.metadata.document_type,
                page_number=chunk.page_number,
                text_snippet=_shorten_text(chunk.text, max_chars=900),
                relevance_score=max(0.0, min(1.0, float(score))),
                source_url=document.metadata.source_url,
            )
        )
        if len(citations) >= 3:
            break

    if not citations:
        return {
            "question": question,
            "answer": "I could not find grounded evidence in the loaded documents for that question.",
            "citations": [],
            "status": "completed",
            "error": None,
            "confidence": 0.0,
            "intent": {"kind": "general", "metric_name": None},
            "metric_rows": [],
        }

    answer = citations[0].text_snippet
    if len(citations) > 1:
        answer += f"\n\nSupporting evidence: {citations[1].text_snippet}"

    confidence = sum(citation.relevance_score for citation in citations) / len(citations)
    if confidence < 0.25:
        answer = (
            "I found only weakly matching evidence, so I can't state a definitive answer yet. "
            f"Closest grounded passage:\n\n{citations[0].text_snippet}"
        )

    return {
        "question": question,
        "answer": answer,
        "citations": [
            {
                "page": citation.page_number,
                "text": citation.text_snippet,
                "score": citation.relevance_score,
                "source_url": citation.source_url,
                "document_type": citation.document_type,
                "document_name": citation.document_name,
            }
            for citation in citations
        ],
        "status": "completed",
        "error": None,
        "confidence": confidence,
        "intent": {"kind": "general", "metric_name": None},
        "metric_rows": [],
    }


def query_company(
    question: str,
    documents: list[Document],
    store: FAISSStore,
) -> dict:
    intent = detect_financial_intent(question)

    if intent["kind"] == "metric":
        metric_result = answer_metric_question(question, documents, store, intent)
        metric_result["question"] = question
        metric_result["citations"] = [
            {
                "page": citation.page_number,
                "text": citation.text_snippet,
                "score": citation.relevance_score,
                "source_url": citation.source_url,
                "document_type": citation.document_type,
                "document_name": citation.document_name,
            }
            for citation in metric_result["citations"]
        ]
        return metric_result

    return _general_retrieval_answer(question, documents, store)


def query_document(
    question: str,
    document: Document,
    store: FAISSStore,
) -> dict:
    return query_company(question, [document], store)
