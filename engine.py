"""
engine.py — Pure Retrieval-Based Question Answering Engine

Implements extractive question answering using ONLY vector similarity.
NO LLM calls, fully offline capability.

Approach:
1. Embed the question using sentence-transformers
2. Retrieve top-k most similar chunks from FAISS
3. Compute cosine similarity between question and each chunk
4. Return the best matching chunk as the answer
5. Include citations with page numbers
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from schemas import (
    CellStatus,
    Document,
    Question,
    SourceCitation,
    TaskCell,
)
from ingestion import FAISSStore, get_embeddings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extractive Answer Extraction
# ---------------------------------------------------------------------------

def extract_answer(
    question: str,
    chunks: list[tuple[str, int, float]],
    top_n: int = 3
) -> tuple[str, list[SourceCitation]]:
    """
    Extract answer using pure retrieval and semantic similarity.
    
    Args:
        question: User question text
        chunks: List of (text, page_number, retrieval_score) tuples from FAISS
        top_n: Number of top chunks to consider for answer
    
    Returns:
        Tuple of (answer_text, list_of_citations)
    """
    if not chunks:
        return "No relevant information found in the document.", []
    
    # Take top N chunks
    top_chunks = chunks[:top_n]
    
    # Embed question and chunks for fine-grained similarity
    try:
        question_emb = get_embeddings([question])[0]
        chunk_texts = [chunk[0] for chunk in top_chunks]
        chunk_embs = get_embeddings(chunk_texts)
        
        # Compute cosine similarity
        similarities = cosine_similarity([question_emb], chunk_embs)[0]
        
        # Find best matching chunk
        best_idx = np.argmax(similarities)
        best_score = float(similarities[best_idx])
        best_chunk_text = chunk_texts[best_idx]
        
        # Build answer from best chunk
        answer = best_chunk_text.strip()
        
        # Create citations from top chunks
        citations = []
        for i, (text, page_num, retrieval_score) in enumerate(top_chunks):
            # Use re-ranking score if available, otherwise use retrieval score
            relevance = float(similarities[i]) if i < len(similarities) else retrieval_score
            
            citations.append(
                SourceCitation(
                    document_name="Company Document",
                    page_number=page_num,
                    text_snippet=text[:300],  # First 300 chars
                    relevance_score=relevance,
                )
            )
        
        # Sort citations by relevance
        citations.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return answer, citations[:3]  # Return top 3 citations
        
    except Exception as e:
        logger.error(f"Error in extract_answer: {e}")
        # Fallback: return best FAISS chunk
        best_chunk = chunks[0]
        answer = best_chunk[0].strip()
        citations = [
            SourceCitation(
                document_name="Company Document",
                page_number=best_chunk[1],
                text_snippet=best_chunk[0][:300],
                relevance_score=best_chunk[2],
            )
        ]
        return answer, citations

# ---------------------------------------------------------------------------
# Single Document Question Answering
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    document: Document,
    store: FAISSStore,
    top_k: int = 5
) -> TaskCell:
    """
    Answer a question for a single document using pure retrieval.
    
    Args:
        question: User question text
        document: Document object to query
        store: FAISS store containing document chunks
        top_k: Number of chunks to retrieve
    
    Returns:
        TaskCell with answer and citations
    """
    cell = TaskCell(
        doc_id=document.doc_id,
        q_id=question,
        status=CellStatus.PROCESSING,
    )
    
    try:
        # 1. Embed the question
        query_emb = get_embeddings([question])[0]
        
        # 2. Retrieve top-k chunks from FAISS
        results = store.search(query_emb, top_k=top_k)
        
        # 3. Filter chunks belonging to this document
        doc_chunks = [
            (chunk.text, chunk.page_number, score)
            for chunk, score in results
            if chunk.doc_id == document.doc_id
        ]
        
        if not doc_chunks:
            cell.answer = "No relevant information found in this document."
            cell.status = CellStatus.COMPLETED
            return cell
        
        # 4. Extract answer using semantic similarity
        answer, citations = extract_answer(question, doc_chunks, top_n=3)
        
        # Update document name in citations
        for citation in citations:
            citation.document_name = document.filename
        
        cell.answer = answer
        cell.citations = citations
        cell.status = CellStatus.COMPLETED
        
    except Exception as e:
        logger.error(f"Error answering question for doc {document.doc_id}: {e}")
        cell.status = CellStatus.ERROR
        cell.error_message = str(e)
    
    return cell

# ---------------------------------------------------------------------------
# Batch Processing (Multiple Questions)
# ---------------------------------------------------------------------------

def answer_multiple_questions(
    questions: list[str],
    document: Document,
    store: FAISSStore,
) -> list[TaskCell]:
    """
    Answer multiple questions for a single document.
    
    Args:
        questions: List of question texts
        document: Document to query
        store: FAISS store
    
    Returns:
        List of TaskCell objects with answers
    """
    cells = []
    
    for question in questions:
        cell = answer_question(question, document, store)
        cells.append(cell)
    
    return cells

# ---------------------------------------------------------------------------
# Simple Query Interface (No Grid)
# ---------------------------------------------------------------------------

def query_document(
    question: str,
    document: Document,
    store: FAISSStore,
) -> dict:
    """
    Simple query interface returning answer and citations as dict.
    
    Args:
        question: User question
        document: Document to query
        store: FAISS store
    
    Returns:
        Dict with 'answer', 'citations', and 'status'
    """
    cell = answer_question(question, document, store)
    
    return {
        "question": question,
        "answer": cell.answer,
        "citations": [
            {
                "page": c.page_number,
                "text": c.text_snippet,
                "score": c.relevance_score,
            }
            for c in cell.citations
        ],
        "status": cell.status.value,
        "error": cell.error_message,
    }