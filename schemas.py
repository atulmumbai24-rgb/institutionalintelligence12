"""
Pydantic data models for the company intelligence app.
"""

from __future__ import annotations

import enum
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class CellStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class DocumentType(str, enum.Enum):
    ANNUAL_REPORT = "Annual Report"
    INTEGRATED_REPORT = "Integrated Annual Report"
    INVESTOR_PRESENTATION = "Investor Presentation"
    RESULTS_RELEASE = "Results Release"
    CONCALL_TRANSCRIPT = "Concall Transcript"
    EXCHANGE_FILING = "Exchange Filing"


class SourceCitation(BaseModel):
    document_name: str = Field(..., description="Name of the source document")
    page_number: int = Field(..., ge=1, description="1-indexed PDF page number")
    text_snippet: str = Field(..., max_length=800, description="Relevant excerpt")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Similarity score")
    source_url: Optional[str] = Field(None, description="Source document URL")


class DocumentMetadata(BaseModel):
    company_name: str
    document_type: str = Field(default=DocumentType.INTEGRATED_REPORT.value)
    financial_year: Optional[str] = None
    filing_date: Optional[date] = None
    language: str = Field(default="en", description="Primary language code")
    source_url: Optional[str] = None
    source_domain: Optional[str] = None
    search_query: Optional[str] = None


class Document(BaseModel):
    doc_id: str = Field(..., description="Unique document identifier")
    filename: str
    metadata: DocumentMetadata
    num_pages: int = 0
    num_chunks: int = 0
    pii_masked: bool = False


class Question(BaseModel):
    q_id: str = Field(..., description="Unique question identifier")
    text: str = Field(..., min_length=5, description="User question")


class TaskCell(BaseModel):
    doc_id: str
    q_id: str
    status: CellStatus = CellStatus.PENDING
    answer: Optional[str] = None
    citations: list[SourceCitation] = Field(default_factory=list)
    error_message: Optional[str] = None


class TextChunk(BaseModel):
    chunk_id: str
    doc_id: str
    page_number: int
    text: str
    embedding: Optional[list[float]] = Field(None, exclude=True)
