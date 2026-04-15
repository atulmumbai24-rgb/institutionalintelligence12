"""
Core schemas for the company intelligence platform.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class CellStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class DocumentType(str, enum.Enum):
    ANNUAL_REPORT = "Annual Report"
    INVESTOR_PRESENTATION = "Investor Presentation"
    RESULTS_RELEASE = "Results Release"
    CONCALL_TRANSCRIPT = "Concall Transcript"
    EXCHANGE_FILING = "Exchange Filing"


class CompanyProfile(BaseModel):
    canonical_name: str
    display_name: str
    ticker: Optional[str] = None
    nse_symbol: Optional[str] = None
    investor_relations_url: Optional[str] = None
    aliases: list[str] = Field(default_factory=list)
    official_domains: list[str] = Field(default_factory=list)


class DiscoveredDocument(BaseModel):
    company_name: str
    document_type: str
    source_kind: str = "Web Search"
    title: str
    source_url: str
    source_domain: str
    search_query: str
    financial_year: Optional[str] = None
    period_label: Optional[str] = None
    local_path: Optional[str] = None
    confidence: float = 0.0
    discovered_at: datetime = Field(default_factory=datetime.utcnow)


class SourceCitation(BaseModel):
    document_name: str
    document_type: str
    page_number: int = Field(..., ge=1)
    text_snippet: str = Field(..., max_length=1200)
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)
    source_url: Optional[str] = None


class DocumentMetadata(BaseModel):
    company_name: str
    document_type: str = Field(default=DocumentType.ANNUAL_REPORT.value)
    financial_year: Optional[str] = None
    source_url: Optional[str] = None
    source_domain: Optional[str] = None
    source_title: Optional[str] = None
    search_query: Optional[str] = None
    period_label: Optional[str] = None


class Document(BaseModel):
    doc_id: str
    filename: str
    metadata: DocumentMetadata
    num_pages: int = 0
    num_chunks: int = 0
    pii_masked: bool = False


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


class MetricPoint(BaseModel):
    metric_name: str
    period_label: str
    value_text: str
    value_numeric: Optional[float] = None
    unit: Optional[str] = None
    page_number: int
    document_name: str
    document_type: str
    source_url: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
