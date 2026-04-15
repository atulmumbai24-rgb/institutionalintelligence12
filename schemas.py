"""
schemas.py — Pydantic Data Models for Institutional Intelligence Platform

Defines structured schemas for documents, questions, task cells, grid results,
and source citations. Includes mock Indian corporate data for demonstration.
"""

from __future__ import annotations

import enum
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PIIType(str, enum.Enum):
    """Types of Personally Identifiable Information relevant to Indian regulations."""
    AADHAAR = "aadhaar"  # 12-digit unique ID
    PAN = "pan"  # ABCDE1234F format
    DIN = "din"  # 8-digit Director Identification Number

class CellStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------

class SourceCitation(BaseModel):
    """A specific citation pointing back to a source document."""
    document_name: str = Field(..., description="Name of the source PDF")
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    text_snippet: str = Field(..., max_length=500, description="Relevant text excerpt")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Cosine similarity score")

class DocumentMetadata(BaseModel):
    """Metadata for an ingested corporate document."""
    company_name: str
    document_type: str = Field(default="Annual Report")
    financial_year: Optional[str] = None
    filing_date: Optional[date] = None
    cik_or_cin: Optional[str] = Field(None, description="CIN for Indian companies")
    language: str = Field(default="en", description="Primary language code (en, hi, hinglish)")

class Document(BaseModel):
    """Represents a parsed and indexed document."""
    doc_id: str = Field(..., description="Unique document identifier")
    filename: str
    metadata: DocumentMetadata
    num_pages: int = 0
    num_chunks: int = 0
    pii_masked: bool = False

class Question(BaseModel):
    """A user-defined question to run across all documents."""
    q_id: str = Field(..., description="Unique question identifier")
    text: str = Field(..., min_length=5, description="The question text")

class TaskCell(BaseModel):
    """One cell in the document × question matrix."""
    doc_id: str
    q_id: str
    status: CellStatus = CellStatus.PENDING
    answer: Optional[str] = None
    citations: list[SourceCitation] = Field(default_factory=list)
    error_message: Optional[str] = None

class GridResult(BaseModel):
    """Complete result grid: rows=documents, columns=questions."""
    documents: list[Document]
    questions: list[Question]
    cells: list[TaskCell] = Field(default_factory=list)

    def get_cell(self, doc_id: str, q_id: str) -> Optional[TaskCell]:
        """Retrieve a specific cell by document and question IDs."""
        for cell in self.cells:
            if cell.doc_id == doc_id and cell.q_id == q_id:
                return cell
        return None

class TextChunk(BaseModel):
    """A chunk of text extracted from a document, ready for embedding."""
    chunk_id: str
    doc_id: str
    page_number: int
    text: str
    embedding: Optional[list[float]] = Field(None, exclude=True)

# ---------------------------------------------------------------------------
# Mock Indian Corporate Data (for demo / quick start)
# ---------------------------------------------------------------------------

MOCK_COMPANIES: list[DocumentMetadata] = [
    DocumentMetadata(
        company_name="Reliance Industries Ltd",
        document_type="Annual Report",
        financial_year="2024-25",
        filing_date=date(2025, 8, 15),
        cik_or_cin="L17110MH1973PLC019786",
        language="en",
    ),
    DocumentMetadata(
        company_name="Tata Consultancy Services Ltd",
        document_type="Annual Report",
        financial_year="2024-25",
        filing_date=date(2025, 7, 1),
        cik_or_cin="L22210MH2004PLC146615",
        language="en",
    ),
    DocumentMetadata(
        company_name="Infosys Ltd",
        document_type="Annual Report",
        financial_year="2024-25",
        filing_date=date(2025, 6, 20),
        cik_or_cin="L85110KA1981PLC013115",
        language="en",
    ),
    DocumentMetadata(
        company_name="HDFC Bank Ltd",
        document_type="Annual Report",
        financial_year="2024-25",
        filing_date=date(2025, 7, 10),
        cik_or_cin="L65920MH1994PLC080618",
        language="en",
    ),
    DocumentMetadata(
        company_name="Bharat Heavy Electricals Ltd",
        document_type="Annual Report",
        financial_year="2024-25",
        filing_date=date(2025, 9, 1),
        cik_or_cin="L74899DL1964GOI004281",
        language="hinglish",
    ),
]

SAMPLE_QUESTIONS: list[str] = [
    "What is the Contingent Liability as reported?",
    "Is there a change in Auditor during the year?",
    "What are the key risks mentioned under Risk Management?",
    "Does the report mention DPDP Act or data privacy compliance?",
    "What is the Revenue from Operations for the current year?",
]
