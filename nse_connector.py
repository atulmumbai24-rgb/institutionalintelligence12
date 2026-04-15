"""
NSE-first connector for annual reports and result documents.
"""

from __future__ import annotations

from typing import Optional

from schemas import CompanyProfile, DiscoveredDocument, DocumentType
from source_utils import best_candidate_from_queries, build_session, current_fiscal_candidates


NSE_ALLOWED_DOMAINS = ["nseindia.com", "nsearchives.nseindia.com"]


def _build_nse_queries(profile: CompanyProfile, document_type: str) -> list[str]:
    fiscal = current_fiscal_candidates()
    symbol = profile.nse_symbol or profile.ticker or profile.display_name
    company = profile.display_name
    queries: list[str] = []

    if document_type == DocumentType.ANNUAL_REPORT.value:
        for label in fiscal["annual"][:2]:
            queries.extend(
                [
                    f"site:nsearchives.nseindia.com {symbol} annual report {label} pdf",
                    f"site:nseindia.com {symbol} annual report {label} pdf",
                    f"site:nsearchives.nseindia.com {company} annual report {label} pdf",
                ]
            )
    elif document_type == DocumentType.RESULTS_RELEASE.value:
        for label in fiscal["quarterly"][:3]:
            queries.extend(
                [
                    f"site:nseindia.com {symbol} financial results {label} pdf",
                    f"site:nseindia.com {company} financial results {label} pdf",
                    f"site:nseindia.com {symbol} integrated filing financials {label}",
                ]
            )
    elif document_type == DocumentType.INVESTOR_PRESENTATION.value:
        for label in fiscal["quarterly"][:3]:
            queries.extend(
                [
                    f"site:nseindia.com {symbol} investor presentation {label} pdf",
                    f"site:nseindia.com {company} investor presentation {label} pdf",
                ]
            )
    elif document_type == DocumentType.CONCALL_TRANSCRIPT.value:
        for label in fiscal["quarterly"][:2]:
            queries.extend(
                [
                    f"site:nseindia.com {symbol} transcript {label} pdf",
                    f"site:nseindia.com {company} conference call transcript {label} pdf",
                ]
            )

    return list(dict.fromkeys(queries))


def search_nse_documents(
    profile: CompanyProfile,
    include_types: Optional[list[str]] = None,
) -> list[DiscoveredDocument]:
    session = build_session()
    document_types = include_types or [
        DocumentType.ANNUAL_REPORT.value,
        DocumentType.RESULTS_RELEASE.value,
    ]

    discovered: list[DiscoveredDocument] = []
    for document_type in document_types:
        queries = _build_nse_queries(profile, document_type)
        if not queries:
            continue
        candidate = best_candidate_from_queries(
            profile,
            document_type,
            queries,
            source_kind="NSE",
            allowed_domains=NSE_ALLOWED_DOMAINS,
            session=session,
        )
        if candidate:
            discovered.append(candidate)

    return discovered
