"""
BSE connector for annual reports and results.
"""

from __future__ import annotations

from typing import Optional

from schemas import CompanyProfile, DiscoveredDocument, DocumentType
from source_utils import best_candidate_from_queries, build_session, current_fiscal_candidates


BSE_ALLOWED_DOMAINS = ["bseindia.com"]


def _build_bse_queries(profile: CompanyProfile, document_type: str) -> list[str]:
    fiscal = current_fiscal_candidates()
    symbol = profile.ticker or profile.nse_symbol or profile.display_name
    company = profile.display_name
    queries: list[str] = []

    if document_type == DocumentType.ANNUAL_REPORT.value:
        for label in fiscal["annual"][:2]:
            queries.extend(
                [
                    f"site:bseindia.com {company} annual report {label} pdf",
                    f"site:bseindia.com {symbol} annual report {label} pdf",
                    f"site:bseindia.com financials annual reports {company}",
                ]
            )
    elif document_type == DocumentType.RESULTS_RELEASE.value:
        for label in fiscal["quarterly"][:3]:
            queries.extend(
                [
                    f"site:bseindia.com {company} financial results {label} pdf",
                    f"site:bseindia.com {symbol} quarterly results {label} pdf",
                    f"site:bseindia.com corporate announcements {company} results {label}",
                ]
            )
    elif document_type == DocumentType.INVESTOR_PRESENTATION.value:
        for label in fiscal["quarterly"][:2]:
            queries.extend(
                [
                    f"site:bseindia.com {company} investor presentation {label} pdf",
                    f"site:bseindia.com {symbol} investor presentation {label} pdf",
                ]
            )

    return list(dict.fromkeys(queries))


def search_bse_documents(
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
        queries = _build_bse_queries(profile, document_type)
        if not queries:
            continue
        candidate = best_candidate_from_queries(
            profile,
            document_type,
            queries,
            source_kind="BSE",
            allowed_domains=BSE_ALLOWED_DOMAINS,
            session=session,
        )
        if candidate:
            discovered.append(candidate)

    return discovered
