"""
Official investor-relations connector.
"""

from __future__ import annotations

from typing import Optional

from schemas import CompanyProfile, DiscoveredDocument, DocumentType
from source_utils import (
    alias_variants,
    best_candidate_from_queries,
    build_session,
    current_fiscal_candidates,
    inspect_page_for_document,
)


def _build_ir_queries(profile: CompanyProfile, document_type: str) -> list[str]:
    fiscal = current_fiscal_candidates()
    queries: list[str] = []

    for domain in profile.official_domains:
        for alias in alias_variants(profile):
            if document_type == DocumentType.ANNUAL_REPORT.value:
                for label in fiscal["annual"][:2]:
                    queries.extend(
                        [
                            f"site:{domain} {alias} investor relations annual report {label} pdf",
                            f"site:{domain} {alias} annual report {label} pdf",
                        ]
                    )
            elif document_type == DocumentType.INVESTOR_PRESENTATION.value:
                for label in fiscal["quarterly"][:3]:
                    queries.extend(
                        [
                            f"site:{domain} {alias} investor presentation {label} pdf",
                            f"site:{domain} {alias} earnings presentation {label} pdf",
                        ]
                    )
            elif document_type == DocumentType.RESULTS_RELEASE.value:
                for label in fiscal["quarterly"][:3]:
                    queries.extend(
                        [
                            f"site:{domain} {alias} results {label} pdf",
                            f"site:{domain} {alias} financial results {label} pdf",
                        ]
                    )
            elif document_type == DocumentType.CONCALL_TRANSCRIPT.value:
                for label in fiscal["quarterly"][:2]:
                    queries.extend(
                        [
                            f"site:{domain} {alias} transcript {label} pdf",
                            f"site:{domain} {alias} conference call transcript {label} pdf",
                        ]
                    )

    return list(dict.fromkeys(queries))


def search_ir_documents(
    profile: CompanyProfile,
    include_types: Optional[list[str]] = None,
) -> list[DiscoveredDocument]:
    session = build_session()
    document_types = include_types or [
        DocumentType.ANNUAL_REPORT.value,
        DocumentType.INVESTOR_PRESENTATION.value,
        DocumentType.RESULTS_RELEASE.value,
        DocumentType.CONCALL_TRANSCRIPT.value,
    ]

    discovered: list[DiscoveredDocument] = []
    allowed_domains = profile.official_domains or None

    for document_type in document_types:
        candidate: Optional[DiscoveredDocument] = None

        if profile.investor_relations_url:
            candidate = inspect_page_for_document(
                profile,
                document_type,
                profile.investor_relations_url,
                query=f"{profile.display_name} investor relations {document_type}",
                source_kind="Investor Relations",
                session=session,
                allowed_domains=allowed_domains,
            )

        if not candidate:
            queries = _build_ir_queries(profile, document_type)
            candidate = best_candidate_from_queries(
                profile,
                document_type,
                queries,
                source_kind="Investor Relations",
                allowed_domains=allowed_domains,
                session=session,
            )

        if candidate:
            discovered.append(candidate)

    return discovered
