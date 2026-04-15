"""
Company document discovery and download orchestration.
Prioritizes official sources: NSE -> BSE -> Investor Relations -> Web fallback.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from typing import Optional

import requests

from bse_connector import search_bse_documents
from company_registry import (
    extract_company_name_from_query,
    get_company_profile,
    list_companies,
    normalize_company_name,
)
from ir_connector import search_ir_documents
from nse_connector import search_nse_documents
from schemas import DiscoveredDocument, DocumentType
from source_utils import (
    alias_variants,
    best_candidate_from_queries,
    build_session,
    current_fiscal_candidates,
    looks_like_pdf,
)

logger = logging.getLogger(__name__)


def extract_company_name(query: str) -> Optional[str]:
    return extract_company_name_from_query(query)


def list_available_companies() -> list[str]:
    return list_companies()


def _build_fallback_queries(profile, document_type: str) -> list[str]:
    fiscal = current_fiscal_candidates()
    queries: list[str] = []

    for alias in alias_variants(profile):
        quoted = f"\"{alias}\""
        if document_type == DocumentType.ANNUAL_REPORT.value:
            for label in fiscal["annual"][:2]:
                queries.extend(
                    [
                        f"{quoted} integrated annual report {label} pdf",
                        f"{quoted} annual report {label} pdf",
                    ]
                )
        elif document_type == DocumentType.RESULTS_RELEASE.value:
            for label in fiscal["quarterly"][:3]:
                queries.extend(
                    [
                        f"{quoted} financial results {label} pdf",
                        f"{quoted} quarterly results {label} pdf",
                    ]
                )
        elif document_type == DocumentType.INVESTOR_PRESENTATION.value:
            for label in fiscal["quarterly"][:3]:
                queries.extend(
                    [
                        f"{quoted} investor presentation {label} pdf",
                        f"{quoted} earnings presentation {label} pdf",
                    ]
                )
        elif document_type == DocumentType.CONCALL_TRANSCRIPT.value:
            for label in fiscal["quarterly"][:2]:
                queries.extend(
                    [
                        f"{quoted} conference call transcript {label} pdf",
                        f"{quoted} earnings call transcript {label} pdf",
                    ]
                )

    return list(dict.fromkeys(queries))


def _search_web_fallback(profile, document_type: str) -> Optional[DiscoveredDocument]:
    queries = _build_fallback_queries(profile, document_type)
    if not queries:
        return None

    return best_candidate_from_queries(
        profile,
        document_type,
        queries,
        source_kind="Web Fallback",
        allowed_domains=None,
        session=build_session(),
    )


def search_company_documents(
    company_name: str,
    max_documents: int = 3,
    include_types: Optional[list[str]] = None,
) -> list[DiscoveredDocument]:
    normalized_company = normalize_company_name(company_name)
    profile = get_company_profile(normalized_company)
    target_types = include_types or [
        DocumentType.ANNUAL_REPORT.value,
        DocumentType.INVESTOR_PRESENTATION.value,
        DocumentType.RESULTS_RELEASE.value,
    ]

    selected_by_type: dict[str, DiscoveredDocument] = {}
    seen_urls: set[str] = set()
    connector_results = [
        search_nse_documents(profile, include_types=target_types),
        search_bse_documents(profile, include_types=target_types),
        search_ir_documents(profile, include_types=target_types),
    ]

    for results in connector_results:
        for document in results:
            if document.document_type in selected_by_type:
                continue
            if document.source_url in seen_urls:
                continue
            selected_by_type[document.document_type] = document
            seen_urls.add(document.source_url)

    for document_type in target_types:
        if document_type in selected_by_type:
            continue
        fallback = _search_web_fallback(profile, document_type)
        if fallback and fallback.source_url not in seen_urls:
            selected_by_type[document_type] = fallback
            seen_urls.add(fallback.source_url)

    ordered_documents = [
        selected_by_type[document_type]
        for document_type in target_types
        if document_type in selected_by_type
    ]
    return ordered_documents[:max_documents]


def _default_save_dir() -> str:
    return os.path.join(tempfile.gettempdir(), "company_intelligence", "downloads")


def _build_filename(document: DiscoveredDocument) -> str:
    year_label = (document.financial_year or document.period_label or "latest").replace("/", "-")
    digest = hashlib.sha1(document.source_url.encode("utf-8")).hexdigest()[:10]
    safe_company = re.sub(r"[^a-z0-9]+", "_", document.company_name.lower()).strip("_")
    safe_type = re.sub(r"[^a-z0-9]+", "_", document.document_type.lower()).strip("_")
    safe_source = re.sub(r"[^a-z0-9]+", "_", document.source_kind.lower()).strip("_")
    return f"{safe_company}_{safe_type}_{safe_source}_{year_label}_{digest}.pdf"


def download_document(document: DiscoveredDocument, save_dir: Optional[str] = None) -> DiscoveredDocument:
    save_dir = save_dir or _default_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, _build_filename(document))

    if os.path.exists(filepath):
        document.local_path = filepath
        return document

    session = build_session()
    try:
        response = session.get(
            document.source_url,
            timeout=45,
            stream=True,
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise Exception(f"Document download failed: {exc}") from exc

    content_type = response.headers.get("Content-Type", "").lower()
    final_url = response.url
    if "pdf" not in content_type and "octet-stream" not in content_type and not looks_like_pdf(final_url):
        raise Exception(f"Downloaded content was not a PDF: {content_type or final_url}")

    with open(filepath, "wb") as file_handle:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file_handle.write(chunk)

    document.local_path = filepath
    return document


def fetch_company_documents(
    query: str,
    company_name: Optional[str] = None,
    max_documents: int = 3,
) -> tuple[Optional[str], list[DiscoveredDocument]]:
    resolved_company = normalize_company_name(company_name) if company_name else extract_company_name(query)
    if not resolved_company:
        return None, []

    discovered = search_company_documents(resolved_company, max_documents=max_documents)
    downloaded: list[DiscoveredDocument] = []

    for document in discovered:
        try:
            downloaded.append(download_document(document))
        except Exception as exc:
            logger.warning(
                "Failed to download %s from %s (%s): %s",
                document.document_type,
                document.source_url,
                document.source_kind,
                exc,
            )

    return resolved_company, downloaded


def get_company_pdf(company_name: str) -> Optional[str]:
    results = search_company_documents(
        company_name,
        max_documents=1,
        include_types=[DocumentType.ANNUAL_REPORT.value],
    )
    return results[0].source_url if results else None


def fetch_company_document(
    query: str,
    company_name: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    resolved_company, documents = fetch_company_documents(query, company_name=company_name, max_documents=1)
    if not documents:
        return resolved_company, None, None
    return resolved_company, documents[0].source_url, documents[0].local_path
