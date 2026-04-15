"""
Live source discovery and download utilities for company filings and reports.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from datetime import date
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from company_registry import (
    extract_company_name_from_query,
    get_company_profile,
    list_companies,
    normalize_company_name,
)
from schemas import DiscoveredDocument, DocumentType

logger = logging.getLogger(__name__)


EXCHANGE_DOMAIN_HINTS = ("nseindia.com", "nsearchives.nseindia.com", "bseindia.com")
BACKUP_SOURCE_HINTS = ("annualreports.com",)


def extract_company_name(query: str) -> Optional[str]:
    return extract_company_name_from_query(query)


def list_available_companies() -> list[str]:
    return list_companies()


def _browser_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
    }


def _build_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(403, 429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update(_browser_headers())
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _looks_like_pdf(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".pdf") or ".pdf" in path


def _extract_financial_year(text: str) -> Optional[str]:
    explicit = re.search(r"(20\d{2}-\d{2})", text)
    if explicit:
        return explicit.group(1)

    fy_match = re.search(r"FY\s?(\d{2})", text, re.IGNORECASE)
    if fy_match:
        fy_two_digits = fy_match.group(1)
        start_year = 2000 + int(fy_two_digits) - 1
        return f"{start_year}-{fy_two_digits}"

    return None


def _current_fiscal_candidates() -> dict[str, list[str]]:
    today = date.today()
    completed_fy_end_year = today.year if today.month > 3 else today.year - 1
    previous_completed_fy_end_year = completed_fy_end_year - 1

    annual_labels = [
        f"{completed_fy_end_year - 1}-{str(completed_fy_end_year)[-2:]}",
        f"{previous_completed_fy_end_year - 1}-{str(previous_completed_fy_end_year)[-2:]}",
        f"FY{completed_fy_end_year % 100:02d}",
        f"FY{previous_completed_fy_end_year % 100:02d}",
    ]
    quarterly_labels = [
        f"FY{completed_fy_end_year % 100:02d}",
        f"FY{previous_completed_fy_end_year % 100:02d}",
        f"Q4 FY{completed_fy_end_year % 100:02d}",
        f"Q3 FY{completed_fy_end_year % 100:02d}",
    ]
    return {
        "annual": annual_labels,
        "quarterly": quarterly_labels,
    }


def _alias_variants(company_name: str) -> list[str]:
    profile = get_company_profile(company_name)
    aliases = [profile.display_name, profile.canonical_name.title()]
    if profile.ticker:
        aliases.append(profile.ticker)
    aliases.extend(alias.title() for alias in profile.aliases[:4])

    deduped: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        cleaned = alias.strip()
        if cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            deduped.append(cleaned)
    return deduped


def _build_queries(company_name: str, document_type: str) -> list[str]:
    fiscal_labels = _current_fiscal_candidates()
    queries: list[str] = []

    for alias in _alias_variants(company_name):
        quoted = f"\"{alias}\""
        if document_type == DocumentType.ANNUAL_REPORT.value:
            for label in fiscal_labels["annual"][:2]:
                queries.extend(
                    [
                        f"{quoted} integrated annual report {label} pdf",
                        f"{quoted} annual report {label} pdf",
                        f"site:nsearchives.nseindia.com {alias} annual report {label} pdf",
                        f"site:bseindia.com {alias} annual report {label} pdf",
                    ]
                )
            queries.append(f"{alias} investor relations annual report pdf")
        elif document_type == DocumentType.INVESTOR_PRESENTATION.value:
            for label in fiscal_labels["quarterly"][:3]:
                queries.extend(
                    [
                        f"{quoted} investor presentation {label} pdf",
                        f"{quoted} earnings presentation {label} pdf",
                        f"{quoted} results presentation {label} pdf",
                    ]
                )
        elif document_type == DocumentType.RESULTS_RELEASE.value:
            for label in fiscal_labels["quarterly"][:3]:
                queries.extend(
                    [
                        f"{quoted} results {label} pdf",
                        f"{quoted} quarterly results {label} pdf",
                        f"{quoted} financial results {label} pdf",
                    ]
                )
        elif document_type == DocumentType.CONCALL_TRANSCRIPT.value:
            for label in fiscal_labels["quarterly"][:2]:
                queries.extend(
                    [
                        f"{quoted} earnings call transcript {label} pdf",
                        f"{quoted} conference call transcript {label} pdf",
                    ]
                )

    return list(dict.fromkeys(queries))


def _page_terms_for_document_type(document_type: str) -> list[str]:
    mapping = {
        DocumentType.ANNUAL_REPORT.value: ["annual report", "integrated report"],
        DocumentType.INVESTOR_PRESENTATION.value: ["investor presentation", "results presentation", "earnings presentation"],
        DocumentType.RESULTS_RELEASE.value: ["results", "financial results", "quarterly results"],
        DocumentType.CONCALL_TRANSCRIPT.value: ["transcript", "conference call", "earnings call"],
    }
    return mapping.get(document_type, [])


def _score_search_result(
    company_name: str,
    document_type: str,
    title: str,
    body: str,
    url: str,
    query: str,
) -> int:
    title_lower = title.lower()
    body_lower = body.lower()
    url_lower = url.lower()
    domain = urlparse(url).netloc.lower()
    profile = get_company_profile(company_name)

    score = 0
    if _looks_like_pdf(url):
        score += 30
    if any(domain_hint in domain for domain_hint in EXCHANGE_DOMAIN_HINTS):
        score += 45
    if any(domain_hint in domain for domain_hint in BACKUP_SOURCE_HINTS):
        score += 10
    if any(official_domain in domain for official_domain in profile.official_domains):
        score += 60

    for term in _page_terms_for_document_type(document_type):
        if term in title_lower:
            score += 20
        if term in body_lower or term in url_lower:
            score += 10

    alias_hits = 0
    for alias in _alias_variants(company_name):
        alias_lower = alias.lower()
        if alias_lower in title_lower or alias_lower in body_lower or alias_lower in url_lower:
            alias_hits += 1
        alias_token = re.sub(r"[^a-z0-9]", "", alias_lower)
        if alias_token and alias_token in re.sub(r"[^a-z0-9]", "", domain):
            alias_hits += 2
    score += min(20, alias_hits * 4)

    extracted_year = _extract_financial_year(f"{title} {body} {url}")
    if extracted_year:
        score += 12
    if "nse" in query.lower():
        score += 8
    if "investor" in url_lower:
        score += 10

    return score


def _candidate_from_result(
    company_name: str,
    document_type: str,
    title: str,
    url: str,
    query: str,
    score: int,
) -> DiscoveredDocument:
    return DiscoveredDocument(
        company_name=normalize_company_name(company_name),
        document_type=document_type,
        title=title or document_type,
        source_url=url,
        source_domain=urlparse(url).netloc,
        search_query=query,
        financial_year=_extract_financial_year(f"{title} {url}"),
        period_label=_extract_financial_year(f"{title} {url}"),
        confidence=min(1.0, score / 120.0),
    )


def _extract_pdf_links_from_page(page_url: str, html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[tuple[str, str]] = []

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if not href:
            continue
        absolute_url = urljoin(page_url, href)
        anchor_text = " ".join(anchor.stripped_strings)
        candidate_text = f"{anchor_text} {absolute_url}".lower()
        if ".pdf" in absolute_url.lower() or any(
            term in candidate_text
            for term in ["annual report", "presentation", "results", "transcript", "pdf"]
        ):
            links.append((absolute_url, anchor_text or absolute_url))

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for url, text in links:
        if url not in seen:
            seen.add(url)
            deduped.append((url, text))
    return deduped


def _discover_pdf_from_page(
    company_name: str,
    document_type: str,
    page_url: str,
    query: str,
    session: requests.Session,
) -> Optional[DiscoveredDocument]:
    try:
        response = session.get(page_url, timeout=30, allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to inspect page %s: %s", page_url, exc)
        return None

    best_candidate: Optional[DiscoveredDocument] = None
    best_score = -1
    for candidate_url, candidate_text in _extract_pdf_links_from_page(page_url, response.text):
        score = _score_search_result(
            company_name,
            document_type,
            candidate_text,
            candidate_text,
            candidate_url,
            query,
        )
        if score > best_score:
            best_score = score
            best_candidate = _candidate_from_result(
                company_name,
                document_type,
                candidate_text,
                candidate_url,
                query,
                score,
            )

    return best_candidate


def _search_best_document(
    company_name: str,
    document_type: str,
    session: requests.Session,
) -> Optional[DiscoveredDocument]:
    queries = _build_queries(company_name, document_type)
    best_pdf_candidate: Optional[DiscoveredDocument] = None
    best_pdf_score = -1
    page_candidates: list[tuple[int, str, str]] = []

    with DDGS() as ddgs:
        for query in queries[:8]:
            logger.info("Searching %s for %s with query: %s", document_type, company_name, query)
            try:
                results = list(
                    ddgs.text(
                        query,
                        region="in-en",
                        safesearch="off",
                        backend="html",
                        max_results=8,
                    )
                )
            except Exception as exc:
                logger.warning("Search query failed for '%s': %s", query, exc)
                continue

            for result in results:
                url = (result.get("href") or result.get("url") or "").strip()
                title = (result.get("title") or "").strip()
                body = (result.get("body") or "").strip()
                if not url:
                    continue

                score = _score_search_result(company_name, document_type, title, body, url, query)
                if _looks_like_pdf(url):
                    if score > best_pdf_score:
                        best_pdf_score = score
                        best_pdf_candidate = _candidate_from_result(
                            company_name,
                            document_type,
                            title,
                            url,
                            query,
                            score,
                        )
                else:
                    page_candidates.append((score, url, query))

            if best_pdf_candidate and best_pdf_score >= 85:
                return best_pdf_candidate

    page_candidates.sort(key=lambda item: item[0], reverse=True)
    for score, page_url, query in page_candidates[:6]:
        if score < 35:
            continue
        page_candidate = _discover_pdf_from_page(company_name, document_type, page_url, query, session)
        if page_candidate:
            return page_candidate

    return best_pdf_candidate


def search_company_documents(
    company_name: str,
    max_documents: int = 3,
    include_types: Optional[list[str]] = None,
) -> list[DiscoveredDocument]:
    normalized_company = normalize_company_name(company_name)
    session = _build_session()
    document_types = include_types or [
        DocumentType.ANNUAL_REPORT.value,
        DocumentType.INVESTOR_PRESENTATION.value,
        DocumentType.RESULTS_RELEASE.value,
    ]

    discovered: list[DiscoveredDocument] = []
    seen_urls: set[str] = set()

    for document_type in document_types:
        candidate = _search_best_document(normalized_company, document_type, session)
        if candidate and candidate.source_url not in seen_urls:
            seen_urls.add(candidate.source_url)
            discovered.append(candidate)
        if len(discovered) >= max_documents:
            break

    return discovered


def _default_save_dir() -> str:
    return os.path.join(tempfile.gettempdir(), "company_intelligence", "downloads")


def _build_filename(document: DiscoveredDocument) -> str:
    year_label = (document.financial_year or document.period_label or "latest").replace("/", "-")
    digest = hashlib.sha1(document.source_url.encode("utf-8")).hexdigest()[:10]
    safe_company = re.sub(r"[^a-z0-9]+", "_", document.company_name.lower()).strip("_")
    safe_type = re.sub(r"[^a-z0-9]+", "_", document.document_type.lower()).strip("_")
    return f"{safe_company}_{safe_type}_{year_label}_{digest}.pdf"


def download_document(document: DiscoveredDocument, save_dir: Optional[str] = None) -> DiscoveredDocument:
    save_dir = save_dir or _default_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, _build_filename(document))

    if os.path.exists(filepath):
        document.local_path = filepath
        return document

    session = _build_session()
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
    if "pdf" not in content_type and not _looks_like_pdf(final_url):
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
            logger.warning("Failed to download %s from %s: %s", document.document_type, document.source_url, exc)

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
