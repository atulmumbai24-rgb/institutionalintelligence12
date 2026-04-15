"""
Shared helpers for official-source document connectors.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from schemas import CompanyProfile, DiscoveredDocument

logger = logging.getLogger(__name__)


EXCHANGE_DOMAINS = ("nseindia.com", "nsearchives.nseindia.com", "bseindia.com")
BACKUP_DOMAINS = ("annualreports.com",)


def browser_headers() -> dict[str, str]:
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


def build_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(403, 429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update(browser_headers())
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def looks_like_pdf(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".pdf") or ".pdf" in path


def extract_financial_year(text: str) -> Optional[str]:
    explicit = re.search(r"(20\d{2}-\d{2})", text)
    if explicit:
        return explicit.group(1)

    fy_match = re.search(r"FY\s?(\d{2})", text, re.IGNORECASE)
    if fy_match:
        fy_two_digits = fy_match.group(1)
        start_year = 2000 + int(fy_two_digits) - 1
        return f"{start_year}-{fy_two_digits}"

    return None


def current_fiscal_candidates() -> dict[str, list[str]]:
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
    return {"annual": annual_labels, "quarterly": quarterly_labels}


def alias_variants(profile: CompanyProfile) -> list[str]:
    aliases = [profile.display_name, profile.canonical_name.title()]
    if profile.ticker:
        aliases.append(profile.ticker)
    if profile.nse_symbol and profile.nse_symbol != profile.ticker:
        aliases.append(profile.nse_symbol)
    aliases.extend(alias.title() for alias in profile.aliases[:4])

    deduped: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        cleaned = alias.strip()
        if cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            deduped.append(cleaned)
    return deduped


def document_terms(document_type: str) -> list[str]:
    mapping = {
        "Annual Report": ["annual report", "integrated report"],
        "Investor Presentation": ["investor presentation", "results presentation", "earnings presentation"],
        "Results Release": ["results", "financial results", "quarterly results"],
        "Concall Transcript": ["transcript", "conference call", "earnings call"],
        "Exchange Filing": ["filing", "announcement", "corporate filing"],
    }
    return mapping.get(document_type, [])


def domain_matches(url: str, allowed_domains: Optional[list[str]]) -> bool:
    if not allowed_domains:
        return True
    domain = urlparse(url).netloc.lower()
    return any(allowed in domain for allowed in allowed_domains)


def score_candidate(
    profile: CompanyProfile,
    document_type: str,
    title: str,
    body: str,
    url: str,
    source_kind: str,
    query: str,
) -> int:
    title_lower = title.lower()
    body_lower = body.lower()
    url_lower = url.lower()
    domain = urlparse(url).netloc.lower()

    score = 0
    if looks_like_pdf(url):
        score += 30
    if source_kind == "NSE" and any(domain_hint in domain for domain_hint in ("nseindia.com", "nsearchives.nseindia.com")):
        score += 70
    if source_kind == "BSE" and "bseindia.com" in domain:
        score += 70
    if source_kind == "Investor Relations" and any(official_domain in domain for official_domain in profile.official_domains):
        score += 70
    if any(official_domain in domain for official_domain in profile.official_domains):
        score += 30
    if any(exchange_domain in domain for exchange_domain in EXCHANGE_DOMAINS):
        score += 20
    if any(backup_domain in domain for backup_domain in BACKUP_DOMAINS):
        score += 8

    for term in document_terms(document_type):
        if term in title_lower:
            score += 20
        if term in body_lower or term in url_lower:
            score += 10

    if "investor" in url_lower:
        score += 10
    if extract_financial_year(f"{title} {body} {url}"):
        score += 12
    if "nse" in query.lower() and source_kind == "NSE":
        score += 8
    if "bse" in query.lower() and source_kind == "BSE":
        score += 8

    alias_hits = 0
    for alias in alias_variants(profile):
        alias_lower = alias.lower()
        if alias_lower in title_lower or alias_lower in body_lower or alias_lower in url_lower:
            alias_hits += 1
        alias_token = re.sub(r"[^a-z0-9]", "", alias_lower)
        if alias_token and alias_token in re.sub(r"[^a-z0-9]", "", domain):
            alias_hits += 2
    score += min(24, alias_hits * 4)

    return score


def build_candidate(
    profile: CompanyProfile,
    document_type: str,
    title: str,
    url: str,
    query: str,
    source_kind: str,
    score: int,
) -> DiscoveredDocument:
    extracted_year = extract_financial_year(f"{title} {url}")
    return DiscoveredDocument(
        company_name=profile.canonical_name,
        document_type=document_type,
        source_kind=source_kind,
        title=title or document_type,
        source_url=url,
        source_domain=urlparse(url).netloc,
        search_query=query,
        financial_year=extracted_year,
        period_label=extracted_year,
        confidence=min(1.0, score / 140.0),
    )


def search_results_for_queries(queries: list[str], max_results: int = 8) -> list[tuple[str, dict]]:
    output: list[tuple[str, dict]] = []
    with DDGS() as ddgs:
        for query in queries:
            try:
                results = list(ddgs.text(query, region="in-en", safesearch="off", max_results=max_results))
            except Exception as exc:
                logger.warning("Search query failed for '%s': %s", query, exc)
                continue
            for result in results:
                output.append((query, result))
    return output


def extract_links_from_html(
    page_url: str,
    html: str,
    document_type: str,
    allowed_domains: Optional[list[str]] = None,
) -> tuple[list[tuple[str, str]], list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    pdf_links: list[tuple[str, str]] = []
    internal_pages: list[str] = []
    doc_terms = document_terms(document_type)
    base_domain = urlparse(page_url).netloc.lower()

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if not href:
            continue
        absolute_url = urljoin(page_url, href)
        if not domain_matches(absolute_url, allowed_domains):
            continue

        anchor_text = " ".join(anchor.stripped_strings)
        candidate_text = f"{anchor_text} {absolute_url}".lower()
        if looks_like_pdf(absolute_url) or any(term in candidate_text for term in doc_terms + ["pdf"]):
            pdf_links.append((absolute_url, anchor_text or absolute_url))

        if (
            urlparse(absolute_url).netloc.lower() == base_domain
            and not looks_like_pdf(absolute_url)
            and any(term in candidate_text for term in doc_terms + ["investor", "financial", "reports", "filings"])
        ):
            internal_pages.append(absolute_url)

    deduped_pdfs: list[tuple[str, str]] = []
    seen_urls: set[str] = set()
    for url, text in pdf_links:
        if url not in seen_urls:
            seen_urls.add(url)
            deduped_pdfs.append((url, text))

    deduped_pages: list[str] = []
    seen_pages: set[str] = set()
    for page in internal_pages:
        if page not in seen_pages:
            seen_pages.add(page)
            deduped_pages.append(page)

    return deduped_pdfs, deduped_pages


def inspect_page_for_document(
    profile: CompanyProfile,
    document_type: str,
    page_url: str,
    query: str,
    source_kind: str,
    session: requests.Session,
    allowed_domains: Optional[list[str]] = None,
) -> Optional[DiscoveredDocument]:
    try:
        response = session.get(page_url, timeout=30, allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to inspect page %s: %s", page_url, exc)
        return None

    best_candidate: Optional[DiscoveredDocument] = None
    best_score = -1
    pdf_links, internal_pages = extract_links_from_html(
        page_url,
        response.text,
        document_type,
        allowed_domains=allowed_domains,
    )

    for candidate_url, candidate_text in pdf_links:
        score = score_candidate(
            profile,
            document_type,
            candidate_text,
            candidate_text,
            candidate_url,
            source_kind,
            query,
        )
        if score > best_score:
            best_score = score
            best_candidate = build_candidate(
                profile,
                document_type,
                candidate_text,
                candidate_url,
                query,
                source_kind,
                score,
            )

    if best_candidate:
        return best_candidate

    for subpage in internal_pages[:3]:
        try:
            subpage_response = session.get(subpage, timeout=25, allow_redirects=True)
            subpage_response.raise_for_status()
        except requests.RequestException:
            continue

        subpage_pdfs, _ = extract_links_from_html(
            subpage,
            subpage_response.text,
            document_type,
            allowed_domains=allowed_domains,
        )
        for candidate_url, candidate_text in subpage_pdfs:
            score = score_candidate(
                profile,
                document_type,
                candidate_text,
                candidate_text,
                candidate_url,
                source_kind,
                query,
            )
            if score > best_score:
                best_score = score
                best_candidate = build_candidate(
                    profile,
                    document_type,
                    candidate_text,
                    candidate_url,
                    query,
                    source_kind,
                    score,
                )

    return best_candidate


def best_candidate_from_queries(
    profile: CompanyProfile,
    document_type: str,
    queries: list[str],
    source_kind: str,
    allowed_domains: Optional[list[str]] = None,
    session: Optional[requests.Session] = None,
) -> Optional[DiscoveredDocument]:
    session = session or build_session()
    best_pdf_candidate: Optional[DiscoveredDocument] = None
    best_pdf_score = -1
    page_candidates: list[tuple[int, str, str]] = []

    for query, result in search_results_for_queries(queries):
        url = (result.get("href") or result.get("url") or "").strip()
        title = (result.get("title") or "").strip()
        body = (result.get("body") or "").strip()
        if not url or not domain_matches(url, allowed_domains):
            continue

        score = score_candidate(profile, document_type, title, body, url, source_kind, query)
        if looks_like_pdf(url):
            if score > best_pdf_score:
                best_pdf_score = score
                best_pdf_candidate = build_candidate(
                    profile,
                    document_type,
                    title,
                    url,
                    query,
                    source_kind,
                    score,
                )
        else:
            page_candidates.append((score, url, query))

        if best_pdf_candidate and best_pdf_score >= 100:
            return best_pdf_candidate

    page_candidates.sort(key=lambda item: item[0], reverse=True)
    for score, page_url, query in page_candidates[:6]:
        if score < 30:
            continue
        page_candidate = inspect_page_for_document(
            profile,
            document_type,
            page_url,
            query,
            source_kind,
            session,
            allowed_domains=allowed_domains,
        )
        if page_candidate:
            if not best_pdf_candidate or page_candidate.confidence > best_pdf_candidate.confidence:
                best_pdf_candidate = page_candidate
                best_pdf_score = int(page_candidate.confidence * 140)

    return best_pdf_candidate
