"""
Dynamic document discovery and download helpers for company reports.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from typing import Optional
from urllib.parse import urlparse

import requests
from duckduckgo_search import DDGS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


COMPANY_ALIASES = {
    "infy": "infosys",
    "ril": "reliance industries",
    "reliance": "reliance industries",
    "tata": "tata consultancy services",
    "tcs": "tata consultancy services",
    "hdfc": "hdfc bank",
    "bharti": "bharti airtel",
    "airtel": "bharti airtel",
    "icici": "icici bank",
}

WATCHLIST_COMPANIES = [
    "infosys",
    "reliance industries",
    "tata consultancy services",
    "hdfc bank",
    "wipro",
    "bharti airtel",
    "icici bank",
    "sbi",
    "ltimindtree",
    "sun pharma",
]

OFFICIAL_DOMAIN_HINTS = (
    "nseindia.com",
    "nsearchives.nseindia.com",
    "bseindia.com",
)

QUESTION_STOPWORDS = {
    "what",
    "why",
    "when",
    "where",
    "which",
    "who",
    "is",
    "are",
    "does",
    "do",
    "did",
    "how",
    "tell",
    "show",
    "give",
    "compare",
    "analyze",
}


def _normalize_company_name(company_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9&.\- ]+", " ", company_name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return COMPANY_ALIASES.get(cleaned, cleaned)


def _looks_like_pdf(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.path.lower().endswith(".pdf")


def _candidate_company_names(query: str) -> list[str]:
    candidates: list[str] = []
    query_lower = query.lower()

    for alias, company in COMPANY_ALIASES.items():
        if alias in query_lower:
            candidates.append(company)

    for company in WATCHLIST_COMPANIES:
        if company in query_lower:
            candidates.append(company)

    possessive_match = re.search(
        r"\b([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,4})'s\b",
        query,
    )
    if possessive_match:
        candidates.append(possessive_match.group(1))

    keyword_match = re.search(
        r"(?:about|for|of|on)\s+([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,4})",
        query,
    )
    if keyword_match:
        candidates.append(keyword_match.group(1))

    capitalized_spans = re.finditer(
        r"\b([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,4})\b",
        query,
    )
    for match in capitalized_spans:
        value = match.group(1).strip()
        normalized = _normalize_company_name(value)
        if normalized in QUESTION_STOPWORDS:
            continue
        if value.lower() in QUESTION_STOPWORDS:
            continue
        candidates.append(value)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_company_name(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)

    return deduped


def extract_company_name(query: str) -> Optional[str]:
    candidates = _candidate_company_names(query)
    if not candidates:
        logger.warning("Could not extract company name from query: %s", query)
        return None

    candidates.sort(key=len, reverse=True)
    company = candidates[0]
    logger.info("Extracted company: %s", company)
    return company


def _build_search_queries(company_name: str) -> list[str]:
    quoted_name = f"\"{company_name}\""
    return [
        f"{quoted_name} Integrated Annual Report 2024-25 filetype:pdf",
        f"{quoted_name} Annual Report 2024-25 investor relations filetype:pdf",
        f"site:nsearchives.nseindia.com {quoted_name} annual report filetype:pdf",
        f"site:nseindia.com {quoted_name} annual report filetype:pdf",
        f"site:bseindia.com {quoted_name} annual report filetype:pdf",
    ]


def _score_pdf_result(company_name: str, result: dict, query: str) -> int:
    url = (result.get("href") or result.get("url") or "").strip()
    title = (result.get("title") or "").lower()
    body = (result.get("body") or "").lower()

    if not url or not _looks_like_pdf(url):
        return -1

    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    company_tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", company_name.lower())
        if token and token not in {"ltd", "limited", "bank"}
    ]

    score = 0
    if any(hint in domain for hint in OFFICIAL_DOMAIN_HINTS):
        score += 60
    if "investor" in url.lower() or "annual-report" in url.lower():
        score += 20
    if "integrated annual report" in f"{title} {body}":
        score += 20
    if "annual report" in f"{title} {body}":
        score += 15
    if "2024-25" in f"{title} {body} {path}":
        score += 20
    if "fy25" in f"{title} {body} {path}":
        score += 12
    if "nse" in query.lower():
        score += 10

    for token in company_tokens:
        if token in domain:
            score += 12
        if token in title or token in body or token in path:
            score += 6

    return score


def search_for_pdf(company_name: str) -> Optional[str]:
    normalized_name = _normalize_company_name(company_name)
    queries = _build_search_queries(normalized_name)
    best_url: Optional[str] = None
    best_score = -1

    try:
        with DDGS() as ddgs:
            for query in queries:
                logger.info("Searching for report PDF with query: %s", query)
                results = list(ddgs.text(query, max_results=10))

                for result in results:
                    url = (result.get("href") or result.get("url") or "").strip()
                    score = _score_pdf_result(normalized_name, result, query)
                    if score > best_score:
                        best_url = url
                        best_score = score

                if best_url and best_score >= 80:
                    logger.info("Selected PDF URL for %s: %s", normalized_name, best_url)
                    return best_url

    except Exception as exc:
        logger.error("Dynamic PDF search failed for %s: %s", normalized_name, exc)

    if best_url:
        logger.info("Falling back to best available PDF URL for %s: %s", normalized_name, best_url)
        return best_url

    logger.warning("No PDF URL found for company: %s", normalized_name)
    return None


def _default_save_dir() -> str:
    return os.path.join(tempfile.gettempdir(), "company_intelligence", "pdfs")


def _build_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(403, 429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_pdf(url: str, save_dir: Optional[str] = None) -> str:
    if save_dir is None:
        save_dir = _default_save_dir()

    os.makedirs(save_dir, exist_ok=True)

    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename.lower().endswith(".pdf"):
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
        filename = f"report_{digest}.pdf"

    filepath = os.path.join(save_dir, filename)
    if os.path.exists(filepath):
        logger.info("PDF already exists: %s", filepath)
        return filepath

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
    }

    try:
        logger.info("Downloading PDF from: %s", url)
        session = _build_session()
        response = session.get(
            url,
            headers=headers,
            timeout=45,
            stream=True,
            allow_redirects=True,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type and not _looks_like_pdf(response.url):
            raise Exception(f"URL did not return a PDF. Content-Type: {content_type}")

        with open(filepath, "wb") as file_handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_handle.write(chunk)

        logger.info("PDF downloaded successfully: %s", filepath)
        return filepath

    except requests.exceptions.RequestException as exc:
        logger.error("Failed to download PDF from %s: %s", url, exc)
        raise Exception(f"PDF download failed: {exc}") from exc


def get_company_pdf(company_name: str) -> Optional[str]:
    normalized_name = _normalize_company_name(company_name)
    return search_for_pdf(normalized_name)


def fetch_company_document(
    query: str,
    company_name: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    company = _normalize_company_name(company_name) if company_name else extract_company_name(query)
    if not company:
        return None, None, None

    pdf_url = get_company_pdf(company)
    if not pdf_url:
        return company, None, None

    try:
        filepath = download_pdf(pdf_url)
        return company, pdf_url, filepath
    except Exception as exc:
        logger.error("Failed to fetch document for %s: %s", company, exc)
        return company, pdf_url, None


def list_available_companies() -> list[str]:
    return sorted(WATCHLIST_COMPANIES)
