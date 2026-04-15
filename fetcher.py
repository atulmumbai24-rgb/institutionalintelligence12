"""
Dynamic document discovery and download helpers for company reports.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import os
import re
import tempfile
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from duckduckgo_search import DDGS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


COMPANY_ALIASES = {
    "infy": "infosys",
    "infosys ltd": "infosys",
    "infosys limited": "infosys",
    "ril": "reliance industries",
    "reliance": "reliance industries",
    "reliance industries ltd": "reliance industries",
    "reliance industries limited": "reliance industries",
    "tata": "tata consultancy services",
    "tcs": "tata consultancy services",
    "tata consultancy service": "tata consultancy services",
    "tata consultancy services ltd": "tata consultancy services",
    "tata consultancy services limited": "tata consultancy services",
    "hdfc": "hdfc bank",
    "hdfc bank ltd": "hdfc bank",
    "hdfc bank limited": "hdfc bank",
    "bharti": "bharti airtel",
    "airtel": "bharti airtel",
    "bharti airtel ltd": "bharti airtel",
    "icici": "icici bank",
    "icici bank ltd": "icici bank",
    "state bank of india": "sbi",
    "sbi bank": "sbi",
    "sun pharmaceutical": "sun pharma",
    "sun pharmaceutical industries": "sun pharma",
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

COMPANY_SEARCH_HINTS = {
    "infosys": {
        "aliases": ["Infosys", "Infosys Ltd"],
        "official_domains": ["infosys.com"],
    },
    "reliance industries": {
        "aliases": ["Reliance Industries", "RIL"],
        "official_domains": ["ril.com"],
    },
    "tata consultancy services": {
        "aliases": ["Tata Consultancy Services", "TCS"],
        "official_domains": ["tcs.com"],
    },
    "hdfc bank": {
        "aliases": ["HDFC Bank"],
        "official_domains": ["hdfcbank.com"],
    },
    "wipro": {
        "aliases": ["Wipro", "Wipro Ltd"],
        "official_domains": ["wipro.com"],
    },
    "bharti airtel": {
        "aliases": ["Bharti Airtel", "Airtel"],
        "official_domains": ["airtel.in"],
    },
    "icici bank": {
        "aliases": ["ICICI Bank"],
        "official_domains": ["icicibank.com"],
    },
    "sbi": {
        "aliases": ["State Bank of India", "SBI"],
        "official_domains": ["sbi.co.in", "bank.sbi"],
    },
    "ltimindtree": {
        "aliases": ["LTIMindtree", "LTI Mindtree"],
        "official_domains": ["ltimindtree.com"],
    },
    "sun pharma": {
        "aliases": ["Sun Pharma", "Sun Pharmaceutical Industries"],
        "official_domains": ["sunpharma.com"],
    },
}

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
    direct_match = COMPANY_ALIASES.get(cleaned)
    if direct_match:
        return direct_match

    if cleaned in WATCHLIST_COMPANIES:
        return cleaned

    if cleaned.endswith(" service"):
        plural_candidate = f"{cleaned}s"
        if plural_candidate in WATCHLIST_COMPANIES or plural_candidate in COMPANY_ALIASES:
            return COMPANY_ALIASES.get(plural_candidate, plural_candidate)

    all_candidates = sorted(set(WATCHLIST_COMPANIES) | set(COMPANY_ALIASES.keys()))
    close_matches = difflib.get_close_matches(cleaned, all_candidates, n=1, cutoff=0.84)
    if close_matches:
        matched = close_matches[0]
        return COMPANY_ALIASES.get(matched, matched)

    return cleaned


def _looks_like_pdf(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    return path.endswith(".pdf") or ".pdf" in path


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
    hints = COMPANY_SEARCH_HINTS.get(company_name, {})
    aliases = hints.get("aliases", [company_name.title()])
    queries: list[str] = []

    for alias in aliases:
        quoted = f"\"{alias}\""
        queries.extend(
            [
                f"{quoted} Integrated Annual Report 2024-25 filetype:pdf",
                f"{quoted} Annual Report 2024-25 filetype:pdf",
                f"{alias} FY25 annual report pdf",
                f"{alias} investor relations annual report pdf",
                f"site:nsearchives.nseindia.com {alias} annual report pdf",
                f"site:nseindia.com {alias} annual report pdf",
                f"site:bseindia.com {alias} annual report pdf",
            ]
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        if query not in seen:
            seen.add(query)
            deduped.append(query)
    return deduped


def _build_page_queries(company_name: str) -> list[str]:
    hints = COMPANY_SEARCH_HINTS.get(company_name, {})
    aliases = hints.get("aliases", [company_name.title()])
    queries: list[str] = []

    for alias in aliases:
        queries.extend(
            [
                f"{alias} investor relations annual report",
                f"{alias} integrated annual report 2024-25",
                f"{alias} annual report 2024-25",
                f"{alias} annual report pdf",
            ]
        )

    return list(dict.fromkeys(queries))


def _score_pdf_result(company_name: str, result: dict, query: str) -> int:
    url = (result.get("href") or result.get("url") or "").strip()
    title = (result.get("title") or "").lower()
    body = (result.get("body") or "").lower()

    if not url or not _looks_like_pdf(url):
        return -1

    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    hints = COMPANY_SEARCH_HINTS.get(company_name, {})
    official_domains = hints.get("official_domains", [])
    company_tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", company_name.lower())
        if token and token not in {"ltd", "limited", "bank"}
    ]

    score = 0
    if any(hint in domain for hint in OFFICIAL_DOMAIN_HINTS):
        score += 60
    if any(official_domain in domain for official_domain in official_domains):
        score += 45
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


def _score_page_result(company_name: str, result: dict) -> int:
    url = (result.get("href") or result.get("url") or "").strip()
    title = (result.get("title") or "").lower()
    body = (result.get("body") or "").lower()
    if not url:
        return -1

    domain = urlparse(url).netloc.lower()
    hints = COMPANY_SEARCH_HINTS.get(company_name, {})
    official_domains = hints.get("official_domains", [])

    score = 0
    if any(official_domain in domain for official_domain in official_domains):
        score += 60
    if any(hint in domain for hint in OFFICIAL_DOMAIN_HINTS):
        score += 25
    if "investor" in url.lower() or "investor" in title or "investor" in body:
        score += 20
    if "annual report" in title or "annual report" in body:
        score += 20
    if "integrated report" in title or "integrated report" in body:
        score += 20

    return score


def _extract_pdf_links_from_html(base_url: str, html: str) -> list[str]:
    matches = re.findall(r"""href=["']([^"']+?\.pdf[^"']*)["']""", html, flags=re.IGNORECASE)
    links: list[str] = []
    for match in matches:
        absolute = urljoin(base_url, match)
        if absolute not in links:
            links.append(absolute)
    return links


def _discover_pdf_from_page(company_name: str, page_url: str, session: requests.Session) -> Optional[str]:
    try:
        response = session.get(page_url, timeout=25, allow_redirects=True)
        response.raise_for_status()
        html = response.text
    except requests.RequestException as exc:
        logger.warning("Failed to inspect page %s: %s", page_url, exc)
        return None

    candidate_links = _extract_pdf_links_from_html(page_url, html)
    if not candidate_links:
        return None

    best_url: Optional[str] = None
    best_score = -1
    for candidate_url in candidate_links:
        mock_result = {"url": candidate_url, "title": page_url, "body": html[:2000]}
        score = _score_pdf_result(company_name, mock_result, page_url)
        if score > best_score:
            best_score = score
            best_url = candidate_url

    return best_url


def search_for_pdf(company_name: str) -> Optional[str]:
    normalized_name = _normalize_company_name(company_name)
    queries = _build_search_queries(normalized_name)
    session = _build_session()
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

            for query in _build_page_queries(normalized_name):
                logger.info("Searching for investor page with query: %s", query)
                page_results = list(ddgs.text(query, max_results=8))
                ranked_pages = sorted(
                    page_results,
                    key=lambda result: _score_page_result(normalized_name, result),
                    reverse=True,
                )

                for result in ranked_pages[:3]:
                    page_url = (result.get("href") or result.get("url") or "").strip()
                    if not page_url:
                        continue
                    discovered_pdf = _discover_pdf_from_page(normalized_name, page_url, session)
                    if discovered_pdf:
                        logger.info(
                            "Discovered PDF from investor page for %s: %s",
                            normalized_name,
                            discovered_pdf,
                        )
                        return discovered_pdf

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
