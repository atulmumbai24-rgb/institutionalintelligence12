"""
fetcher.py - Company Document Fetching Module

Handles:
1. Company name extraction from user queries
2. Dynamic PDF discovery via web search
3. PDF downloading and local storage
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional
from urllib.parse import urlparse

import requests
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known Companies / Aliases (used only for extraction, not hardcoded PDF URLs)
# ---------------------------------------------------------------------------

KNOWN_COMPANIES = [
    "infosys",
    "reliance",
    "reliance industries",
    "tcs",
    "tata consultancy services",
    "hdfc",
    "hdfc bank",
    "wipro",
    "bharti airtel",
    "airtel",
    "icici",
    "icici bank",
]

COMPANY_ALIASES = {
    "infy": "infosys",
    "ril": "reliance",
    "tata": "tcs",
    "hdfc bank ltd": "hdfc bank",
    "wipro limited": "wipro",
    "bharti": "bharti airtel",
    "icici bank limited": "icici bank",
}

OFFICIAL_DOMAIN_HINTS = [
    "nseindia.com",
    "bseindia.com",
    "annualreports.com",
]

# ---------------------------------------------------------------------------
# Company Name Extraction
# ---------------------------------------------------------------------------

def extract_company_name(query: str) -> Optional[str]:
    """
    Extract company name from user query using heuristic matching.
    """
    query_lower = query.lower()

    company_names = sorted(KNOWN_COMPANIES, key=len, reverse=True)
    for company in company_names:
        if company in query_lower:
            logger.info("Extracted company: %s", company)
            return company

    for alias, company in COMPANY_ALIASES.items():
        if alias in query_lower:
            logger.info("Extracted company via alias '%s': %s", alias, company)
            return company

    words = query.split()
    for i, word in enumerate(words):
        if word and word[0].isupper() and len(word) > 2:
            potential = word.lower()
            if i + 1 < len(words) and words[i + 1] and words[i + 1][0].isupper():
                potential = f"{word} {words[i + 1]}".lower()

            if potential in KNOWN_COMPANIES or potential in COMPANY_ALIASES:
                return COMPANY_ALIASES.get(potential, potential)

    logger.warning("Could not extract company name from query: %s", query)
    return None

# ---------------------------------------------------------------------------
# Search Helpers
# ---------------------------------------------------------------------------

def _normalize_company_name(company_name: str) -> str:
    return COMPANY_ALIASES.get(company_name.lower().strip(), company_name.lower().strip())

def _looks_like_pdf(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    return path.endswith(".pdf") or ".pdf" in path

def _domain_score(url: str, company_name: str) -> int:
    domain = urlparse(url).netloc.lower()
    company_slug = re.sub(r"[^a-z0-9]", "", company_name.lower())

    score = 0
    if any(hint in domain for hint in OFFICIAL_DOMAIN_HINTS):
        score += 50
    if "nseindia.com" in domain:
        score += 30
    if "bseindia.com" in domain:
        score += 25
    if company_slug and company_slug in re.sub(r"[^a-z0-9]", "", domain):
        score += 35
    if any(token in domain for token in ["investor", "investors", "ir.", "annualreport"]):
        score += 15

    return score

def _rank_search_results(company_name: str, results: list[dict]) -> list[dict]:
    ranked = []
    for item in results:
        url = item.get("href") or item.get("url") or ""
        title = (item.get("title") or "").lower()
        body = (item.get("body") or "").lower()

        if not url or not _looks_like_pdf(url):
            continue

        score = _domain_score(url, company_name)

        text_blob = f"{title} {body} {url.lower()}"
        if "integrated annual report" in text_blob:
            score += 20
        if "annual report" in text_blob:
            score += 15
        if "2024-25" in text_blob or "2024 25" in text_blob:
            score += 20
        if "2025" in text_blob:
            score += 5

        ranked.append(
            {
                "url": url,
                "title": item.get("title", ""),
                "score": score,
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked

def search_for_pdf(company_name: str) -> Optional[str]:
    """
    Search the web for a company's latest integrated annual report PDF.

    Primary query:
        "{company_name} Integrated Annual Report 2024-25 filetype:pdf"

    Fallback query:
        "{company_name} NSE archive annual report filetype:pdf"
    """
    normalized_name = _normalize_company_name(company_name)

    queries = [
        f"{normalized_name} Integrated Annual Report 2024-25 filetype:pdf",
        f"{normalized_name} annual report 2024-25 investor relations filetype:pdf",
        f"{normalized_name} NSE archive annual report filetype:pdf",
    ]

    try:
        with DDGS() as ddgs:
            for query in queries:
                logger.info("Searching for PDF with query: %s", query)
                raw_results = list(ddgs.text(query, max_results=10))
                ranked_results = _rank_search_results(normalized_name, raw_results)

                if ranked_results:
                    best_match = ranked_results[0]["url"]
                    logger.info("Best PDF match for %s: %s", normalized_name, best_match)
                    return best_match

    except Exception as exc:
        logger.error("Dynamic PDF search failed for %s: %s", normalized_name, exc)

    logger.warning("No PDF found dynamically for company: %s", normalized_name)
    return None

# ---------------------------------------------------------------------------
# PDF Downloading
# ---------------------------------------------------------------------------

def download_pdf(url: str, save_dir: str = "/tmp/company_pdfs") -> str:
    """
    Download a PDF from URL and save it locally.
    """
    os.makedirs(save_dir, exist_ok=True)

    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename.lower().endswith(".pdf"):
        filename = f"document_{abs(hash(url))}.pdf"

    filepath = os.path.join(save_dir, filename)

    if os.path.exists(filepath):
        logger.info("PDF already exists: %s", filepath)
        return filepath

    try:
        logger.info("Downloading PDF from: %s", url)
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

        response = requests.get(
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

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.info("PDF downloaded successfully: %s", filepath)
        return filepath

    except requests.exceptions.RequestException as exc:
        logger.error("Failed to download PDF from %s: %s", url, exc)
        raise Exception(f"PDF download failed: {exc}")

# ---------------------------------------------------------------------------
# High-Level API
# ---------------------------------------------------------------------------

def get_company_pdf(company_name: str) -> Optional[str]:
    """
    Resolve a company document URL dynamically.
    """
    company_name = _normalize_company_name(company_name)
    return search_for_pdf(company_name)

def fetch_company_document(query: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Complete pipeline: extract company -> search PDF URL -> download.
    """
    company = extract_company_name(query)
    if not company:
        return None, None, None

    pdf_url = get_company_pdf(company)
    if not pdf_url:
        logger.warning("No PDF URL found for company: %s", company)
        return company, None, None

    try:
        filepath = download_pdf(pdf_url)
        return company, pdf_url, filepath
    except Exception as exc:
        logger.error("Failed to fetch document for %s: %s", company, exc)
        return company, pdf_url, None

def list_available_companies() -> list[str]:
    """
    Keep sidebar compatibility with app.py.
    """
    return sorted(set(KNOWN_COMPANIES))