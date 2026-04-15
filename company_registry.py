"""
Canonical company registry and normalization helpers.
"""

from __future__ import annotations

import difflib
import re
from typing import Optional

from schemas import CompanyProfile


COMPANY_PROFILES: list[CompanyProfile] = [
    CompanyProfile(
        canonical_name="infosys",
        display_name="Infosys",
        ticker="INFY",
        nse_symbol="INFY",
        investor_relations_url="https://www.infosys.com/investors.html",
        aliases=["infosys ltd", "infosys limited", "infy"],
        official_domains=["infosys.com"],
    ),
    CompanyProfile(
        canonical_name="reliance industries",
        display_name="Reliance Industries",
        ticker="RELIANCE",
        nse_symbol="RELIANCE",
        aliases=["ril", "reliance", "reliance industries ltd", "reliance industries limited"],
        official_domains=["ril.com"],
    ),
    CompanyProfile(
        canonical_name="tata consultancy services",
        display_name="Tata Consultancy Services",
        ticker="TCS",
        nse_symbol="TCS",
        investor_relations_url="https://www.tcs.com/investor-relations",
        aliases=[
            "tata consultancy service",
            "tcs",
            "tata consultancy services ltd",
            "tata consultancy services limited",
        ],
        official_domains=["tcs.com"],
    ),
    CompanyProfile(
        canonical_name="hdfc bank",
        display_name="HDFC Bank",
        ticker="HDFCBANK",
        nse_symbol="HDFCBANK",
        investor_relations_url="https://www.hdfcbank.com/personal/about-us/investor-relations",
        aliases=["hdfc", "hdfc bank ltd", "hdfc bank limited"],
        official_domains=["hdfcbank.com"],
    ),
    CompanyProfile(
        canonical_name="wipro",
        display_name="Wipro",
        ticker="WIPRO",
        nse_symbol="WIPRO",
        aliases=["wipro ltd", "wipro limited"],
        official_domains=["wipro.com"],
    ),
    CompanyProfile(
        canonical_name="bharti airtel",
        display_name="Bharti Airtel",
        ticker="BHARTIARTL",
        nse_symbol="BHARTIARTL",
        aliases=["airtel", "bharti", "bharti airtel ltd"],
        official_domains=["airtel.in"],
    ),
    CompanyProfile(
        canonical_name="icici bank",
        display_name="ICICI Bank",
        ticker="ICICIBANK",
        nse_symbol="ICICIBANK",
        aliases=["icici", "icici bank ltd", "icici bank limited"],
        official_domains=["icicibank.com"],
    ),
    CompanyProfile(
        canonical_name="sbi",
        display_name="State Bank of India",
        ticker="SBIN",
        nse_symbol="SBIN",
        aliases=["state bank of india", "sbi bank"],
        official_domains=["sbi.co.in", "bank.sbi"],
    ),
    CompanyProfile(
        canonical_name="ltimindtree",
        display_name="LTIMindtree",
        ticker="LTIM",
        nse_symbol="LTIM",
        aliases=["lti mindtree", "larsen and toubro infotech mindtree"],
        official_domains=["ltimindtree.com"],
    ),
    CompanyProfile(
        canonical_name="sun pharma",
        display_name="Sun Pharmaceutical Industries",
        ticker="SUNPHARMA",
        nse_symbol="SUNPHARMA",
        aliases=["sun pharmaceutical", "sun pharmaceutical industries"],
        official_domains=["sunpharma.com"],
    ),
    CompanyProfile(
        canonical_name="itc",
        display_name="ITC",
        ticker="ITC",
        nse_symbol="ITC",
        aliases=["itc ltd", "itc limited"],
        official_domains=["itcportal.com"],
    ),
    CompanyProfile(
        canonical_name="maruti suzuki",
        display_name="Maruti Suzuki India",
        ticker="MARUTI",
        nse_symbol="MARUTI",
        aliases=["maruti suzuki india", "maruti"],
        official_domains=["marutisuzuki.com"],
    ),
]


_PROFILE_BY_CANONICAL = {profile.canonical_name: profile for profile in COMPANY_PROFILES}
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for profile in COMPANY_PROFILES:
    _ALIAS_TO_CANONICAL[profile.canonical_name] = profile.canonical_name
    _ALIAS_TO_CANONICAL[profile.display_name.lower()] = profile.canonical_name
    if profile.ticker:
        _ALIAS_TO_CANONICAL[profile.ticker.lower()] = profile.canonical_name
    for alias in profile.aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = profile.canonical_name


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
    "analysis",
    "financial",
    "detail",
    "details",
    "revenue",
    "profit",
}


def _clean_company_text(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9&.\- ]+", " ", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def normalize_company_name(company_name: str) -> str:
    cleaned = _clean_company_text(company_name)
    direct = _ALIAS_TO_CANONICAL.get(cleaned)
    if direct:
        return direct

    if cleaned.endswith(" service"):
        plural_candidate = f"{cleaned}s"
        direct = _ALIAS_TO_CANONICAL.get(plural_candidate)
        if direct:
            return direct

    all_candidates = sorted(set(_ALIAS_TO_CANONICAL.keys()))
    close_matches = difflib.get_close_matches(cleaned, all_candidates, n=1, cutoff=0.82)
    if close_matches:
        matched = close_matches[0]
        return _ALIAS_TO_CANONICAL.get(matched, matched)

    return cleaned


def get_company_profile(company_name: str) -> CompanyProfile:
    canonical_name = normalize_company_name(company_name)
    profile = _PROFILE_BY_CANONICAL.get(canonical_name)
    if profile:
        return profile

    display_name = " ".join(part.capitalize() for part in canonical_name.split())
    return CompanyProfile(
        canonical_name=canonical_name,
        display_name=display_name,
        aliases=[],
        official_domains=[],
    )


def list_companies() -> list[str]:
    return sorted(profile.display_name for profile in COMPANY_PROFILES)


def extract_company_name_from_query(query: str) -> Optional[str]:
    query_lower = query.lower()

    for alias, canonical in sorted(_ALIAS_TO_CANONICAL.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in query_lower:
            return canonical

    possessive_match = re.search(
        r"\b([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,4})'s\b",
        query,
    )
    if possessive_match:
        return normalize_company_name(possessive_match.group(1))

    keyword_match = re.search(
        r"(?:about|for|of|on)\s+([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,4})",
        query,
    )
    if keyword_match:
        candidate = normalize_company_name(keyword_match.group(1))
        if candidate not in QUESTION_STOPWORDS:
            return candidate

    capitalized_spans = re.finditer(
        r"\b([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,4})\b",
        query,
    )
    for match in capitalized_spans:
        candidate = normalize_company_name(match.group(1))
        if candidate not in QUESTION_STOPWORDS:
            return candidate

    return None
