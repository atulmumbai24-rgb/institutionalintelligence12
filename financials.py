"""
Rule-based finance question routing and metric extraction.
"""

from __future__ import annotations

import re
from typing import Optional

from ingestion import FAISSStore, get_embeddings
from schemas import Document, MetricPoint, SourceCitation


METRIC_DEFINITIONS = {
    "revenue": {
        "display_name": "Revenue",
        "question_terms": ["revenue", "sales", "income from operations", "topline"],
        "page_terms": ["revenue from operations", "revenue", "income from operations", "total income"],
    },
    "pat": {
        "display_name": "Profit After Tax",
        "question_terms": ["profit after tax", "pat", "net profit", "profit"],
        "page_terms": ["profit after tax", "net profit", "profit for the year", "profit attributable"],
    },
    "ebitda": {
        "display_name": "EBITDA",
        "question_terms": ["ebitda", "operating profit"],
        "page_terms": ["ebitda", "operating profit"],
    },
    "debt": {
        "display_name": "Debt",
        "question_terms": ["debt", "borrowings", "net debt"],
        "page_terms": ["borrowings", "debt", "net debt"],
    },
    "cash": {
        "display_name": "Cash",
        "question_terms": ["cash", "cash balance", "cash and cash equivalents"],
        "page_terms": ["cash and cash equivalents", "cash balance", "cash"],
    },
}

PERIOD_PATTERN = re.compile(r"(?:FY\s?\d{2}|20\d{2}-\d{2}|20\d{2})", re.IGNORECASE)
AMOUNT_PATTERN = re.compile(
    r"(?:Rs\.?|INR)?\s*\(?(?:(?:\d{1,3}(?:,\d{2,3})+)|(?:\d{4,}))(?:\.\d+)?\)?",
    re.IGNORECASE,
)


def detect_financial_intent(question: str) -> dict:
    question_lower = question.lower()
    metric_name: Optional[str] = None

    for candidate_metric, config in METRIC_DEFINITIONS.items():
        if any(term in question_lower for term in config["question_terms"]):
            metric_name = candidate_metric
            break

    if not metric_name:
        return {"kind": "general", "metric_name": None, "display_name": None, "lookback_years": 0}

    lookback_years = 1
    explicit_match = re.search(r"(?:past|last|previous)\s+(\d+)\s+years?", question_lower)
    if explicit_match:
        lookback_years = max(1, min(5, int(explicit_match.group(1))))
    elif "years" in question_lower or "trend" in question_lower or "compare" in question_lower:
        lookback_years = 2

    return {
        "kind": "metric",
        "metric_name": metric_name,
        "display_name": METRIC_DEFINITIONS[metric_name]["display_name"],
        "lookback_years": lookback_years,
    }


def _normalize_period_label(value: str) -> str:
    token = value.upper().replace(" ", "")
    if token.startswith("FY"):
        return token
    if re.fullmatch(r"20\d{2}-\d{2}", token):
        return f"FY{token[-2:]}"
    return token


def _metadata_period_labels(financial_year: Optional[str]) -> list[str]:
    if not financial_year:
        return []

    year_match = re.search(r"(20\d{2})-(\d{2})", financial_year)
    if year_match:
        end_year_two_digits = year_match.group(2)
        end_year_full = int(year_match.group(1)[:2] + end_year_two_digits)
        latest = f"FY{end_year_two_digits}"
        previous = f"FY{(end_year_full - 1) % 100:02d}"
        return [latest, previous, f"FY{(end_year_full - 2) % 100:02d}"]

    fy_match = re.search(r"FY\s?(\d{2})", financial_year, re.IGNORECASE)
    if fy_match:
        latest_two_digits = int(fy_match.group(1))
        return [f"FY{latest_two_digits:02d}", f"FY{(latest_two_digits - 1) % 100:02d}"]

    return []


def _to_float(value_text: str) -> Optional[float]:
    cleaned = value_text.lower()
    negative = "(" in cleaned and ")" in cleaned
    cleaned = cleaned.replace("rs.", "")
    cleaned = cleaned.replace("rs", "")
    cleaned = cleaned.replace("inr", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("(", "")
    cleaned = cleaned.replace(")", "")
    cleaned = cleaned.strip()

    try:
        value = float(cleaned)
        return -value if negative else value
    except ValueError:
        return None


def _detect_unit(text: str) -> str:
    text_lower = text.lower()
    if "crore" in text_lower or "crores" in text_lower:
        return "crore"
    if "million" in text_lower:
        return "million"
    if "billion" in text_lower:
        return "billion"
    if "lakhs" in text_lower or "lakh" in text_lower:
        return "lakh"
    return "reported units"


def _extract_amounts(text: str) -> list[str]:
    values = []
    for match in AMOUNT_PATTERN.findall(text):
        stripped = match.strip()
        if stripped not in values:
            values.append(stripped)
    return values


def _filter_amounts(amounts: list[str], metric_name: str) -> list[str]:
    filtered: list[str] = []
    for amount in amounts:
        numeric_value = _to_float(amount)
        if numeric_value is None:
            continue
        if 1900 <= abs(numeric_value) <= 2099:
            continue
        if metric_name != "cash" and abs(numeric_value) < 1000 and "," not in amount:
            continue
        filtered.append(amount)
    return filtered


def _sort_metric_points(points: list[MetricPoint]) -> list[MetricPoint]:
    def sort_key(point: MetricPoint) -> tuple[int, float]:
        fy_match = re.search(r"FY(\d{2})", point.period_label)
        if fy_match:
            return (2000 + int(fy_match.group(1)), point.confidence)
        year_match = re.search(r"(20\d{2})", point.period_label)
        if year_match:
            return (int(year_match.group(1)), point.confidence)
        return (0, point.confidence)

    return sorted(points, key=sort_key, reverse=True)


def _extract_points_from_page(
    metric_name: str,
    page_text: str,
    page_number: int,
    document: Document,
) -> list[MetricPoint]:
    config = METRIC_DEFINITIONS[metric_name]
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    points: list[MetricPoint] = []

    for index, line in enumerate(lines):
        if not any(term in line.lower() for term in config["page_terms"]):
            continue

        window_lines = lines[max(0, index - 1): min(len(lines), index + 4)]
        window_text = " ".join(window_lines)
        amounts = _filter_amounts(_extract_amounts(window_text), metric_name)
        if not amounts:
            continue

        periods = [_normalize_period_label(match) for match in PERIOD_PATTERN.findall(window_text)]
        deduped_periods: list[str] = []
        for period in periods:
            if period not in deduped_periods:
                deduped_periods.append(period)

        if len(deduped_periods) < len(amounts):
            for fallback_period in _metadata_period_labels(document.metadata.financial_year):
                if fallback_period not in deduped_periods:
                    deduped_periods.append(fallback_period)

        if not deduped_periods:
            deduped_periods = _metadata_period_labels(document.metadata.financial_year)

        if not deduped_periods:
            deduped_periods = [document.metadata.period_label or "Reported"]

        pair_count = min(len(amounts), len(deduped_periods))
        if pair_count == 1 and len(amounts) >= 2:
            metadata_periods = _metadata_period_labels(document.metadata.financial_year)
            if len(metadata_periods) >= 2:
                deduped_periods = metadata_periods
                pair_count = min(len(amounts), len(deduped_periods))

        confidence = 0.55
        if document.metadata.document_type in {"Annual Report", "Results Release", "Investor Presentation"}:
            confidence += 0.15
        if len(PERIOD_PATTERN.findall(window_text)) >= 1:
            confidence += 0.1
        if _detect_unit(window_text) != "reported units":
            confidence += 0.05
        if "revenue from operations" in window_text.lower() or "profit after tax" in window_text.lower():
            confidence += 0.1

        for period_label, amount_text in zip(deduped_periods[:pair_count], amounts[:pair_count]):
            points.append(
                MetricPoint(
                    metric_name=config["display_name"],
                    period_label=period_label,
                    value_text=amount_text,
                    value_numeric=_to_float(amount_text),
                    unit=_detect_unit(window_text),
                    page_number=page_number,
                    document_name=document.filename,
                    document_type=document.metadata.document_type,
                    source_url=document.metadata.source_url,
                    confidence=min(1.0, confidence),
                )
            )

    return points


def extract_metric_points(
    metric_name: str,
    documents: list[Document],
    store: FAISSStore,
) -> list[MetricPoint]:
    config = METRIC_DEFINITIONS[metric_name]
    search_query = " ".join([config["display_name"], *config["page_terms"][:2], "financial statements"])
    query_embedding = get_embeddings([search_query])[0]
    search_results = store.search(query_embedding, top_k=max(16, len(documents) * 6))
    doc_map = {document.doc_id: document for document in documents}

    candidate_pages: list[tuple[Document, int]] = []
    seen_pages: set[tuple[str, int]] = set()

    for chunk, _score in search_results:
        if not any(term in chunk.text.lower() for term in config["page_terms"]):
            continue
        page_key = (chunk.doc_id, chunk.page_number)
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        candidate_pages.append((doc_map[chunk.doc_id], chunk.page_number))

    if not candidate_pages:
        for document in documents:
            for page in store.get_document_pages(document.doc_id):
                if any(term in page["text"].lower() for term in config["page_terms"]):
                    candidate_pages.append((document, page["page_number"]))

    all_points: list[MetricPoint] = []
    for document, page_number in candidate_pages:
        pages = store.get_document_pages(document.doc_id)
        matching_pages = [page for page in pages if page["page_number"] == page_number]
        if not matching_pages:
            continue
        all_points.extend(
            _extract_points_from_page(metric_name, matching_pages[0]["text"], page_number, document)
        )

    best_points_by_period: dict[str, MetricPoint] = {}
    for point in all_points:
        existing = best_points_by_period.get(point.period_label)
        if existing is None or point.confidence > existing.confidence:
            best_points_by_period[point.period_label] = point

    return _sort_metric_points(list(best_points_by_period.values()))


def answer_metric_question(
    question: str,
    documents: list[Document],
    store: FAISSStore,
    intent: dict,
) -> dict:
    metric_name = intent["metric_name"]
    metric_points = extract_metric_points(metric_name, documents, store)
    lookback_years = intent["lookback_years"]
    selected_points = metric_points[:lookback_years]

    citations = [
        SourceCitation(
            document_name=point.document_name,
            document_type=point.document_type,
            page_number=point.page_number,
            text_snippet=f"{point.metric_name}: {point.value_text} ({point.period_label})",
            relevance_score=point.confidence,
            source_url=point.source_url,
        )
        for point in selected_points[:3]
    ]

    metric_rows = [
        {
            "Period": point.period_label,
            "Metric": point.metric_name,
            "Value": point.value_text,
            "Unit": point.unit or "",
            "Document Type": point.document_type,
        }
        for point in selected_points
    ]

    if not selected_points:
        return {
            "answer": (
                f"I couldn't verify {intent['display_name'].lower()} definitively from the retrieved filings. "
                "The app found sources, but not a reliable enough metric extraction for this question."
            ),
            "citations": [],
            "status": "completed",
            "error": None,
            "metric_rows": [],
            "confidence": 0.0,
            "intent": intent,
        }

    company_name = documents[0].metadata.company_name if documents else "The company"
    if len(selected_points) < lookback_years:
        answer = (
            f"I found {intent['display_name'].lower()} evidence for {company_name}, but not enough to confirm "
            f"the last {lookback_years} years definitively. "
        )
    else:
        answer = f"Based on the retrieved filings, {company_name} reported "

    joined_values = ", ".join(
        f"{point.value_text} in {point.period_label}" for point in selected_points
    )
    answer += f"{intent['display_name'].lower()} of {joined_values}."

    if len(selected_points) >= 2:
        latest = selected_points[0]
        previous = selected_points[1]
        if (
            latest.value_numeric is not None
            and previous.value_numeric is not None
            and previous.value_numeric != 0
            and latest.unit == previous.unit
        ):
            growth = (latest.value_numeric - previous.value_numeric) / abs(previous.value_numeric)
            answer += f" That implies a year-over-year change of {growth:.1%}."

    average_confidence = sum(point.confidence for point in selected_points) / len(selected_points)
    return {
        "answer": answer,
        "citations": citations,
        "status": "completed",
        "error": None,
        "metric_rows": metric_rows,
        "confidence": average_confidence,
        "intent": intent,
    }
