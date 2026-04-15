"""
Deployable Streamlit website for grounded Indian-equity company intelligence.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from company_registry import get_company_profile, normalize_company_name
from engine import query_company
from fetcher import extract_company_name, fetch_company_documents, list_available_companies
from ingestion import FAISSStore, ingest_document
from schemas import Document, DocumentMetadata, DiscoveredDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Indian Equity Intelligence",
    layout="wide",
)


if "knowledge_bases" not in st.session_state:
    st.session_state.knowledge_bases: dict[str, dict] = {}

if "query_history" not in st.session_state:
    st.session_state.query_history: list[dict] = []


def _build_company_knowledge(
    company_name: str,
    downloaded_sources: list[DiscoveredDocument],
) -> tuple[list[Document], FAISSStore]:
    store = FAISSStore()
    documents: list[Document] = []
    profile = get_company_profile(company_name)

    for source in downloaded_sources:
        if not source.local_path:
            continue

        metadata = DocumentMetadata(
            company_name=profile.display_name,
            document_type=source.document_type,
            financial_year=source.financial_year,
            source_url=source.source_url,
            source_domain=source.source_domain,
            source_title=source.title,
            search_query=source.search_query,
            period_label=source.period_label,
        )
        document = ingest_document(source.local_path, metadata, store)
        documents.append(document)

    return documents, store


def _source_dataframe(sources: list[DiscoveredDocument]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Source Kind": source.source_kind,
                "Document Type": source.document_type,
                "FY / Period": source.financial_year or source.period_label or "Latest",
                "Domain": source.source_domain,
                "Confidence": f"{source.confidence:.0%}",
                "Source URL": source.source_url,
            }
            for source in sources
        ]
    )


with st.sidebar:
    st.header("Grounded Research")
    st.caption(
        "This website fetches live filings and company documents, then answers only from retrieved evidence."
    )
    st.info(
        "Source priority is NSE, then BSE, then official investor-relations pages, and finally web fallback. "
        "If evidence is weak or incomplete, the app says so instead of pretending certainty."
    )
    st.subheader("Sample companies")
    for company in list_available_companies()[:12]:
        st.write(f"- {company}")

    st.divider()
    st.metric("Companies cached", len(st.session_state.knowledge_bases))
    st.metric("Questions answered", len(st.session_state.query_history))

    if st.button("Clear cache", use_container_width=True):
        st.session_state.knowledge_bases = {}
        st.session_state.query_history = []
        st.rerun()


st.title("Indian Equity Intelligence")
st.caption(
    "Ask for financial metrics or company details. The app fetches official exchange and investor-relations "
    "documents first, ingests them, and answers with citations."
)

st.divider()
st.header("Ask a company question")

company_input = st.text_input(
    "Company name",
    placeholder="e.g. Tata Consultancy Services, TCS, Infosys, Reliance Industries",
    key="company_input",
)
question_input = st.text_input(
    "Question",
    placeholder="e.g. revenue for the past 2 years",
    key="question_input",
)

col1, col2 = st.columns([1, 1])
with col1:
    refresh_sources = st.checkbox("Refresh live sources", value=False)
with col2:
    max_sources = st.selectbox("Sources to ingest", options=[2, 3, 4], index=1)

example_queries = [
    ("TCS", "revenue for the past 2 years"),
    ("Infosys", "profit after tax for the last 2 years"),
    ("HDFC Bank", "what does the latest annual report say about risks"),
    ("Reliance Industries", "revenue for the past 2 years"),
]

with st.expander("Example questions"):
    for index, (example_company, example_question) in enumerate(example_queries, start=1):
        if st.button(f"{index}. {example_company} - {example_question}", key=f"example_{index}"):
            st.session_state["question_input"] = example_question
            st.session_state["company_input"] = example_company
            st.rerun()

if st.button("Get grounded answer", type="primary", use_container_width=True):
    if not question_input.strip():
        st.error("Enter a question first.")
        st.stop()

    detected_company = company_input.strip() or extract_company_name(question_input)
    if not detected_company:
        st.error("Please enter the company name explicitly so the app can fetch the right live sources.")
        st.stop()

    canonical_company = normalize_company_name(detected_company)
    profile = get_company_profile(canonical_company)
    cache_key = canonical_company

    if refresh_sources or cache_key not in st.session_state.knowledge_bases:
        with st.spinner(f"Finding live sources for {profile.display_name}..."):
            resolved_company, downloaded_sources = fetch_company_documents(
                question_input,
                company_name=detected_company,
                max_documents=max_sources,
            )

        if not downloaded_sources:
            st.error(
                f"I couldn't fetch usable live documents for {profile.display_name}. "
                "That usually means the source pages blocked scraping or did not expose clean PDF links."
            )
            st.stop()

        with st.spinner("Ingesting retrieved documents..."):
            documents, store = _build_company_knowledge(resolved_company or canonical_company, downloaded_sources)

        if not documents:
            st.error("Documents were downloaded, but no readable text could be extracted from them.")
            st.stop()

        st.session_state.knowledge_bases[cache_key] = {
            "company_name": resolved_company or canonical_company,
            "profile_name": profile.display_name,
            "documents": documents,
            "store": store,
            "sources": downloaded_sources,
        }

    knowledge = st.session_state.knowledge_bases[cache_key]

    st.success(f"Loaded {len(knowledge['documents'])} live documents for {knowledge['profile_name']}.")

    sources_tab, answer_tab = st.tabs(["Sources", "Answer"])
    with sources_tab:
        st.dataframe(_source_dataframe(knowledge["sources"]), use_container_width=True, hide_index=True)

    with answer_tab:
        with st.spinner("Answering from grounded evidence..."):
            result = query_company(question_input, knowledge["documents"], knowledge["store"])

        confidence = float(result.get("confidence", 0.0) or 0.0)
        confidence_label = "High" if confidence >= 0.7 else "Medium" if confidence >= 0.4 else "Low"
        st.subheader("Answer")
        st.write(result["answer"])
        st.caption(f"Groundedness: {confidence_label} ({confidence:.0%})")

        metric_rows = result.get("metric_rows") or []
        if metric_rows:
            st.subheader("Structured metric view")
            st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

        citations = result.get("citations") or []
        if citations:
            st.subheader("Citations")
            for index, citation in enumerate(citations, start=1):
                title = (
                    f"Citation {index} - {citation.get('document_type', 'Document')} - "
                    f"Page {citation['page']} - Score {citation['score']:.0%}"
                )
                with st.expander(title):
                    st.write(citation["text"])
                    if citation.get("source_url"):
                        st.caption(citation["source_url"])

    st.session_state.query_history.append(
        {
            "company": knowledge["profile_name"],
            "question": question_input,
            "answer": result["answer"],
            "confidence": confidence,
        }
    )


if st.session_state.query_history:
    st.divider()
    st.header("Recent questions")
    for item in reversed(st.session_state.query_history[-5:]):
        with st.expander(f"{item['company']} - {item['question']}"):
            st.write(item["answer"])
            st.caption(f"Groundedness: {item['confidence']:.0%}")
