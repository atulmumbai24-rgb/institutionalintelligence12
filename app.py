"""
Streamlit frontend for the V1 company intelligence baseline.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import streamlit as st

from engine import query_document
from fetcher import extract_company_name, fetch_company_document, list_available_companies
from ingestion import FAISSStore, ingest_document
from schemas import Document, DocumentMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Company Intelligence V1",
    page_icon="C",
    layout="wide",
)


if "company_stores" not in st.session_state:
    st.session_state.company_stores: dict[str, tuple[Document, FAISSStore]] = {}

if "query_history" not in st.session_state:
    st.session_state.query_history: list[dict] = []


with st.sidebar:
    st.header("V1 Focus")
    st.caption("Dynamic annual report discovery for Indian equities.")

    st.subheader("Sample companies")
    for company in list_available_companies():
        if company in st.session_state.company_stores:
            st.write(f"- {company.title()} (ingested)")
        else:
            st.write(f"- {company.title()}")

    st.divider()
    st.metric("Documents ingested", len(st.session_state.company_stores))
    st.metric("Questions answered", len(st.session_state.query_history))

    if st.button("Clear session data", use_container_width=True):
        st.session_state.company_stores = {}
        st.session_state.query_history = []
        st.rerun()


st.title("Automated Company Intelligence Platform - V1")
st.caption(
    "Ask grounded questions about Indian companies. The app finds an annual report online, "
    "ingests it, and answers with citations."
)

st.divider()
st.header("Ask a question")

company_override = st.text_input(
    "Company name (optional but recommended for broader coverage):",
    placeholder="e.g. Infosys or Tata Consultancy Services",
)

example_queries = [
    "What is Infosys revenue growth?",
    "Does HDFC Bank mention credit risk?",
    "What are the key business segments of Reliance Industries?",
    "What does TCS say about AI opportunities?",
    "What is Wipro's profit margin?",
]

col1, col2 = st.columns([3, 1])

with col1:
    user_query = st.text_input(
        "Question:",
        placeholder="e.g. What is the company's revenue from operations?",
        key="query_input",
    )

with col2:
    st.write("")
    st.write("")
    if st.button("Get answer", use_container_width=True, type="primary"):
        if user_query.strip():
            st.session_state["process_query"] = True

with st.expander("Example questions"):
    for index, example in enumerate(example_queries, start=1):
        if st.button(f"{index}. {example}", key=f"example_{index}"):
            st.session_state["query_input"] = example
            st.session_state["process_query"] = True
            st.rerun()


if st.session_state.get("process_query") and user_query.strip():
    st.session_state["process_query"] = False

    with st.container():
        st.divider()
        st.header("Processing")

        detected_company = company_override.strip() or extract_company_name(user_query)
        if not detected_company:
            st.error(
                "Could not confidently detect the company. Mention it explicitly in the question "
                "or use the company name field above."
            )
            st.stop()

        company_key = detected_company.lower().strip()
        st.success(f"Detected company: {detected_company.title()}")

        if company_key in st.session_state.company_stores:
            document, store = st.session_state.company_stores[company_key]
            st.info(f"Using cached document for {detected_company.title()}")
        else:
            with st.spinner(f"Searching and downloading the latest annual report for {detected_company.title()}..."):
                company_name, pdf_url, filepath = fetch_company_document(
                    user_query,
                    company_name=detected_company,
                )

            if not pdf_url or not filepath:
                st.error(
                    f"Could not fetch a report for {detected_company.title()}. "
                    "Try a more specific company name."
                )
                st.stop()

            st.write(f"Source URL: {pdf_url}")

            metadata = DocumentMetadata(
                company_name=(company_name or detected_company).title(),
                document_type="Integrated Annual Report",
                financial_year="2024-25",
                source_url=pdf_url,
                source_domain=urlparse(pdf_url).netloc,
                search_query=f"{detected_company} Integrated Annual Report 2024-25 filetype:pdf",
            )

            with st.spinner("Ingesting document into the vector store..."):
                store = FAISSStore()
                document = ingest_document(filepath=filepath, metadata=metadata, store=store)

            st.session_state.company_stores[company_key] = (document, store)
            st.success(
                f"Ingested {document.num_pages} pages and {document.num_chunks} chunks for "
                f"{metadata.company_name}."
            )

        with st.spinner("Searching the document for the best grounded answer..."):
            result = query_document(user_query, document, store)

        st.divider()
        st.header("Answer")

        if result["status"] != "completed":
            st.error(result["error"] or "Unable to answer the question.")
            st.stop()

        st.write(result["answer"])

        if document.metadata.source_url:
            st.caption(f"Source: {document.metadata.source_url}")

        if result["citations"]:
            st.subheader("Citations")
            for index, citation in enumerate(result["citations"], start=1):
                label = f"Citation {index} - Page {citation['page']} - Score {citation['score']:.2%}"
                with st.expander(label):
                    st.write(citation["text"])
                    if citation.get("source_url"):
                        st.caption(citation["source_url"])

        st.session_state.query_history.append(
            {
                "query": user_query,
                "company": company_key,
                "answer": result["answer"],
                "citations": len(result["citations"]),
            }
        )


if st.session_state.query_history:
    st.divider()
    st.header("Recent queries")
    for index, item in enumerate(reversed(st.session_state.query_history), start=1):
        with st.expander(f"{index}. {item['query']} ({item['company'].title()})"):
            st.write(item["answer"])
            st.caption(f"Citations used: {item['citations']}")
