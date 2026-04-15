"""
app.py — Streamlit Frontend for Automated Company Intelligence Platform

NEW FLOW:
1. User enters a question (e.g., "Is Infosys doing well financially?")
2. System automatically extracts company name
3. System fetches and ingests company PDF
4. System answers using PURE RETRIEVAL (no LLM)

Run: streamlit run app.py
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import streamlit as st

from schemas import Document, DocumentMetadata
from ingestion import FAISSStore, ingest_document
from engine import query_document
from fetcher import (
    extract_company_name,
    fetch_company_document,
    list_available_companies,
    get_company_pdf,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Company Intelligence — Automated RAG Platform",
    page_icon="🏢",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

if "company_stores" not in st.session_state:
    # Dictionary mapping company_name -> (Document, FAISSStore)
    st.session_state.company_stores: dict[str, tuple[Document, FAISSStore]] = {}

if "query_history" not in st.session_state:
    st.session_state.query_history: list[dict] = []

# ---------------------------------------------------------------------------
# Sidebar — Available Companies
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🏢 Available Companies")
    st.caption("System can automatically fetch reports for these companies:")
    
    available_companies = list_available_companies()
    for i, company in enumerate(available_companies, 1):
        # Check if already ingested
        if company in st.session_state.company_stores:
            st.write(f"{i}. ✅ **{company.title()}** (Ingested)")
        else:
            st.write(f"{i}. {company.title()}")
    
    st.divider()
    
    st.subheader("📊 System Stats")
    st.metric("Documents Ingested", len(st.session_state.company_stores))
    st.metric("Total Queries", len(st.session_state.query_history))
    
    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.company_stores = {}
        st.session_state.query_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

st.title("🏢 Automated Company Intelligence Platform")
st.caption(
    "Ask questions about Indian companies. The system automatically fetches annual reports "
    "and answers using **pure retrieval** (no LLM required)."
)

st.divider()

# ---------------------------------------------------------------------------
# Query Input
# ---------------------------------------------------------------------------

st.header("💬 Ask a Question")

# Example queries
example_queries = [
    "Is Infosys doing well financially?",
    "What is the revenue of TCS?",
    "Does HDFC Bank mention any risks?",
    "What are Reliance's key business segments?",
    "What is Wipro's profit margin?",
]

col1, col2 = st.columns([3, 1])

with col1:
    user_query = st.text_input(
        "Enter your question about any company:",
        placeholder="e.g., What is Infosys' revenue growth?",
        key="query_input",
    )

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    if st.button("🔍 Get Answer", use_container_width=True, type="primary"):
        if user_query.strip():
            st.session_state["process_query"] = True

# Show example queries
with st.expander("📝 Example Questions"):
    for i, example in enumerate(example_queries, 1):
        if st.button(f"{i}. {example}", key=f"example_{i}"):
            st.session_state["query_input"] = example
            st.session_state["process_query"] = True
            st.rerun()

# ---------------------------------------------------------------------------
# Query Processing
# ---------------------------------------------------------------------------

if st.session_state.get("process_query") and user_query.strip():
    st.session_state["process_query"] = False
    
    with st.container():
        st.divider()
        st.header("🔎 Processing Your Query...")
        
        # Step 1: Extract company name
        with st.spinner("Extracting company name..."):
            company = extract_company_name(user_query)
        
        if not company:
            st.error(
                "❌ Could not identify company from your question. "
                "Please mention one of the supported companies explicitly."
            )
            st.info(f"Supported companies: {', '.join([c.title() for c in available_companies])}")
        else:
            st.success(f"✅ Detected company: **{company.title()}**")
            
            # Step 2: Check if already ingested
            if company in st.session_state.company_stores:
                st.info(f"ℹ️ Using previously ingested data for {company.title()}")
                document, store = st.session_state.company_stores[company]
            else:
                # Step 3: Fetch and ingest document
                st.subheader("📥 Fetching Company Document")
                
                pdf_url = get_company_pdf(company)
                if not pdf_url:
                    st.error(f"❌ No PDF available for {company.title()}")
                    st.stop()
                
                st.write(f"**PDF URL:** {pdf_url}")
                
                with st.spinner(f"Downloading and ingesting {company.title()} annual report..."):
                    company_name, pdf_url, filepath = fetch_company_document(user_query)
                    
                    if not filepath:
                        st.error(f"❌ Failed to download PDF for {company.title()}")
                        st.stop()
                    
                    st.success(f"✅ Downloaded: {filepath}")
                    
                    # Ingest document
                    metadata = DocumentMetadata(
                        company_name=company.title(),
                        document_type="Annual Report",
                    )
                    
                    store = FAISSStore()
                    document = ingest_document(
                        filepath=filepath,
                        metadata=metadata,
                        store=store,
                    )
                    
                    # Cache for future queries
                    st.session_state.company_stores[company] = (document, store)
                    
                    st.success(
                        f"✅ Ingested {document.num_pages} pages, "
                        f"{document.num_chunks} chunks"
                    )
            
            # Step 4: Answer question using pure retrieval
            st.subheader("🤖 Generating Answer")
            
            with st.spinner("Searching document and extracting answer..."):
                result = query_document(user_query, document, store)
            
            # Display answer
            st.divider()
            st.header("💡 Answer")
            
            if result["status"] == "completed":
                st.write(result["answer"])
                
                # Show citations
                if result["citations"]:
                    st.subheader("📚 Source Citations")
                    
                    for i, citation in enumerate(result["citations"], 1):
                        with st.expander(
                            f"📄 Citation {i} — Page {citation['page']} "
                            f"(Relevance: {citation['score']:.2%})"
                        ):
                            st.write(citation["text"])
                
                # Save to history
                st.session_state.query_history.append({
                    "query": user_query,
                    "company": company,
                    "answer": result["answer"],
                    "citations": len(result["citations"]),
                })
                
            else:
                st.error(f"❌ Error: {result['error']}")

# ---------------------------------------------------------------------------
# Query History
# ---------------------------------------------------------------------------

if st.session_state.query_history:
    st.divider()
    st.header("📜 Query History")
    
    for i, item in enumerate(reversed(st.session_state.query_history), 1):
        with st.expander(f"{i}. {item['query']} (Company: {item['company'].title()})"):
            st.write(f"**Answer:** {item['answer'][:200]}...")
            st.caption(f"Sources: {item['citations']} citations")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "🏢 Automated Company Intelligence Platform v2.0 | "
    "Fully Offline • No API Keys Required • Pure Retrieval-Based Answering"
)