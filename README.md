# Indian Equity Intelligence

A deployable Streamlit website for grounded financial Q&A on Indian companies.

## What it does

- Discovers live public company documents such as annual reports, investor presentations, and results releases.
- Prioritizes official sources in this order: NSE, BSE, official investor-relations pages, then web fallback.
- Downloads and ingests multiple sources per company.
- Routes finance questions like revenue and profit into a structured extraction path.
- Falls back to grounded retrieval for broader qualitative questions.
- Shows citations and source URLs so answers stay auditable.

## Important truthfulness note

No web research system can honestly guarantee perfect truth on every question. This app is designed to be safer than a generic chatbot:

- It answers from retrieved documents, not from unsupported guesses.
- It shows citations for the answer.
- It says when evidence is weak or incomplete instead of pretending certainty.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

This project is structured for easy deployment on Streamlit Community Cloud or any environment that can run Streamlit.

Required entrypoint:

- `app.py`

Required Python dependencies:

- `requirements.txt`

## Current MVP scope

- Best for publicly available Indian large-cap company documents.
- Strongest on questions like revenue, PAT, EBITDA, debt, and basic qualitative evidence lookup.
- Source discovery still depends on live public websites and may fail when sites block scraping or change structure.
- The official connectors are connector-first, not licensed market-data feeds. They are for public filings and disclosures, not tick-level live pricing.
