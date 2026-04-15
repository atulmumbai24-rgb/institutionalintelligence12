"""
fetcher.py — Company Document Fetching Module

Handles:
1. Company name extraction from user queries
2. PDF URL mapping for Indian companies
3. PDF downloading and local storage
"""

import re
import os
import logging
from typing import Optional
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Company PDF Mapping (Hardcoded for stability)
# ---------------------------------------------------------------------------

COMPANY_PDFS = {
    # Note: Some URLs may require authentication or have rate limits.
    # Users can add their own PDF URLs or provide local file paths.
    # For production, consider using official investor relations APIs or SEC EDGAR.
    
    "infosys": "https://www.infosys.com/investors/reports-filings/annual-report/annual-report-2023-24.pdf",
    "reliance": "https://www.ril.com/DownloadFiles/Annual-Report/Reliance-Industries-Limited-Integrated-Annual-Report-2023-24.pdf",
    "tcs": "https://www.tcs.com/content/dam/global-tcs/en/investors/financial-statements/2023-24/ar/annual-report-2023-2024.pdf",
    "tata consultancy services": "https://www.tcs.com/content/dam/global-tcs/en/investors/financial-statements/2023-24/ar/annual-report-2023-2024.pdf",
    "hdfc": "https://www.hdfcbank.com/content/api/contentstream-id/723fb80a-2dde-42a3-9793-7ae1be57c87f/b73e1d7f-f34c-4b2f-a2f1-6d4d1c46c26f/Annual%20Reports/HDFC-Bank-Annual-Report-FY2023-24.pdf",
    "hdfc bank": "https://www.hdfcbank.com/content/api/contentstream-id/723fb80a-2dde-42a3-9793-7ae1be57c87f/b73e1d7f-f34c-4b2f-a2f1-6d4d1c46c26f/Annual%20Reports/HDFC-Bank-Annual-Report-FY2023-24.pdf",
    "wipro": "https://www.wipro.com/content/dam/nexus/en/investor/annual-report/2023-2024/wipro-annual-report-2023-24.pdf",
    "bharti airtel": "https://www.airtel.in/content/dam/airtelin/india/investorrelation/pdf/annual-report/Airtel_Integrated_Report_FY24.pdf",
    "airtel": "https://www.airtel.in/content/dam/airtelin/india/investorrelation/pdf/annual-report/Airtel_Integrated_Report_FY24.pdf",
    "icici": "https://www.icicibank.com/content/dam/icicibank/india/about-us/annual-report/ICICI-Bank-Annual-Report-2023-24.pdf",
    "icici bank": "https://www.icicibank.com/content/dam/icicibank/india/about-us/annual-report/ICICI-Bank-Annual-Report-2023-24.pdf",
    
    # Sample working PDFs for testing (public domain / open access)
    "tesla": "https://www.sec.gov/Archives/edgar/data/1318605/000095017024013644/tsla-20231231.htm",
    "sample": "https://www.africau.edu/images/default/sample.pdf",
}

# Company name aliases for better matching
COMPANY_ALIASES = {
    "infy": "infosys",
    "ril": "reliance",
    "reliance industries": "reliance",
    "tata": "tcs",
    "hdfc bank ltd": "hdfc bank",
    "wipro limited": "wipro",
    "bharti": "bharti airtel",
    "icici bank limited": "icici bank",
}

# ---------------------------------------------------------------------------
# Company Name Extraction
# ---------------------------------------------------------------------------

def extract_company_name(query: str) -> Optional[str]:
    """
    Extract company name from user query using heuristic matching.
    
    Strategy:
    1. Convert query to lowercase
    2. Look for known company names in the query
    3. Check aliases
    4. Return normalized company name or None
    
    Args:
        query: User question (e.g., "Is Infosys doing well financially?")
    
    Returns:
        Normalized company name (e.g., "infosys") or None if not found
    
    Examples:
        >>> extract_company_name("Is Infosys profitable?")
        'infosys'
        >>> extract_company_name("How is TCS performing?")
        'tcs'
    """
    query_lower = query.lower()
    
    # First, check for direct company name matches (longest first for better matching)
    company_names = sorted(COMPANY_PDFS.keys(), key=len, reverse=True)
    for company in company_names:
        if company in query_lower:
            logger.info(f"Extracted company: {company}")
            return company
    
    # Check aliases
    for alias, company in COMPANY_ALIASES.items():
        if alias in query_lower:
            logger.info(f"Extracted company via alias '{alias}': {company}")
            return company
    
    # Fallback: extract first capitalized words (simple heuristic)
    # This can be improved with NER models later
    words = query.split()
    for i, word in enumerate(words):
        # Check if word is capitalized and might be a company name
        if word and word[0].isupper() and len(word) > 2:
            potential = word.lower()
            # Check next word too for multi-word companies
            if i + 1 < len(words) and words[i + 1][0].isupper():
                potential = f"{word} {words[i + 1]}".lower()
            
            if potential in COMPANY_PDFS or potential in COMPANY_ALIASES:
                return COMPANY_ALIASES.get(potential, potential)
    
    logger.warning(f"Could not extract company name from query: {query}")
    return None

# ---------------------------------------------------------------------------
# PDF Downloading
# ---------------------------------------------------------------------------

def download_pdf(url: str, save_dir: str = "/tmp/company_pdfs") -> str:
    """
    Download a PDF from URL and save it locally.
    
    Args:
        url: Direct URL to the PDF file
        save_dir: Directory to save downloaded PDFs
    
    Returns:
        Local file path of the downloaded PDF
    
    Raises:
        Exception: If download fails
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract filename from URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename.endswith('.pdf'):
        filename = f"document_{hash(url)}.pdf"
    
    filepath = os.path.join(save_dir, filename)
    
    # Check if already downloaded
    if os.path.exists(filepath):
        logger.info(f"PDF already exists: {filepath}")
        return filepath
    
    # Download PDF with headers to avoid 403 errors
    try:
        logger.info(f"Downloading PDF from: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,application/octet-stream,*/*',
        }
        response = requests.get(url, headers=headers, timeout=30, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"PDF downloaded successfully: {filepath}")
        return filepath
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download PDF from {url}: {e}")
        raise Exception(f"PDF download failed: {e}")

# ---------------------------------------------------------------------------
# High-Level API
# ---------------------------------------------------------------------------

def get_company_pdf(company_name: str) -> Optional[str]:
    """
    Get the PDF URL for a given company name.
    
    Args:
        company_name: Normalized company name (lowercase)
    
    Returns:
        PDF URL or None if company not found
    """
    # Check aliases first
    company_name = COMPANY_ALIASES.get(company_name, company_name)
    return COMPANY_PDFS.get(company_name)

def fetch_company_document(query: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Complete pipeline: extract company → get PDF URL → download.
    
    Args:
        query: User question
    
    Returns:
        Tuple of (company_name, pdf_url, local_filepath) or (None, None, None)
    """
    # Extract company name
    company = extract_company_name(query)
    if not company:
        return None, None, None
    
    # Get PDF URL
    pdf_url = get_company_pdf(company)
    if not pdf_url:
        logger.warning(f"No PDF URL found for company: {company}")
        return company, None, None
    
    # Download PDF
    try:
        filepath = download_pdf(pdf_url)
        return company, pdf_url, filepath
    except Exception as e:
        logger.error(f"Failed to fetch document for {company}: {e}")
        return company, pdf_url, None

def list_available_companies() -> list[str]:
    """
    Get list of all available companies.
    
    Returns:
        List of company names
    """
    return sorted(COMPANY_PDFS.keys())


