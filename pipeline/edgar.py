"""
pipeline/edgar.py
Fetches 10-K and 10-Q filings from the SEC EDGAR public API.
Completely free — no API key, no sign-up required.
"""

import os
import re
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "PortfolioProject student@university.edu",
    "Accept-Encoding": "gzip, deflate",
}

EDGAR_COMPANY = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

KNOWN_COMPANIES = {
    "Apple":     "0000320193",
    "Microsoft": "0000789019",
    "Tesla":     "0001318605",
    "Amazon":    "0001018724",
    "Google":    "0001652044",
    "Meta":      "0001326801",
    "Netflix":   "0001065280",
    "Nvidia":    "0001045810",
}


def get_recent_filings(cik: str, form_type: str = "10-K", count: int = 5) -> list[dict]:
    """Get recent filings for a company from SEC EDGAR."""
    cik_int = int(cik.lstrip("0") or "0")
    url = EDGAR_COMPANY.format(cik=cik_int)

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise RuntimeError(f"Could not fetch company data for CIK {cik}: {e}")

    filings_data = data.get("filings", {}).get("recent", {})
    forms        = filings_data.get("form", [])
    dates        = filings_data.get("filingDate", [])
    accessions   = filings_data.get("accessionNumber", [])
    descriptions = filings_data.get("primaryDocument", [])

    results = []
    for form, date, acc, desc in zip(forms, dates, accessions, descriptions):
        if form == form_type:
            acc_clean = acc.replace("-", "")
            results.append({
                "form":        form,
                "date":        date,
                "accession":   acc,
                "acc_clean":   acc_clean,
                "primary_doc": desc,
                "cik":         str(cik_int),
                "company":     data.get("name", "Unknown"),
            })
            if len(results) >= count:
                break

    return results


def _get_full_doc_url(cik_int: int, acc_clean: str, primary_doc: str) -> str:
    """
    Find the URL of the actual full 10-K document.
    Uses the filing index JSON which is more reliable than the HTML index.
    """
    # Try the filing index JSON first (most reliable)
    index_json_url = f"https://data.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{acc_clean}-index.json"
    base_archive   = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}"

    try:
        r = requests.get(index_json_url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            docs = data.get("directory", {}).get("item", [])
            # Find the largest .htm file — that's the full 10-K
            htm_docs = [
                d for d in docs
                if d.get("name", "").lower().endswith(".htm")
                and "index" not in d.get("name", "").lower()
                and not d.get("name", "").lower().startswith("r")  # skip inline XBRL
            ]
            if htm_docs:
                # Sort by size descending — largest file is the full report
                htm_docs.sort(key=lambda d: int(d.get("size", 0)), reverse=True)
                best = htm_docs[0]["name"]
                return f"{base_archive}/{best}"
    except Exception:
        pass

    # Fall back to constructing the URL directly from primary_doc
    if primary_doc and not primary_doc.endswith("-index.htm"):
        return f"{base_archive}/{primary_doc}"

    return f"{base_archive}/{primary_doc}"


def fetch_filing_text(filing: dict, max_chars: int = 500_000) -> str:
    """
    Downloads and cleans the full text of a 10-K filing.
    """
    cik_int     = int(filing["cik"])
    acc_clean   = filing["acc_clean"]
    primary_doc = filing["primary_doc"]

    url = _get_full_doc_url(cik_int, acc_clean, primary_doc)
    filing["url"] = url
    print(f"  Fetching: {url}")

    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Could not download filing: {e}")

    print(f"  Downloaded {len(r.content):,} bytes")

    soup = BeautifulSoup(r.content, "lxml")

    # Remove non-content elements
    for tag in soup(["script", "style", "ix:header", "xbrli:xbrl", "head", "meta"]):
        tag.decompose()

    # Convert tables to readable text
    for table in soup.find_all("table"):
        rows = []
        for row in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            table.replace_with("\n" + "\n".join(rows) + "\n")

    text = soup.get_text(separator="\n")

    # Clean whitespace
    lines = [line.strip() for line in text.splitlines()]
    lines = [l for l in lines if len(l) > 20]
    text  = "\n".join(lines)
    text  = re.sub(r"\n{3,}", "\n\n", text)

    print(f"  Extracted {len(text):,} characters")
    return text[:max_chars]
