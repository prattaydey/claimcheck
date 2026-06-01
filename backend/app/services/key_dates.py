"""
Key dates and milestones extraction from patents.
Identifies filing dates, grant dates, expiration dates, and key milestones.
"""

import re
from datetime import datetime
from typing import Optional
import google.genai as genai
from app.config import settings

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


DATE_EXTRACTION_PROMPT = """Extract key dates and milestones from patent documents. Return JSON with:
{
    "filing_date": "YYYY-MM-DD or null",
    "grant_date": "YYYY-MM-DD or null",
    "publication_date": "YYYY-MM-DD or null",
    "expiration_date": "YYYY-MM-DD or null",
    "milestones": [
        {
            "date": "YYYY-MM-DD",
            "event": "string - what happened",
            "significance": "string - why it matters"
        }
    ]
}"""


def extract_patent_dates(patent_data: dict) -> dict:
    """
    Extract key dates from patent metadata.

    Args:
        patent_data: Dict with patent_id, title, abstract, etc.

    Returns:
        {
            "filing_date": "YYYY-MM-DD",
            "grant_date": "YYYY-MM-DD",
            "publication_date": "YYYY-MM-DD",
            "expiration_date": "YYYY-MM-DD",
            "milestones": [...]
        }
    """
    dates = {
        "filing_date": None,
        "grant_date": patent_data.get("grant_date"),
        "publication_date": None,
        "expiration_date": None,
        "milestones": [],
    }

    # Extract filing date if present in patent data
    if "filing_date" in patent_data:
        dates["filing_date"] = patent_data["filing_date"]

    # Calculate expiration date (typically 20 years from filing)
    if dates["filing_date"]:
        try:
            filing = datetime.strptime(dates["filing_date"], "%Y-%m-%d")
            expiration = filing.replace(year=filing.year + 20)
            dates["expiration_date"] = expiration.strftime("%Y-%m-%d")
        except:
            pass

    # If grant date exists, calculate publication date (typically 18 months after filing)
    if dates["filing_date"]:
        try:
            filing = datetime.strptime(dates["filing_date"], "%Y-%m-%d")
            publication = filing.replace(month=filing.month + 6)
            if publication.month > 12:
                publication = publication.replace(year=publication.year + 1, month=publication.month - 12)
            dates["publication_date"] = publication.strftime("%Y-%m-%d")
        except:
            pass

    return dates


def calculate_patent_timeline(grant_date: str) -> dict:
    """
    Calculate important dates based on grant date.
    Standard US patent: 20 years from filing (roughly 18-24 months before grant).
    """
    if not grant_date:
        return {
            "years_remaining": None,
            "expiration_year": None,
            "status": "Unknown",
            "milestones": [],
        }

    try:
        grant = datetime.strptime(grant_date, "%Y-%m-%d")
        now = datetime.now()

        # Estimate filing date (assume 18 months before grant)
        # Add 18 months by adding 1 year and 6 months
        filing_year = grant.year
        filing_month = grant.month - 6
        if filing_month <= 0:
            filing_year -= 1
            filing_month += 12

        filing_estimate = datetime(filing_year, filing_month, 1)

        # Patent expires 20 years from filing
        expiration_year = filing_estimate.year + 20
        expiration = datetime(expiration_year, filing_estimate.month, 1)

        years_remaining = (expiration - now).days / 365.25
        status = "Active" if years_remaining > 0 else "Expired"

        milestones = [
            {
                "date": grant_date,
                "event": "Patent Granted",
                "significance": "Patent protection became enforceable",
            },
            {
                "date": filing_estimate.strftime("%Y-%m-%d"),
                "event": "Patent Filed (estimated)",
                "significance": "Start of patent term (20 years)",
            },
            {
                "date": expiration.strftime("%Y-%m-%d"),
                "event": "Patent Expires",
                "significance": "Patent protection ends, claims enter public domain",
            },
        ]

        return {
            "years_remaining": max(0, round(years_remaining, 1)),
            "expiration_year": expiration_year,
            "status": status,
            "milestones": milestones,
        }
    except Exception as e:
        return {
            "years_remaining": None,
            "expiration_year": None,
            "status": "Unknown",
            "milestones": [],
        }


def get_licensing_timeline(patent_id: str, grant_date: str) -> dict:
    """
    Provide timeline guidance for licensing negotiations.
    """
    timeline = calculate_patent_timeline(grant_date)
    years_left = timeline["years_remaining"] or 0

    if years_left < 3:
        urgency = "CRITICAL: Patent expiring soon, act immediately"
        strategy = "Negotiate expedited licensing before patent expires"
    elif years_left < 7:
        urgency = "HIGH: Limited patent life remaining"
        strategy = "Prioritize licensing negotiations, design-around may be preferable"
    elif years_left < 14:
        urgency = "MEDIUM: Mid-term patent protection"
        strategy = "Standard licensing or redesign approach viable"
    else:
        urgency = "LOW: Long patent life remaining"
        strategy = "Time for thorough prior art analysis before licensing"

    return {
        "patent_id": patent_id,
        "urgency": urgency,
        "strategy": strategy,
        "years_remaining": years_left,
        "milestones": timeline["milestones"],
    }
