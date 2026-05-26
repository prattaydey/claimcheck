"""
Utility for scraping patent abstracts and claims from Google Patents.
Usage (standalone): python -m app.ingestion.scraper US-11234567-B2
"""
import re
import sys
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ClaimCheckBot/1.0; "
        "+https://github.com/claimcheck)"
    )
}
BASE_URL = "https://patents.google.com/patent/{patent_id}/en"


@dataclass
class ScrapedPatent:
    patent_id: str
    title: str = ""
    inventor: str = ""
    grant_date: str = ""
    abstract: str = ""
    claims: list[str] = field(default_factory=list)
    detailed_description: str = ""


def scrape_google_patent(patent_id: str, timeout: int = 10) -> ScrapedPatent | None:
    """Fetch and parse a Google Patents page for the given patent ID."""
    url = BASE_URL.format(patent_id=patent_id.replace("-", ""))
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[scraper] Request failed for {patent_id}: {exc}", file=sys.stderr)
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    patent = ScrapedPatent(patent_id=patent_id)

    # Title
    if title_tag := soup.find("h1", id="title"):
        patent.title = title_tag.get_text(strip=True)

    # Inventors
    inventors = soup.select("dd.inventor span[itemprop='name']")
    patent.inventor = ", ".join(i.get_text(strip=True) for i in inventors)

    # Grant date
    if date_tag := soup.find("time", itemprop="datePublished"):
        patent.grant_date = date_tag.get("datetime", "")

    # Abstract
    if abstract_section := soup.find("section", itemprop="abstract"):
        patent.abstract = abstract_section.get_text(separator=" ", strip=True)

    # Claims
    claims_section = soup.find("section", itemprop="claims")
    if claims_section:
        for claim_div in claims_section.find_all("div", class_="claim"):
            text = claim_div.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text)
            if text:
                patent.claims.append(text)

    # Detailed description
    if desc_section := soup.find("section", itemprop="description"):
        patent.detailed_description = desc_section.get_text(separator=" ", strip=True)

    return patent


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.ingestion.scraper <patent_id>")
        sys.exit(1)

    result = scrape_google_patent(sys.argv[1])
    if result:
        import json
        print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))
    else:
        print("Scraping failed.")
        sys.exit(1)
