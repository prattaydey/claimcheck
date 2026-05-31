import asyncio
from langchain_core.documents import Document
from app.config import settings
from app.database import get_vector_store


def query_prior_art(text: str, section_filter: str | None = None, domain_filter: str | None = None) -> list[dict]:
    """
    Perform a similarity search and return top-k matches with scores.
    Optionally filter by section or domain for targeted searches.

    Args:
        text: Query text
        section_filter: Filter by section ("Claims", "Abstract", "Detailed Description", "Full Text")
        domain_filter: Filter by technology domain ("software", "hardware", "biotech", etc.)

    Returns a list of dicts, each dict format is:
        {
            "patent_id", "title", "inventor", "grant_date",
            "section", "claim_number", "text", "similarity_score", "domain"
        }
    """
    store = get_vector_store()

    where: dict | None = None
    if section_filter or domain_filter:
        where = {}
        if section_filter:
            where["section"] = section_filter
        if domain_filter:
            where["domain"] = domain_filter

    results: list[tuple[Document, float]] = store.similarity_search_with_relevance_scores(
        text,
        k=settings.top_k_results,
        filter=where,
    )

    hits = []
    for doc, score in results:
        hits.append({
            "patent_id": doc.metadata.get("patent_id", ""),
            "title": doc.metadata.get("title", ""),
            "inventor": doc.metadata.get("inventor", ""),
            "grant_date": doc.metadata.get("grant_date", ""),
            "section": doc.metadata.get("section", ""),
            "claim_number": doc.metadata.get("claim_number", 0),
            "text": doc.page_content,
            "similarity_score": round(float(score), 4),
            "domain": doc.metadata.get("domain", "unknown"),
        })

    return hits


def query_prior_art_batch(texts: list[str], section_filter: str | None = None, domain_filter: str | None = None) -> list[list[dict]]:
    """
    Perform parallel similarity searches for multiple texts.
    Optionally filter by section or domain for targeted searches.

    Args:
        texts: List of query texts
        section_filter: Filter by section ("Claims", "Abstract", "Detailed Description", "Full Text")
        domain_filter: Filter by technology domain ("software", "hardware", "biotech", etc.)

    Returns a list of result lists, one per input text.
    """
    store = get_vector_store()
    where: dict | None = None
    if section_filter or domain_filter:
        where = {}
        if section_filter:
            where["section"] = section_filter
        if domain_filter:
            where["domain"] = domain_filter

    all_results = []
    for text in texts:
        results: list[tuple[Document, float]] = store.similarity_search_with_relevance_scores(
            text,
            k=settings.top_k_results,
            filter=where,
        )

        hits = []
        for doc, score in results:
            hits.append({
                "patent_id": doc.metadata.get("patent_id", ""),
                "title": doc.metadata.get("title", ""),
                "inventor": doc.metadata.get("inventor", ""),
                "grant_date": doc.metadata.get("grant_date", ""),
                "section": doc.metadata.get("section", ""),
                "claim_number": doc.metadata.get("claim_number", 0),
                "text": doc.page_content,
                "similarity_score": round(float(score), 4),
                "domain": doc.metadata.get("domain", "unknown"),
            })
        all_results.append(hits)

    return all_results


def query_prior_art_by_domains(text: str, primary_domain: str, secondary_domains: list[str] | None = None, section_filter: str | None = None) -> dict:
    """
    Query prior art using domain-aware search strategy.
    Searches primary domain first, then secondary domains, and returns results organized by domain.

    Args:
        text: Query text
        primary_domain: Primary technology domain from document classification
        secondary_domains: List of secondary domains (optional)
        section_filter: Filter by section (optional)

    Returns:
        {
            "primary_domain_results": list[dict],  # Results from primary domain
            "secondary_domain_results": list[list[dict]],  # Results grouped by secondary domain
            "all_results": list[dict],  # All results merged and sorted by similarity
            "domains_searched": list[str]  # Which domains were queried
        }
    """
    primary_results = query_prior_art(text, section_filter=section_filter, domain_filter=primary_domain)

    secondary_results_by_domain = {}
    domains_searched = [primary_domain]

    if secondary_domains:
        for domain in secondary_domains:
            if domain != primary_domain:  # Avoid duplicate searches
                results = query_prior_art(text, section_filter=section_filter, domain_filter=domain)
                secondary_results_by_domain[domain] = results
                domains_searched.append(domain)

    # Merge all results and sort by similarity score
    all_results = primary_results.copy()
    for domain_results in secondary_results_by_domain.values():
        all_results.extend(domain_results)

    all_results.sort(key=lambda x: x["similarity_score"], reverse=True)

    return {
        "primary_domain_results": primary_results,
        "secondary_domain_results": secondary_results_by_domain,
        "all_results": all_results[:settings.top_k_results],  # Limit to top-k after merge
        "domains_searched": domains_searched,
    }
