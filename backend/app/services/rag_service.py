from langchain_core.documents import Document
from app.config import settings
from app.database import get_vector_store


def query_prior_art(text: str, section_filter: str | None = None) -> list[dict]:
    """
    Perform a similarity search and return top-k matches with scores.

    Returns a list of dicts, each dict format is:
        {
            "patent_id", "title", "inventor", "grant_date",
            "section", "claim_number", "text", "similarity_score"
        }
    """
    store = get_vector_store()

    where: dict | None = None
    if section_filter:
        where = {"section": section_filter}

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
        })

    return hits
