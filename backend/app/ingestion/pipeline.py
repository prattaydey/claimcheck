import json
import uuid
from pathlib import Path
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

from app.config import settings
from app.ingestion.extractor import classify_document


def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# JSON patent ingestion: splits into 3 sections (Abstract, Claims, Detailed Description)
# attaches metadata (patent_id, title, inventor, grant_date, section, claim_number, domain)
def load_patent_json(path: str | Path) -> list[Document]:
    """Parse a structured patent JSON file into LangChain Documents."""
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    splitter = _make_splitter()
    docs: list[Document] = []

    # Classify patent to determine domain
    patent_text = f"{data.get('title', '')} {data.get('abstract', '')} {data.get('detailed_description', '')}"
    classification = classify_document(patent_text)
    patent_domain = classification.get("primary_domain", "unknown")

    base_meta = {
        "patent_id": data.get("patent_id", ""),
        "title": data.get("title", ""),
        "inventor": data.get("inventor", ""),
        "grant_date": data.get("grant_date", ""),
        "domain": patent_domain,
    }

    # Abstract
    if abstract := data.get("abstract", "").strip():
        for chunk in splitter.split_text(abstract):
            docs.append(Document(
                page_content=chunk,
                metadata={**base_meta, "section": "Abstract", "claim_number": 0},
            ))

    # Claims
    for claim in data.get("claims", []):
        text = claim.get("text", "").strip()
        if not text:
            continue
        for chunk in splitter.split_text(text):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    **base_meta,
                    "section": "Claims",
                    "claim_number": claim.get("claim_number", 0),
                },
            ))

    # Detailed Description
    if description := data.get("detailed_description", "").strip():
        for chunk in splitter.split_text(description):
            docs.append(Document(
                page_content=chunk,
                metadata={**base_meta, "section": "Detailed Description", "claim_number": 0},
            ))

    return docs


# PDF patent ingestion: stays under one section (Full Text)
# attaches same metadata (patent_id, title, inventor, grant_date, section, claim_number, domain)
def load_patent_pdf(path: str | Path) -> list[Document]:
    """Extract text from a patent PDF and chunk it."""
    reader = PdfReader(str(path))
    full_text = "\n\n".join(
        page.extract_text() or "" for page in reader.pages
    ).strip()

    stem = Path(path).stem

    # Classify PDF to determine domain
    classification = classify_document(full_text)
    patent_domain = classification.get("primary_domain", "unknown")

    splitter = _make_splitter()
    return [
        Document(
            page_content=chunk,
            metadata={
                "patent_id": stem,
                "title": stem.replace("-", " "),
                "inventor": "Unknown",
                "grant_date": "",
                "section": "Full Text",
                "claim_number": 0,
                "domain": patent_domain,
            },
        )
        for chunk in splitter.split_text(full_text)
        if chunk.strip()
    ]


# User document parsing (for /api/analyze-document)
def parse_user_document(text: str) -> list[dict]:
    """
    Split a user's draft specification into paragraphs, assigning each a
    stable paragraph_id. Filters out diagram/metadata pages.
    Returns a list of {paragraph_id, text} dicts.
    """
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    filtered = []
    for para in raw_paragraphs:
        # Skip if too short (likely metadata/diagram labels)
        if len(para) < 300:
            continue

        # Skip if mostly special characters (diagram metadata)
        alpha_chars = sum(1 for c in para if c.isalpha())
        if alpha_chars / len(para) < 0.4:
            continue

        # Skip if fewer than 3 sentences (mostly noise/diagrams)
        if para.count('.') < 3:
            continue

        filtered.append(para)

    return [
        {"paragraph_id": str(uuid.uuid4()), "text": para}
        for para in filtered
    ]


# Seed loader: ingest everything in raw_patents/, returns len of raw_patents/
def ingest_all_patents(vector_store) -> int:
    """Load all JSON/PDF patent files from raw_patents/ into ChromaDB."""
    raw_dir = Path(settings.raw_patents_dir)
    docs: list[Document] = []

    for json_file in sorted(raw_dir.glob("*.json")):
        docs.extend(load_patent_json(json_file))

    for pdf_file in sorted(raw_dir.glob("*.pdf")):
        docs.extend(load_patent_pdf(pdf_file))

    if not docs:
        return 0

    # Assign stable IDs so re-runs don't duplicate
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, d.page_content)) for d in docs]
    vector_store.add_documents(docs, ids=ids)
    return len(docs)
