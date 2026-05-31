import io
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.database import get_vector_store, reset_vector_store
from app.ingestion.pipeline import ingest_all_patents, parse_user_document
from app.ingestion.extractor import classify_document
from app.services.analytical import generate_report
from app.services.rag_service import query_prior_art, query_prior_art_batch, query_prior_art_by_domains
from app.services.synthesis import analyze_patterns, suggest_workarounds


# Lifespan: seed ChromaDB on startup if collection is empty
@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_vector_store()
    count = store._collection.count()
    if count == 0:
        print("[startup] ChromaDB empty — seeding from raw_patents/ …")
        n = ingest_all_patents(store)
        reset_vector_store() # ensures freshly seeded data is visible for next req
        print(f"[startup] Ingested {n} chunks.")
    else:
        print(f"[startup] ChromaDB ready ({count} chunks).")
    yield


app = FastAPI(title="ClaimCheck API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request / Response models
class AnalyzeDocumentResponse(BaseModel):
    paragraphs: list[dict]  # [{paragraph_id, text}]
    total_paragraphs: int
    classification: dict  # {primary_domain, confidence, secondary_domains, key_terms, technical_summary}


class PriorArtRequest(BaseModel):
    text: str
    paragraph_id: str | None = None
    section_filter: str | None = None  # "Claims", "Abstract", "Detailed Description"
    domain_filter: str | None = None  # "software", "hardware", "biotech", etc.


class PriorArtResponse(BaseModel):
    results: list[dict]
    query_text: str


class GenerateReportRequest(BaseModel):
    user_text: str
    prior_art_hits: list[dict]


class BatchPriorArtRequest(BaseModel):
    texts: list[str]
    section_filter: str | None = None


class BatchPriorArtResponse(BaseModel):
    results: list[list[dict]]
    query_texts: list[str]


class DomainAwarePriorArtRequest(BaseModel):
    text: str
    primary_domain: str
    secondary_domains: list[str] | None = None
    section_filter: str | None = None


class DomainAwarePriorArtResponse(BaseModel):
    primary_domain_results: list[dict]
    secondary_domain_results: dict
    all_results: list[dict]
    domains_searched: list[str]
    query_text: str


class SynthesisAnalysisRequest(BaseModel):
    paragraphs: list[dict]  # [{paragraph_id, text}]
    prior_art_by_paragraph: dict  # {paragraph_id: [prior art results]}


class SynthesisAnalysisResponse(BaseModel):
    cross_paragraph_themes: list[dict]
    conflict_clusters: list[dict]
    core_exposures: list[dict]
    narrative_summary: str


class WorkaroundRequest(BaseModel):
    exposure: str
    patent_claims: list[str]


class WorkaroundResponse(BaseModel):
    exposure_summary: str
    redesign_scope: str
    workarounds: list[dict]
    recommended_path: str


# Routes
@app.get("/api/health")
async def health():
    store = get_vector_store()
    return {"status": "ok", "chunks_indexed": store._collection.count()}


@app.post("/api/analyze-document", response_model=AnalyzeDocumentResponse)
async def analyze_document(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
):
    """
    Accept a plain-text/markdown body OR a file upload.
    Returns the document split into paragraphs with stable IDs and domain classification.
    """
    content = ""

    if file is not None:
        raw = await file.read()
        if file.filename and file.filename.lower().endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            content = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        else:
            content = raw.decode("utf-8", errors="replace")
    elif text is not None:
        content = text
    else:
        raise HTTPException(status_code=422, detail="Provide 'file' or 'text' field.")

    if not content.strip():
        raise HTTPException(status_code=422, detail="Document appears to be empty.")

    paragraphs = parse_user_document(content)
    classification = classify_document(content)
    return AnalyzeDocumentResponse(paragraphs=paragraphs, total_paragraphs=len(paragraphs), classification=classification)


@app.post("/api/query-prior-art", response_model=PriorArtResponse)
async def query_prior_art_endpoint(body: PriorArtRequest):
    """Semantic search for the 5 most relevant patent chunks."""
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="'text' must not be empty.")

    results = query_prior_art(body.text, section_filter=body.section_filter, domain_filter=body.domain_filter)
    return PriorArtResponse(results=results, query_text=body.text)


@app.post("/api/query-prior-art-batch", response_model=BatchPriorArtResponse)
async def query_prior_art_batch_endpoint(body: BatchPriorArtRequest):
    """Batch semantic search for multiple texts (parallel)."""
    if not body.texts:
        raise HTTPException(status_code=422, detail="'texts' must not be empty.")

    if any(not text.strip() for text in body.texts):
        raise HTTPException(status_code=422, detail="All texts must not be empty.")

    results = query_prior_art_batch(body.texts, section_filter=body.section_filter)
    return BatchPriorArtResponse(results=results, query_texts=body.texts)


@app.post("/api/query-prior-art-domains", response_model=DomainAwarePriorArtResponse)
async def query_prior_art_domains_endpoint(body: DomainAwarePriorArtRequest):
    """Domain-aware prior art search using document classification."""
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="'text' must not be empty.")

    results = query_prior_art_by_domains(
        body.text,
        primary_domain=body.primary_domain,
        secondary_domains=body.secondary_domains,
        section_filter=body.section_filter,
    )
    return DomainAwarePriorArtResponse(
        primary_domain_results=results["primary_domain_results"],
        secondary_domain_results=results["secondary_domain_results"],
        all_results=results["all_results"],
        domains_searched=results["domains_searched"],
        query_text=body.text,
    )


@app.post("/api/generate-report")
async def generate_report_endpoint(body: GenerateReportRequest):
    """Generate an AI infringement risk report as structured JSON."""
    if not body.user_text.strip():
        raise HTTPException(status_code=422, detail="'user_text' must not be empty.")

    report = generate_report(body.user_text, body.prior_art_hits)
    return JSONResponse(content=report)


@app.post("/api/analyze-patterns", response_model=SynthesisAnalysisResponse)
async def analyze_patterns_endpoint(body: SynthesisAnalysisRequest):
    """Analyze cross-paragraph patterns and conflict clusters."""
    if not body.paragraphs:
        raise HTTPException(status_code=422, detail="'paragraphs' must not be empty.")

    if not body.prior_art_by_paragraph:
        raise HTTPException(status_code=422, detail="'prior_art_by_paragraph' must not be empty.")

    result = analyze_patterns(body.paragraphs, body.prior_art_by_paragraph)
    return SynthesisAnalysisResponse(
        cross_paragraph_themes=result.get("cross_paragraph_themes", []),
        conflict_clusters=result.get("conflict_clusters", []),
        core_exposures=result.get("core_exposures", []),
        narrative_summary=result.get("narrative_summary", ""),
    )


@app.post("/api/suggest-workarounds", response_model=WorkaroundResponse)
async def suggest_workarounds_endpoint(body: WorkaroundRequest):
    """Generate workaround suggestions for a specific exposure."""
    if not body.exposure.strip():
        raise HTTPException(status_code=422, detail="'exposure' must not be empty.")

    if not body.patent_claims:
        raise HTTPException(status_code=422, detail="'patent_claims' must not be empty.")

    result = suggest_workarounds(body.exposure, body.patent_claims)
    return WorkaroundResponse(
        exposure_summary=result.get("exposure_summary", ""),
        redesign_scope=result.get("redesign_scope", ""),
        workarounds=result.get("workarounds", []),
        recommended_path=result.get("recommended_path", ""),
    )


@app.post("/api/ingest")
async def trigger_ingest():
    """Manually re-trigger patent ingestion from raw_patents/."""
    store = get_vector_store()
    n = ingest_all_patents(store)
    reset_vector_store()
    return {"ingested_chunks": n}
