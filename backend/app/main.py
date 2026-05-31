import io
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.database import get_vector_store, reset_vector_store
from app.ingestion.pipeline import ingest_all_patents, parse_user_document
from app.services.analytical import generate_report
from app.services.rag_service import query_prior_art, query_prior_art_batch


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


class PriorArtRequest(BaseModel):
    text: str
    paragraph_id: str | None = None
    section_filter: str | None = None  # "Claims", "Abstract", "Detailed Description"


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
    Returns the document split into paragraphs with stable IDs.
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
    return AnalyzeDocumentResponse(paragraphs=paragraphs, total_paragraphs=len(paragraphs))


@app.post("/api/query-prior-art", response_model=PriorArtResponse)
async def query_prior_art_endpoint(body: PriorArtRequest):
    """Semantic search for the 5 most relevant patent chunks."""
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="'text' must not be empty.")

    results = query_prior_art(body.text, section_filter=body.section_filter)
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


@app.post("/api/generate-report")
async def generate_report_endpoint(body: GenerateReportRequest):
    """Generate an AI infringement risk report as structured JSON."""
    if not body.user_text.strip():
        raise HTTPException(status_code=422, detail="'user_text' must not be empty.")

    report = generate_report(body.user_text, body.prior_art_hits)
    return JSONResponse(content=report)


@app.post("/api/ingest")
async def trigger_ingest():
    """Manually re-trigger patent ingestion from raw_patents/."""
    store = get_vector_store()
    n = ingest_all_patents(store)
    reset_vector_store()
    return {"ingested_chunks": n}
