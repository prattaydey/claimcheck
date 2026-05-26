from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings
from app.config import settings


class BatchEmbeddings(Embeddings):
    """Wrapper around GoogleGenerativeAIEmbeddings to handle batching properly."""

    def __init__(self, model: str, api_key: str):
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=api_key,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts one at a time to avoid batching issues."""
        return [self._embeddings.embed_documents([text])[0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._embeddings.embed_documents([text])[0]

_chroma_client: Optional[chromadb.PersistentClient] = None
_vector_store: Optional[Chroma] = None


def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        embeddings = BatchEmbeddings(
            model=settings.gemini_embedding_model,
            api_key=settings.gemini_api_key,
        )
        _vector_store = Chroma(
            client=get_chroma_client(),
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
        )
    return _vector_store


def reset_vector_store() -> None:
    """Force re-initialization (useful after ingestion)."""
    global _vector_store
    _vector_store = None
