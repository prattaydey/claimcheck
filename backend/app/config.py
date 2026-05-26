from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    gemini_api_key: str
    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_embedding_model: str = "models/gemini-embedding-2"

    chroma_persist_dir: str = str(BASE_DIR / "data" / "chroma_db")
    chroma_collection_name: str = "patents"

    raw_patents_dir: str = str(BASE_DIR / "data" / "raw_patents")

    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 5

    class Config:
        env_file = BASE_DIR / ".env"
        env_file_encoding = "utf-8"


settings = Settings()
