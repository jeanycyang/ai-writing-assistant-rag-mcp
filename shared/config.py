import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = Field(default="postgresql+psycopg://ai_writer:ai_writer@localhost:5432/ai_writing_assistance")
    local_database_url: str = Field(default="postgresql+psycopg://ai_writer:ai_writer@localhost:5432/ai_writing_assistance")

    rag_api_host: str = "0.0.0.0"
    rag_api_port: int = 8001
    rag_api_url: str = "http://localhost:8001"

    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8

    raw_chunk_size: int = 650
    raw_chunk_overlap: int = 120

    summary_data_dir: str = "data/sample/summaries"
    raw_data_dir: str = "data/sample/raw"

    @property
    def effective_database_url(self) -> str:
        if os.getenv("SERVICE_NAME"):
            return self.database_url
        return self.local_database_url

    def database_url_for_work(self, work: str | None = None) -> str:
        if not work:
            return self.effective_database_url
        return make_url(self.effective_database_url).set(database=work).render_as_string(hide_password=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
