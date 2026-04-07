from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = Field(default="postgresql+psycopg://fanfic:fanfic@localhost:5432/fanfic_rag")
    local_database_url: str = Field(default="postgresql+psycopg://fanfic:fanfic@localhost:5432/fanfic_rag")

    rag_api_host: str = "0.0.0.0"
    rag_api_port: int = 8001
    agent_api_host: str = "0.0.0.0"
    agent_api_port: int = 8002
    rag_api_url: str = "http://localhost:8001"

    llm_provider: str = "ollama"
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "hauhau-gemma4-e4b-q4km"
    ollama_request_timeout: float = 120.0

    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8

    raw_chunk_size: int = 650
    raw_chunk_overlap: int = 120

    summary_data_dir: str = "data/sample/summaries"
    raw_data_dir: str = "data/sample/raw"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
