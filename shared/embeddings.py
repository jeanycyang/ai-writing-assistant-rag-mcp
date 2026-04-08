from __future__ import annotations

from abc import ABC, abstractmethod
from threading import Lock

from sentence_transformers import SentenceTransformer

from shared.config import Settings, get_settings


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._settings.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()


_provider_lock = Lock()
_provider_instance: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance

    with _provider_lock:
        if _provider_instance is not None:
            return _provider_instance

        settings = get_settings()
        if settings.embedding_provider == "sentence_transformers":
            _provider_instance = SentenceTransformerEmbeddingProvider(settings)
            return _provider_instance
        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


def preload_embedding_provider() -> EmbeddingProvider:
    return get_embedding_provider()
