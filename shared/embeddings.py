from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

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


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider == "sentence_transformers":
        return SentenceTransformerEmbeddingProvider(settings)
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


def preload_embedding_provider() -> EmbeddingProvider:
    return get_embedding_provider()
