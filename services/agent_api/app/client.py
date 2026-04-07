from __future__ import annotations

from typing import Any

import httpx

from shared.embeddings import get_embedding_provider
from shared.config import get_settings


class RagApiClient:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = httpx.Client(base_url=self._settings.rag_api_url, timeout=60.0)
        self._embedding_provider = get_embedding_provider()

    def _with_query_embedding(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = payload.get("query")
        if not isinstance(query, str):
            raise ValueError("Expected string query for vectorized rag-api request")
        enriched = dict(payload)
        enriched["query_embedding"] = self._embedding_provider.embed_text(query)
        return enriched

    def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post("/search/summaries", json=self._with_query_embedding(payload))
        response.raise_for_status()
        return response.json()

    def get_linked_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post("/retrieve/linked-raw", json=payload)
        response.raise_for_status()
        return response.json()

    def search_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post("/search/raw", json=self._with_query_embedding(payload))
        response.raise_for_status()
        return response.json()

    def healthcheck(self) -> dict[str, Any]:
        response = self._client.get("/healthz")
        response.raise_for_status()
        payload = response.json()
        return {
            "status": payload.get("status", "ok"),
            "base_url": self._settings.rag_api_url,
        }
