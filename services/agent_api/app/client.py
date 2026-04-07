from __future__ import annotations

from typing import Any

import httpx

from shared.config import get_settings


class RagApiClient:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = httpx.Client(base_url=self._settings.rag_api_url, timeout=60.0)

    def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post("/search/summaries", json=payload)
        response.raise_for_status()
        return response.json()

    def get_linked_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post("/retrieve/linked-raw", json=payload)
        response.raise_for_status()
        return response.json()

    def search_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post("/search/raw", json=payload)
        response.raise_for_status()
        return response.json()
