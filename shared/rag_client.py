from __future__ import annotations

from time import perf_counter
from typing import Any

import httpx

from shared.config import get_settings


class RagApiClient:
    def __init__(self, work: str | None = None) -> None:
        self._settings = get_settings()
        self._work = work
        self._client = httpx.Client(base_url=self._settings.rag_api_url, timeout=60.0)
        self._embedding_provider = None

    def _path(self, path: str) -> str:
        if self._work:
            return f"/works/{self._work}{path}"
        return path

    def _get_embedding_provider(self):
        if self._embedding_provider is None:
            from shared.embeddings import get_embedding_provider

            self._embedding_provider = get_embedding_provider()
        return self._embedding_provider

    def _with_query_embedding(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = payload.get("query")
        if not isinstance(query, str):
            raise ValueError("Expected string query for vectorized rag-api request")
        enriched = dict(payload)
        enriched["query_embedding"] = self._get_embedding_provider().embed_text(query)
        return enriched

    def _vectorized_request(self, path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
        embedding_started_at = perf_counter()
        enriched_payload = self._with_query_embedding(payload)
        embedding_elapsed_ms = round((perf_counter() - embedding_started_at) * 1000, 2)

        request_started_at = perf_counter()
        response = self._client.post(self._path(path), json=enriched_payload)
        response.raise_for_status()
        request_elapsed_ms = round((perf_counter() - request_started_at) * 1000, 2)
        return response.json(), {
            "embedding_ms": embedding_elapsed_ms,
            "rag_api_ms": request_elapsed_ms,
        }

    def _normalize_linked_raw_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        summary_hit_ids: list[str] = []
        raw_summary_hit_ids = payload.get("summary_hit_ids", [])
        if not raw_summary_hit_ids:
            for alternate_key in ("summary_hits", "hits", "summaries"):
                alternate_value = payload.get(alternate_key)
                if alternate_value:
                    raw_summary_hit_ids = alternate_value
                    break

        if isinstance(raw_summary_hit_ids, str):
            raw_summary_hit_ids = [raw_summary_hit_ids]
        elif isinstance(raw_summary_hit_ids, dict):
            raw_summary_hit_ids = [raw_summary_hit_ids]
        elif not isinstance(raw_summary_hit_ids, list):
            raw_summary_hit_ids = []

        for item in raw_summary_hit_ids:
            if isinstance(item, str):
                summary_hit_ids.append(item)
            elif isinstance(item, dict):
                for key in ("id", "summary_id", "summary_hit_id"):
                    value = item.get(key)
                    if isinstance(value, str):
                        summary_hit_ids.append(value)
                        break

        normalized = {"summary_hit_ids": summary_hit_ids}
        top_k_per_hit = payload.get("top_k_per_hit")
        if top_k_per_hit is None:
            top_k_per_hit = payload.get("top_k")
        if top_k_per_hit is not None:
            normalized["top_k_per_hit"] = top_k_per_hit
        return normalized

    def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
        result, _ = self._vectorized_request("/search/summaries", payload)
        return result

    def search_summaries_with_timings(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
        return self._vectorized_request("/search/summaries", payload)

    def search_summary_characters(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(
            self._path("/search/summary-characters"),
            json={
                "characters": payload.get("characters", []),
                "operator": payload.get("operator", "or"),
                "chapter_id": payload.get("chapter_id"),
                "from_chapter": payload.get("from_chapter"),
                "to_chapter": payload.get("to_chapter"),
                "min_priority_score": payload.get("min_priority_score"),
                "top_k": payload.get("top_k"),
            },
        )
        response.raise_for_status()
        return response.json()

    def get_linked_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(self._path("/retrieve/linked-raw"), json=self._normalize_linked_raw_payload(payload))
        response.raise_for_status()
        return response.json()

    def get_summary_paragraph(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(
            self._path("/retrieve/summary-paragraph"),
            json={
                "chapter_id": payload["chapter_id"],
                "paragraph_id": payload["paragraph_id"],
            },
        )
        response.raise_for_status()
        return response.json()

    def get_summary_chapter(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(
            self._path("/retrieve/summary-chapter"),
            json={"chapter_id": payload["chapter_id"]},
        )
        response.raise_for_status()
        return response.json()

    def get_raw_paragraph(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(
            self._path("/retrieve/raw-paragraph"),
            json={
                "chapter_id": payload["chapter_id"],
                "paragraph_id": payload["paragraph_id"],
            },
        )
        response.raise_for_status()
        return response.json()

    def get_raw_chapter(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(
            self._path("/retrieve/raw-chapter"),
            json={"chapter_id": payload["chapter_id"]},
        )
        response.raise_for_status()
        return response.json()

    def search_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
        result, _ = self._vectorized_request("/search/raw", payload)
        return result

    def search_raw_with_timings(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
        return self._vectorized_request("/search/raw", payload)

    def healthcheck(self) -> dict[str, Any]:
        response = self._client.get("/healthz")
        response.raise_for_status()
        payload = response.json()
        return {
            "status": payload.get("status", "ok"),
            "base_url": self._settings.rag_api_url,
        }
