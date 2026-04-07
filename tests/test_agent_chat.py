from typing import Any

import httpx
import pytest

from services.agent_api.app import chat
from shared.schemas import ChatRequest


class FakeProvider:
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        think: bool | str | None = None,
    ) -> dict[str, Any]:
        assert tools is None
        assert think is False
        return {
            "message": {
                "role": "assistant",
                "content": "任隊長在 episode_01 的第 1 段先被提到，當時人還沒出現。",
            }
        }


class FakeRagClient:
    def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert payload == {"query": "任隊長第一次被提到是在哪裡？", "top_k": 3}
        return {
            "hits": [
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "chapter_id": "episode_01",
                    "paragraph_id": 1,
                    "scene": "走廊低語",
                    "plot": "任隊長先被傳聞提到，本人尚未出場。",
                    "key_events": ["值勤表異樣", "任隊長被提到"],
                    "score": 0.95,
                    "citation": {
                        "summary_id": "11111111-1111-1111-1111-111111111111",
                        "chapter_id": "episode_01",
                        "paragraph_id": 1,
                        "source_path": "data/sample/summaries/episode_01.md",
                        "score": 0.95,
                        "citation_type": "summary",
                    },
                }
            ]
        }

    def get_linked_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert payload == {
            "summary_hit_ids": ["11111111-1111-1111-1111-111111111111"],
            "top_k_per_hit": 1,
        }
        return {
            "hits": [
                {
                    "chapter_id": "episode_01",
                    "paragraph_id": 1,
                    "chunk_id": 0,
                    "original_text": "林妍看見值勤表時，先聽見眾人提起任隊長。",
                    "score": 0.88,
                    "citation": {
                        "summary_id": "11111111-1111-1111-1111-111111111111",
                        "raw_chunk_id": "22222222-2222-2222-2222-222222222222",
                        "chapter_id": "episode_01",
                        "paragraph_id": 1,
                        "chunk_id": 0,
                        "source_path": "data/sample/raw/episode_01.md",
                        "score": 0.88,
                        "citation_type": "raw",
                    },
                }
            ]
        }

    def search_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("Should not call raw search in this test")


def test_run_chat_uses_deterministic_summary_first_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(chat, "get_llm_provider", lambda: FakeProvider())
    monkeypatch.setattr(chat, "RagApiClient", FakeRagClient)

    response = chat.run_chat(ChatRequest(message="任隊長第一次被提到是在哪裡？"))

    assert "第 1 段" in response.answer
    assert response.debug.iterations == 1
    assert response.debug.unique_citation_count == 2
    assert [call.tool_name for call in response.debug.tool_calls] == [
        "search_episode_summaries",
        "get_linked_original_text",
    ]
    assert response.debug.elapsed_ms is None
    assert response.debug.step_timings == []


def test_run_chat_includes_elapsed_ms_when_requested(monkeypatch) -> None:
    monkeypatch.setattr(chat, "get_llm_provider", lambda: FakeProvider())
    monkeypatch.setattr(chat, "RagApiClient", FakeRagClient)

    response = chat.run_chat(ChatRequest(message="任隊長第一次被提到是在哪裡？", include_timing=True))

    assert response.debug.elapsed_ms is not None
    assert response.debug.elapsed_ms >= 0
    assert [timing.step for timing in response.debug.step_timings] == [
        "search_episode_summaries",
        "get_linked_original_text",
        "final_generation",
    ]
    assert all(timing.elapsed_ms >= 0 for timing in response.debug.step_timings)


def test_run_chat_falls_back_to_raw_search(monkeypatch) -> None:
    class RawFallbackRagClient:
        def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {"hits": []}

        def get_linked_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("not expected")

        def search_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload == {"query": "任隊長第一次被提到是在哪裡？", "top_k": 3}
            return {
                "hits": [
                    {
                        "chapter_id": "episode_01",
                        "paragraph_id": 1,
                        "chunk_id": 0,
                        "original_text": "走廊裡先有人提起任隊長。",
                        "score": 0.81,
                        "citation": {
                            "raw_chunk_id": "33333333-3333-3333-3333-333333333333",
                            "chapter_id": "episode_01",
                            "paragraph_id": 1,
                            "chunk_id": 0,
                            "source_path": "data/sample/raw/episode_01.md",
                            "score": 0.81,
                            "citation_type": "raw",
                        },
                    }
                ]
            }

    monkeypatch.setattr(chat, "get_llm_provider", lambda: FakeProvider())
    monkeypatch.setattr(chat, "RagApiClient", RawFallbackRagClient)

    response = chat.run_chat(ChatRequest(message="任隊長第一次被提到是在哪裡？"))

    assert response.debug.tool_calls[-1].tool_name == "search_original_text"
    assert len(response.citations) == 1


def test_run_chat_includes_raw_search_timing_when_used(monkeypatch) -> None:
    class RawFallbackRagClient:
        def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {"hits": []}

        def get_linked_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("not expected")

        def search_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {
                "hits": [
                    {
                        "chapter_id": "episode_01",
                        "paragraph_id": 1,
                        "chunk_id": 0,
                        "original_text": "走廊裡先有人提起任隊長。",
                        "score": 0.81,
                        "citation": {
                            "raw_chunk_id": "33333333-3333-3333-3333-333333333333",
                            "chapter_id": "episode_01",
                            "paragraph_id": 1,
                            "chunk_id": 0,
                            "source_path": "data/sample/raw/episode_01.md",
                            "score": 0.81,
                            "citation_type": "raw",
                        },
                    }
                ]
            }

    monkeypatch.setattr(chat, "get_llm_provider", lambda: FakeProvider())
    monkeypatch.setattr(chat, "RagApiClient", RawFallbackRagClient)

    response = chat.run_chat(ChatRequest(message="任隊長第一次被提到是在哪裡？", include_timing=True))

    assert [timing.step for timing in response.debug.step_timings] == [
        "search_episode_summaries",
        "search_original_text",
        "final_generation",
    ]


def test_run_chat_passes_context_to_final_model_call(monkeypatch) -> None:
    captured_messages: list[list[dict[str, Any]]] = []

    class ContextProvider(FakeProvider):
        def complete(
            self,
            messages: list[dict[str, Any]],
            *,
            tools: list[dict[str, Any]] | None = None,
            think: bool | str | None = None,
        ) -> dict[str, Any]:
            captured_messages.append(messages)
            return super().complete(messages, tools=tools, think=think)

    monkeypatch.setattr(chat, "get_llm_provider", lambda: ContextProvider())
    monkeypatch.setattr(chat, "RagApiClient", FakeRagClient)

    chat.run_chat(ChatRequest(message="任隊長第一次被提到是在哪裡？"))

    assert len(captured_messages) == 1
    assert "Summary search results:" in captured_messages[0][1]["content"]
    assert "Linked original text:" in captured_messages[0][1]["content"]


def test_run_chat_returns_503_when_rag_api_fails(monkeypatch) -> None:
    class FailingRagClient:
        def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise httpx.ConnectError("rag down")

        def get_linked_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("not expected")

        def search_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("not expected")

    monkeypatch.setattr(chat, "get_llm_provider", lambda: FakeProvider())
    monkeypatch.setattr(chat, "RagApiClient", FailingRagClient)

    with pytest.raises(Exception) as exc_info:
        chat.run_chat(ChatRequest(message="任隊長第一次被提到是在哪裡？"))

    assert "search_episode_summaries" in str(exc_info.value)


def test_run_chat_returns_503_when_llm_provider_fails(monkeypatch) -> None:
    class FailingProvider:
        def complete(
            self,
            messages: list[dict[str, Any]],
            *,
            tools: list[dict[str, Any]] | None = None,
            think: bool | str | None = None,
        ) -> dict[str, Any]:
            raise httpx.ConnectError("ollama down")

    monkeypatch.setattr(chat, "get_llm_provider", lambda: FailingProvider())
    monkeypatch.setattr(chat, "RagApiClient", FakeRagClient)

    with pytest.raises(Exception) as exc_info:
        chat.run_chat(ChatRequest(message="任隊長第一次被提到是在哪裡？"))

    assert "LLM provider request failed" in str(exc_info.value)
