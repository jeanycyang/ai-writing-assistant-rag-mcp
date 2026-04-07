from typing import Any

import httpx
import pytest
from fastapi import HTTPException

from services.agent_api.app import chat
from shared.schemas import ChatRequest


class FakeProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_episode_summaries",
                                "arguments": {"query": "任隊長第一次被提到是在哪裡？", "top_k": 2},
                            }
                        }
                    ],
                }
            }
        return {
            "message": {
                "role": "assistant",
                "content": "任隊長在 episode_01 的第 1 段先被提到，之後才正式現身。",
            }
        }


class FakeRagClient:
    def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["query"] == "任隊長第一次被提到是在哪裡？"
        return {
            "hits": [
                {
                    "citation": {
                        "summary_id": "11111111-1111-1111-1111-111111111111",
                        "chapter_id": "episode_01",
                        "paragraph_id": 1,
                        "source_path": "data/sample/summaries/episode_01.md",
                        "score": 0.95,
                        "citation_type": "summary",
                    }
                }
            ]
        }

    def get_linked_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("Should not call linked raw in this test")

    def search_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("Should not call raw search in this test")


def test_run_chat_uses_summary_search_first(monkeypatch) -> None:
    monkeypatch.setattr(chat, "get_llm_provider", lambda: FakeProvider())
    monkeypatch.setattr(chat, "RagApiClient", FakeRagClient)

    response = chat.run_chat(ChatRequest(message="任隊長第一次被提到是在哪裡？"))

    assert "先被提到" in response.answer
    assert response.debug.iterations == 2
    assert response.debug.unique_citation_count == 1
    assert response.debug.completed_without_tool_call is False
    assert response.debug.tool_calls[0].tool_name == "search_episode_summaries"
    assert response.citations[0].chapter_id == "episode_01"


def test_run_chat_dedupes_citations(monkeypatch) -> None:
    class DuplicateCitationRagClient:
        def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
            citation = {
                "summary_id": "11111111-1111-1111-1111-111111111111",
                "chapter_id": "episode_01",
                "paragraph_id": 1,
                "source_path": "data/sample/summaries/episode_01.md",
                "score": 0.95,
                "citation_type": "summary",
            }
            return {"hits": [{"citation": citation}, {"citation": citation}]}

        def get_linked_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("not expected")

        def search_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("not expected")

    monkeypatch.setattr(chat, "get_llm_provider", lambda: FakeProvider())
    monkeypatch.setattr(chat, "RagApiClient", DuplicateCitationRagClient)

    response = chat.run_chat(ChatRequest(message="任隊長第一次被提到是在哪裡？"))

    assert len(response.citations) == 1


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

    assert "rag-api request failed" in str(exc_info.value)


def test_run_chat_returns_502_for_unsupported_tool(monkeypatch) -> None:
    class UnsupportedToolProvider:
        def complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "unknown_tool", "arguments": {}}},
                    ],
                }
            }

    monkeypatch.setattr(chat, "get_llm_provider", lambda: UnsupportedToolProvider())
    monkeypatch.setattr(chat, "RagApiClient", FakeRagClient)

    with pytest.raises(HTTPException) as exc_info:
        chat.run_chat(ChatRequest(message="test"))

    assert exc_info.value.status_code == 502
    assert "Unsupported tool call" in str(exc_info.value.detail)


def test_run_chat_returns_502_when_tool_loop_exhausts(monkeypatch) -> None:
    class EndlessToolProvider:
        def complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_episode_summaries",
                                "arguments": {"query": "loop", "top_k": 1},
                            }
                        }
                    ],
                }
            }

    class LoopRagClient:
        def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {"hits": []}

        def get_linked_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("not expected")

        def search_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("not expected")

    monkeypatch.setattr(chat, "get_llm_provider", lambda: EndlessToolProvider())
    monkeypatch.setattr(chat, "RagApiClient", LoopRagClient)

    with pytest.raises(HTTPException) as exc_info:
        chat.run_chat(ChatRequest(message="loop"))

    assert exc_info.value.status_code == 502
    assert "tool loop limit" in str(exc_info.value.detail)
