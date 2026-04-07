from typing import Any

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
                                "arguments": {"query": "When was Ren first mentioned?", "top_k": 2},
                            }
                        }
                    ],
                }
            }
        return {
            "message": {
                "role": "assistant",
                "content": "Ren is first mentioned in episode_01 paragraph 1 before appearing in person.",
            }
        }


class FakeRagClient:
    def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
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

    response = chat.run_chat(ChatRequest(message="When was Ren first mentioned?"))

    assert "first mentioned" in response.answer
    assert response.debug.tool_calls[0].tool_name == "search_episode_summaries"
    assert response.citations[0].chapter_id == "episode_01"
