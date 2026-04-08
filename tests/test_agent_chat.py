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
        assert payload in (
            {"query": "任隊長第一次被提到是在哪裡？", "top_k": 5},
            {"query": "任隊長第一次被提到是在哪裡？", "top_k": 8},
        )
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
        assert payload in (
            {
                "summary_hit_ids": ["11111111-1111-1111-1111-111111111111"],
                "top_k_per_hit": 2,
            },
            {
                "summary_hit_ids": ["11111111-1111-1111-1111-111111111111"],
                "top_k_per_hit": 3,
            },
        )
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
        assert payload == {"query": "任隊長第一次被提到是在哪裡？", "top_k": 8}
        return {"hits": []}

    def search_summaries_with_timings(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
        return self.search_summaries(payload), {"embedding_ms": 12.5, "rag_api_ms": 7.25}

    def search_raw_with_timings(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
        return self.search_raw(payload), {"embedding_ms": 10.0, "rag_api_ms": 5.0}


def test_run_chat_uses_deterministic_summary_first_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(chat, "get_llm_provider", lambda: FakeProvider())
    monkeypatch.setattr(chat, "RagApiClient", FakeRagClient)

    response = chat.run_chat(ChatRequest(message="任隊長第一次被提到是在哪裡？"))

    assert "第 1 段" in response.answer
    assert response.debug.iterations == 1
    assert response.debug.unique_citation_count == 2
    assert len(response.citations) == 2
    assert response.citations[0].citation_type == "raw"
    assert [call.tool_name for call in response.debug.tool_calls] == [
        "search_episode_summaries",
        "get_linked_original_text",
        "search_episode_summaries",
        "get_linked_original_text",
        "search_original_text",
    ]
    assert response.debug.elapsed_ms is None
    assert response.debug.step_timings == []


def test_run_chat_includes_elapsed_ms_when_requested(monkeypatch) -> None:
    monkeypatch.setattr(chat, "get_llm_provider", lambda: FakeProvider())
    monkeypatch.setattr(chat, "RagApiClient", FakeRagClient)

    response = chat.run_chat(ChatRequest(message="任隊長第一次被提到是在哪裡？", include_timing=True))

    assert response.debug.elapsed_ms is not None
    assert response.debug.elapsed_ms >= 0
    steps = [timing.step for timing in response.debug.step_timings]
    assert steps[-1] == "final_generation"
    assert steps.count("search_episode_summaries_embed_query") == 2
    assert steps.count("search_episode_summaries_rag_api") == 2
    assert steps.count("get_linked_original_text") == 2
    assert steps.count("search_original_text_embed_query") == 1
    assert steps.count("search_original_text_rag_api") == 1
    assert all(timing.elapsed_ms >= 0 for timing in response.debug.step_timings)


def test_run_chat_falls_back_to_raw_search(monkeypatch) -> None:
    class RawFallbackRagClient:
        def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload in (
                {"query": "任隊長第一次被提到是在哪裡？", "top_k": 5},
                {"query": "任隊長第一次被提到是在哪裡？", "top_k": 8},
            )
            return {"hits": []}

        def search_summaries_with_timings(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
            return self.search_summaries(payload), {"embedding_ms": 11.0, "rag_api_ms": 6.0}

        def get_linked_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("not expected")

        def search_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload in (
                {"query": "任隊長第一次被提到是在哪裡？", "top_k": 5},
                {"query": "任隊長第一次被提到是在哪裡？", "top_k": 8},
            )
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

        def search_raw_with_timings(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
            return self.search_raw(payload), {"embedding_ms": 13.0, "rag_api_ms": 4.0}

    monkeypatch.setattr(chat, "get_llm_provider", lambda: FakeProvider())
    monkeypatch.setattr(chat, "RagApiClient", RawFallbackRagClient)

    response = chat.run_chat(ChatRequest(message="任隊長第一次被提到是在哪裡？"))

    assert response.debug.tool_calls[-1].tool_name == "search_original_text"
    assert len(response.citations) == 1


def test_run_chat_includes_raw_search_timing_when_used(monkeypatch) -> None:
    class RawFallbackRagClient:
        def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload in (
                {"query": "任隊長第一次被提到是在哪裡？", "top_k": 5},
                {"query": "任隊長第一次被提到是在哪裡？", "top_k": 8},
            )
            return {"hits": []}

        def search_summaries_with_timings(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
            return self.search_summaries(payload), {"embedding_ms": 11.0, "rag_api_ms": 6.0}

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

        def search_raw_with_timings(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
            return self.search_raw(payload), {"embedding_ms": 13.0, "rag_api_ms": 4.0}

    monkeypatch.setattr(chat, "get_llm_provider", lambda: FakeProvider())
    monkeypatch.setattr(chat, "RagApiClient", RawFallbackRagClient)

    response = chat.run_chat(ChatRequest(message="任隊長第一次被提到是在哪裡？", include_timing=True))

    steps = [timing.step for timing in response.debug.step_timings]
    assert steps[-1] == "final_generation"
    assert steps.count("search_episode_summaries_embed_query") == 2
    assert steps.count("search_episode_summaries_rag_api") == 2
    assert steps.count("search_original_text_embed_query") == 2
    assert steps.count("search_original_text_rag_api") == 2


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
    assert captured_messages[0][0]["role"] == "system"
    assert captured_messages[0][1]["role"] == "user"
    assert "Summary search results:" in captured_messages[0][1]["content"]
    assert "Linked original text:" in captured_messages[0][1]["content"]
    assert "Do not restate the entire context." in captured_messages[0][1]["content"]
    assert "do not add unrelated scene summaries" in captured_messages[0][1]["content"]


def test_run_chat_records_final_model_input_in_debug(monkeypatch) -> None:
    monkeypatch.setattr(chat, "get_llm_provider", lambda: FakeProvider())
    monkeypatch.setattr(chat, "RagApiClient", FakeRagClient)

    response = chat.run_chat(ChatRequest(message="任隊長第一次被提到是在哪裡？"))

    assert len(response.debug.model_inputs) == 1
    assert response.debug.model_inputs[0].phase == "final_generation"
    assert response.debug.model_inputs[0].tools == []
    assert response.debug.model_inputs[0].messages[0]["role"] == "system"
    assert "Summary search results:" in response.debug.model_inputs[0].messages[-1]["content"]


def test_run_chat_passes_prior_turns_as_real_chat_messages(monkeypatch) -> None:
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

    class GenericRagClient:
        def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload in (
                {"query": "what's my name?", "top_k": 5},
                {"query": "what's my name?", "top_k": 8},
            )
            return {"hits": []}

        def get_linked_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("not expected")

        def search_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload in (
                {"query": "what's my name?", "top_k": 5},
                {"query": "what's my name?", "top_k": 8},
            )
            return {"hits": []}

    monkeypatch.setattr(chat, "get_llm_provider", lambda: ContextProvider())
    monkeypatch.setattr(chat, "RagApiClient", GenericRagClient)

    chat.run_chat(
        ChatRequest(
            message="what's my name?",
            history=[
                {"role": "user", "content": "my name is Jean"},
                {"role": "assistant", "content": "hello Jean"},
            ],
        )
    )

    assert [message["role"] for message in captured_messages[0]] == ["system", "user", "assistant", "user"]
    assert captured_messages[0][1]["content"] == "my name is Jean"
    assert captured_messages[0][2]["content"] == "hello Jean"
    assert "Conversation history:" not in captured_messages[0][3]["content"]
    assert "Current user question:\nwhat's my name?" in captured_messages[0][3]["content"]
    assert "conversation state for follow-up questions in this session" in captured_messages[0][3]["content"]


def test_run_chat_allows_model_to_call_exact_raw_lookup_tool(monkeypatch) -> None:
    class ToolCallingProvider:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, Any]]] = []

        def build_tool_definitions(self) -> list[dict[str, Any]]:
            return [{"type": "function", "function": {"name": "get_raw_paragraph", "parameters": {"type": "object"}}}]

        def complete(
            self,
            messages: list[dict[str, Any]],
            *,
            tools: list[dict[str, Any]] | None = None,
            format_schema: dict[str, Any] | str | None = None,
            think: bool | str | None = None,
        ) -> dict[str, Any]:
            self.calls.append(messages)
            if format_schema is not None:
                return {
                    "message": {
                        "role": "assistant",
                        "content": '{"supported": true, "direct_answer": true, "reason": "raw paragraph directly answers the question", "recommended_next_action": "answer"}',
                    }
                }
            if len(self.calls) == 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_raw_paragraph",
                                    "arguments": {
                                        "chapter_id": "Chapter_16",
                                        "paragraph_id": 18,
                                    },
                                }
                            }
                        ],
                    }
                }
            return {
                "message": {
                    "role": "assistant",
                    "content": "Chapter_16 paragraph 18 說的是指定段落內容。",
                }
            }

    class ExactRawRagClient:
        def get_raw_paragraph(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload == {
                "chapter_id": "Chapter_16",
                "paragraph_id": 18,
            }
            return {
                "hits": [
                    {
                        "chapter_id": "Chapter_16",
                        "paragraph_id": 18,
                        "chunk_id": 0,
                        "original_text": "指定段落內容。",
                        "score": 0.99,
                        "citation": {
                            "raw_chunk_id": "33333333-3333-3333-3333-333333333333",
                            "chapter_id": "Chapter_16",
                            "paragraph_id": 18,
                            "chunk_id": 0,
                            "source_path": "data/sample/raw/Chapter_16.md",
                            "score": 0.99,
                            "citation_type": "raw",
                        },
                    }
                ]
            }

    provider = ToolCallingProvider()
    monkeypatch.setattr(chat, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(chat, "RagApiClient", ExactRawRagClient)

    response = chat.run_chat(ChatRequest(message="Chapter_16 paragraph=18 說了什麼？"))

    assert response.answer == "Chapter_16 paragraph 18 說的是指定段落內容。"
    assert response.citations[0].chapter_id == "Chapter_16"
    assert response.citations[0].paragraph_id == 18
    assert response.debug.tool_calls[0].tool_name == "get_raw_paragraph"
    assert response.debug.tool_calls[0].arguments["chapter_id"] == "Chapter_16"
    assert response.debug.tool_calls[0].arguments["paragraph_id"] == 18


def test_run_chat_records_tool_and_verifier_inputs_in_debug(monkeypatch) -> None:
    class ToolCallingProvider:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, Any]]] = []

        def build_tool_definitions(self) -> list[dict[str, Any]]:
            return [{"type": "function", "function": {"name": "get_raw_paragraph", "parameters": {"type": "object"}}}]

        def complete(
            self,
            messages: list[dict[str, Any]],
            *,
            tools: list[dict[str, Any]] | None = None,
            format_schema: dict[str, Any] | str | None = None,
            think: bool | str | None = None,
        ) -> dict[str, Any]:
            self.calls.append(messages)
            if format_schema is not None:
                return {
                    "message": {
                        "role": "assistant",
                        "content": '{"supported": true, "direct_answer": true, "reason": "raw paragraph directly answers the question", "recommended_next_action": "answer"}',
                    }
                }
            if len(self.calls) == 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_raw_paragraph",
                                    "arguments": {
                                        "chapter_id": "Chapter_16",
                                        "paragraph_id": 18,
                                    },
                                }
                            }
                        ],
                    }
                }
            return {
                "message": {
                    "role": "assistant",
                    "content": "Chapter_16 paragraph 18 說的是指定段落內容。",
                }
            }

    class ExactRawRagClient:
        def get_raw_paragraph(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {
                "hits": [
                    {
                        "chapter_id": "Chapter_16",
                        "paragraph_id": 18,
                        "chunk_id": 0,
                        "original_text": "指定段落內容。",
                        "score": 0.99,
                        "citation": {
                            "raw_chunk_id": "33333333-3333-3333-3333-333333333333",
                            "chapter_id": "Chapter_16",
                            "paragraph_id": 18,
                            "chunk_id": 0,
                            "source_path": "data/sample/raw/Chapter_16.md",
                            "score": 0.99,
                            "citation_type": "raw",
                        },
                    }
                ]
            }

    provider = ToolCallingProvider()
    monkeypatch.setattr(chat, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(chat, "RagApiClient", ExactRawRagClient)

    response = chat.run_chat(ChatRequest(message="Chapter_16 paragraph=18 說了什麼？"))

    assert response.debug.model_inputs[0].phase == "tool_generation"
    assert response.debug.model_inputs[0].tools == ["get_raw_paragraph"]
    assert response.debug.model_inputs[0].messages[-1]["content"] == "Chapter_16 paragraph=18 說了什麼？"


def test_run_chat_normalizes_tool_supplied_chapter_id_for_exact_lookup(monkeypatch) -> None:
    class ToolCallingProvider:
        def __init__(self) -> None:
            self.calls = 0

        def build_tool_definitions(self) -> list[dict[str, Any]]:
            return [{"type": "function", "function": {"name": "get_raw_paragraph", "parameters": {"type": "object"}}}]

        def complete(
            self,
            messages: list[dict[str, Any]],
            *,
            tools: list[dict[str, Any]] | None = None,
            format_schema: dict[str, Any] | str | None = None,
            think: bool | str | None = None,
        ) -> dict[str, Any]:
            if format_schema is not None:
                return {
                    "message": {
                        "role": "assistant",
                        "content": '{"supported": true, "direct_answer": true, "reason": "raw paragraph directly answers the question", "recommended_next_action": "answer"}',
                    }
                }
            self.calls += 1
            if self.calls == 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_raw_paragraph",
                                    "arguments": {
                                        "chapter_id": "Chapter 16",
                                        "paragraph_id": 18,
                                    },
                                }
                            }
                        ],
                    }
                }
            return {
                "message": {
                    "role": "assistant",
                    "content": "Chapter_16 第 18 段說的是指定段落內容。",
                }
            }

    class ExactRawRagClient:
        def get_raw_paragraph(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload == {
                "chapter_id": "Chapter_16",
                "paragraph_id": 18,
            }
            return {
                "hits": [
                    {
                        "chapter_id": "Chapter_16",
                        "paragraph_id": 18,
                        "chunk_id": 0,
                        "original_text": "指定段落內容。",
                        "score": 0.99,
                        "citation": {
                            "raw_chunk_id": "33333333-3333-3333-3333-333333333333",
                            "chapter_id": "Chapter_16",
                            "paragraph_id": 18,
                            "chunk_id": 0,
                            "source_path": "data/sample/raw/Chapter_16.md",
                            "score": 0.99,
                            "citation_type": "raw",
                        },
                    }
                ]
            }

    monkeypatch.setattr(chat, "get_llm_provider", lambda: ToolCallingProvider())
    monkeypatch.setattr(chat, "RagApiClient", ExactRawRagClient)

    response = chat.run_chat(ChatRequest(message="Chapter 16 的第 18 段說了什麼？"))

    assert response.answer == "Chapter_16 第 18 段說的是指定段落內容。"
    assert response.debug.tool_calls[0].arguments["chapter_id"] == "Chapter_16"


def test_run_chat_repairs_wrong_exact_address_tool_choice(monkeypatch) -> None:
    class RepairingProvider:
        def __init__(self) -> None:
            self.normal_calls = 0
            self.verifier_calls = 0

        def build_tool_definitions(self) -> list[dict[str, Any]]:
            return [
                {"type": "function", "function": {"name": "search_episode_summaries", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "get_raw_paragraph", "parameters": {"type": "object"}}},
            ]

        def complete(
            self,
            messages: list[dict[str, Any]],
            *,
            tools: list[dict[str, Any]] | None = None,
            format_schema: dict[str, Any] | str | None = None,
            think: bool | str | None = None,
        ) -> dict[str, Any]:
            if format_schema is not None:
                self.verifier_calls += 1
                if self.verifier_calls == 1:
                    return {
                        "message": {
                            "role": "assistant",
                            "content": '{"supported": false, "direct_answer": false, "reason": "summary evidence is indirect and from the wrong paragraph", "recommended_next_action": "get_raw_paragraph"}',
                        }
                    }
                return {
                    "message": {
                        "role": "assistant",
                        "content": '{"supported": true, "direct_answer": true, "reason": "raw paragraph directly answers the question", "recommended_next_action": "answer"}',
                    }
                }
            self.normal_calls += 1
            if self.normal_calls == 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_episode_summaries",
                                    "arguments": {"query": "Chapter_16 paragraph=18 說了什麼？", "top_k": 5},
                                }
                            }
                        ],
                    }
                }
            if self.normal_calls == 2:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "根據摘要，Chapter_17 的第 21 段提到相關內容。",
                    }
                }
            if self.normal_calls == 3:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_raw_paragraph",
                                    "arguments": {"chapter_id": "Chapter_16", "paragraph_id": 18},
                                }
                            }
                        ],
                    }
                }
            return {
                "message": {
                    "role": "assistant",
                    "content": "Chapter_16 第 18 段說的是指定段落內容。",
                }
            }

    class RepairRagClient:
        def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload == {
                "query": "Chapter_16 paragraph=18 說了什麼？",
                "chapter_id": None,
                "timeline_layer": None,
                "character": None,
                "mentioned_character": None,
                "min_priority_score": None,
                "tags": [],
                "top_k": 5,
            }
            return {
                "hits": [
                    {
                        "id": "11111111-1111-1111-1111-111111111111",
                        "chapter_id": "Chapter_17",
                        "paragraph_id": 21,
                        "scene": "預告",
                        "plot": "提到下一章。",
                        "key_events": ["預告"],
                        "score": 0.6,
                        "citation": {
                            "summary_id": "11111111-1111-1111-1111-111111111111",
                            "chapter_id": "Chapter_17",
                            "paragraph_id": 21,
                            "source_path": "data/sample/summaries/Chapter_17.md",
                            "score": 0.6,
                            "citation_type": "summary",
                        },
                    }
                ]
            }

        def get_raw_paragraph(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload == {"chapter_id": "Chapter_16", "paragraph_id": 18}
            return {
                "hits": [
                    {
                        "chapter_id": "Chapter_16",
                        "paragraph_id": 18,
                        "chunk_id": 0,
                        "original_text": "指定段落內容。",
                        "score": 1.0,
                        "citation": {
                            "raw_chunk_id": "22222222-2222-2222-2222-222222222222",
                            "chapter_id": "Chapter_16",
                            "paragraph_id": 18,
                            "chunk_id": 0,
                            "source_path": "data/sample/raw/Chapter_16.md",
                            "score": 1.0,
                            "citation_type": "raw",
                        },
                    }
                ]
            }

    monkeypatch.setattr(chat, "get_llm_provider", lambda: RepairingProvider())
    monkeypatch.setattr(chat, "RagApiClient", RepairRagClient)

    response = chat.run_chat(ChatRequest(message="Chapter_16 paragraph=18 說了什麼？"))

    assert response.answer == "Chapter_16 第 18 段說的是指定段落內容。"
    assert response.citations[0].chapter_id == "Chapter_16"
    assert response.citations[0].paragraph_id == 18
    assert [call.tool_name for call in response.debug.tool_calls] == [
        "search_episode_summaries",
        "get_raw_paragraph",
    ]


def test_run_chat_accepts_answer_after_successful_exact_lookup_without_verifier(monkeypatch) -> None:
    class ToolCallingProvider:
        def __init__(self) -> None:
            self.normal_calls = 0
            self.verifier_calls = 0

        def build_tool_definitions(self) -> list[dict[str, Any]]:
            return [{"type": "function", "function": {"name": "get_raw_paragraph", "parameters": {"type": "object"}}}]

        def complete(
            self,
            messages: list[dict[str, Any]],
            *,
            tools: list[dict[str, Any]] | None = None,
            format_schema: dict[str, Any] | str | None = None,
            think: bool | str | None = None,
        ) -> dict[str, Any]:
            if format_schema is not None:
                self.verifier_calls += 1
                return {
                    "message": {
                        "role": "assistant",
                        "content": '{"supported": false, "direct_answer": false, "reason": "should not run", "recommended_next_action": "search_original_text"}',
                    }
                }
            self.normal_calls += 1
            if self.normal_calls == 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_raw_paragraph",
                                    "arguments": {"chapter_id": "Chapter 16", "paragraph_id": 18},
                                }
                            }
                        ],
                    }
                }
            return {
                "message": {
                    "role": "assistant",
                    "content": "Chapter_16 第 18 段說的是指定段落內容。",
                }
            }

    class ExactRawRagClient:
        def get_raw_paragraph(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload == {"chapter_id": "Chapter_16", "paragraph_id": 18}
            return {
                "hits": [
                    {
                        "chapter_id": "Chapter_16",
                        "paragraph_id": 18,
                        "chunk_id": 0,
                        "original_text": "指定段落內容。",
                        "score": 1.0,
                        "citation": {
                            "raw_chunk_id": "55555555-5555-5555-5555-555555555555",
                            "chapter_id": "Chapter_16",
                            "paragraph_id": 18,
                            "chunk_id": 0,
                            "source_path": "data/sample/raw/Chapter_16.md",
                            "score": 1.0,
                            "citation_type": "raw",
                        },
                    }
                ]
            }

    provider = ToolCallingProvider()
    monkeypatch.setattr(chat, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(chat, "RagApiClient", ExactRawRagClient)

    response = chat.run_chat(ChatRequest(message="Chapter 16 的第 18 段說了什麼？"))

    assert response.answer == "Chapter_16 第 18 段說的是指定段落內容。"
    assert provider.verifier_calls == 0
    assert response.citations[0].chapter_id == "Chapter_16"


def test_run_chat_accepts_answer_after_successful_chapter_scoped_lookup_without_verifier(monkeypatch) -> None:
    class ToolCallingProvider:
        def __init__(self) -> None:
            self.normal_calls = 0
            self.verifier_calls = 0

        def build_tool_definitions(self) -> list[dict[str, Any]]:
            return [
                {"type": "function", "function": {"name": "search_episode_summaries", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "search_original_text", "parameters": {"type": "object"}}},
            ]

        def complete(
            self,
            messages: list[dict[str, Any]],
            *,
            tools: list[dict[str, Any]] | None = None,
            format_schema: dict[str, Any] | str | None = None,
            think: bool | str | None = None,
        ) -> dict[str, Any]:
            if format_schema is not None:
                self.verifier_calls += 1
                return {
                    "message": {
                        "role": "assistant",
                        "content": '{"supported": false, "direct_answer": false, "reason": "should not run", "recommended_next_action": "search_original_text"}',
                    }
                }
            self.normal_calls += 1
            if self.normal_calls == 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_episode_summaries",
                                    "arguments": {"chapter_id": "Chapter 46", "query": "Chapter 46 內容摘要", "top_k": 1},
                                }
                            },
                            {
                                "function": {
                                    "name": "search_original_text",
                                    "arguments": {"chapter_id": "Chapter 46", "query": "Chapter 46 內容摘要", "top_k": 3},
                                }
                            },
                        ],
                    }
                }
            return {
                "message": {
                    "role": "assistant",
                    "content": "Chapter 46 描述了指定章節的主要事件。",
                }
            }

    class ChapterScopedRagClient:
        def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload == {
                "query": "Chapter 46 內容摘要",
                "chapter_id": "Chapter_46",
                "timeline_layer": None,
                "character": None,
                "mentioned_character": None,
                "min_priority_score": None,
                "tags": [],
                "top_k": 1,
            }
            return {
                "hits": [
                    {
                        "id": "33333333-3333-3333-3333-333333333333",
                        "chapter_id": "Chapter_46",
                        "paragraph_id": 18,
                        "scene": "章節摘要",
                        "plot": "主要事件。",
                        "key_events": ["事件"],
                        "score": 0.47,
                        "citation": {
                            "summary_id": "33333333-3333-3333-3333-333333333333",
                            "chapter_id": "Chapter_46",
                            "paragraph_id": 18,
                            "source_path": "data/sample/summaries/Chapter_46.md",
                            "score": 0.47,
                            "citation_type": "summary",
                        },
                    }
                ]
            }

        def search_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload == {
                "query": "Chapter 46 內容摘要",
                "chapter_id": "Chapter_46",
                "paragraph_id": None,
                "tags": [],
                "top_k": 3,
            }
            return {
                "hits": [
                    {
                        "chapter_id": "Chapter_46",
                        "paragraph_id": 18,
                        "chunk_id": 0,
                        "original_text": "章節中的主要事件。",
                        "score": 0.71,
                        "citation": {
                            "raw_chunk_id": "66666666-6666-6666-6666-666666666666",
                            "chapter_id": "Chapter_46",
                            "paragraph_id": 18,
                            "chunk_id": 0,
                            "source_path": "data/sample/raw/Chapter_46.md",
                            "score": 0.71,
                            "citation_type": "raw",
                        },
                    }
                ]
            }

    provider = ToolCallingProvider()
    monkeypatch.setattr(chat, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(chat, "RagApiClient", ChapterScopedRagClient)

    response = chat.run_chat(ChatRequest(message="Chapter 46 說了什麼？"))

    assert response.answer == "Chapter 46 描述了指定章節的主要事件。"
    assert provider.verifier_calls == 0
    assert response.debug.tool_calls[0].arguments["chapter_id"] == "Chapter_46"
    assert response.debug.tool_calls[1].arguments["chapter_id"] == "Chapter_46"


def test_run_chat_normalizes_tool_supplied_chapter_id_for_raw_search(monkeypatch) -> None:
    class ToolCallingProvider:
        def __init__(self) -> None:
            self.calls = 0

        def build_tool_definitions(self) -> list[dict[str, Any]]:
            return [{"type": "function", "function": {"name": "search_original_text", "parameters": {"type": "object"}}}]

        def complete(
            self,
            messages: list[dict[str, Any]],
            *,
            tools: list[dict[str, Any]] | None = None,
            format_schema: dict[str, Any] | str | None = None,
            think: bool | str | None = None,
        ) -> dict[str, Any]:
            if format_schema is not None:
                return {
                    "message": {
                        "role": "assistant",
                        "content": '{"supported": true, "direct_answer": true, "reason": "raw search directly answers the question", "recommended_next_action": "answer"}',
                    }
                }
            self.calls += 1
            if self.calls == 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_original_text",
                                    "arguments": {
                                        "chapter_id": "Chapter 16",
                                        "paragraph_id": 18,
                                        "query": "第 18 段內容",
                                        "top_k": 1,
                                    },
                                }
                            }
                        ],
                    }
                }
            return {
                "message": {
                    "role": "assistant",
                    "content": "Chapter_16 第 18 段說的是指定段落內容。",
                }
            }

    class RawSearchRagClient:
        def search_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload == {
                "query": "第 18 段內容",
                "chapter_id": "Chapter_16",
                "paragraph_id": 18,
                "tags": [],
                "top_k": 1,
            }
            return {
                "hits": [
                    {
                        "chapter_id": "Chapter_16",
                        "paragraph_id": 18,
                        "chunk_id": 0,
                        "original_text": "指定段落內容。",
                        "score": 0.9,
                        "citation": {
                            "raw_chunk_id": "44444444-4444-4444-4444-444444444444",
                            "chapter_id": "Chapter_16",
                            "paragraph_id": 18,
                            "chunk_id": 0,
                            "source_path": "data/sample/raw/Chapter_16.md",
                            "score": 0.9,
                            "citation_type": "raw",
                        },
                    }
                ]
            }

    monkeypatch.setattr(chat, "get_llm_provider", lambda: ToolCallingProvider())
    monkeypatch.setattr(chat, "RagApiClient", RawSearchRagClient)

    response = chat.run_chat(ChatRequest(message="Chapter 16 的第 18 段說了什麼？"))

    assert response.answer == "Chapter_16 第 18 段說的是指定段落內容。"
    assert response.debug.tool_calls[0].tool_name == "search_original_text"
    assert response.debug.tool_calls[0].arguments["chapter_id"] == "Chapter_16"


def test_parse_verification_result_tolerates_fenced_json_with_trailing_text() -> None:
    result = chat._parse_verification_result(
        """```json
{
  "supported": true,
  "direct_answer": true,
  "reason": "raw paragraph directly answers the question",
  "recommended_next_action": "answer"
}
```
extra explanation that should be ignored
"""
    )

    assert result.supported is True
    assert result.direct_answer is True
    assert result.recommended_next_action == "answer"


def test_parse_verification_result_falls_back_on_incomplete_json() -> None:
    result = chat._parse_verification_result("```json\n{\"supported\": false}\n```\nextra text")

    assert result.supported is False
    assert result.direct_answer is False
    assert result.reason == "verifier returned incomplete JSON"
    assert result.recommended_next_action == "search_original_text"


def test_run_chat_broadens_search_when_initial_evidence_is_thin(monkeypatch) -> None:
    class ThinFirstPassRagClient:
        def __init__(self) -> None:
            self.summary_top_ks: list[int] = []
            self.raw_top_ks: list[int] = []
            self.linked_top_ks: list[int] = []

        def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.summary_top_ks.append(payload["top_k"])
            if payload["top_k"] == 5:
                return {
                    "hits": [
                        {
                            "id": "11111111-1111-1111-1111-111111111111",
                            "chapter_id": "episode_01",
                            "paragraph_id": 1,
                            "scene": "初次登場",
                            "plot": "角色被提到。",
                            "key_events": ["角色被提到"],
                            "score": 0.6,
                            "citation": {
                                "summary_id": "11111111-1111-1111-1111-111111111111",
                                "chapter_id": "episode_01",
                                "paragraph_id": 1,
                                "source_path": "data/sample/summaries/episode_01.md",
                                "score": 0.6,
                                "citation_type": "summary",
                            },
                        }
                    ]
                }
            return {
                "hits": [
                    {
                        "id": "11111111-1111-1111-1111-111111111111",
                        "chapter_id": "episode_01",
                        "paragraph_id": 1,
                        "scene": "初次登場",
                        "plot": "角色被提到。",
                        "key_events": ["角色被提到"],
                        "score": 0.6,
                        "citation": {
                            "summary_id": "11111111-1111-1111-1111-111111111111",
                            "chapter_id": "episode_01",
                            "paragraph_id": 1,
                            "source_path": "data/sample/summaries/episode_01.md",
                            "score": 0.6,
                            "citation_type": "summary",
                        },
                    },
                    {
                        "id": "22222222-2222-2222-2222-222222222222",
                        "chapter_id": "episode_03",
                        "paragraph_id": 5,
                        "scene": "補充資料",
                        "plot": "角色職業被明確提到。",
                        "key_events": ["角色職業"],
                        "score": 0.58,
                        "citation": {
                            "summary_id": "22222222-2222-2222-2222-222222222222",
                            "chapter_id": "episode_03",
                            "paragraph_id": 5,
                            "source_path": "data/sample/summaries/episode_03.md",
                            "score": 0.58,
                            "citation_type": "summary",
                        },
                    },
                ]
            }

        def get_linked_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.linked_top_ks.append(payload["top_k_per_hit"])
            if payload["top_k_per_hit"] == 2:
                return {
                    "hits": [
                        {
                            "chapter_id": "episode_01",
                            "paragraph_id": 1,
                            "chunk_id": 0,
                            "original_text": "只提到角色名字。",
                            "score": 0.55,
                            "citation": {
                                "summary_id": "11111111-1111-1111-1111-111111111111",
                                "raw_chunk_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                                "chapter_id": "episode_01",
                                "paragraph_id": 1,
                                "chunk_id": 0,
                                "source_path": "data/sample/raw/episode_01.md",
                                "score": 0.55,
                                "citation_type": "raw",
                            },
                        }
                    ]
                }
            return {
                "hits": [
                    {
                        "chapter_id": "episode_01",
                        "paragraph_id": 1,
                        "chunk_id": 0,
                        "original_text": "只提到角色名字。",
                        "score": 0.55,
                        "citation": {
                            "summary_id": "11111111-1111-1111-1111-111111111111",
                            "raw_chunk_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                            "chapter_id": "episode_01",
                            "paragraph_id": 1,
                            "chunk_id": 0,
                            "source_path": "data/sample/raw/episode_01.md",
                            "score": 0.55,
                            "citation_type": "raw",
                        },
                    },
                    {
                        "chapter_id": "episode_03",
                        "paragraph_id": 5,
                        "chunk_id": 1,
                        "original_text": "他的職業是醫師。",
                        "score": 0.91,
                        "citation": {
                            "summary_id": "22222222-2222-2222-2222-222222222222",
                            "raw_chunk_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                            "chapter_id": "episode_03",
                            "paragraph_id": 5,
                            "chunk_id": 1,
                            "source_path": "data/sample/raw/episode_03.md",
                            "score": 0.91,
                            "citation_type": "raw",
                        },
                    },
                ]
            }

        def search_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.raw_top_ks.append(payload["top_k"])
            return {"hits": []}

    client = ThinFirstPassRagClient()
    monkeypatch.setattr(chat, "get_llm_provider", lambda: FakeProvider())
    monkeypatch.setattr(chat, "RagApiClient", lambda: client)

    response = chat.run_chat(ChatRequest(message="character_A 的職業是什麼？"))

    assert client.summary_top_ks == [5, 8]
    assert client.linked_top_ks == [2, 3]
    assert client.raw_top_ks == [8]
    assert any(call.arguments.get("top_k") == 8 for call in response.debug.tool_calls if call.tool_name == "search_episode_summaries")
    assert any(call.arguments.get("top_k_per_hit") == 3 for call in response.debug.tool_calls if call.tool_name == "get_linked_original_text")


def test_run_chat_trims_citations_to_top_ranked_evidence(monkeypatch) -> None:
    class NoisyRagClient(FakeRagClient):
        def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
            result = super().search_summaries(payload)
            result["hits"].append(
                {
                    "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "chapter_id": "episode_02",
                    "paragraph_id": 1,
                    "scene": "會議後",
                    "plot": "較弱的次要線索。",
                    "key_events": ["旁證"],
                    "score": 0.42,
                    "citation": {
                        "summary_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                        "chapter_id": "episode_02",
                        "paragraph_id": 1,
                        "source_path": "data/sample/summaries/episode_02.md",
                        "score": 0.42,
                        "citation_type": "summary",
                    },
                }
            )
            return result

        def get_linked_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload == {
                "summary_hit_ids": [
                    "11111111-1111-1111-1111-111111111111",
                    "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                ],
                "top_k_per_hit": 2,
            }
            result = {
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
            result["hits"].extend(
                [
                    {
                        "chapter_id": "episode_01",
                        "paragraph_id": 2,
                        "chunk_id": 1,
                        "original_text": "次要段落。",
                        "score": 0.41,
                        "citation": {
                            "summary_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                            "raw_chunk_id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
                            "chapter_id": "episode_01",
                            "paragraph_id": 2,
                            "chunk_id": 1,
                            "source_path": "data/sample/raw/episode_01.md",
                            "score": 0.41,
                            "citation_type": "raw",
                        },
                    },
                    {
                        "chapter_id": "episode_02",
                        "paragraph_id": 1,
                        "chunk_id": 0,
                        "original_text": "更弱的段落。",
                        "score": 0.2,
                        "citation": {
                            "summary_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                            "raw_chunk_id": "dddddddd-dddd-dddd-dddd-dddddddddddd",
                            "chapter_id": "episode_02",
                            "paragraph_id": 1,
                            "chunk_id": 0,
                            "source_path": "data/sample/raw/episode_02.md",
                            "score": 0.2,
                            "citation_type": "raw",
                        },
                    },
                ]
            )
            return result

    monkeypatch.setattr(chat, "get_llm_provider", lambda: FakeProvider())
    monkeypatch.setattr(chat, "RagApiClient", NoisyRagClient)

    response = chat.run_chat(ChatRequest(message="任隊長第一次被提到是在哪裡？"))

    assert len(response.citations) == 2
    assert [citation.citation_type for citation in response.citations] == ["raw", "summary"]
    assert response.citations[0].score == 0.88
    assert {citation.chapter_id for citation in response.citations} == {"episode_01"}


def test_run_chat_uses_primary_summary_cluster_for_final_context(monkeypatch) -> None:
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

    class MultiSummaryRagClient(FakeRagClient):
        def search_summaries(self, payload: dict[str, Any]) -> dict[str, Any]:
            result = super().search_summaries(payload)
            result["hits"].append(
                {
                    "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "chapter_id": "episode_02",
                    "paragraph_id": 1,
                    "scene": "會議後",
                    "plot": "次要線索。",
                    "key_events": ["旁證"],
                    "score": 0.42,
                    "citation": {
                        "summary_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                        "chapter_id": "episode_02",
                        "paragraph_id": 1,
                        "source_path": "data/sample/summaries/episode_02.md",
                        "score": 0.42,
                        "citation_type": "summary",
                    },
                }
            )
            return result

        def get_linked_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload["summary_hit_ids"] == [
                "11111111-1111-1111-1111-111111111111",
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            ]
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
                    },
                    {
                        "chapter_id": "episode_02",
                        "paragraph_id": 1,
                        "chunk_id": 0,
                        "original_text": "這段不該進 final context。",
                        "score": 0.2,
                        "citation": {
                            "summary_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                            "raw_chunk_id": "dddddddd-dddd-dddd-dddd-dddddddddddd",
                            "chapter_id": "episode_02",
                            "paragraph_id": 1,
                            "chunk_id": 0,
                            "source_path": "data/sample/raw/episode_02.md",
                            "score": 0.2,
                            "citation_type": "raw",
                        },
                    },
                ]
            }

    monkeypatch.setattr(chat, "get_llm_provider", lambda: ContextProvider())
    monkeypatch.setattr(chat, "RagApiClient", MultiSummaryRagClient)

    chat.run_chat(ChatRequest(message="任隊長第一次被提到是在哪裡？"))

    prompt = captured_messages[0][-1]["content"]
    assert "chapter=episode_01 paragraph=1" in prompt
    assert "這段不該進 final context" not in prompt


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
