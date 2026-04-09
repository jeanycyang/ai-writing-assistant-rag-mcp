import json

from services.codex_mcp.server import WritingAssistantMcpServer


class FakeRagClient:
    def search_summary_characters(self, payload):
        assert payload == {
            "characters": ["Character Alpha", "Character Beta"],
            "operator": "or",
            "chapter_id": "Chapter_16",
            "from_chapter": 1,
            "to_chapter": 40,
            "min_priority_score": 0.8,
            "top_k": 3,
        }
        return {
            "hits": [
                {
                    "id": "33333333-3333-3333-3333-333333333333",
                    "chapter_id": "Chapter_16",
                    "paragraph_id": 5,
                    "characters": ["Character Alpha"],
                    "plot": "Character Alpha appears in a key scene.",
                    "citation": {
                        "summary_id": "33333333-3333-3333-3333-333333333333",
                        "chapter_id": "Chapter_16",
                        "paragraph_id": 5,
                        "source_path": "data/summary.md",
                        "citation_type": "summary",
                    },
                }
            ]
        }

    def search_summaries(self, payload):
        assert payload["query"] == "任隊長第一次被提到是在哪裡？"
        assert payload["chapter_id"] == "Chapter_16"
        return {
            "hits": [
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "chapter_id": "Chapter_16",
                    "paragraph_id": 3,
                    "plot": "任隊長先被提到。",
                    "citation": {
                        "summary_id": "11111111-1111-1111-1111-111111111111",
                        "chapter_id": "Chapter_16",
                        "paragraph_id": 3,
                        "source_path": "data/summary.md",
                        "citation_type": "summary",
                    },
                }
            ]
        }

    def get_linked_raw(self, payload):
        assert payload["top_k_per_hit"] == 2
        return {
            "hits": [
                {
                    "chapter_id": "Chapter_16",
                    "paragraph_id": 3,
                    "original_text": "有人先提到任隊長。",
                    "citation": {
                        "raw_chunk_id": "22222222-2222-2222-2222-222222222222",
                        "chapter_id": "Chapter_16",
                        "paragraph_id": 3,
                        "chunk_id": 0,
                        "source_path": "data/raw.md",
                        "citation_type": "raw",
                    },
                }
            ]
        }

    def search_raw(self, payload):
        return {"hits": []}

    def get_summary_paragraph(self, payload):
        return {"hits": [payload]}

    def get_raw_paragraph(self, payload):
        return {"hits": [payload]}

    def get_summary_chapter(self, payload):
        return {"chapter_id": payload["chapter_id"], "paragraphs": [], "full_summary_text": ""}

    def get_raw_chapter(self, payload):
        return {"chapter_id": payload["chapter_id"], "paragraphs": [], "full_text": ""}


def test_tools_list_exposes_writing_oriented_tools() -> None:
    server = WritingAssistantMcpServer(rag_client=FakeRagClient())

    response = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    assert response is not None
    tool_names = [tool["name"] for tool in response["result"]["tools"]]
    assert tool_names == [
        "writing_lookup",
        "search_summary_by_characters",
        "get_summary_paragraph",
        "get_raw_paragraph",
        "get_chapter_summary",
        "get_chapter_text",
    ]


def test_writing_lookup_orchestrates_summary_first_retrieval() -> None:
    server = WritingAssistantMcpServer(rag_client=FakeRagClient())

    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "writing_lookup",
                "arguments": {
                    "question": "任隊長第一次被提到是在哪裡？",
                    "chapter_id": "Chapter 16",
                },
            },
        }
    )

    assert response is not None
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["chapter_id"] == "Chapter_16"
    assert payload["confidence"] == "high"
    assert payload["suggested_next_step"] == "answer_from_evidence"
    assert len(payload["summary_hits"]) == 1
    assert len(payload["raw_hits"]) == 1


def test_get_chapter_text_normalizes_chapter_id() -> None:
    server = WritingAssistantMcpServer(rag_client=FakeRagClient())

    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "get_chapter_text", "arguments": {"chapter_id": "Chapter 16"}},
        }
    )

    assert response is not None
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["chapter_id"] == "Chapter_16"


def test_search_summary_by_characters_normalizes_chapter_id() -> None:
    server = WritingAssistantMcpServer(rag_client=FakeRagClient())

    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "search_summary_by_characters",
                "arguments": {
                    "characters": ["Character Alpha", "Character Beta"],
                    "operator": "or",
                    "chapter_id": "Chapter 16",
                    "from_chapter": 1,
                    "to_chapter": 40,
                    "min_priority_score": 0.8,
                    "top_k": 3,
                },
            },
        }
    )

    assert response is not None
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["hits"][0]["chapter_id"] == "Chapter_16"
