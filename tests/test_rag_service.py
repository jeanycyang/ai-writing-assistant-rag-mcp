from types import SimpleNamespace
from uuid import uuid4

from services.rag_api.app import service
from shared.schemas import RawParagraphRequest, RawSearchRequest, SummaryParagraphRequest, SummarySearchRequest


class FakeRepo:
    def __init__(self, session):
        self.session = session

    def search_summaries(self, query_embedding, **kwargs):
        assert query_embedding == [0.1, 0.2]
        assert kwargs["query"] == "查詢摘要"
        row = SimpleNamespace(
            id=uuid4(),
            chapter_id="episode_01",
            paragraph_id=1,
            priority_score=0.9,
            timeline_layer="鋪陳",
            scene="走廊",
            characters=["林妍"],
            mentioned_characters=["任隊長"],
            tags=["傳聞"],
            key_events=["提到任隊長"],
            plot="任隊長尚未現身前先被提到。",
            source_path="data/sample/summaries/episode_01.md",
        )
        return [(row, 0.88)]

    def search_raw(self, query_embedding, **kwargs):
        assert query_embedding == [0.3, 0.4]
        row = SimpleNamespace(
            id=uuid4(),
            chapter_id="episode_01",
            paragraph_id=2,
            chunk_id=1,
            original_text="任隊長提醒林妍報告被刪改。",
            source_path="data/sample/raw/episode_01.md",
            linked_summary_id=None,
        )
        return [(row, 0.77)]

    def get_summary_paragraph(self, **kwargs):
        assert kwargs == {"chapter_id": "episode_01", "paragraph_id": 1}
        row = SimpleNamespace(
            id=uuid4(),
            chapter_id="episode_01",
            paragraph_id=1,
            priority_score=0.9,
            timeline_layer="鋪陳",
            scene="走廊",
            characters=["林妍"],
            mentioned_characters=["任隊長"],
            tags=["傳聞"],
            key_events=["提到任隊長"],
            plot="任隊長尚未現身前先被提到。",
            source_path="data/sample/summaries/episode_01.md",
        )
        return [(row, 1.0)]

    def get_raw_paragraph(self, **kwargs):
        assert kwargs == {"chapter_id": "episode_01", "paragraph_id": 2}
        row = SimpleNamespace(
            id=uuid4(),
            chapter_id="episode_01",
            paragraph_id=2,
            chunk_id=1,
            original_text="任隊長提醒林妍報告被刪改。",
            source_path="data/sample/raw/episode_01.md",
            linked_summary_id=None,
        )
        return [(row, 1.0)]


def test_search_summaries_uses_supplied_query_embedding(monkeypatch):
    monkeypatch.setattr(service, "RagRepository", FakeRepo)
    response = service.search_summaries(
        None,
        SummarySearchRequest(query="查詢摘要", query_embedding=[0.1, 0.2], top_k=1),
    )
    assert response.hits[0].chapter_id == "episode_01"


def test_search_raw_uses_supplied_query_embedding(monkeypatch):
    monkeypatch.setattr(service, "RagRepository", FakeRepo)
    response = service.search_raw(
        None,
        RawSearchRequest(query="查詢原文", query_embedding=[0.3, 0.4], top_k=1),
    )
    assert response.hits[0].chunk_id == 1


def test_get_summary_paragraph_uses_exact_metadata_lookup(monkeypatch):
    monkeypatch.setattr(service, "RagRepository", FakeRepo)
    response = service.get_summary_paragraph(
        None,
        SummaryParagraphRequest(chapter_id="episode_01", paragraph_id=1),
    )
    assert response.hits[0].paragraph_id == 1


def test_get_raw_paragraph_uses_exact_metadata_lookup(monkeypatch):
    monkeypatch.setattr(service, "RagRepository", FakeRepo)
    response = service.get_raw_paragraph(
        None,
        RawParagraphRequest(chapter_id="episode_01", paragraph_id=2),
    )
    assert response.hits[0].chunk_id == 1
