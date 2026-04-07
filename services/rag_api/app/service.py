from __future__ import annotations

from sqlalchemy.orm import Session

from shared.embeddings import get_embedding_provider
from shared.repository import RagRepository
from shared.schemas import (
    Citation,
    LinkedRawResponse,
    RawHit,
    RawSearchRequest,
    RawSearchResponse,
    SummaryHit,
    SummarySearchRequest,
    SummarySearchResponse,
)


def _build_summary_hit(row, score: float) -> SummaryHit:
    citation = Citation(
        summary_id=row.id,
        chapter_id=row.chapter_id,
        paragraph_id=row.paragraph_id,
        source_path=row.source_path,
        score=score,
        citation_type="summary",
    )
    return SummaryHit(
        id=row.id,
        chapter_id=row.chapter_id,
        paragraph_id=row.paragraph_id,
        priority_score=row.priority_score,
        timeline_layer=row.timeline_layer,
        scene=row.scene,
        characters=row.characters,
        mentioned_characters=row.mentioned_characters,
        tags=row.tags,
        key_events=row.key_events,
        plot=row.plot,
        source_path=row.source_path,
        score=score,
        citation=citation,
    )


def _build_raw_hit(row, score: float) -> RawHit:
    citation = Citation(
        raw_chunk_id=row.id,
        summary_id=row.linked_summary_id,
        chapter_id=row.chapter_id,
        paragraph_id=row.paragraph_id,
        chunk_id=row.chunk_id,
        source_path=row.source_path,
        score=score,
        citation_type="raw",
    )
    return RawHit(
        id=row.id,
        chapter_id=row.chapter_id,
        paragraph_id=row.paragraph_id,
        chunk_id=row.chunk_id,
        original_text=row.original_text,
        source_path=row.source_path,
        linked_summary_id=row.linked_summary_id,
        score=score,
        citation=citation,
    )


def search_summaries(session: Session, request: SummarySearchRequest) -> SummarySearchResponse:
    embedding_provider = get_embedding_provider()
    query_embedding = embedding_provider.embed_text(request.query)
    repo = RagRepository(session)
    rows = repo.search_summaries(
        query_embedding,
        query=request.query,
        chapter_id=request.chapter_id,
        timeline_layer=request.timeline_layer,
        character=request.character,
        mentioned_character=request.mentioned_character,
        min_priority_score=request.min_priority_score,
        tags=request.tags,
        top_k=request.top_k,
    )
    return SummarySearchResponse(hits=[_build_summary_hit(row, score) for row, score in rows])


def search_raw(session: Session, request: RawSearchRequest) -> RawSearchResponse:
    embedding_provider = get_embedding_provider()
    query_embedding = embedding_provider.embed_text(request.query)
    repo = RagRepository(session)
    rows = repo.search_raw(
        query_embedding,
        chapter_id=request.chapter_id,
        paragraph_id=request.paragraph_id,
        top_k=request.top_k,
    )
    return RawSearchResponse(hits=[_build_raw_hit(row, score) for row, score in rows])


def get_linked_raw(session: Session, summary_hit_ids, top_k_per_hit: int) -> LinkedRawResponse:
    repo = RagRepository(session)
    rows = repo.get_linked_raw(summary_hit_ids, top_k_per_hit)
    return LinkedRawResponse(hits=[_build_raw_hit(row, score) for row, score in rows])
