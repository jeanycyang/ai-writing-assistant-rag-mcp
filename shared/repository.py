from __future__ import annotations

import re
from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from shared.models import RawChunk, SummaryChunk
from shared.parsing import ParsedRawChunkRecord, ParsedSummaryRecord

QUERY_FLAG_HINTS = {
    "mention": ("提到", "提及", "聽見", "傳聞", "mentioned"),
    "not_present": ("沒出現", "未出現", "尚未現身", "不在場", "還沒出現", "未登場", "not yet present"),
    "first": ("第一次", "最早", "先", "首先", "first"),
}


def _summary_query_filters(
    stmt: Select[tuple[SummaryChunk]],
    *,
    chapter_id: str | None,
    timeline_layer: str | None,
    character: str | None,
    mentioned_character: str | None,
    min_priority_score: float | None,
    tags: list[str],
) -> Select[tuple[SummaryChunk]]:
    if chapter_id:
        stmt = stmt.where(SummaryChunk.chapter_id == chapter_id)
    if timeline_layer:
        stmt = stmt.where(SummaryChunk.timeline_layer == timeline_layer)
    if character:
        stmt = stmt.where(SummaryChunk.characters.contains([character]))
    if mentioned_character:
        stmt = stmt.where(SummaryChunk.mentioned_characters.contains([mentioned_character]))
    if min_priority_score is not None:
        stmt = stmt.where(SummaryChunk.priority_score >= min_priority_score)
    if tags:
        stmt = stmt.where(SummaryChunk.tags.overlap(tags))
    return stmt


def _normalize_query_text(query: str) -> str:
    return re.sub(r"\s+", "", query)


def _query_has_hint(query: str, hint_group: str) -> bool:
    normalized = _normalize_query_text(query)
    return any(hint in normalized for hint in QUERY_FLAG_HINTS[hint_group])


def _candidate_text(row: SummaryChunk) -> str:
    return " ".join(
        [
            row.scene,
            row.plot,
            " ".join(row.characters),
            " ".join(row.mentioned_characters),
            " ".join(row.tags),
            " ".join(row.key_events),
        ]
    )


def _rerank_summary_score(query: str, row: SummaryChunk, semantic_score: float) -> float:
    score = semantic_score
    normalized_query = _normalize_query_text(query)
    candidate_text = _candidate_text(row)
    mentioned_not_present = [item for item in row.mentioned_characters if item not in row.characters]

    if _query_has_hint(query, "mention") and mentioned_not_present:
        score += 0.03
    if _query_has_hint(query, "not_present") and mentioned_not_present:
        score += 0.08
    if _query_has_hint(query, "first") and mentioned_not_present:
        score += 0.04
        score += 0.02 / max(row.paragraph_id, 1)

    if _query_has_hint(query, "not_present") and any(
        phrase in candidate_text for phrase in ("尚未現身", "未現身", "未出現", "未登場")
    ):
        score += 0.06

    for name in mentioned_not_present:
        if name and name in normalized_query:
            score += 0.06
    for name in row.characters:
        if name and name in normalized_query:
            score += 0.02

    return score


class RagRepository:
    def __init__(self, session: Session):
        self.session = session

    def upsert_summary_chunks(self, records: list[ParsedSummaryRecord], embeddings: list[list[float]]) -> None:
        values = []
        for record, embedding in zip(records, embeddings, strict=True):
            values.append(
                {
                    "external_id": record.external_id,
                    "chapter_id": record.chapter_id,
                    "paragraph_id": record.paragraph_id,
                    "priority_score": record.priority_score,
                    "timeline_layer": record.timeline_layer,
                    "scene": record.scene,
                    "characters": record.characters,
                    "mentioned_characters": record.mentioned_characters,
                    "tags": record.tags,
                    "key_events": record.key_events,
                    "plot": record.plot,
                    "embedding_text": record.embedding_text,
                    "source_path": record.source_path,
                    "source_hash": record.source_hash,
                    "embedding": embedding,
                }
            )
        stmt = insert(SummaryChunk).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[SummaryChunk.external_id],
            set_={
                "chapter_id": stmt.excluded.chapter_id,
                "paragraph_id": stmt.excluded.paragraph_id,
                "priority_score": stmt.excluded.priority_score,
                "timeline_layer": stmt.excluded.timeline_layer,
                "scene": stmt.excluded.scene,
                "characters": stmt.excluded.characters,
                "mentioned_characters": stmt.excluded.mentioned_characters,
                "tags": stmt.excluded.tags,
                "key_events": stmt.excluded.key_events,
                "plot": stmt.excluded.plot,
                "embedding_text": stmt.excluded.embedding_text,
                "source_path": stmt.excluded.source_path,
                "source_hash": stmt.excluded.source_hash,
                "embedding": stmt.excluded.embedding,
            },
        )
        self.session.execute(stmt)
        self.session.flush()

    def upsert_raw_chunks(self, records: list[ParsedRawChunkRecord], embeddings: list[list[float]]) -> None:
        summary_rows = self.session.execute(select(SummaryChunk)).scalars().all()
        summary_lookup = {(row.chapter_id, row.paragraph_id): row.id for row in summary_rows}

        values = []
        for record, embedding in zip(records, embeddings, strict=True):
            values.append(
                {
                    "external_id": record.external_id,
                    "chapter_id": record.chapter_id,
                    "paragraph_id": record.paragraph_id,
                    "chunk_id": record.chunk_id,
                    "original_text": record.original_text,
                    "embedding_text": record.embedding_text,
                    "source_path": record.source_path,
                    "source_hash": record.source_hash,
                    "embedding": embedding,
                    "linked_summary_id": summary_lookup.get((record.chapter_id, record.paragraph_id)),
                }
            )
        stmt = insert(RawChunk).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[RawChunk.external_id],
            set_={
                "chapter_id": stmt.excluded.chapter_id,
                "paragraph_id": stmt.excluded.paragraph_id,
                "chunk_id": stmt.excluded.chunk_id,
                "original_text": stmt.excluded.original_text,
                "embedding_text": stmt.excluded.embedding_text,
                "source_path": stmt.excluded.source_path,
                "source_hash": stmt.excluded.source_hash,
                "embedding": stmt.excluded.embedding,
                "linked_summary_id": stmt.excluded.linked_summary_id,
            },
        )
        self.session.execute(stmt)
        self.session.flush()

    def search_summaries(
        self,
        query_embedding: list[float],
        *,
        query: str,
        chapter_id: str | None,
        timeline_layer: str | None,
        character: str | None,
        mentioned_character: str | None,
        min_priority_score: float | None,
        tags: list[str],
        top_k: int,
    ) -> list[tuple[SummaryChunk, float]]:
        distance = SummaryChunk.embedding.cosine_distance(query_embedding)
        stmt = select(SummaryChunk, (1 - distance).label("score"))
        stmt = _summary_query_filters(
            stmt,
            chapter_id=chapter_id,
            timeline_layer=timeline_layer,
            character=character,
            mentioned_character=mentioned_character,
            min_priority_score=min_priority_score,
            tags=tags,
        )
        stmt = stmt.order_by(distance).limit(max(top_k * 5, top_k))
        candidates = [(row[0], float(row[1])) for row in self.session.execute(stmt).all()]
        reranked = [
            (row, _rerank_summary_score(query, row, semantic_score))
            for row, semantic_score in candidates
        ]
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked[:top_k]

    def search_raw(
        self,
        query_embedding: list[float],
        *,
        chapter_id: str | None,
        paragraph_id: int | None,
        top_k: int,
    ) -> list[tuple[RawChunk, float]]:
        distance = RawChunk.embedding.cosine_distance(query_embedding)
        stmt = select(RawChunk, (1 - distance).label("score"))
        if chapter_id:
            stmt = stmt.where(RawChunk.chapter_id == chapter_id)
        if paragraph_id is not None:
            stmt = stmt.where(RawChunk.paragraph_id == paragraph_id)
        stmt = stmt.order_by(distance).limit(top_k)
        return [(row[0], float(row[1])) for row in self.session.execute(stmt).all()]

    def get_linked_raw(self, summary_ids: list[UUID], top_k_per_hit: int) -> list[tuple[RawChunk, float]]:
        if not summary_ids:
            return []
        summaries = self.session.execute(
            select(SummaryChunk.id, SummaryChunk.priority_score).where(SummaryChunk.id.in_(summary_ids))
        ).all()
        priority_lookup = {row[0]: float(row[1]) for row in summaries}
        rows = self.session.execute(
            select(RawChunk).where(RawChunk.linked_summary_id.in_(summary_ids)).order_by(RawChunk.chapter_id, RawChunk.chunk_id)
        ).scalars().all()
        grouped: dict[UUID, list[RawChunk]] = {}
        for row in rows:
            if row.linked_summary_id is None:
                continue
            grouped.setdefault(row.linked_summary_id, []).append(row)
        results: list[tuple[RawChunk, float]] = []
        for summary_id in summary_ids:
            for item in grouped.get(summary_id, [])[:top_k_per_hit]:
                results.append((item, priority_lookup.get(summary_id, 0.0)))
        return results
