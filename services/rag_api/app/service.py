from __future__ import annotations

from sqlalchemy.orm import Session

from shared.repository import RagRepository
from shared.schemas import (
    ChapterRequest,
    Citation,
    LinkedRawResponse,
    RawChapterParagraph,
    RawChapterResponse,
    RawParagraphRequest,
    RawParagraphResponse,
    RawHit,
    RawSearchRequest,
    RawSearchResponse,
    SummaryChapterParagraph,
    SummaryChapterResponse,
    SummaryParagraphRequest,
    SummaryParagraphResponse,
    SummaryHit,
    SummaryCharacterSearchRequest,
    SummaryCharacterSearchResponse,
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


def _build_summary_chapter_paragraph(row) -> SummaryChapterParagraph:
    return SummaryChapterParagraph(
        paragraph_id=row.paragraph_id,
        priority_score=row.priority_score,
        timeline_layer=row.timeline_layer,
        scene=row.scene,
        characters=row.characters,
        mentioned_characters=row.mentioned_characters,
        tags=row.tags,
        key_events=row.key_events,
        plot=row.plot,
        citation=Citation(
            summary_id=row.id,
            chapter_id=row.chapter_id,
            paragraph_id=row.paragraph_id,
            source_path=row.source_path,
            score=1.0,
            citation_type="summary",
        ),
    )


def _raw_overlap_size(left: str, right: str) -> int:
    max_overlap = min(len(left), len(right))
    for size in range(max_overlap, 0, -1):
        if left.endswith(right[:size]):
            return size
    return 0


def _merge_raw_chunks(chunks: list[str]) -> str:
    if not chunks:
        return ""
    merged = chunks[0]
    for chunk in chunks[1:]:
        overlap = _raw_overlap_size(merged, chunk)
        merged += chunk[overlap:]
    return merged


def search_summaries(session: Session, request: SummarySearchRequest) -> SummarySearchResponse:
    repo = RagRepository(session)
    rows = repo.search_summaries(
        request.query_embedding,
        query=request.query or "",
        chapter_id=request.chapter_id,
        timeline_layer=request.timeline_layer,
        character=request.character,
        mentioned_character=request.mentioned_character,
        min_priority_score=request.min_priority_score,
        tags=request.tags,
        top_k=request.top_k,
    )
    return SummarySearchResponse(hits=[_build_summary_hit(row, score) for row, score in rows])


def search_summary_characters(session: Session, request: SummaryCharacterSearchRequest) -> SummaryCharacterSearchResponse:
    repo = RagRepository(session)
    rows = repo.search_summary_characters(
        characters=request.characters,
        operator=request.operator,
        chapter_id=request.chapter_id,
        from_chapter=request.from_chapter,
        to_chapter=request.to_chapter,
        min_priority_score=request.min_priority_score,
        top_k=request.top_k,
    )
    return SummaryCharacterSearchResponse(hits=[_build_summary_hit(row, score) for row, score in rows])


def search_raw(session: Session, request: RawSearchRequest) -> RawSearchResponse:
    repo = RagRepository(session)
    rows = repo.search_raw(
        request.query_embedding,
        chapter_id=request.chapter_id,
        paragraph_id=request.paragraph_id,
        top_k=request.top_k,
    )
    return RawSearchResponse(hits=[_build_raw_hit(row, score) for row, score in rows])


def get_linked_raw(session: Session, summary_hit_ids, top_k_per_hit: int) -> LinkedRawResponse:
    repo = RagRepository(session)
    rows = repo.get_linked_raw(summary_hit_ids, top_k_per_hit)
    return LinkedRawResponse(hits=[_build_raw_hit(row, score) for row, score in rows])


def get_summary_paragraph(session: Session, request: SummaryParagraphRequest) -> SummaryParagraphResponse:
    repo = RagRepository(session)
    rows = repo.get_summary_paragraph(chapter_id=request.chapter_id, paragraph_id=request.paragraph_id)
    return SummaryParagraphResponse(hits=[_build_summary_hit(row, score) for row, score in rows])


def get_summary_chapter(session: Session, request: ChapterRequest) -> SummaryChapterResponse:
    repo = RagRepository(session)
    rows = repo.get_summary_chapter(chapter_id=request.chapter_id)
    paragraphs = [_build_summary_chapter_paragraph(row) for row in rows]
    full_summary_text = "\n\n".join(
        f"## {paragraph.paragraph_id}\nscene: {paragraph.scene}\nplot: {paragraph.plot}"
        for paragraph in paragraphs
    )
    return SummaryChapterResponse(
        chapter_id=request.chapter_id,
        source_path=rows[0].source_path if rows else None,
        full_summary_text=full_summary_text,
        paragraphs=paragraphs,
    )


def get_raw_paragraph(session: Session, request: RawParagraphRequest) -> RawParagraphResponse:
    repo = RagRepository(session)
    rows = repo.get_raw_paragraph(chapter_id=request.chapter_id, paragraph_id=request.paragraph_id)
    return RawParagraphResponse(hits=[_build_raw_hit(row, score) for row, score in rows])


def get_raw_chapter(session: Session, request: ChapterRequest) -> RawChapterResponse:
    repo = RagRepository(session)
    rows = repo.get_raw_chapter(chapter_id=request.chapter_id)

    grouped: dict[int | None, list] = {}
    for row in rows:
        grouped.setdefault(row.paragraph_id, []).append(row)

    paragraphs: list[RawChapterParagraph] = []
    for paragraph_id in sorted(grouped, key=lambda value: (value is None, value if value is not None else 10**9)):
        chunk_rows = grouped[paragraph_id]
        text = _merge_raw_chunks([row.original_text for row in chunk_rows])
        citations = [
            Citation(
                raw_chunk_id=row.id,
                summary_id=row.linked_summary_id,
                chapter_id=row.chapter_id,
                paragraph_id=row.paragraph_id,
                chunk_id=row.chunk_id,
                source_path=row.source_path,
                score=1.0,
                citation_type="raw",
            )
            for row in chunk_rows
        ]
        paragraphs.append(RawChapterParagraph(paragraph_id=paragraph_id, text=text, citations=citations))

    full_text = "\n\n".join(
        f"## {paragraph.paragraph_id}\n{paragraph.text}" if paragraph.paragraph_id is not None else paragraph.text
        for paragraph in paragraphs
    )
    return RawChapterResponse(
        chapter_id=request.chapter_id,
        source_path=rows[0].source_path if rows else None,
        full_text=full_text,
        paragraphs=paragraphs,
    )
