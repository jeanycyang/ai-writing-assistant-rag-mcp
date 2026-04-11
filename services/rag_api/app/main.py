from functools import lru_cache
from typing import Any

from fastapi import Body, Depends, FastAPI, Response, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from services.codex_mcp.server import WritingAssistantMcpServer
from services.rag_api.app import service
from shared.db import create_session, get_db
from shared.schemas import (
    ChapterRequest,
    LinkedRawRequest,
    LinkedRawResponse,
    RawChapterResponse,
    RawParagraphRequest,
    RawParagraphResponse,
    RawSearchRequest,
    RawSearchResponse,
    SummaryChapterResponse,
    SummaryParagraphRequest,
    SummaryParagraphResponse,
    SummaryCharacterSearchRequest,
    SummaryCharacterSearchResponse,
    SummarySearchRequest,
    SummarySearchResponse,
)

app = FastAPI(title="ai-writing-assistance-rag-api")


@lru_cache(maxsize=1)
def get_mcp_server() -> WritingAssistantMcpServer:
    return WritingAssistantMcpServer()


@lru_cache(maxsize=32)
def get_mcp_server_for_work(work: str) -> WritingAssistantMcpServer:
    return WritingAssistantMcpServer(work=work)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz(db: Session = Depends(get_db)) -> dict[str, str]:
    db.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.post("/mcp")
def mcp(
    request: dict[str, Any] = Body(...),
    server: WritingAssistantMcpServer = Depends(get_mcp_server),
):
    response = server.handle_request(request)
    if response is None:
        return Response(status_code=status.HTTP_202_ACCEPTED)
    return response


@app.post("/mcp/{work}")
def mcp_for_work(work: str, request: dict[str, Any] = Body(...)):
    response = get_mcp_server_for_work(work).handle_request(request)
    if response is None:
        return Response(status_code=status.HTTP_202_ACCEPTED)
    return response


@app.post("/search/summaries", response_model=SummarySearchResponse)
def search_summaries(request: SummarySearchRequest, db: Session = Depends(get_db)) -> SummarySearchResponse:
    return service.search_summaries(db, request)


def _with_work_session(work: str):
    session = create_session(work)
    try:
        yield session
    finally:
        session.close()


@app.post("/search/summary-characters", response_model=SummaryCharacterSearchResponse)
def search_summary_characters(
    request: SummaryCharacterSearchRequest,
    db: Session = Depends(get_db),
) -> SummaryCharacterSearchResponse:
    return service.search_summary_characters(db, request)


@app.post("/search/raw", response_model=RawSearchResponse)
def search_raw(request: RawSearchRequest, db: Session = Depends(get_db)) -> RawSearchResponse:
    return service.search_raw(db, request)


@app.post("/retrieve/linked-raw", response_model=LinkedRawResponse)
def retrieve_linked_raw(request: LinkedRawRequest, db: Session = Depends(get_db)) -> LinkedRawResponse:
    return service.get_linked_raw(db, request.summary_hit_ids, request.top_k_per_hit)


@app.post("/retrieve/summary-paragraph", response_model=SummaryParagraphResponse)
def retrieve_summary_paragraph(
    request: SummaryParagraphRequest,
    db: Session = Depends(get_db),
) -> SummaryParagraphResponse:
    return service.get_summary_paragraph(db, request)


@app.post("/retrieve/summary-chapter", response_model=SummaryChapterResponse)
def retrieve_summary_chapter(
    request: ChapterRequest,
    db: Session = Depends(get_db),
) -> SummaryChapterResponse:
    return service.get_summary_chapter(db, request)


@app.post("/retrieve/raw-paragraph", response_model=RawParagraphResponse)
def retrieve_raw_paragraph(
    request: RawParagraphRequest,
    db: Session = Depends(get_db),
) -> RawParagraphResponse:
    return service.get_raw_paragraph(db, request)


@app.post("/retrieve/raw-chapter", response_model=RawChapterResponse)
def retrieve_raw_chapter(
    request: ChapterRequest,
    db: Session = Depends(get_db),
) -> RawChapterResponse:
    return service.get_raw_chapter(db, request)


@app.post("/works/{work}/search/summaries", response_model=SummarySearchResponse)
def search_summaries_for_work(
    work: str,
    request: SummarySearchRequest,
    db: Session = Depends(_with_work_session),
) -> SummarySearchResponse:
    return service.search_summaries(db, request)


@app.post("/works/{work}/search/summary-characters", response_model=SummaryCharacterSearchResponse)
def search_summary_characters_for_work(
    work: str,
    request: SummaryCharacterSearchRequest,
    db: Session = Depends(_with_work_session),
) -> SummaryCharacterSearchResponse:
    return service.search_summary_characters(db, request)


@app.post("/works/{work}/search/raw", response_model=RawSearchResponse)
def search_raw_for_work(
    work: str,
    request: RawSearchRequest,
    db: Session = Depends(_with_work_session),
) -> RawSearchResponse:
    return service.search_raw(db, request)


@app.post("/works/{work}/retrieve/linked-raw", response_model=LinkedRawResponse)
def retrieve_linked_raw_for_work(
    work: str,
    request: LinkedRawRequest,
    db: Session = Depends(_with_work_session),
) -> LinkedRawResponse:
    return service.get_linked_raw(db, request.summary_hit_ids, request.top_k_per_hit)


@app.post("/works/{work}/retrieve/summary-paragraph", response_model=SummaryParagraphResponse)
def retrieve_summary_paragraph_for_work(
    work: str,
    request: SummaryParagraphRequest,
    db: Session = Depends(_with_work_session),
) -> SummaryParagraphResponse:
    return service.get_summary_paragraph(db, request)


@app.post("/works/{work}/retrieve/summary-chapter", response_model=SummaryChapterResponse)
def retrieve_summary_chapter_for_work(
    work: str,
    request: ChapterRequest,
    db: Session = Depends(_with_work_session),
) -> SummaryChapterResponse:
    return service.get_summary_chapter(db, request)


@app.post("/works/{work}/retrieve/raw-paragraph", response_model=RawParagraphResponse)
def retrieve_raw_paragraph_for_work(
    work: str,
    request: RawParagraphRequest,
    db: Session = Depends(_with_work_session),
) -> RawParagraphResponse:
    return service.get_raw_paragraph(db, request)


@app.post("/works/{work}/retrieve/raw-chapter", response_model=RawChapterResponse)
def retrieve_raw_chapter_for_work(
    work: str,
    request: ChapterRequest,
    db: Session = Depends(_with_work_session),
) -> RawChapterResponse:
    return service.get_raw_chapter(db, request)
