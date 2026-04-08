from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.orm import Session

from services.rag_api.app import service
from shared.db import get_db
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

app = FastAPI(title="fanfiction-rag-api")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz(db: Session = Depends(get_db)) -> dict[str, str]:
    db.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.post("/search/summaries", response_model=SummarySearchResponse)
def search_summaries(request: SummarySearchRequest, db: Session = Depends(get_db)) -> SummarySearchResponse:
    return service.search_summaries(db, request)


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
