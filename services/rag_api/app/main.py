from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.orm import Session

from services.rag_api.app import service
from shared.db import get_db
from shared.schemas import (
    LinkedRawRequest,
    LinkedRawResponse,
    RawSearchRequest,
    RawSearchResponse,
    SummarySearchRequest,
    SummarySearchResponse,
)

app = FastAPI(title="fanfiction-rag-api")


@app.get("/healthz")
def healthz(db: Session = Depends(get_db)) -> dict[str, str]:
    db.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.post("/search/summaries", response_model=SummarySearchResponse)
def search_summaries(request: SummarySearchRequest, db: Session = Depends(get_db)) -> SummarySearchResponse:
    return service.search_summaries(db, request)


@app.post("/search/raw", response_model=RawSearchResponse)
def search_raw(request: RawSearchRequest, db: Session = Depends(get_db)) -> RawSearchResponse:
    return service.search_raw(db, request)


@app.post("/retrieve/linked-raw", response_model=LinkedRawResponse)
def retrieve_linked_raw(request: LinkedRawRequest, db: Session = Depends(get_db)) -> LinkedRawResponse:
    return service.get_linked_raw(db, request.summary_hit_ids, request.top_k_per_hit)
