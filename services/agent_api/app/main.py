from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock, Thread

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from services.agent_api.app.chat import build_session_chat_response, run_chat, run_chat_turn
from services.agent_api.app.client import RagApiClient
from services.agent_api.app.provider import get_llm_provider
from services.agent_api.app.sessions import get_session_manager
from shared.embeddings import preload_embedding_provider
from shared.schemas import (
    ChatRequest,
    ChatResponse,
    SessionChatRequest,
    SessionChatResponse,
    SessionCreateResponse,
    SessionDeleteResponse,
    SessionDetailResponse,
    SessionListResponse,
)

_warmup_lock = Lock()
_warmup_started = False
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _run_embedding_warmup() -> None:
    preload_embedding_provider()


def start_agent_warmup() -> bool:
    global _warmup_started
    with _warmup_lock:
        if _warmup_started:
            return False
        _warmup_started = True
    Thread(target=_run_embedding_warmup, name="agent-embedding-warmup", daemon=True).start()
    return True


@asynccontextmanager
async def lifespan(_: FastAPI):
    start_agent_warmup()
    yield


app = FastAPI(title="fanfiction-agent-api", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, object]:
    dependencies: dict[str, object] = {}
    overall_status = "ok"

    try:
        dependencies["rag_api"] = RagApiClient().healthcheck()
    except httpx.HTTPError as exc:
        overall_status = "degraded"
        dependencies["rag_api"] = {"status": "error", "detail": str(exc)}

    try:
        dependencies["llm_provider"] = get_llm_provider().healthcheck()
        if not dependencies["llm_provider"].get("model_available", False):
            overall_status = "degraded"
    except httpx.HTTPError as exc:
        overall_status = "degraded"
        dependencies["llm_provider"] = {"status": "error", "detail": str(exc)}

    return {"status": overall_status, "dependencies": dependencies}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return run_chat(request)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/sessions", response_model=SessionCreateResponse)
def create_session() -> SessionCreateResponse:
    return get_session_manager().create_session()


@app.get("/sessions", response_model=SessionListResponse)
def list_sessions() -> SessionListResponse:
    return SessionListResponse(sessions=get_session_manager().list_sessions())


@app.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session(session_id: str) -> SessionDetailResponse:
    return get_session_manager().get_session_detail(session_id)


@app.post("/sessions/{session_id}/chat", response_model=SessionChatResponse)
def session_chat(session_id: str, request: SessionChatRequest) -> SessionChatResponse:
    manager = get_session_manager()
    manager.append_user_message(session_id, request.message)
    history = manager.get_recent_history(session_id)[:-1]
    response = run_chat_turn(message=request.message, history=history, include_timing=request.include_timing)
    updated_session = manager.append_assistant_message(session_id, response.answer, response.debug)
    return build_session_chat_response(
        response,
        session_id=session_id,
        turn_index=manager.assistant_turn_index(session_id),
        updated_at=updated_session.detail().updated_at,
    )


@app.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
def delete_session(session_id: str) -> SessionDeleteResponse:
    get_session_manager().delete_session(session_id)
    return SessionDeleteResponse(status="deleted", session_id=session_id)
