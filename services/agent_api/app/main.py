import httpx
from fastapi import FastAPI

from services.agent_api.app.chat import run_chat
from services.agent_api.app.client import RagApiClient
from services.agent_api.app.provider import get_llm_provider
from shared.schemas import ChatRequest, ChatResponse

app = FastAPI(title="fanfiction-agent-api")


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
