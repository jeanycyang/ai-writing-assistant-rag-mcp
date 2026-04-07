from fastapi import FastAPI

from services.agent_api.app.chat import run_chat
from shared.schemas import ChatRequest, ChatResponse

app = FastAPI(title="fanfiction-agent-api")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return run_chat(request)
