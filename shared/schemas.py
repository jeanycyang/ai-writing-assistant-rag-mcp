from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class Citation(BaseModel):
    summary_id: UUID | None = None
    raw_chunk_id: UUID | None = None
    chapter_id: str
    paragraph_id: int | None = None
    chunk_id: int | None = None
    source_path: str
    score: float | None = None
    citation_type: str


class SummaryHit(BaseModel):
    id: UUID
    chapter_id: str
    paragraph_id: int
    priority_score: float
    timeline_layer: str
    scene: str
    characters: list[str]
    mentioned_characters: list[str]
    tags: list[str]
    key_events: list[str]
    plot: str
    source_path: str
    score: float
    citation: Citation


class RawHit(BaseModel):
    id: UUID
    chapter_id: str
    paragraph_id: int | None
    chunk_id: int
    original_text: str
    source_path: str
    linked_summary_id: UUID | None = None
    score: float
    citation: Citation


class SummarySearchRequest(BaseModel):
    query: str | None = None
    query_embedding: list[float]
    chapter_id: str | None = None
    timeline_layer: str | None = None
    character: str | None = None
    mentioned_character: str | None = None
    min_priority_score: float | None = None
    tags: list[str] = Field(default_factory=list)
    top_k: int = 5


class SummarySearchResponse(BaseModel):
    hits: list[SummaryHit]


class RawSearchRequest(BaseModel):
    query: str | None = None
    query_embedding: list[float]
    chapter_id: str | None = None
    paragraph_id: int | None = None
    tags: list[str] = Field(default_factory=list)
    top_k: int = 5


class RawSearchResponse(BaseModel):
    hits: list[RawHit]


class LinkedRawRequest(BaseModel):
    summary_hit_ids: list[UUID]
    top_k_per_hit: int = 2


class LinkedRawResponse(BaseModel):
    hits: list[RawHit]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    include_timing: bool = False


class ToolCallDebug(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    result_count: int


class ChatStepTiming(BaseModel):
    step: str
    elapsed_ms: float


class ChatDebugInfo(BaseModel):
    provider: str
    model: str
    iterations: int = 0
    unique_citation_count: int = 0
    completed_without_tool_call: bool = False
    elapsed_ms: float | None = None
    step_timings: list[ChatStepTiming] = Field(default_factory=list)
    tool_calls: list[ToolCallDebug] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    debug: ChatDebugInfo


class SessionMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    created_at: str


class SessionCreateResponse(BaseModel):
    session_id: str
    title: str | None = None
    created_at: str
    updated_at: str
    message_count: int


class SessionSummary(BaseModel):
    session_id: str
    title: str | None = None
    created_at: str
    updated_at: str
    message_count: int


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class SessionDetailResponse(BaseModel):
    session_id: str
    title: str | None = None
    created_at: str
    updated_at: str
    message_count: int
    messages: list[SessionMessage]
    last_debug: ChatDebugInfo | None = None


class SessionChatRequest(BaseModel):
    message: str
    include_timing: bool = False


class SessionChatResponse(ChatResponse):
    session_id: str
    turn_index: int
    updated_at: str


class SessionDeleteResponse(BaseModel):
    status: str
    session_id: str
