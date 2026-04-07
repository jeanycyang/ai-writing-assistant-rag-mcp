from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi import HTTPException

from services.agent_api.app.client import RagApiClient
from services.agent_api.app.provider import extract_message, extract_tool_calls, get_llm_provider
from shared.config import get_settings
from shared.schemas import ChatDebugInfo, ChatRequest, ChatResponse, Citation, ToolCallDebug

SYSTEM_PROMPT = """You are a local fanfiction canon assistant for Traditional Chinese (Taiwan) source material.

Rules:
- Use search_episode_summaries first for canon lookup whenever retrieval is needed.
- If summaries are insufficient, ambiguous, or too compressed, call get_linked_original_text or search_original_text.
- Prefer original text when the user asks for exact evidence, exact wording, scene nuance, or dialogue details.
- Do not answer from memory when retrieval is needed.
- Do not invent unsupported facts.
- If evidence is weak or conflicting, say so explicitly.
- Preserve Traditional Chinese wording from Taiwan source material in your answer unless the user explicitly asks for translation or normalization.
- Cite chapter / paragraph / chunk provenance from retrieved results in the final answer.
"""


def _tool_result_message(tool_name: str, payload: dict[str, Any]) -> str:
    return json.dumps({"tool_name": tool_name, "result": payload}, ensure_ascii=False)


def _collect_citations(result: dict[str, Any]) -> list[Citation]:
    citations: list[Citation] = []
    for hit in result.get("hits", []):
        citation = hit.get("citation")
        if citation:
            citations.append(Citation.model_validate(citation))
    return citations


def _citation_key(citation: Citation) -> tuple[Any, ...]:
    return (
        str(citation.summary_id) if citation.summary_id else None,
        str(citation.raw_chunk_id) if citation.raw_chunk_id else None,
        citation.chapter_id,
        citation.paragraph_id,
        citation.chunk_id,
        citation.source_path,
        citation.citation_type,
    )


def _dedupe_citations(citations: list[Citation]) -> list[Citation]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[Citation] = []
    for citation in citations:
        key = _citation_key(citation)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def run_chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings()
    provider = get_llm_provider()
    rag_client = RagApiClient()

    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in request.history:
        messages.append({"role": item.role, "content": item.content})
    messages.append({"role": "user", "content": request.message})

    all_citations: list[Citation] = []
    debug = ChatDebugInfo(provider=settings.llm_provider, model=settings.ollama_model, tool_calls=[])

    for _ in range(8):
        try:
            payload = provider.complete(messages)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"LLM provider request failed: {exc}") from exc
        message = extract_message(payload)
        tool_calls = extract_tool_calls(message)
        if not tool_calls:
            answer = message.get("content", "")
            return ChatResponse(answer=answer, citations=_dedupe_citations(all_citations), debug=debug)

        messages.append(
            {
                "role": "assistant",
                "content": message.get("content", ""),
                "tool_calls": tool_calls,
            }
        )

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            arguments = tool_call["arguments"]
            try:
                if tool_name == "search_episode_summaries":
                    result = rag_client.search_summaries(arguments)
                elif tool_name == "get_linked_original_text":
                    result = rag_client.get_linked_raw(arguments)
                elif tool_name == "search_original_text":
                    result = rag_client.search_raw(arguments)
                else:
                    raise HTTPException(status_code=502, detail=f"Unsupported tool call from model: {tool_name}")
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=503, detail=f"rag-api request failed during {tool_name}: {exc}") from exc

            all_citations.extend(_collect_citations(result))
            debug.tool_calls.append(
                ToolCallDebug(
                    tool_name=tool_name,
                    arguments=arguments,
                    result_count=len(result.get("hits", [])),
                )
            )
            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": _tool_result_message(tool_name, result),
                }
            )

    raise HTTPException(status_code=502, detail="Model did not return a final answer within the tool loop limit")
