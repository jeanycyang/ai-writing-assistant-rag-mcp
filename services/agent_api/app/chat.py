from __future__ import annotations

import json
from time import perf_counter
from typing import Any

import httpx
from fastapi import HTTPException

from services.agent_api.app.client import RagApiClient
from services.agent_api.app.provider import extract_message, get_llm_provider
from shared.config import get_settings
from shared.schemas import ChatDebugInfo, ChatRequest, ChatResponse, ChatStepTiming, Citation, ToolCallDebug

SYSTEM_PROMPT = """You are a local fanfiction canon assistant for Traditional Chinese (Taiwan) source material.

Rules:
- Answer only from the retrieved evidence provided to you.
- Prefer episode summaries for canon lookup.
- Use linked original text when available to support scene detail and wording.
- If the evidence is insufficient or conflicting, say so explicitly.
- Preserve Traditional Chinese (Taiwan) wording.
- Keep the answer concise and directly answer the user's question first.
- After the answer, add a short evidence note that cites chapter / paragraph references when available.
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


def _extract_summary_hit_ids(result: dict[str, Any]) -> list[str]:
    summary_hit_ids: list[str] = []
    for hit in result.get("hits", []):
        summary_id = hit.get("id")
        if not summary_id:
            citation = hit.get("citation") or {}
            summary_id = citation.get("summary_id")
        if summary_id:
            summary_hit_ids.append(str(summary_id))
    return summary_hit_ids


def _format_summary_hits(result: dict[str, Any]) -> str:
    lines: list[str] = []
    for idx, hit in enumerate(result.get("hits", []), start=1):
        citation = hit.get("citation") or {}
        lines.append(
            (
                f"{idx}. chapter={hit.get('chapter_id')} paragraph={hit.get('paragraph_id')} "
                f"score={hit.get('score')}\n"
                f"scene={hit.get('scene')}\n"
                f"plot={hit.get('plot')}\n"
                f"key_events={', '.join(hit.get('key_events', []))}\n"
                f"source={citation.get('source_path')}"
            )
        )
    return "\n\n".join(lines) if lines else "(none)"


def _format_raw_hits(result: dict[str, Any]) -> str:
    lines: list[str] = []
    for idx, hit in enumerate(result.get("hits", []), start=1):
        citation = hit.get("citation") or {}
        lines.append(
            (
                f"{idx}. chapter={hit.get('chapter_id')} paragraph={hit.get('paragraph_id')} "
                f"chunk={hit.get('chunk_id')} score={hit.get('score')}\n"
                f"text={hit.get('original_text')}\n"
                f"source={citation.get('source_path')}"
            )
        )
    return "\n\n".join(lines) if lines else "(none)"


def _needs_direct_raw_search(summary_result: dict[str, Any], linked_raw_result: dict[str, Any]) -> bool:
    return not summary_result.get("hits") or not linked_raw_result.get("hits")


def _build_final_messages(
    request: ChatRequest,
    summary_result: dict[str, Any],
    linked_raw_result: dict[str, Any],
    raw_result: dict[str, Any],
) -> list[dict[str, str]]:
    history_lines = [f"{item.role}: {item.content}" for item in request.history]
    history_block = "\n".join(history_lines) if history_lines else "(none)"
    user_prompt = (
        f"User question:\n{request.message}\n\n"
        f"Conversation history:\n{history_block}\n\n"
        f"Summary search results:\n{_format_summary_hits(summary_result)}\n\n"
        f"Linked original text:\n{_format_raw_hits(linked_raw_result)}\n\n"
        f"Direct raw search fallback:\n{_format_raw_hits(raw_result)}\n"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _record_step_timing(debug: ChatDebugInfo, step: str, started_at: float, *, enabled: bool) -> None:
    if not enabled:
        return
    debug.step_timings.append(ChatStepTiming(step=step, elapsed_ms=round((perf_counter() - started_at) * 1000, 2)))


def run_chat(request: ChatRequest) -> ChatResponse:
    started_at = perf_counter()
    settings = get_settings()
    provider = get_llm_provider()
    rag_client = RagApiClient()

    debug = ChatDebugInfo(provider=settings.llm_provider, model=settings.ollama_model, tool_calls=[])
    all_citations: list[Citation] = []

    include_timing = request.include_timing

    step_started_at = perf_counter()
    try:
        summary_result = rag_client.search_summaries({"query": request.message, "top_k": 3})
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"rag-api request failed during search_episode_summaries: {exc}") from exc
    _record_step_timing(debug, "search_episode_summaries", step_started_at, enabled=include_timing)
    all_citations.extend(_collect_citations(summary_result))
    debug.tool_calls.append(
        ToolCallDebug(
            tool_name="search_episode_summaries",
            arguments={"query": request.message, "top_k": 3},
            result_count=len(summary_result.get("hits", [])),
        )
    )

    linked_raw_result: dict[str, Any] = {"hits": []}
    summary_hit_ids = _extract_summary_hit_ids(summary_result)
    if summary_hit_ids:
        step_started_at = perf_counter()
        try:
            linked_raw_result = rag_client.get_linked_raw({"summary_hit_ids": summary_hit_ids, "top_k_per_hit": 1})
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"rag-api request failed during get_linked_original_text: {exc}",
            ) from exc
        _record_step_timing(debug, "get_linked_original_text", step_started_at, enabled=include_timing)
        all_citations.extend(_collect_citations(linked_raw_result))
        debug.tool_calls.append(
            ToolCallDebug(
                tool_name="get_linked_original_text",
                arguments={"summary_hit_ids": summary_hit_ids, "top_k_per_hit": 1},
                result_count=len(linked_raw_result.get("hits", [])),
            )
        )

    raw_result: dict[str, Any] = {"hits": []}
    if _needs_direct_raw_search(summary_result, linked_raw_result):
        step_started_at = perf_counter()
        try:
            raw_result = rag_client.search_raw({"query": request.message, "top_k": 3})
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"rag-api request failed during search_original_text: {exc}") from exc
        _record_step_timing(debug, "search_original_text", step_started_at, enabled=include_timing)
        all_citations.extend(_collect_citations(raw_result))
        debug.tool_calls.append(
            ToolCallDebug(
                tool_name="search_original_text",
                arguments={"query": request.message, "top_k": 3},
                result_count=len(raw_result.get("hits", [])),
            )
        )

    final_messages = _build_final_messages(request, summary_result, linked_raw_result, raw_result)
    debug.iterations = 1

    step_started_at = perf_counter()
    try:
        payload = provider.complete(final_messages, tools=None, think=False)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"LLM provider request failed: {exc}") from exc
    _record_step_timing(debug, "final_generation", step_started_at, enabled=include_timing)

    message = extract_message(payload)
    deduped_citations = _dedupe_citations(all_citations)
    debug.unique_citation_count = len(deduped_citations)
    debug.completed_without_tool_call = False
    if include_timing:
        debug.elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
    return ChatResponse(answer=message.get("content", ""), citations=deduped_citations, debug=debug)
