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
- If the answer is not stated in the evidence, say that directly and stop. Do not add nearby plot details just to fill space.
- Preserve Traditional Chinese (Taiwan) wording.
- Start with the direct answer in the first sentence.
- Keep the whole answer brief: usually 2-4 sentences, no headings, no bullet lists.
- Prefer the strongest single supporting passage instead of listing every possible clue.
- End with one short evidence sentence citing chapter / paragraph references when available.
"""

SUMMARY_TOP_K = 2
RAW_TOP_K = 2
LINKED_RAW_TOP_K_PER_HIT = 1
MAX_FINAL_SUMMARY_HITS = 1
MAX_FINAL_LINKED_RAW_HITS = 1
MAX_FINAL_RAW_HITS = 1
MAX_RETURNED_CITATIONS = 4


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


def _sort_citations(citations: list[Citation]) -> list[Citation]:
    return sorted(
        citations,
        key=lambda citation: (
            0 if citation.citation_type == "raw" else 1,
            -(citation.score or 0.0),
            citation.chapter_id,
            citation.paragraph_id if citation.paragraph_id is not None else 10**9,
            citation.chunk_id if citation.chunk_id is not None else 10**9,
        ),
    )


def _select_hits(result: dict[str, Any], limit: int) -> dict[str, Any]:
    return {"hits": list(result.get("hits", []))[:limit]}


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


def _primary_summary_hit(result: dict[str, Any]) -> dict[str, Any] | None:
    hits = result.get("hits", [])
    if not hits:
        return None
    return hits[0]


def _primary_summary_id(result: dict[str, Any]) -> str | None:
    hit = _primary_summary_hit(result)
    if not hit:
        return None
    summary_id = hit.get("id")
    if summary_id:
        return str(summary_id)
    citation = hit.get("citation") or {}
    if citation.get("summary_id"):
        return str(citation["summary_id"])
    return None


def _filter_linked_raw_hits(result: dict[str, Any], summary_id: str | None) -> dict[str, Any]:
    if not summary_id:
        return result
    filtered_hits = [
        hit
        for hit in result.get("hits", [])
        if str((hit.get("citation") or {}).get("summary_id") or "") == summary_id
    ]
    return {"hits": filtered_hits or list(result.get("hits", []))}


def _select_citations(
    citations: list[Citation],
    *,
    primary_summary_id: str | None,
    primary_chapter_id: str | None,
    limit: int = MAX_RETURNED_CITATIONS,
) -> list[Citation]:
    ranked = _sort_citations(_dedupe_citations(citations))
    preferred: list[Citation] = []
    for citation in ranked:
        citation_summary_id = str(citation.summary_id) if citation.summary_id else None
        if primary_summary_id and citation_summary_id == primary_summary_id:
            preferred.append(citation)
        elif primary_chapter_id and citation.chapter_id == primary_chapter_id and citation not in preferred:
            preferred.append(citation)

    if primary_summary_id and preferred:
        primary_cluster = [
            citation
            for citation in preferred
            if (str(citation.summary_id) if citation.summary_id else None) == primary_summary_id
        ]
        if primary_cluster:
            return primary_cluster[:limit]

    ordered = preferred + [citation for citation in ranked if citation not in preferred]
    return ordered[:limit]


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
    primary_summary_id = _primary_summary_id(summary_result)
    summary_context = _select_hits(summary_result, MAX_FINAL_SUMMARY_HITS)
    linked_raw_context = _select_hits(_filter_linked_raw_hits(linked_raw_result, primary_summary_id), MAX_FINAL_LINKED_RAW_HITS)
    raw_context = _select_hits(raw_result, MAX_FINAL_RAW_HITS)
    history_lines = [f"{item.role}: {item.content}" for item in request.history]
    history_block = "\n".join(history_lines) if history_lines else "(none)"
    user_prompt = (
        f"User question:\n{request.message}\n\n"
        f"Conversation history:\n{history_block}\n\n"
        "Use only the most relevant evidence below. Do not restate the entire context.\n\n"
        "If the evidence does not directly answer the question, reply that the answer is not stated or evidence is insufficient."
        " In that case, do not add unrelated scene summaries.\n\n"
        f"Summary search results:\n{_format_summary_hits(summary_context)}\n\n"
        f"Linked original text:\n{_format_raw_hits(linked_raw_context)}\n\n"
        f"Direct raw search fallback:\n{_format_raw_hits(raw_context)}\n"
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

    try:
        if include_timing:
            summary_result, summary_timings = rag_client.search_summaries_with_timings(
                {"query": request.message, "top_k": SUMMARY_TOP_K}
            )
            debug.step_timings.append(
                ChatStepTiming(step="search_episode_summaries_embed_query", elapsed_ms=summary_timings["embedding_ms"])
            )
            debug.step_timings.append(
                ChatStepTiming(step="search_episode_summaries_rag_api", elapsed_ms=summary_timings["rag_api_ms"])
            )
        else:
            summary_result = rag_client.search_summaries({"query": request.message, "top_k": SUMMARY_TOP_K})
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"rag-api request failed during search_episode_summaries: {exc}") from exc
    all_citations.extend(_collect_citations(summary_result))
    debug.tool_calls.append(
        ToolCallDebug(
            tool_name="search_episode_summaries",
            arguments={"query": request.message, "top_k": SUMMARY_TOP_K},
            result_count=len(summary_result.get("hits", [])),
        )
    )

    linked_raw_result: dict[str, Any] = {"hits": []}
    summary_hit_ids = _extract_summary_hit_ids(summary_result)
    if summary_hit_ids:
        step_started_at = perf_counter()
        try:
            linked_raw_result = rag_client.get_linked_raw(
                {"summary_hit_ids": summary_hit_ids, "top_k_per_hit": LINKED_RAW_TOP_K_PER_HIT}
            )
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
                arguments={"summary_hit_ids": summary_hit_ids, "top_k_per_hit": LINKED_RAW_TOP_K_PER_HIT},
                result_count=len(linked_raw_result.get("hits", [])),
            )
        )

    raw_result: dict[str, Any] = {"hits": []}
    if _needs_direct_raw_search(summary_result, linked_raw_result):
        try:
            if include_timing:
                raw_result, raw_timings = rag_client.search_raw_with_timings(
                    {"query": request.message, "top_k": RAW_TOP_K}
                )
                debug.step_timings.append(
                    ChatStepTiming(step="search_original_text_embed_query", elapsed_ms=raw_timings["embedding_ms"])
                )
                debug.step_timings.append(
                    ChatStepTiming(step="search_original_text_rag_api", elapsed_ms=raw_timings["rag_api_ms"])
                )
            else:
                raw_result = rag_client.search_raw({"query": request.message, "top_k": RAW_TOP_K})
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"rag-api request failed during search_original_text: {exc}") from exc
        all_citations.extend(_collect_citations(raw_result))
        debug.tool_calls.append(
            ToolCallDebug(
                tool_name="search_original_text",
                arguments={"query": request.message, "top_k": RAW_TOP_K},
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
    primary_hit = _primary_summary_hit(summary_result)
    primary_summary_id = _primary_summary_id(summary_result)
    primary_chapter_id = primary_hit.get("chapter_id") if primary_hit else None
    ranked_citations = _sort_citations(_dedupe_citations(all_citations))
    trimmed_citations = _select_citations(
        all_citations,
        primary_summary_id=primary_summary_id,
        primary_chapter_id=primary_chapter_id,
    )
    debug.unique_citation_count = len(ranked_citations)
    debug.completed_without_tool_call = False
    if include_timing:
        debug.elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
    return ChatResponse(answer=message.get("content", ""), citations=trimmed_citations, debug=debug)
