from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any

import httpx
from fastapi import HTTPException

from services.agent_api.app.client import RagApiClient
from services.agent_api.app.provider import extract_message, extract_tool_calls, get_llm_provider
from shared.config import get_settings
from shared.schemas import (
    ChatDebugInfo,
    ChatModelInput,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStepTiming,
    ChatVerificationResult,
    Citation,
    SessionChatResponse,
    SessionMessage,
    ToolCallDebug,
)

SYSTEM_PROMPT = """You are a local fanfiction canon assistant for Traditional Chinese (Taiwan) source material.

Rules:
- Use tools for canon/source lookup questions before answering.
- When the user asks for a specific chapter or paragraph, prefer exact tool arguments such as `chapter_id` and `paragraph_id` instead of relying on semantic similarity alone.
- If the first tool result is partial, weak, or only answers part of the question, keep searching before saying the answer is unknown.
- For follow-up questions about an already identified character, scene, or event, search again for the missing detail instead of assuming the first evidence bundle is sufficient.
- Use retrieved evidence for canon claims about the source material.
- Prior conversation turns in the same session are part of the conversation state and must be considered when answering follow-up questions.
- Do not present conversational history as canon evidence.
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

SUMMARY_TOP_K = 5
RAW_TOP_K = 5
LINKED_RAW_TOP_K_PER_HIT = 2
EXPANDED_SUMMARY_TOP_K = 8
EXPANDED_RAW_TOP_K = 8
EXPANDED_LINKED_RAW_TOP_K_PER_HIT = 3
MIN_SUMMARY_HITS_FOR_CONFIDENCE = 2
MIN_RAW_HITS_FOR_CONFIDENCE = 2
MAX_FINAL_SUMMARY_HITS = 3
MAX_FINAL_LINKED_RAW_HITS = 3
MAX_FINAL_RAW_HITS = 3
MAX_RETURNED_CITATIONS = 6
MAX_TOOL_ITERATIONS = 4
MAX_VERIFIER_REPAIRS = 3

VERIFIER_SYSTEM_PROMPT = """You verify whether a draft answer is directly supported by retrieved evidence.

Rules:
- Return JSON matching the requested schema.
- `supported` is true only if the cited evidence supports the draft answer.
- `direct_answer` is true only if the evidence directly answers the user's current question rather than only giving nearby or partial context.
- If the evidence type is wrong for the question, recommend a better next action.
- Prefer `get_raw_paragraph` for exact chapter/paragraph questions.
- Prefer `search_original_text` for concrete follow-up details like occupation, relationship, wording, or quotes.
- Prefer `search_episode_summaries` for broad canon overview or timeline questions.
- Prefer `get_summary_paragraph` only when the user explicitly wants the summary of an exact paragraph.
- Recommend `answer` only when the draft should be accepted as-is.
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


def _needs_broader_search(
    summary_result: dict[str, Any],
    linked_raw_result: dict[str, Any],
    raw_result: dict[str, Any],
) -> bool:
    return (
        len(summary_result.get("hits", [])) < MIN_SUMMARY_HITS_FOR_CONFIDENCE
        or len(linked_raw_result.get("hits", [])) + len(raw_result.get("hits", [])) < MIN_RAW_HITS_FOR_CONFIDENCE
    )


def _build_final_messages(
    *,
    message: str,
    history: list[ChatMessage] | list[SessionMessage],
    summary_result: dict[str, Any],
    linked_raw_result: dict[str, Any],
    raw_result: dict[str, Any],
) -> list[dict[str, str]]:
    primary_summary_id = _primary_summary_id(summary_result)
    summary_context = _select_hits(summary_result, MAX_FINAL_SUMMARY_HITS)
    linked_raw_context = _select_hits(_filter_linked_raw_hits(linked_raw_result, primary_summary_id), MAX_FINAL_LINKED_RAW_HITS)
    raw_context = _select_hits(raw_result, MAX_FINAL_RAW_HITS)
    final_prompt = (
        f"Current user question:\n{message}\n\n"
        "Use the prior chat messages as conversation state for follow-up questions in this session.\n"
        "Use only the retrieved evidence below for canon claims about the source material.\n"
        "Do not restate the entire context.\n\n"
        "If the retrieved evidence does not directly answer a canon question, reply that the answer is not stated or evidence is insufficient."
        " In that case, do not add unrelated scene summaries.\n\n"
        f"Summary search results:\n{_format_summary_hits(summary_context)}\n\n"
        f"Linked original text:\n{_format_raw_hits(linked_raw_context)}\n\n"
        f"Direct raw search fallback:\n{_format_raw_hits(raw_context)}\n"
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend({"role": item.role, "content": item.content} for item in history)
    messages.append({"role": "user", "content": final_prompt})
    return messages


def _build_tool_loop_messages(
    *,
    message: str,
    history: list[ChatMessage] | list[SessionMessage],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend({"role": item.role, "content": item.content} for item in history)
    messages.append({"role": "user", "content": message})
    return messages


def _build_verifier_messages(
    *,
    message: str,
    history: list[ChatMessage] | list[SessionMessage],
    draft_answer: str,
    summary_result: dict[str, Any],
    linked_raw_result: dict[str, Any],
    raw_result: dict[str, Any],
    attempted_tools: list[str],
) -> list[dict[str, str]]:
    attempted_block = "\n".join(f"- {item}" for item in attempted_tools) if attempted_tools else "(none)"
    verifier_prompt = (
        f"Current user question:\n{message}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        f"Attempted tool actions:\n{attempted_block}\n\n"
        "Use the conversation history only to understand the follow-up question. Do not treat it as canon evidence.\n\n"
        f"Summary search results:\n{_format_summary_hits(summary_result)}\n\n"
        f"Linked original text:\n{_format_raw_hits(linked_raw_result)}\n\n"
        f"Raw/or exact paragraph evidence:\n{_format_raw_hits(raw_result)}\n"
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": VERIFIER_SYSTEM_PROMPT}]
    messages.extend({"role": item.role, "content": item.content} for item in history)
    messages.append({"role": "user", "content": verifier_prompt})
    return messages


def _tool_call_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    return f"{tool_name}({json.dumps(arguments, ensure_ascii=False, sort_keys=True)})"


def _attempted_tool_signatures(tool_calls: list[ToolCallDebug]) -> list[str]:
    return [_tool_call_signature(call.tool_name, call.arguments) for call in tool_calls]


def _provider_complete(
    provider: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    format_schema: dict[str, Any] | str | None = None,
    think: bool | str | None = None,
) -> dict[str, Any]:
    try:
        return provider.complete(messages, tools=tools, format_schema=format_schema, think=think)
    except TypeError:
        return provider.complete(messages, tools=tools, think=think)


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        newline_index = stripped.find("\n")
        if newline_index != -1:
            closing_fence_index = stripped.find("\n```", newline_index)
            if closing_fence_index != -1:
                stripped = stripped[newline_index + 1 : closing_fence_index].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return stripped[start : end + 1]
    return stripped


def _parse_verification_result(text: str) -> ChatVerificationResult:
    candidate = _extract_json_object(text)
    try:
        return ChatVerificationResult.model_validate_json(candidate)
    except Exception:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return ChatVerificationResult.model_validate(
                    {
                        "supported": payload.get("supported", False),
                        "direct_answer": payload.get("direct_answer", payload.get("direct", False)),
                        "reason": payload.get("reason", payload.get("why", "verifier returned incomplete JSON")),
                        "recommended_next_action": payload.get(
                            "recommended_next_action",
                            payload.get("recommended_action", payload.get("next_action", payload.get("action", "search_original_text"))),
                        ),
                    }
                )
        except Exception:
            pass
        supported_match = re.search(r'"supported"\s*:\s*(true|false)', text, re.IGNORECASE)
        direct_match = re.search(r'"direct_answer"\s*:\s*(true|false)', text, re.IGNORECASE)
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
        action_match = re.search(
            r'"recommended_next_action"\s*:\s*"(answer|search_original_text|search_episode_summaries|get_raw_paragraph|get_summary_paragraph)"',
            text,
        )
        if not (supported_match and direct_match and reason_match and action_match):
            return ChatVerificationResult(
                supported=False,
                direct_answer=False,
                reason="verifier returned incomplete JSON",
                recommended_next_action="search_original_text",
            )
        return ChatVerificationResult.model_validate(
            {
                "supported": supported_match.group(1).lower() == "true",
                "direct_answer": direct_match.group(1).lower() == "true",
                "reason": reason_match.group(1),
                "recommended_next_action": action_match.group(1),
            }
        )


def _verify_answer(
    *,
    provider: Any,
    message: str,
    history: list[ChatMessage] | list[SessionMessage],
    draft_answer: str,
    summary_result: dict[str, Any],
    linked_raw_result: dict[str, Any],
    raw_result: dict[str, Any],
    debug: ChatDebugInfo,
    include_timing: bool,
) -> ChatVerificationResult:
    step_started_at = perf_counter()
    verifier_messages = _build_verifier_messages(
        message=message,
        history=history,
        draft_answer=draft_answer,
        summary_result=summary_result,
        linked_raw_result=linked_raw_result,
        raw_result=raw_result,
        attempted_tools=_attempted_tool_signatures(debug.tool_calls),
    )
    verifier_schema = ChatVerificationResult.model_json_schema()
    _record_model_input(
        debug,
        phase="verification",
        messages=verifier_messages,
        tools=None,
        format_schema=verifier_schema,
    )
    try:
        payload = _provider_complete(
            provider,
            verifier_messages,
            tools=None,
            format_schema=verifier_schema,
            think=False,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"LLM provider request failed: {exc}") from exc
    _record_step_timing(debug, "verification", step_started_at, enabled=include_timing)
    return _parse_verification_result(extract_message(payload).get("content", "{}"))


def _build_repair_message(
    verification: ChatVerificationResult,
    attempted_tools: list[str],
) -> str:
    attempted_block = "\n".join(f"- {item}" for item in attempted_tools) if attempted_tools else "(none)"
    return (
        "The previous draft answer was rejected by the verifier.\n"
        f"Reason: {verification.reason}\n"
        f"Preferred next action: {verification.recommended_next_action}\n"
        "Do not repeat the same failed tool actions if they did not directly answer the question.\n"
        f"Already attempted:\n{attempted_block}\n"
        "You must either call a better tool now or, if the evidence is now directly sufficient, answer."
    )


def _build_exhausted_repair_answer(verification: ChatVerificationResult) -> str:
    return (
        "目前在額外檢索後，仍找不到可直接支持答案的證據。"
        f" 先前檢索問題是：{verification.reason}"
    )


def _build_tool_enforcement_message() -> str:
    return (
        "You did not call any retrieval tool. For canon or source lookup questions, you must call a tool before answering.\n"
        "For exact chapter/paragraph questions, prefer `get_raw_paragraph` or `get_summary_paragraph`.\n"
        "For concrete missing details, prefer `search_original_text`.\n"
        "Do not answer yet. Call the best retrieval tool now."
    )


def _should_use_ranked_citations(tool_calls: list[ToolCallDebug]) -> bool:
    return any(call.tool_name in {"get_raw_paragraph", "get_summary_paragraph"} for call in tool_calls)


def _has_direct_location_hit(
    tool_calls: list[ToolCallDebug],
    *,
    summary_result: dict[str, Any],
    raw_result: dict[str, Any],
) -> bool:
    if any(call.tool_name == "get_raw_paragraph" and call.result_count > 0 for call in tool_calls):
        return bool(raw_result.get("hits"))
    if any(call.tool_name == "get_summary_paragraph" and call.result_count > 0 for call in tool_calls):
        return bool(summary_result.get("hits"))
    return False


def _has_chapter_scoped_hit(
    tool_calls: list[ToolCallDebug],
    *,
    summary_result: dict[str, Any],
    raw_result: dict[str, Any],
) -> bool:
    chapter_scoped_tools = [
        call
        for call in tool_calls
        if call.tool_name in {"search_episode_summaries", "search_original_text"}
        and call.arguments.get("chapter_id")
        and call.arguments.get("paragraph_id") is None
        and call.result_count > 0
    ]
    if not chapter_scoped_tools:
        return False

    requested_chapters = {str(call.arguments.get("chapter_id")) for call in chapter_scoped_tools}
    if len(requested_chapters) != 1:
        return False
    requested_chapter = next(iter(requested_chapters))

    candidate_hits = list(summary_result.get("hits", [])) + list(raw_result.get("hits", []))
    if not candidate_hits:
        return False
    return any(hit.get("chapter_id") == requested_chapter for hit in candidate_hits)


def _record_step_timing(debug: ChatDebugInfo, step: str, started_at: float, *, enabled: bool) -> None:
    if not enabled:
        return
    debug.step_timings.append(ChatStepTiming(step=step, elapsed_ms=round((perf_counter() - started_at) * 1000, 2)))


def _record_model_input(
    debug: ChatDebugInfo,
    *,
    phase: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    format_schema: dict[str, Any] | str | None = None,
) -> None:
    tool_names: list[str] = []
    for tool in tools or []:
        function_data = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function_data, dict) and function_data.get("name"):
            tool_names.append(str(function_data["name"]))
    debug.model_inputs.append(
        ChatModelInput(
            phase=phase,
            messages=json.loads(json.dumps(messages, ensure_ascii=False)),
            tools=tool_names,
            format_schema=format_schema,
        )
    )


def _canonicalize_chapter_id(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    chapter_id = value.strip()
    if not chapter_id:
        return chapter_id
    match = re.fullmatch(r"(?i)chapter(?:[\s_:-]+)?0*(\d+)", chapter_id)
    if not match:
        return chapter_id
    return f"Chapter_{int(match.group(1))}"


def _normalize_tool_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    if tool_name in {
        "search_episode_summaries",
        "search_original_text",
        "get_raw_paragraph",
        "get_summary_paragraph",
    }:
        normalized["chapter_id"] = _canonicalize_chapter_id(normalized.get("chapter_id"))
    return normalized


def _execute_tool_call(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    rag_client: RagApiClient,
    debug: ChatDebugInfo,
    include_timing: bool,
) -> dict[str, Any]:
    normalized_arguments = _normalize_tool_arguments(tool_name, arguments)
    try:
        if tool_name == "search_episode_summaries":
            payload = {
                "query": normalized_arguments.get("query", ""),
                "chapter_id": normalized_arguments.get("chapter_id"),
                "timeline_layer": normalized_arguments.get("timeline_layer"),
                "character": normalized_arguments.get("character"),
                "mentioned_character": normalized_arguments.get("mentioned_character"),
                "min_priority_score": normalized_arguments.get("min_priority_score"),
                "tags": normalized_arguments.get("tags") or [],
                "top_k": normalized_arguments.get("top_k", SUMMARY_TOP_K),
            }
            if include_timing:
                result, timings = rag_client.search_summaries_with_timings(payload)
                debug.step_timings.append(
                    ChatStepTiming(step="search_episode_summaries_embed_query", elapsed_ms=timings["embedding_ms"])
                )
                debug.step_timings.append(
                    ChatStepTiming(step="search_episode_summaries_rag_api", elapsed_ms=timings["rag_api_ms"])
                )
            else:
                result = rag_client.search_summaries(payload)
        elif tool_name == "get_linked_original_text":
            payload = {
                "summary_hit_ids": normalized_arguments.get("summary_hit_ids") or [],
                "top_k_per_hit": normalized_arguments.get("top_k_per_hit", LINKED_RAW_TOP_K_PER_HIT),
            }
            step_started_at = perf_counter()
            result = rag_client.get_linked_raw(payload)
            _record_step_timing(debug, "get_linked_original_text", step_started_at, enabled=include_timing)
        elif tool_name == "search_original_text":
            payload = {
                "query": normalized_arguments.get("query", ""),
                "chapter_id": normalized_arguments.get("chapter_id"),
                "paragraph_id": normalized_arguments.get("paragraph_id"),
                "tags": normalized_arguments.get("tags") or [],
                "top_k": normalized_arguments.get("top_k", RAW_TOP_K),
            }
            if include_timing:
                result, timings = rag_client.search_raw_with_timings(payload)
                debug.step_timings.append(
                    ChatStepTiming(step="search_original_text_embed_query", elapsed_ms=timings["embedding_ms"])
                )
                debug.step_timings.append(
                    ChatStepTiming(step="search_original_text_rag_api", elapsed_ms=timings["rag_api_ms"])
                )
            else:
                result = rag_client.search_raw(payload)
        elif tool_name == "get_raw_paragraph":
            payload = {
                "chapter_id": normalized_arguments.get("chapter_id"),
                "paragraph_id": normalized_arguments.get("paragraph_id"),
            }
            step_started_at = perf_counter()
            result = rag_client.get_raw_paragraph(payload)
            _record_step_timing(debug, "get_raw_paragraph", step_started_at, enabled=include_timing)
        elif tool_name == "get_summary_paragraph":
            payload = {
                "chapter_id": normalized_arguments.get("chapter_id"),
                "paragraph_id": normalized_arguments.get("paragraph_id"),
            }
            step_started_at = perf_counter()
            result = rag_client.get_summary_paragraph(payload)
            _record_step_timing(debug, "get_summary_paragraph", step_started_at, enabled=include_timing)
        else:
            raise HTTPException(status_code=500, detail=f"Unsupported tool call: {tool_name}")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"rag-api request failed during {tool_name}: {exc}") from exc

    debug.tool_calls.append(
        ToolCallDebug(
            tool_name=tool_name,
            arguments=normalized_arguments,
            result_count=len(result.get("hits", [])),
        )
    )
    return result


def _run_tool_loop(
    *,
    message: str,
    history: list[ChatMessage] | list[SessionMessage],
    provider,
    rag_client: RagApiClient,
    debug: ChatDebugInfo,
    include_timing: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any], dict[str, Any], list[Citation]]:
    if not hasattr(provider, "build_tool_definitions"):
        return None, {"hits": []}, {"hits": []}, {"hits": []}, []

    messages = _build_tool_loop_messages(message=message, history=history)
    summary_result: dict[str, Any] = {"hits": []}
    linked_raw_result: dict[str, Any] = {"hits": []}
    raw_result: dict[str, Any] = {"hits": []}
    all_citations: list[Citation] = []
    repair_attempts = 0
    last_verification = ChatVerificationResult(
        supported=False,
        direct_answer=False,
        reason="tool loop ended without a directly supported answer",
        recommended_next_action="answer",
    )

    for _ in range(MAX_TOOL_ITERATIONS):
        step_started_at = perf_counter()
        tool_definitions = provider.build_tool_definitions()
        _record_model_input(debug, phase="tool_generation", messages=messages, tools=tool_definitions)
        try:
            payload = _provider_complete(provider, messages, tools=tool_definitions, think=False)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"LLM provider request failed: {exc}") from exc
        _record_step_timing(debug, "tool_generation", step_started_at, enabled=include_timing)

        assistant_message = extract_message(payload)
        tool_calls = extract_tool_calls(assistant_message)
        if not tool_calls:
            if _has_direct_location_hit(
                debug.tool_calls,
                summary_result=summary_result,
                raw_result=raw_result,
            ):
                return assistant_message, summary_result, linked_raw_result, raw_result, all_citations
            if _has_chapter_scoped_hit(
                debug.tool_calls,
                summary_result=summary_result,
                raw_result=raw_result,
            ):
                return assistant_message, summary_result, linked_raw_result, raw_result, all_citations
            if not debug.tool_calls:
                if repair_attempts >= MAX_VERIFIER_REPAIRS:
                    return None, summary_result, linked_raw_result, raw_result, all_citations
                repair_attempts += 1
                if assistant_message.get("content"):
                    messages.append(assistant_message)
                messages.append({"role": "user", "content": _build_tool_enforcement_message()})
                continue
            verification = _verify_answer(
                provider=provider,
                message=message,
                history=history,
                draft_answer=assistant_message.get("content", ""),
                summary_result=summary_result,
                linked_raw_result=linked_raw_result,
                raw_result=raw_result,
                debug=debug,
                include_timing=include_timing,
            )
            last_verification = verification
            if verification.supported and verification.direct_answer and verification.recommended_next_action == "answer":
                return assistant_message, summary_result, linked_raw_result, raw_result, all_citations
            if repair_attempts >= MAX_VERIFIER_REPAIRS:
                return (
                    {"role": "assistant", "content": _build_exhausted_repair_answer(verification)},
                    summary_result,
                    linked_raw_result,
                    raw_result,
                    all_citations,
                )
            repair_attempts += 1
            messages.append(assistant_message)
            messages.append(
                {
                    "role": "user",
                    "content": _build_repair_message(verification, _attempted_tool_signatures(debug.tool_calls)),
                }
            )
            continue

        messages.append(assistant_message)
        for tool_call in tool_calls:
            tool_name = tool_call.get("name") or ""
            arguments = tool_call.get("arguments") or {}
            result = _execute_tool_call(
                tool_name=tool_name,
                arguments=arguments,
                rag_client=rag_client,
                debug=debug,
                include_timing=include_timing,
            )
            if tool_name == "search_episode_summaries":
                summary_result = result
            elif tool_name == "get_linked_original_text":
                linked_raw_result = result
            elif tool_name in {"search_original_text", "get_raw_paragraph"}:
                raw_result = result
            elif tool_name == "get_summary_paragraph":
                summary_result = result
            all_citations.extend(_collect_citations(result))
            messages.append({"role": "tool", "tool_name": tool_name, "content": _tool_result_message(tool_name, result)})

    if debug.tool_calls:
        return (
            {"role": "assistant", "content": _build_exhausted_repair_answer(last_verification)},
            summary_result,
            linked_raw_result,
            raw_result,
            all_citations,
        )
    return None, summary_result, linked_raw_result, raw_result, all_citations


def _run_deterministic_retrieval(
    *,
    message: str,
    rag_client: RagApiClient,
    debug: ChatDebugInfo,
    include_timing: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[Citation]]:
    all_citations: list[Citation] = []

    try:
        if include_timing:
            summary_result, summary_timings = rag_client.search_summaries_with_timings({"query": message, "top_k": SUMMARY_TOP_K})
            debug.step_timings.append(
                ChatStepTiming(step="search_episode_summaries_embed_query", elapsed_ms=summary_timings["embedding_ms"])
            )
            debug.step_timings.append(
                ChatStepTiming(step="search_episode_summaries_rag_api", elapsed_ms=summary_timings["rag_api_ms"])
            )
        else:
            summary_result = rag_client.search_summaries({"query": message, "top_k": SUMMARY_TOP_K})
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"rag-api request failed during search_episode_summaries: {exc}") from exc
    all_citations.extend(_collect_citations(summary_result))
    debug.tool_calls.append(
        ToolCallDebug(
            tool_name="search_episode_summaries",
            arguments={"query": message, "top_k": SUMMARY_TOP_K},
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
                raw_result, raw_timings = rag_client.search_raw_with_timings({"query": message, "top_k": RAW_TOP_K})
                debug.step_timings.append(
                    ChatStepTiming(step="search_original_text_embed_query", elapsed_ms=raw_timings["embedding_ms"])
                )
                debug.step_timings.append(
                    ChatStepTiming(step="search_original_text_rag_api", elapsed_ms=raw_timings["rag_api_ms"])
                )
            else:
                raw_result = rag_client.search_raw({"query": message, "top_k": RAW_TOP_K})
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"rag-api request failed during search_original_text: {exc}") from exc
        all_citations.extend(_collect_citations(raw_result))
        debug.tool_calls.append(
            ToolCallDebug(
                tool_name="search_original_text",
                arguments={"query": message, "top_k": RAW_TOP_K},
                result_count=len(raw_result.get("hits", [])),
            )
        )

    if _needs_broader_search(summary_result, linked_raw_result, raw_result):
        try:
            if include_timing:
                broader_summary_result, summary_timings = rag_client.search_summaries_with_timings(
                    {"query": message, "top_k": EXPANDED_SUMMARY_TOP_K}
                )
                debug.step_timings.append(
                    ChatStepTiming(step="search_episode_summaries_embed_query", elapsed_ms=summary_timings["embedding_ms"])
                )
                debug.step_timings.append(
                    ChatStepTiming(step="search_episode_summaries_rag_api", elapsed_ms=summary_timings["rag_api_ms"])
                )
            else:
                broader_summary_result = rag_client.search_summaries({"query": message, "top_k": EXPANDED_SUMMARY_TOP_K})
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"rag-api request failed during search_episode_summaries: {exc}") from exc
        summary_result = broader_summary_result
        all_citations.extend(_collect_citations(summary_result))
        debug.tool_calls.append(
            ToolCallDebug(
                tool_name="search_episode_summaries",
                arguments={"query": message, "top_k": EXPANDED_SUMMARY_TOP_K},
                result_count=len(summary_result.get("hits", [])),
            )
        )

        summary_hit_ids = _extract_summary_hit_ids(summary_result)
        if summary_hit_ids:
            step_started_at = perf_counter()
            try:
                linked_raw_result = rag_client.get_linked_raw(
                    {"summary_hit_ids": summary_hit_ids, "top_k_per_hit": EXPANDED_LINKED_RAW_TOP_K_PER_HIT}
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
                    arguments={"summary_hit_ids": summary_hit_ids, "top_k_per_hit": EXPANDED_LINKED_RAW_TOP_K_PER_HIT},
                    result_count=len(linked_raw_result.get("hits", [])),
                )
            )

        try:
            if include_timing:
                raw_result, raw_timings = rag_client.search_raw_with_timings({"query": message, "top_k": EXPANDED_RAW_TOP_K})
                debug.step_timings.append(
                    ChatStepTiming(step="search_original_text_embed_query", elapsed_ms=raw_timings["embedding_ms"])
                )
                debug.step_timings.append(
                    ChatStepTiming(step="search_original_text_rag_api", elapsed_ms=raw_timings["rag_api_ms"])
                )
            else:
                raw_result = rag_client.search_raw({"query": message, "top_k": EXPANDED_RAW_TOP_K})
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"rag-api request failed during search_original_text: {exc}") from exc
        all_citations.extend(_collect_citations(raw_result))
        debug.tool_calls.append(
            ToolCallDebug(
                tool_name="search_original_text",
                arguments={"query": message, "top_k": EXPANDED_RAW_TOP_K},
                result_count=len(raw_result.get("hits", [])),
            )
        )

    return summary_result, linked_raw_result, raw_result, all_citations


def run_chat_turn(*, message: str, history: list[ChatMessage] | list[SessionMessage], include_timing: bool) -> ChatResponse:
    started_at = perf_counter()
    settings = get_settings()
    provider = get_llm_provider()
    rag_client = RagApiClient()

    debug = ChatDebugInfo(provider=settings.llm_provider, model=settings.ollama_model, tool_calls=[])
    all_citations: list[Citation] = []

    assistant_message, summary_result, linked_raw_result, raw_result, tool_loop_citations = _run_tool_loop(
        message=message,
        history=history,
        provider=provider,
        rag_client=rag_client,
        debug=debug,
        include_timing=include_timing,
    )
    if assistant_message is not None:
        all_citations.extend(tool_loop_citations)
        ranked_citations = _sort_citations(_dedupe_citations(all_citations))
        if _should_use_ranked_citations(debug.tool_calls):
            trimmed_citations = ranked_citations[:MAX_RETURNED_CITATIONS]
        else:
            trimmed_citations = _select_citations(
                all_citations,
                primary_summary_id=_primary_summary_id(summary_result),
                primary_chapter_id=(_primary_summary_hit(summary_result) or {}).get("chapter_id"),
            )
        debug.iterations = max(1, len(debug.tool_calls) + 1)
        debug.unique_citation_count = len(ranked_citations)
        debug.completed_without_tool_call = len(debug.tool_calls) == 0
        if include_timing:
            debug.elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        return ChatResponse(answer=assistant_message.get("content", ""), citations=trimmed_citations, debug=debug)

    summary_result, linked_raw_result, raw_result, deterministic_citations = _run_deterministic_retrieval(
        message=message,
        rag_client=rag_client,
        debug=debug,
        include_timing=include_timing,
    )
    all_citations.extend(deterministic_citations)

    final_messages = _build_final_messages(
        message=message,
        history=history,
        summary_result=summary_result,
        linked_raw_result=linked_raw_result,
        raw_result=raw_result,
    )
    debug.iterations = 1

    step_started_at = perf_counter()
    _record_model_input(debug, phase="final_generation", messages=final_messages, tools=None)
    try:
        payload = _provider_complete(provider, final_messages, tools=None, think=False)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"LLM provider request failed: {exc}") from exc
    _record_step_timing(debug, "final_generation", step_started_at, enabled=include_timing)

    message = extract_message(payload)
    primary_hit = _primary_summary_hit(summary_result)
    primary_summary_id = _primary_summary_id(summary_result)
    primary_chapter_id = primary_hit.get("chapter_id") if primary_hit else None
    ranked_citations = _sort_citations(_dedupe_citations(all_citations))
    if _should_use_ranked_citations(debug.tool_calls):
        trimmed_citations = ranked_citations[:MAX_RETURNED_CITATIONS]
    else:
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


def run_chat(request: ChatRequest) -> ChatResponse:
    return run_chat_turn(message=request.message, history=request.history, include_timing=request.include_timing)


def build_session_chat_response(
    response: ChatResponse,
    *,
    session_id: str,
    turn_index: int,
    updated_at: str,
) -> SessionChatResponse:
    return SessionChatResponse(
        answer=response.answer,
        citations=response.citations,
        debug=response.debug,
        session_id=session_id,
        turn_index=turn_index,
        updated_at=updated_at,
    )
