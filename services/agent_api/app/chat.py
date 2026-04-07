from __future__ import annotations

import json
from typing import Any

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
        payload = provider.complete(messages)
        message = extract_message(payload)
        tool_calls = extract_tool_calls(message)
        if not tool_calls:
            answer = message.get("content", "")
            return ChatResponse(answer=answer, citations=all_citations, debug=debug)

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
            if tool_name == "search_episode_summaries":
                result = rag_client.search_summaries(arguments)
            elif tool_name == "get_linked_original_text":
                result = rag_client.get_linked_raw(arguments)
            elif tool_name == "search_original_text":
                result = rag_client.search_raw(arguments)
            else:
                raise ValueError(f"Unsupported tool call: {tool_name}")

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

    raise RuntimeError("Model did not return a final answer within the tool loop limit")
