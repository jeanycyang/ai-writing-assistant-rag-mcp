from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import httpx

from shared.config import Settings, get_settings
from shared.tooling import TOOL_SPECS, ToolSpec


class LLMProvider(ABC):
    @abstractmethod
    def build_tool_definitions(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = httpx.Client(timeout=settings.ollama_request_timeout)

    def build_tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in TOOL_SPECS
        ]

    def complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        response = self._client.post(
            f"{self._settings.ollama_base_url}/api/chat",
            json={
                "model": self._settings.ollama_model,
                "messages": messages,
                "tools": self.build_tool_definitions(),
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json()


def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    if settings.llm_provider == "ollama":
        return OllamaProvider(settings)
    raise ValueError(f"Unsupported llm provider: {settings.llm_provider}")


def extract_message(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("message", {})


def extract_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = message.get("tool_calls") or []
    normalized: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        function_data = tool_call.get("function", {})
        arguments = function_data.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        normalized.append(
            {
                "name": function_data.get("name"),
                "arguments": arguments,
            }
        )
    return normalized
