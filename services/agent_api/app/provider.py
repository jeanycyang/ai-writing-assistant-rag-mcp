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
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        format_schema: dict[str, Any] | str | None = None,
        think: bool | str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def healthcheck(self) -> dict[str, Any]:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = httpx.Client(timeout=settings.ollama_request_timeout)
        self._health_client = httpx.Client(timeout=settings.ollama_health_timeout)

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

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        format_schema: dict[str, Any] | str | None = None,
        think: bool | str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._settings.ollama_model,
            "messages": messages,
            "stream": False,
            "keep_alive": self._settings.ollama_keep_alive,
            "options": {
                "num_predict": self._settings.ollama_num_predict,
            },
        }
        if tools is not None:
            payload["tools"] = tools
        if format_schema is not None:
            payload["format"] = format_schema
        if think is not None:
            payload["think"] = think
        response = self._client.post(
            f"{self._settings.ollama_base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def healthcheck(self) -> dict[str, Any]:
        response = self._health_client.get(f"{self._settings.ollama_base_url}/api/tags")
        response.raise_for_status()
        payload = response.json()
        model_names = [model.get("name", "") for model in payload.get("models", [])]
        configured_model = self._settings.ollama_model
        model_available = configured_model in model_names or f"{configured_model}:latest" in model_names
        return {
            "status": "ok",
            "base_url": self._settings.ollama_base_url,
            "model_configured": configured_model,
            "model_available": model_available,
        }


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
