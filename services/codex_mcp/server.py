from __future__ import annotations

import json
import re
import sys
import time
import traceback
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

SERVER_NAME = "fanfic_rag"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"
ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    from shared.rag_client import RagApiClient


def _log(message: str) -> None:
    print(f"fanfic_rag mcp: {message}", file=sys.stderr, flush=True)


class ToolError(Exception):
    pass


@dataclass(frozen=True)
class McpTool:
    name: str
    description: str
    input_schema: dict[str, Any]


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


def _tool_text(payload: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}], "isError": False}


def _error_response(code: int, message: str, request_id: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _read_message() -> dict[str, Any] | None:
    first_line = sys.stdin.buffer.readline()
    if not first_line:
        return None

    stripped = first_line.strip()
    if stripped.startswith(b"{"):
        return json.loads(stripped.decode("utf-8"))

    content_length: int | None = None
    line = first_line
    while True:
        if line in {b"\r\n", b"\n"}:
            break
        header = line.decode("utf-8").strip()
        if header.lower().startswith("content-length:"):
            content_length = int(header.split(":", 1)[1].strip())
        line = sys.stdin.buffer.readline()
        if not line:
            return None

    if content_length is None:
        raise RuntimeError("Missing Content-Length header")
    body = sys.stdin.buffer.read(content_length)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _write_message(payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    _log(f"writing response id={payload.get('id')} length={len(body)}")
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()
    _log(f"flushed response id={payload.get('id')}")


class FanficMcpServer:
    def __init__(self, rag_client: "RagApiClient | None" = None) -> None:
        self._rag_client = rag_client
        self._initialized = False
        self._tools = {
            tool.name: tool
            for tool in (
                McpTool(
                    name="fanfic_lookup",
                    description=(
                        "Look up canon evidence for a fanfic writing question. Use this first for continuity, "
                        "timeline, relationship, and scene-detail questions."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "chapter_id": {"type": ["string", "null"]},
                            "mode": {
                                "type": ["string", "null"],
                                "enum": [None, "canon_overview", "scene_detail", "exact_quote", "exact_location"],
                            },
                        },
                        "required": ["question"],
                    },
                ),
                McpTool(
                    name="search_summary_by_characters",
                    description=(
                        "Search structured summary paragraphs by one or more exact character names, "
                        "using `operator: \"or\"` or `operator: \"and\"`, with optional chapter range and minimum priority filters."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "characters": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                            "operator": {"type": "string", "enum": ["or", "and"], "default": "or"},
                            "chapter_id": {"type": ["string", "null"]},
                            "from_chapter": {"type": ["integer", "null"], "minimum": 1},
                            "to_chapter": {"type": ["integer", "null"], "minimum": 1},
                            "min_priority_score": {"type": ["number", "null"]},
                            "top_k": {"type": ["integer", "null"], "minimum": 1},
                        },
                        "required": ["characters"],
                    },
                ),
                McpTool(
                    name="get_summary_paragraph",
                    description="Retrieve the exact structured summary for one chapter paragraph.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "chapter_id": {"type": "string"},
                            "paragraph_id": {"type": "integer", "minimum": 1},
                        },
                        "required": ["chapter_id", "paragraph_id"],
                    },
                ),
                McpTool(
                    name="get_raw_paragraph",
                    description="Retrieve the exact raw/original paragraph for one chapter paragraph.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "chapter_id": {"type": "string"},
                            "paragraph_id": {"type": "integer", "minimum": 1},
                        },
                        "required": ["chapter_id", "paragraph_id"],
                    },
                ),
                McpTool(
                    name="get_chapter_summary",
                    description="Retrieve the full structured summary of a chapter in paragraph order.",
                    input_schema={
                        "type": "object",
                        "properties": {"chapter_id": {"type": "string"}},
                        "required": ["chapter_id"],
                    },
                ),
                McpTool(
                    name="get_chapter_text",
                    description="Retrieve the full raw/original text of a chapter in paragraph order.",
                    input_schema={
                        "type": "object",
                        "properties": {"chapter_id": {"type": "string"}},
                        "required": ["chapter_id"],
                    },
                ),
            )
        }

    def _get_rag_client(self) -> "RagApiClient":
        if self._rag_client is None:
            from shared.rag_client import RagApiClient

            self._rag_client = RagApiClient()
        return self._rag_client

    def list_tools(self) -> dict[str, Any]:
        return {
            "tools": [
                {"name": tool.name, "description": tool.description, "inputSchema": tool.input_schema}
                for tool in self._tools.values()
            ]
        }

    def call_tool(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        payload = arguments or {}
        if name == "fanfic_lookup":
            return _tool_text(self._fanfic_lookup(payload))
        if name == "search_summary_by_characters":
            return _tool_text(
                self._get_rag_client().search_summary_characters(
                    {
                        "characters": payload["characters"],
                        "operator": payload.get("operator", "or"),
                        "chapter_id": _canonicalize_chapter_id(payload.get("chapter_id")),
                        "from_chapter": payload.get("from_chapter"),
                        "to_chapter": payload.get("to_chapter"),
                        "min_priority_score": payload.get("min_priority_score"),
                        "top_k": payload.get("top_k"),
                    }
                )
            )
        if name == "get_summary_paragraph":
            return _tool_text(
                self._get_rag_client().get_summary_paragraph(
                    {
                        "chapter_id": _canonicalize_chapter_id(payload["chapter_id"]),
                        "paragraph_id": payload["paragraph_id"],
                    }
                )
            )
        if name == "get_raw_paragraph":
            return _tool_text(
                self._get_rag_client().get_raw_paragraph(
                    {
                        "chapter_id": _canonicalize_chapter_id(payload["chapter_id"]),
                        "paragraph_id": payload["paragraph_id"],
                    }
                )
            )
        if name == "get_chapter_summary":
            return _tool_text(
                self._get_rag_client().get_summary_chapter({"chapter_id": _canonicalize_chapter_id(payload["chapter_id"])})
            )
        if name == "get_chapter_text":
            return _tool_text(
                self._get_rag_client().get_raw_chapter({"chapter_id": _canonicalize_chapter_id(payload["chapter_id"])})
            )
        raise ToolError(f"Unknown tool: {name}")

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params") or {}

        if method == "initialize":
            _log("received initialize")
            self._initialized = True
            requested_version = params.get("protocolVersion")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": requested_version or PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "instructions": (
                        "Use tools to retrieve fanfic canon context before answering continuity or drafting questions."
                    ),
                },
            }
        if method == "notifications/initialized":
            return None
        if method.startswith("notifications/"):
            return None
        if method == "ping":
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}
        if method == "resources/list":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"resources": []}}
        if method == "resources/templates/list":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"resourceTemplates": []}}
        if method == "prompts/list":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"prompts": []}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": request_id, "result": self.list_tools()}
        if method == "tools/call":
            try:
                result = self.call_tool(str(params["name"]), params.get("arguments"))
            except KeyError as exc:
                return _error_response(-32602, f"Missing required argument: {exc.args[0]}", request_id)
            except ToolError as exc:
                return _error_response(-32601, str(exc), request_id)
            except Exception as exc:  # pragma: no cover - defensive protocol boundary
                return _error_response(-32000, str(exc), request_id)
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        if method == "shutdown":
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}
        if method == "exit":
            raise SystemExit(0)
        return _error_response(-32601, f"Method not found: {method}", request_id)

    def _fanfic_lookup(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = payload["question"]
        chapter_id = _canonicalize_chapter_id(payload.get("chapter_id"))
        mode = payload.get("mode") or "scene_detail"

        summary_top_k = 5
        raw_top_k = 5
        linked_top_k = 2
        if mode == "canon_overview":
            summary_top_k = 8
            linked_top_k = 3
        elif mode in {"exact_quote", "exact_location"}:
            raw_top_k = 8

        rag_client = self._get_rag_client()
        summary_result = rag_client.search_summaries(
            {"query": question, "chapter_id": chapter_id, "top_k": summary_top_k}
        )
        summary_hits = summary_result.get("hits", [])

        linked_raw = {"hits": []}
        if summary_hits:
            linked_raw = rag_client.get_linked_raw(
                {"summary_hits": summary_hits, "top_k_per_hit": linked_top_k}
            )

        raw_result = {"hits": []}
        if mode in {"exact_quote", "exact_location"} or not summary_hits or not linked_raw.get("hits"):
            raw_result = rag_client.search_raw({"query": question, "chapter_id": chapter_id, "top_k": raw_top_k})

        raw_hits = list(linked_raw.get("hits", [])) + list(raw_result.get("hits", []))
        citations = [hit.get("citation") for hit in summary_hits + raw_hits if hit.get("citation")]

        if summary_hits and raw_hits:
            confidence = "high"
            suggested_next_step = "answer_from_evidence"
        elif summary_hits or raw_hits:
            confidence = "medium"
            suggested_next_step = "inspect_chapter_text" if chapter_id else "answer_with_caution"
        else:
            confidence = "low"
            suggested_next_step = "insufficient_evidence"

        return {
            "question": question,
            "chapter_id": chapter_id,
            "mode": mode,
            "confidence": confidence,
            "suggested_next_step": suggested_next_step,
            "summary_hits": summary_hits,
            "raw_hits": raw_hits,
            "citations": citations,
        }


def main() -> None:
    _log("server main start")
    server = FanficMcpServer()
    try:
        while True:
            request = _read_message()
            if request is None:
                if server._initialized:
                    _log("stdin closed after initialize; keeping process alive until terminated")
                    signal.pause()
                _log("server main exit")
                return
            response = server.handle_request(request)
            if response is not None:
                _write_message(response)
    except BaseException as exc:
        _log(f"fatal error: {exc.__class__.__name__}: {exc}")
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()


def _read_message() -> dict[str, Any] | None:
    content_length: int | None = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            _log("stdin closed before next message")
            return None
        if line in {b"\r\n", b"\n"}:
            break
        header = line.decode("utf-8").strip()
        if header.lower().startswith("content-length:"):
            content_length = int(header.split(":", 1)[1].strip())

    if content_length is None:
        _log("missing Content-Length header")
        raise RuntimeError("Missing Content-Length header")
    body = sys.stdin.buffer.read(content_length)
    if not body:
        _log("empty message body")
        return None
    _log(f"read message body length={content_length}")
    return json.loads(body.decode("utf-8"))


def _write_message(payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    _log(
        "writing response"
        + (f" method_result_id={payload.get('id')}" if isinstance(payload, dict) else "")
        + f" length={len(body)}"
    )
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def main() -> None:
    _log("server main start")
    server = FanficMcpServer()
    while True:
        request = _read_message()
        if request is None:
            _log("server main exit")
            return
        response = server.handle_request(request)
        if response is not None:
            _write_message(response)


if __name__ == "__main__":
    main()
