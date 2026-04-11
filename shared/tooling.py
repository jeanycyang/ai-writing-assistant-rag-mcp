from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="search_episode_summaries",
        description="Search structured episode summaries for broad source-backed lookup, timelines, scene overview, and high-level context. Do not use this as the first choice for exact chapter/paragraph requests.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "chapter_id": {"type": "string"},
                "timeline_layer": {"type": "string"},
                "character": {"type": "string"},
                "mentioned_character": {"type": "string"},
                "min_priority_score": {"type": "number"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="get_linked_original_text",
        description="Fetch raw original text linked to summary search hits when more detail is needed after a summary search.",
        parameters={
            "type": "object",
            "properties": {
                "summary_hit_ids": {"type": "array", "items": {"type": "string"}},
                "top_k_per_hit": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["summary_hit_ids"],
        },
    ),
    ToolSpec(
        name="search_original_text",
        description="Search raw/original text directly for concrete details such as occupation, relationship, quotes, wording, or follow-up facts when summaries are insufficient or ambiguous.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "chapter_id": {"type": "string"},
                "paragraph_id": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="get_raw_paragraph",
        description="Retrieve the exact raw/original paragraph by explicit chapter_id and paragraph_id. Use this first for exact chapter/paragraph questions.",
        parameters={
            "type": "object",
            "properties": {
                "chapter_id": {"type": "string"},
                "paragraph_id": {"type": "integer", "minimum": 1},
            },
            "required": ["chapter_id", "paragraph_id"],
        },
    ),
    ToolSpec(
        name="get_summary_paragraph",
        description="Retrieve the exact structured summary paragraph by explicit chapter_id and paragraph_id. Use this only when the user asks for a summary of an exact paragraph.",
        parameters={
            "type": "object",
            "properties": {
                "chapter_id": {"type": "string"},
                "paragraph_id": {"type": "integer", "minimum": 1},
            },
            "required": ["chapter_id", "paragraph_id"],
        },
    ),
)
