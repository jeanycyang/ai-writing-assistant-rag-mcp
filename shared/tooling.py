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
        description="Search structured episode summaries first for canon lookup.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "chapter_id": {"type": ["string", "null"]},
                "timeline_layer": {"type": ["string", "null"]},
                "character": {"type": ["string", "null"]},
                "mentioned_character": {"type": ["string", "null"]},
                "min_priority_score": {"type": ["number", "null"]},
                "tags": {"type": "array", "items": {"type": "string"}},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="get_linked_original_text",
        description="Fetch raw original text linked to summary search hits when more detail is needed.",
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
        description="Search raw/original text directly when summary evidence is insufficient or ambiguous.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "chapter_id": {"type": ["string", "null"]},
                "paragraph_id": {"type": ["integer", "null"]},
                "tags": {"type": "array", "items": {"type": "string"}},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
    ),
)
