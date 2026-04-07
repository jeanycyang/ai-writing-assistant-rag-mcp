from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


SUMMARY_HEADER_RE = re.compile(r"^##\s+(?P<paragraph>\d+)\s*$")
RAW_HEADER_RE = re.compile(r"^##\s+(?P<paragraph>\d+)\s*$")
LIST_SPLIT_RE = re.compile(r"[,\uFF0C\u3001;\uFF1B]+")

REQUIRED_SUMMARY_FIELDS = {
    "priority_score",
    "timeline_layer",
    "scene",
    "characters",
    "mentioned_characters",
    "tags",
    "key_events",
    "plot",
}


@dataclass
class ParsedSummaryRecord:
    external_id: str
    chapter_id: str
    paragraph_id: int
    priority_score: float
    timeline_layer: str
    scene: str
    characters: list[str]
    mentioned_characters: list[str]
    tags: list[str]
    key_events: list[str]
    plot: str
    embedding_text: str
    source_path: str
    source_hash: str


@dataclass
class ParsedRawChunkRecord:
    external_id: str
    chapter_id: str
    paragraph_id: int | None
    chunk_id: int
    original_text: str
    embedding_text: str
    source_path: str
    source_hash: str


def compute_source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_list(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return [item.strip() for item in value if item.strip()]
    stripped = value.strip()
    if not stripped:
        return []
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1]
    parts = [part.strip() for part in LIST_SPLIT_RE.split(stripped)]
    return [part for part in parts if part]


def _parse_summary_block(block: str) -> dict[str, object]:
    lines = [line.rstrip() for line in block.splitlines() if line.strip()]
    parsed: dict[str, object] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if ":" not in line:
            raise ValueError(f"Malformed summary field line: {line}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if key == "key_events" and not value:
            events: list[str] = []
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                events.append(lines[i].split("- ", 1)[1].strip())
                i += 1
            parsed[key] = events
            continue
        parsed[key] = value
        i += 1
    missing = REQUIRED_SUMMARY_FIELDS - parsed.keys()
    if missing:
        raise ValueError(f"Missing required summary fields: {sorted(missing)}")
    return parsed


def build_summary_embedding_text(
    chapter_id: str,
    paragraph_id: int,
    data: dict[str, object],
) -> str:
    key_events = data["key_events"] if isinstance(data["key_events"], list) else normalize_list(str(data["key_events"]))
    parts = [
        f"chapter_id: {chapter_id}",
        f"paragraph_id: {paragraph_id}",
        f"timeline_layer: {data['timeline_layer']}",
        f"scene: {data['scene']}",
        f"characters: {', '.join(normalize_list(str(data['characters'])))}",
        f"mentioned_characters: {', '.join(normalize_list(str(data['mentioned_characters'])))}",
        f"tags: {', '.join(normalize_list(str(data['tags'])))}",
        f"key_events: {', '.join(key_events)}",
        f"plot: {data['plot']}",
    ]
    return "\n".join(parts)


def parse_summary_file(path: str | Path) -> list[ParsedSummaryRecord]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    source_hash = compute_source_hash(text)
    chapter_id = file_path.stem
    parts = re.split(r"(?m)^##\s+", text)
    records: list[ParsedSummaryRecord] = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        first_line, _, rest = stripped.partition("\n")
        if not first_line.isdigit():
            raise ValueError(f"Invalid summary paragraph header in {path}: {first_line}")
        paragraph_id = int(first_line)
        parsed = _parse_summary_block(rest)
        external_id = f"{chapter_id}:{paragraph_id}"
        records.append(
            ParsedSummaryRecord(
                external_id=external_id,
                chapter_id=chapter_id,
                paragraph_id=paragraph_id,
                priority_score=float(parsed["priority_score"]),
                timeline_layer=str(parsed["timeline_layer"]),
                scene=str(parsed["scene"]),
                characters=normalize_list(str(parsed["characters"])),
                mentioned_characters=normalize_list(str(parsed["mentioned_characters"])),
                tags=normalize_list(str(parsed["tags"])),
                key_events=parsed["key_events"]
                if isinstance(parsed["key_events"], list)
                else normalize_list(str(parsed["key_events"])),
                plot=str(parsed["plot"]),
                embedding_text=build_summary_embedding_text(chapter_id, paragraph_id, parsed),
                source_path=str(file_path),
                source_hash=source_hash,
            )
        )
    if not records:
        raise ValueError(f"No summary records found in {path}")
    return records


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        if end < len(normalized):
            sentence_boundaries = "。！？；.!?;\n"
            boundary = max(normalized.rfind(marker, start, end) for marker in sentence_boundaries)
            if boundary > start:
                end = boundary + 1
            else:
                space_boundary = normalized.rfind(" ", start, end)
                if space_boundary > start:
                    end = space_boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def parse_raw_file(path: str | Path, chunk_size: int, overlap: int) -> list[ParsedRawChunkRecord]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    source_hash = compute_source_hash(text)
    chapter_id = file_path.stem
    parts = re.split(r"(?m)^##\s+", text)
    records: list[ParsedRawChunkRecord] = []
    global_chunk_id = 0
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        first_line, _, rest = stripped.partition("\n")
        paragraph_id = int(first_line) if first_line.isdigit() else None
        body = rest.strip() if paragraph_id is not None else stripped
        chunks = _chunk_text(body, chunk_size=chunk_size, overlap=overlap)
        for local_idx, chunk in enumerate(chunks):
            paragraph_label = paragraph_id if paragraph_id is not None else "na"
            external_id = f"{chapter_id}:{paragraph_label}:{local_idx}"
            prefix = f"chapter_id: {chapter_id}\nparagraph_id: {paragraph_label}\nchunk_id: {global_chunk_id}\n"
            records.append(
                ParsedRawChunkRecord(
                    external_id=external_id,
                    chapter_id=chapter_id,
                    paragraph_id=paragraph_id,
                    chunk_id=global_chunk_id,
                    original_text=chunk,
                    embedding_text=prefix + chunk,
                    source_path=str(file_path),
                    source_hash=source_hash,
                )
            )
            global_chunk_id += 1
    if not records:
        raise ValueError(f"No raw chunks found in {path}")
    return records
