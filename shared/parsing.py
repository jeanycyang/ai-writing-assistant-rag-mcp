from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


SUMMARY_HEADER_RE = re.compile(r"^##\s+(?P<paragraph>\d+)\s*$", re.MULTILINE)
RAW_HEADER_RE = re.compile(r"^##\s+(?P<paragraph>\d+)\s*$", re.MULTILINE)
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

MULTILINE_LIST_FIELDS = {
    "scene",
    "characters",
    "mentioned_characters",
    "tags",
    "key_events",
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


def normalize_chapter_id(path: str | Path) -> str:
    stem = Path(path).stem
    if stem.endswith("_summary"):
        return stem[: -len("_summary")]
    return stem


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
        if not value:
            collected_lines: list[str] = []
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if ":" in next_line and not next_line.lstrip().startswith("- "):
                    break
                if next_line.lstrip().startswith("- "):
                    collected_lines.append(next_line.split("- ", 1)[1].strip())
                else:
                    collected_lines.append(next_line.strip())
                i += 1
            if key in MULTILINE_LIST_FIELDS:
                parsed[key] = collected_lines
            else:
                parsed[key] = "\n".join(collected_lines).strip()
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
    characters = normalize_list(data["characters"])  # type: ignore[arg-type]
    mentioned_characters = normalize_list(data["mentioned_characters"])  # type: ignore[arg-type]
    mentioned_not_present = [item for item in mentioned_characters if item not in characters]
    key_events = data["key_events"] if isinstance(data["key_events"], list) else normalize_list(data["key_events"])  # type: ignore[arg-type]
    scene = " | ".join(data["scene"]) if isinstance(data["scene"], list) else str(data["scene"])
    tags = normalize_list(data["tags"])  # type: ignore[arg-type]
    parts = [
        f"chapter_id: {chapter_id}",
        f"paragraph_id: {paragraph_id}",
        f"timeline_layer: {data['timeline_layer']}",
        f"scene: {scene}",
        f"characters: {', '.join(characters)}",
        f"mentioned_characters: {', '.join(mentioned_characters)}",
        f"mentioned_but_not_present: {', '.join(mentioned_not_present)}",
        f"tags: {', '.join(tags)}",
        f"key_events: {', '.join(key_events)}",
        f"plot: {data['plot']}",
    ]
    return "\n".join(parts)


def parse_summary_file(path: str | Path) -> list[ParsedSummaryRecord]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    source_hash = compute_source_hash(text)
    chapter_id = normalize_chapter_id(file_path)
    records: list[ParsedSummaryRecord] = []
    matches = list(SUMMARY_HEADER_RE.finditer(text))
    for index, match in enumerate(matches):
        paragraph_id = int(match.group("paragraph"))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        rest = text[start:end].strip()
        parsed = _parse_summary_block(rest)
        external_id = f"{chapter_id}:{paragraph_id}"
        records.append(
            ParsedSummaryRecord(
                external_id=external_id,
                chapter_id=chapter_id,
                paragraph_id=paragraph_id,
                priority_score=float(parsed["priority_score"]),
                timeline_layer=str(parsed["timeline_layer"]),
                scene=" | ".join(parsed["scene"]) if isinstance(parsed["scene"], list) else str(parsed["scene"]),
                characters=normalize_list(parsed["characters"]),  # type: ignore[arg-type]
                mentioned_characters=normalize_list(parsed["mentioned_characters"]),  # type: ignore[arg-type]
                tags=normalize_list(parsed["tags"]),  # type: ignore[arg-type]
                key_events=parsed["key_events"]
                if isinstance(parsed["key_events"], list)
                else normalize_list(parsed["key_events"]),  # type: ignore[arg-type]
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
    chapter_id = normalize_chapter_id(file_path)
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
