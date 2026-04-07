from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from shared.config import get_settings
from shared.embeddings import get_embedding_provider
from shared.parsing import parse_raw_file, parse_summary_file
from shared.repository import RagRepository


def ingest_directories(
    session: Session,
    *,
    summary_dir: str | None = None,
    raw_dir: str | None = None,
) -> dict[str, int]:
    settings = get_settings()
    summary_root = Path(summary_dir or settings.summary_data_dir)
    raw_root = Path(raw_dir or settings.raw_data_dir)

    summary_files = sorted(summary_root.glob("*.md"))
    raw_files = sorted(raw_root.glob("*.md"))
    if not summary_files:
        raise FileNotFoundError(f"No summary markdown files found in {summary_root}")
    if not raw_files:
        raise FileNotFoundError(f"No raw markdown files found in {raw_root}")

    repo = RagRepository(session)
    embedding_provider = get_embedding_provider()

    summary_records = []
    for file_path in summary_files:
        summary_records.extend(parse_summary_file(file_path))
    summary_embeddings = embedding_provider.embed_texts([record.embedding_text for record in summary_records])
    repo.upsert_summary_chunks(summary_records, summary_embeddings)

    raw_records = []
    for file_path in raw_files:
        raw_records.extend(
            parse_raw_file(file_path, chunk_size=settings.raw_chunk_size, overlap=settings.raw_chunk_overlap)
        )
    raw_embeddings = embedding_provider.embed_texts([record.embedding_text for record in raw_records])
    repo.upsert_raw_chunks(raw_records, raw_embeddings)
    session.commit()

    return {
        "summary_files": len(summary_files),
        "summary_records": len(summary_records),
        "raw_files": len(raw_files),
        "raw_chunks": len(raw_records),
    }
