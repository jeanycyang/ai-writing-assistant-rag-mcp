from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from time import perf_counter

from sqlalchemy.orm import Session

from shared.config import get_settings
from shared.embeddings import get_embedding_provider
from shared.parsing import parse_raw_file, parse_summary_file
from shared.repository import RagRepository

logger = logging.getLogger(__name__)


def _compute_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_cache_key(
    *,
    summary_files: list[Path],
    raw_files: list[Path],
    embedding_model: str,
    raw_chunk_size: int,
    raw_chunk_overlap: int,
) -> tuple[str, dict[str, object]]:
    manifest = {
        "embedding_model": embedding_model,
        "raw_chunk_size": raw_chunk_size,
        "raw_chunk_overlap": raw_chunk_overlap,
        "summary_files": [
            {"path": str(path), "sha256": _compute_file_hash(path)}
            for path in summary_files
        ],
        "raw_files": [
            {"path": str(path), "sha256": _compute_file_hash(path)}
            for path in raw_files
        ],
    }
    cache_key = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_key, manifest


def _save_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_or_compute_pickle_cache(
    cache_path: Path,
    *,
    refresh_cache: bool,
    stage_name: str,
    compute,
):
    if cache_path.exists() and not refresh_cache:
        logger.info("Cache hit for %s: %s", stage_name, cache_path)
        return _load_pickle(cache_path)
    logger.info("Cache miss for %s", stage_name)
    value = compute()
    _save_pickle(cache_path, value)
    logger.info("Saved %s cache: %s", stage_name, cache_path)
    return value


def ingest_directories(
    session: Session,
    *,
    summary_dir: str | None = None,
    raw_dir: str | None = None,
    cache_dir: str | None = None,
    refresh_cache: bool = False,
) -> dict[str, int]:
    started_at = perf_counter()
    settings = get_settings()
    summary_root = Path(summary_dir or settings.summary_data_dir)
    raw_root = Path(raw_dir or settings.raw_data_dir)

    summary_files = sorted(summary_root.glob("*.md"))
    raw_files = sorted(raw_root.glob("*.md"))
    if not summary_files:
        raise FileNotFoundError(f"No summary markdown files found in {summary_root}")
    if not raw_files:
        raise FileNotFoundError(f"No raw markdown files found in {raw_root}")

    logger.info("Starting ingestion")
    logger.info("Summary root: %s (%d files)", summary_root, len(summary_files))
    logger.info("Raw root: %s (%d files)", raw_root, len(raw_files))

    repo = RagRepository(session)
    resolved_cache_dir = Path(cache_dir or ".cache/ingest")
    cache_key, cache_manifest = _build_cache_key(
        summary_files=summary_files,
        raw_files=raw_files,
        embedding_model=settings.embedding_model,
        raw_chunk_size=settings.raw_chunk_size,
        raw_chunk_overlap=settings.raw_chunk_overlap,
    )
    run_cache_dir = resolved_cache_dir / cache_key
    run_cache_dir.mkdir(parents=True, exist_ok=True)
    (run_cache_dir / "manifest.json").write_text(
        json.dumps(cache_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Using ingestion cache: %s", run_cache_dir)

    logger.info("Loading embedding provider: %s", settings.embedding_model)
    embedding_started_at = perf_counter()
    embedding_provider = get_embedding_provider()
    logger.info("Embedding provider ready in %.2fs", perf_counter() - embedding_started_at)

    def compute_summary_records():
        summary_records = []
        summary_parse_started_at = perf_counter()
        for index, file_path in enumerate(summary_files, start=1):
            records = parse_summary_file(file_path)
            summary_records.extend(records)
            logger.info(
                "Parsed summary file %d/%d: %s (%d records)",
                index,
                len(summary_files),
                file_path.name,
                len(records),
            )
        logger.info(
            "Parsed %d summary records in %.2fs",
            len(summary_records),
            perf_counter() - summary_parse_started_at,
        )
        return summary_records

    summary_records = _load_or_compute_pickle_cache(
        run_cache_dir / "summary_records.pkl",
        refresh_cache=refresh_cache,
        stage_name="summary records",
        compute=compute_summary_records,
    )

    logger.info("Generating summary embeddings for %d records", len(summary_records))
    def compute_summary_embeddings():
        summary_embedding_started_at = perf_counter()
        embeddings = embedding_provider.embed_texts([record.embedding_text for record in summary_records])
        logger.info("Generated summary embeddings in %.2fs", perf_counter() - summary_embedding_started_at)
        return embeddings

    summary_embeddings = _load_or_compute_pickle_cache(
        run_cache_dir / "summary_embeddings.pkl",
        refresh_cache=refresh_cache,
        stage_name="summary embeddings",
        compute=compute_summary_embeddings,
    )

    logger.info("Upserting summary records into PostgreSQL")
    summary_upsert_started_at = perf_counter()
    repo.upsert_summary_chunks(summary_records, summary_embeddings)
    logger.info("Upserted summary records in %.2fs", perf_counter() - summary_upsert_started_at)
    logger.info("Committing summary transaction")
    summary_commit_started_at = perf_counter()
    session.commit()
    logger.info("Committed summary transaction in %.2fs", perf_counter() - summary_commit_started_at)

    def compute_raw_records():
        raw_records = []
        raw_parse_started_at = perf_counter()
        for index, file_path in enumerate(raw_files, start=1):
            records = parse_raw_file(file_path, chunk_size=settings.raw_chunk_size, overlap=settings.raw_chunk_overlap)
            raw_records.extend(records)
            logger.info(
                "Parsed raw file %d/%d: %s (%d chunks)",
                index,
                len(raw_files),
                file_path.name,
                len(records),
            )
        logger.info(
            "Parsed %d raw chunks in %.2fs",
            len(raw_records),
            perf_counter() - raw_parse_started_at,
        )
        return raw_records

    raw_records = _load_or_compute_pickle_cache(
        run_cache_dir / "raw_records.pkl",
        refresh_cache=refresh_cache,
        stage_name="raw records",
        compute=compute_raw_records,
    )

    logger.info("Generating raw embeddings for %d chunks", len(raw_records))
    def compute_raw_embeddings():
        raw_embedding_started_at = perf_counter()
        embeddings = embedding_provider.embed_texts([record.embedding_text for record in raw_records])
        logger.info("Generated raw embeddings in %.2fs", perf_counter() - raw_embedding_started_at)
        return embeddings

    raw_embeddings = _load_or_compute_pickle_cache(
        run_cache_dir / "raw_embeddings.pkl",
        refresh_cache=refresh_cache,
        stage_name="raw embeddings",
        compute=compute_raw_embeddings,
    )

    logger.info("Upserting raw chunks into PostgreSQL")
    raw_upsert_started_at = perf_counter()
    repo.upsert_raw_chunks(raw_records, raw_embeddings)
    logger.info("Upserted raw chunks in %.2fs", perf_counter() - raw_upsert_started_at)

    logger.info("Committing raw transaction")
    commit_started_at = perf_counter()
    session.commit()
    logger.info("Committed raw transaction in %.2fs", perf_counter() - commit_started_at)

    result = {
        "summary_files": len(summary_files),
        "summary_records": len(summary_records),
        "raw_files": len(raw_files),
        "raw_chunks": len(raw_records),
    }
    logger.info("Ingestion finished in %.2fs", perf_counter() - started_at)
    logger.info("Ingestion result: %s", result)
    return result
