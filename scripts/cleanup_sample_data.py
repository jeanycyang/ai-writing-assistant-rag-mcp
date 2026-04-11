from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import delete, func, select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.db import create_session
from shared.models import RawChunk, SummaryChunk

DEFAULT_PREFIX = "data/sample/"


def cleanup_by_source_prefix(source_prefix: str, work: str | None = None) -> dict[str, int | str]:
    session = create_session(work)
    try:
        summary_count = session.scalar(
            select(func.count()).select_from(SummaryChunk).where(SummaryChunk.source_path.like(f"{source_prefix}%"))
        )
        raw_count = session.scalar(
            select(func.count()).select_from(RawChunk).where(RawChunk.source_path.like(f"{source_prefix}%"))
        )

        session.execute(delete(RawChunk).where(RawChunk.source_path.like(f"{source_prefix}%")))
        session.execute(delete(SummaryChunk).where(SummaryChunk.source_path.like(f"{source_prefix}%")))
        session.commit()
    finally:
        session.close()

    return {
        "source_prefix": source_prefix,
        "work": work or "",
        "deleted_summary_records": int(summary_count or 0),
        "deleted_raw_records": int(raw_count or 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete ingested records by source_path prefix.")
    parser.add_argument("--work", default=None, help="Named work/database to target.")
    parser.add_argument(
        "--source-prefix",
        default=DEFAULT_PREFIX,
        help="Delete records whose source_path starts with this prefix.",
    )
    args = parser.parse_args()
    print(json.dumps(cleanup_by_source_prefix(args.source_prefix, args.work), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
