from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.db import SessionLocal
from shared.ingestion import ingest_directories


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest summary and raw markdown files into PostgreSQL.")
    parser.add_argument("--summary-dir", default=None, help="Directory containing summary markdown files.")
    parser.add_argument("--raw-dir", default=None, help="Directory containing raw markdown files.")
    parser.add_argument(
        "--cache-dir",
        default=".cache/ingest",
        help="Directory used to persist parsed-record and embedding caches for resumable ingestion.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recompute caches even if matching cached stages already exist on disk.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    session = SessionLocal()
    try:
        result = ingest_directories(
            session,
            summary_dir=args.summary_dir,
            raw_dir=args.raw_dir,
            cache_dir=args.cache_dir,
            refresh_cache=args.refresh_cache,
        )
    finally:
        session.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
