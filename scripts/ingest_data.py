from __future__ import annotations

import argparse
import json
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
    args = parser.parse_args()

    session = SessionLocal()
    try:
        result = ingest_directories(session, summary_dir=args.summary_dir, raw_dir=args.raw_dir)
    finally:
        session.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
