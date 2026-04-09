---
name: text-source
description: Prepare and validate AI writing assistance RAG source files from local summary markdown and raw episode text. Use when working on ingestion, source parsing, source-path configuration, chapter/paragraph linkage, or OCR-derived source files. Triggers on text-source, summary markdown, raw text, OCR text, ingestion source, Traditional Chinese, Taiwan Chinese.
---

# Text Source

Use this skill when the task involves locating, validating, or adapting the source documents that feed the AI writing assistance RAG system.

Current source locations:

- Summary markdown: `~/Documents/ocr/AI_summary_v2/*.md`
- Raw episode text: `~/Documents/ocr/text/*.md`

Production note:

- `data/sample/*` is demo/test-only content
- do not use `data/sample/*` as a production ingestion source
- for production, ingest from the OCR roots above or make those roots the defaults in `.env`
- if sample data was already imported, remove it first with:
  - `source venv/bin/activate && python scripts/cleanup_sample_data.py`

## Critical Rules

- Treat both summaries and raw text as Traditional Chinese used in Taiwan.
- Preserve original script. Do not convert to Simplified Chinese.
- Preserve punctuation, speaker wording, honorifics, and Taiwan-specific lexical choices unless the user explicitly asks for normalization.
- Default file encoding is UTF-8.
- If OCR noise is obvious, fix only deterministic formatting issues during parsing or preprocessing. Do not silently rewrite meaning-bearing text.
- Keep chapter and paragraph linkage stable between summary and raw layers whenever the source files make that possible.

## Quick Start

1. Confirm the source roots exist before changing ingestion logic.
2. Inspect 1-2 real files from each source directory to verify headings, paragraph markers, and encoding.
3. Preserve a direct mapping from source file path to `chapter_id`.
4. Preserve `## <paragraph number>` headings when available and use them as the primary paragraph-link key.
5. If a file deviates from the expected structure, fail loudly unless the user explicitly wants a tolerant repair path.

### Import Real Data Into PostgreSQL

Use the local ingestion script with the real OCR roots:

```bash
source venv/bin/activate
python scripts/ingest_data.py \
  --summary-dir ~/Documents/ocr/AI_summary_v2 \
  --raw-dir ~/Documents/ocr/text
```

Recommended execution order:

1. `docker compose up -d postgres`
2. `source venv/bin/activate && alembic upgrade head`
3. `source venv/bin/activate && python scripts/cleanup_sample_data.py`
4. inspect one summary file and one raw file
5. run the ingestion command above

Useful spot checks:

```bash
sed -n '1,80p' ~/Documents/ocr/AI_summary_v2/Chapter_34_summary.md
sed -n '1,80p' ~/Documents/ocr/text/Chapter_34.md
```

Expected ingestion result:

```json
{
  "summary_files": 34,
  "summary_records": 812,
  "raw_files": 34,
  "raw_chunks": 965
}
```

Notes:

- file names become `chapter_id`
- both roots should contain `.md` files
- malformed summary files should fail loudly with the offending path
- if paragraph headers do not line up between summary and raw files, escalate before inventing a mapping

## Expected Formats

### Summary markdown

Expected per-paragraph structure:

```md
## 12
priority_score: 0.9
timeline_layer: ...
scene: ...
characters: A, B
mentioned_characters: C, D
tags: ...
key_events:
- ...
- ...
plot: ...
```

Handling rules:

- `## <number>` is the paragraph boundary and primary paragraph id.
- `characters`, `mentioned_characters`, and `tags` should be normalized into arrays without changing the underlying names.
- `key_events` may be a bullet list or a compact list-like string; parse safely and preserve Traditional Chinese wording.
- `plot` should remain semantically intact. Do not compress or paraphrase it during ingestion.

### Raw text

Expected format is markdown text, preferably with matching paragraph headings:

```md
## 12
這一段原文...
```

Handling rules:

- If `## <number>` exists, use it as `paragraph_id`.
- If paragraph markers are absent, preserve source order and record that paragraph linkage is partial.
- Chunk conservatively. Do not split aggressively across dialogue or sentence boundaries when a cleaner boundary is available.

## Implementation Guidance

- Build summary `embedding_text` from structured metadata plus plot content, not by embedding the raw markdown blob.
- Build raw `embedding_text` from the original text with a light metadata prefix only.
- Keep `source_path` and `source_hash` for every ingested record.
- Make re-imports idempotent where practical.
- If adding preprocessing, keep it minimal and explicit:
  - normalize line endings
  - trim accidental surrounding whitespace
  - collapse clearly spurious blank-line noise only when it does not affect paragraph structure

## Validation Checklist

- Summary files and raw files are both readable as UTF-8.
- Traditional Chinese characters are preserved end to end.
- Paragraph ids line up between summary and raw sources when headings exist in both.
- Character names, tags, and key events are parsed without script conversion.
- Malformed summary records fail loudly with the source file path called out.
- Raw chunking preserves enough surrounding context for source retrieval and evidence lookup.

## When To Escalate

Escalate to the user before making assumptions if:

- summary and raw files use incompatible chapter naming
- paragraph numbering does not align and cannot be inferred safely
- OCR corruption appears semantic rather than cosmetic
- a proposed normalization step would alter Traditional Chinese wording or Taiwan-specific usage
