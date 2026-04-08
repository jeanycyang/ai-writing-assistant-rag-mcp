# Local-First Fanfiction RAG

Local-first retrieval-augmented generation for fanfiction writing assistance on macOS. The primary path is now Codex plus a local MCP server backed by `rag-api`, while `agent-api` remains available only as a legacy opt-in service.

## Architecture

- `postgres`: PostgreSQL with `pgvector` for structured summary embeddings and raw-text embeddings.
- `rag-api`: FastAPI retrieval service with vendor-neutral HTTP contracts.
- `services/codex_mcp/server.py`: local STDIO MCP server that exposes writing-oriented retrieval tools to Codex.
- `agent-api`: legacy FastAPI chat service with a provider abstraction. It is suspended from default Docker startup.
- `scripts/ingest_data.py`: local ingestion entrypoint for summary markdown and raw episode text.

The retrieval policy is summary-first:

1. search episode summaries
2. if needed, fetch linked original text or search raw text directly
3. answer only from retrieved evidence

This keeps canon lookup compact and fast while still allowing fallback to scene-level wording and nuance.

Legacy `agent-api /chat` behavior:

1. deterministically call `search_episode_summaries`
2. deterministically call `get_linked_original_text` for the returned summary hits
3. fall back to `search_original_text` only if summary evidence or linked raw evidence is missing
4. make one final Ollama generation call with the retrieved context

The earlier model-driven multi-turn tool loop was removed from `/chat` because it was unstable and too expensive on local hardware.

## Python Setup

Use the project virtual environment for every local Python command.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

## Quick Start

For production import, use this order:

1. `source venv/bin/activate`
2. `docker compose up -d postgres`
3. `alembic upgrade head`
4. `python scripts/cleanup_sample_data.py`
5. import real OCR data from `~/Documents/ocr/AI_summary_v2` and `~/Documents/ocr/text`

The detailed production import procedure is in [Quick Start: Import Real OCR Data](#quick-start-import-real-ocr-data).

## Ollama on macOS Host

Run Ollama on the host, not in Docker. The default model in `.env.example` is:

```bash
hauhau-gemma4-e4b-q4km
```

The containers call Ollama through:

```bash
http://host.docker.internal:11434
```

Change `OLLAMA_MODEL` in `.env` if you want a different local model.

The default writing workflow no longer depends on Ollama. Ollama is only needed if you explicitly start the legacy `agent-api` profile.

## Codex Writing Workspace

For fanfic writing sessions, open [codex-writing-workspace/README.md](/Users/jeanycyang/Documents/fanfiction-rag/codex-writing-workspace/README.md) instead of the repo root.

That workspace is intentionally documentation-only:

- `AGENTS.md`
- `PROMPTS.md`
- `.codex/config.toml`

It does not contain Python or shell launcher files, so Codex is less likely to inspect implementation code by accident during writing sessions.

The workspace MCP config starts the server from the parent repo:

```toml
[mcp_servers.fanfic_rag]
command = "../venv/bin/python"
args = ["-u", "../services/codex_mcp/server.py"]
cwd = "."
startup_timeout_sec = 10
tool_timeout_sec = 120
```

## MCP Notes

The Codex integration is a local STDIO MCP server, not an HTTP MCP server.

Important implementation details that proved necessary:

- the MCP server must stay alive after replying to `initialize`
- the writing workspace should not contain code launchers
- the server reads STDIO input defensively
- the server writes STDIO output as newline-delimited JSON
- all logs go to `stderr`, never `stdout`
- Python is launched with `-u` so stdout/stderr are unbuffered during MCP startup

If you are debugging Codex MCP startup, inspect:

- `~/.codex/log/codex-tui.log`

Useful log markers from `services/codex_mcp/server.py` are:

- `fanfic_rag mcp: server main start`
- `fanfic_rag mcp: received initialize`
- `fanfic_rag mcp: writing response id=...`
- `fanfic_rag mcp: flushed response id=...`

## Start the Stack

Apply migrations locally if you want to manage schema outside containers:

```bash
source venv/bin/activate
alembic upgrade head
```

Or let the `rag-api` container run migrations at startup.

Start the default stack:

```bash
docker compose up --build
```

Startup order:

1. `postgres` becomes healthy
2. `rag-api` runs migrations and starts

Health semantics:

- `/healthz` means the API process is up
- `/readyz` means the service and its dependencies are actually usable
- `rag-api /readyz` checks PostgreSQL connectivity

Start the legacy `agent-api` only when you explicitly want it:

```bash
docker compose --profile legacy-agent up --build
```

## Ingest Data

Sample data is included under [data/sample](/Users/jeanycyang/Documents/fanfiction-rag/data/sample).

Production rule:

- `data/sample` is test/demo data only
- do not use `data/sample` for production ingestion
- for production, always ingest from the real OCR roots explicitly or set `.env` to the real OCR roots

If sample/demo records were already imported into PostgreSQL, remove them before production import:

```bash
source venv/bin/activate
python scripts/cleanup_sample_data.py
```

Or via `make`:

```bash
make cleanup-sample-data
```

This deletes only records whose `source_path` starts with `data/sample/`.

```bash
source venv/bin/activate
python scripts/ingest_data.py --summary-dir data/sample/summaries --raw-dir data/sample/raw
```

The ingestion pipeline:

- parses structured summary markdown
- validates required fields
- builds retrieval-friendly `embedding_text`
- chunks raw text while preserving chapter and paragraph linkage
- computes embeddings with `sentence-transformers`
- upserts records by stable external ids with `source_hash` stored for inspection

Default embedding choice: `BAAI/bge-m3`. It is multilingual and practical for Traditional Chinese (Taiwan) text on a Mac-centric local setup. The embedding layer is abstracted so a different provider can be added later without changing the retrieval API.

Query embeddings are generated outside `rag-api`. The shared vectorized client is now used by both the local MCP server and the legacy `agent-api`, so `rag-api` stays retrieval-only and does not depend on PyTorch.

The `agent-api` Docker image still prefetches the configured embedding model during build, but that now matters only for the legacy profile.

## Quick Start: Import Real OCR Data

Real source roots used by the project:

- summary markdown: `~/Documents/ocr/AI_summary_v2`
- raw episode markdown: `~/Documents/ocr/text`

Use this flow to import the real Traditional Chinese (Taiwan) source data into PostgreSQL.

1. Start PostgreSQL.

```bash
docker compose up -d postgres
```

2. Apply the schema locally.

```bash
source venv/bin/activate
alembic upgrade head
```

3. Remove any previously imported sample/demo data.

```bash
source venv/bin/activate
python scripts/cleanup_sample_data.py
```

4. Inspect one real summary file and one real raw file before importing.

```bash
sed -n '1,80p' ~/Documents/ocr/AI_summary_v2/Chapter_34_summary.md
sed -n '1,80p' ~/Documents/ocr/text/Chapter_34.md
```

What to verify:

- files are readable as UTF-8
- summaries use `## <paragraph number>` headings
- raw text uses matching `## <paragraph number>` headings when available
- Traditional Chinese wording is intact

5. Run ingestion against the real roots.

```bash
source venv/bin/activate
python scripts/ingest_data.py \
  --summary-dir ~/Documents/ocr/AI_summary_v2 \
  --raw-dir ~/Documents/ocr/text
```

6. Read the JSON result.

Expected output shape:

```json
{
  "summary_files": 34,
  "summary_records": 812,
  "raw_files": 34,
  "raw_chunks": 965
}
```

Important import rules:

- summary and raw files must both be `.md`
- `chapter_id` is derived from the source filename stem
- summary parsing fails loudly if required fields are missing
- paragraph linkage works best when both layers preserve the same `## <number>` headings
- the importer preserves Traditional Chinese and does not convert to Simplified Chinese

If the real source files should become the default local roots, set these in `.env`:

```bash
SUMMARY_DATA_DIR=/Users/jeanycyang/Documents/ocr/AI_summary_v2
RAW_DATA_DIR=/Users/jeanycyang/Documents/ocr/text
```

Then the shorter command works:

```bash
source venv/bin/activate
python scripts/ingest_data.py
```

For production use, prefer setting the real OCR roots in `.env` so the default ingestion command never points at demo content.

## API Overview

`rag-api`

- `POST /search/summaries`
- `POST /search/raw`
- `POST /retrieve/linked-raw`
- `POST /retrieve/summary-paragraph`
- `POST /retrieve/raw-paragraph`
- `POST /retrieve/summary-chapter`
- `POST /retrieve/raw-chapter`
- `GET /healthz`
- `GET /readyz`

Legacy `agent-api`

- `POST /chat`
- `GET /healthz`
- `GET /readyz`

`rag-api` uses explicit Pydantic request/response models and standardized citations containing chapter, paragraph, chunk, source path, and score metadata.

## Codex MCP Setup

```bash
source venv/bin/activate
docker compose up --build
```

Codex CLI and the Codex IDE extension will pick up the project-scoped MCP server from [.codex/config.toml](/Users/jeanycyang/Documents/fanfiction-rag/.codex/config.toml) when you open this repo.

The MCP server exposes these tools:

- `fanfic_lookup`
- `get_summary_paragraph`
- `get_raw_paragraph`
- `get_chapter_summary`
- `get_chapter_text`

If you want Codex to avoid the implementation files and stay focused on writing, open the clean workspace at [codex-writing-workspace/README.md](/Users/jeanycyang/Documents/fanfiction-rag/codex-writing-workspace/README.md) instead of the repo root.

Typical Codex prompts:

- `先用 fanfic 工具確認任隊長第一次被提到、但人還沒出現，是在哪一段。`
- `先抓 Chapter_16 的完整摘要，再幫我規劃下一段衝突。`
- `把 Chapter_16 原文抓出來，我想比對語氣和敘事節奏。`
- `先查 canon，再幫我寫一段新的林妍視角場景，並把你新增的創作部分和 canon 事實分開。`

## Legacy `agent-api`

Example call:

```bash
curl -X POST http://localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "任隊長第一次被提到、但人還沒出現，是在哪一段？",
    "include_timing": true
  }'
```

When `include_timing` is enabled, the response includes:

- `debug.elapsed_ms`: total `/chat` time
- `debug.step_timings`: per-step timings for:
  - `search_episode_summaries_embed_query`
  - `search_episode_summaries_rag_api`
  - `get_linked_original_text`
  - `search_original_text_embed_query` when fallback is used
  - `search_original_text_rag_api` when fallback is used
  - `final_generation`

Current profiling finding:

- `search_episode_summaries_embed_query` was the main cold-start bottleneck
- `agent-api` now bakes the configured embedding model into the Docker image and uses a thread-safe singleton for query embeddings
- after rebuild/recreate, live validation showed `search_episode_summaries_embed_query` around `228-420ms` instead of the earlier ~`62s` cold path
- `get_linked_original_text` is fast in comparison, so raw linked retrieval is not the current hotspot
- `final_generation` is now the dominant latency component in normal `/chat` requests

## Health Checks

Check liveness:

```bash
curl -s http://localhost:8001/healthz
```

Check readiness for the default stack:

```bash
curl -s http://localhost:8001/readyz
```

If you start the legacy profile, you can also check:

```bash
curl -s http://localhost:8002/healthz
curl -s http://localhost:8002/readyz
```

Typical legacy `agent-api /readyz` outcomes:

- `status: ok`: `rag-api` is reachable and the configured Ollama model is available
- `status: degraded`: `rag-api` is down, Ollama is down, or the configured model is missing

## Testing

```bash
source venv/bin/activate
pytest
```

Current local status:

- full test suite passes: `32 passed`
- live `/chat` validation has succeeded across direct lookup, evidence summary, cross-episode reasoning, and insufficient-evidence prompts
- the main remaining runtime quality gap is answer shaping for some nuanced “why” questions

## Reset and Rebuild

```bash
docker compose down -v
docker compose up --build
```

## Known Limitations

- v1 uses vector search plus metadata filters only. There is no reranker or BM25 hybrid layer yet.
- Codex tool use still depends on prompt quality and the repo `AGENTS.md` instructions.
- full chapter retrieval can return large payloads for long chapters.
- legacy `agent-api` remains request-scoped and is not part of the recommended writing workflow.
- the sample data and parsing defaults now assume Traditional Chinese (Taiwan) source material.

## Current TODOs

- continue real writing-flow validation of the Codex MCP tools across outline, drafting, and continuity-check prompts
- tune the `fanfic_lookup` output shape so Codex gets enough evidence without overly large payloads
- decide later whether the legacy `agent-api` should be removed entirely

## Future Extension Path

- add richer MCP tools if recurring writing tasks need them
- add alternative embedding providers through `EmbeddingProvider`
- add a public context-bundle endpoint if external clients need preassembled context
- remove the legacy `agent-api` path entirely if it is no longer useful
