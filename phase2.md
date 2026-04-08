# Phase 2

## Current Direction

Phase 2 is now centered on **Codex-native fanfic writing assistance**, not on improving local-model chat quality.

The implemented default path is:

- `postgres` for storage
- `rag-api` for retrieval
- a local STDIO MCP server for Codex at `services/codex_mcp/server.py`
- repo-scoped Codex config in `.codex/config.toml`
- repo guidance in `AGENTS.md`

`agent-api` still exists, but it is now a **legacy opt-in service** and is suspended from default Docker startup.

## Why The Direction Changed

The original Phase 2 plan assumed a local `agent-api` plus Ollama chat experience would remain the main product path.

That is no longer the best fit for the actual use case:

- local Gemma quality is not good enough for production writing help
- the main need is grounded canon lookup for a stronger model
- Codex can already use tools directly through MCP
- the repo already had retrieval contracts worth preserving

So Phase 2 now focuses on:

- exposing the fanfic RAG system to Codex directly
- keeping `rag-api` as the retrieval source of truth
- reducing reliance on Ollama and legacy chat orchestration

## Implemented Architecture

### Default path

1. Codex loads the repo
2. Codex sees `.codex/config.toml`
3. Codex starts the local MCP server
4. the MCP server generates query embeddings locally
5. the MCP server calls `rag-api`
6. Codex uses returned evidence to answer canon questions or assist with drafting

### Writing-only workspace

For actual writing sessions, the preferred workspace is:

- `codex-writing-workspace/`

That folder is intentionally kept free of Python and shell files. It contains only:

- `AGENTS.md`
- `PROMPTS.md`
- `.codex/config.toml`

The MCP server implementation lives outside the writing workspace in:

- `services/codex_mcp/server.py`

This separation exists so Codex does not inspect engineering files by accident during fanfic writing sessions.

### Legacy path

`agent-api` remains available only behind the Docker Compose profile:

- `legacy-agent`

Default startup does **not** include it.

## Implemented Retrieval Surface

### Existing retrieval endpoints still used

- `POST /search/summaries`
- `POST /search/raw`
- `POST /retrieve/linked-raw`
- `POST /retrieve/summary-paragraph`
- `POST /retrieve/raw-paragraph`

### New chapter-level endpoints

- `POST /retrieve/summary-chapter`
- `POST /retrieve/raw-chapter`

These were added so Codex can work with whole-chapter context during:

- outline planning
- continuity checking
- prose/style review
- scene drafting against canon constraints

## Implemented MCP Tools

The MCP server exposes these tools to Codex:

- `fanfic_lookup(question, chapter_id?, mode?)`
- `get_summary_paragraph(chapter_id, paragraph_id)`
- `get_raw_paragraph(chapter_id, paragraph_id)`
- `get_chapter_summary(chapter_id)`
- `get_chapter_text(chapter_id)`

### Tool intent

- `fanfic_lookup`: broad canon lookup and summary-first evidence gathering
- `get_summary_paragraph`: exact structured summary paragraph lookup
- `get_raw_paragraph`: exact raw paragraph lookup
- `get_chapter_summary`: full chapter summary in paragraph order
- `get_chapter_text`: full chapter raw text in paragraph order

`get_linked_original_text` is intentionally kept internal to orchestration rather than exposed directly as a Codex-facing tool.

## Shared Query-Embedding Client

The vectorizing RAG client logic was moved into shared code:

- `shared/rag_client.py`

This matters because:

- `rag-api` still stays retrieval-only
- query embeddings are still generated outside `rag-api`
- the new MCP path can search summaries/raw without depending on `agent-api`
- legacy `agent-api` can still reuse the same client

## Chapter Retrieval Notes

### Summary chapter retrieval

`/retrieve/summary-chapter` returns:

- `chapter_id`
- `source_path`
- `full_summary_text`
- ordered summary paragraphs with citations

### Raw chapter retrieval

`/retrieve/raw-chapter` returns:

- `chapter_id`
- `source_path`
- `full_text`
- ordered raw paragraphs with citations

Important implementation detail:

- raw chunks are stored with overlap
- full chapter text cannot be built by naive concatenation
- the service merges overlapping chunk boundaries before constructing paragraph text

## Codex Repo Guidance

`AGENTS.md` now defines the default working rules for Codex in this repo:

- use MCP tools before answering canon questions from memory
- use chapter-level tools for planning and revision
- use exact paragraph tools for quote verification and precise references
- separate canon-backed facts from newly invented prose
- cite chapter and paragraph when the evidence includes them

The writing-only workspace also has its own `AGENTS.md` and `PROMPTS.md` tuned for drafting and canon-check flows rather than repo engineering.

## MCP Handshake Notes

The MCP server integration required several practical fixes before Codex would attach reliably.

The final state worth documenting is:

- MCP transport is local STDIO
- the server process is launched with unbuffered Python: `python -u`
- the server must not exit immediately after replying to `initialize`
- stdout is reserved for protocol messages only
- stderr is used for logs and handshake debugging
- the server writes newline-delimited JSON responses for STDIO
- the server accepts input defensively during startup

Useful handshake log markers are:

- `server main start`
- `received initialize`
- `writing response id=...`
- `flushed response id=...`

These logs were important for distinguishing:

- startup/import failures
- stale workspace config
- protocol-shape mismatches
- premature post-initialize exit

## Docker Compose State

The default Docker stack is now:

- `postgres`
- `rag-api`

`agent-api` is suspended by default in `docker-compose.yml`.

### Default startup

```bash
docker compose up --build
```

### Explicit legacy startup

```bash
docker compose --profile legacy-agent up --build
```

## What Is Still Legacy

These pieces still exist, but they are not the recommended path:

- `agent-api`
- local web chat UI under `agent-api`
- Ollama-backed `/chat`
- session-oriented legacy chat flow

They remain in the repo for compatibility and manual testing only.

## Acceptance State

Phase 2 is now considered implemented when these conditions hold:

- Codex can load the repo and see the MCP tools
- chapter-level summary and raw retrieval are available through `rag-api`
- default Docker startup does not start `agent-api`
- Codex can use the repo-level instructions in `AGENTS.md`
- the shared vectorized client works without `agent-api`

## Verification Summary

Current implementation was verified with:

- targeted tests for the new chapter retrieval endpoints
- targeted tests for the new MCP server
- full repo test run
- Compose service checks showing:
  - default services: `postgres`, `rag-api`
  - legacy profile services: `postgres`, `rag-api`, `agent-api`
- direct MCP handshake probes against `services/codex_mcp/server.py`
- Codex log inspection showing `initialize` receipt and response flush
