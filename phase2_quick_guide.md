# Phase 2 Quick Guide

## Default Stack

Start the default local stack:

```bash
source venv/bin/activate
docker compose up --build
```

This starts:

- `postgres`
- `rag-api`

It does **not** start `agent-api`.

## Codex Setup

Open the repo in Codex CLI or the Codex IDE extension.

Codex should pick up:

- `.codex/config.toml`
- `AGENTS.md`

The repo MCP server exposes these tools:

- `fanfic_lookup`
- `get_summary_paragraph`
- `get_raw_paragraph`
- `get_chapter_summary`
- `get_chapter_text`

## Typical Prompts

- `先用 fanfic 工具確認任隊長第一次被提到、但人還沒出現，是在哪一段。`
- `先抓 Chapter_16 的完整摘要，再幫我規劃下一段衝突。`
- `把 Chapter_16 原文抓出來，我想比對語氣和敘事節奏。`
- `先查 canon，再幫我寫一段新的林妍視角場景，並把你新增的創作部分和 canon 事實分開。`

## RAG API

Check default health:

```bash
curl -sS http://localhost:8001/healthz
curl -sS http://localhost:8001/readyz
```

Example exact paragraph lookup:

```bash
curl -sS -X POST http://localhost:8001/retrieve/raw-paragraph \
  -H 'Content-Type: application/json' \
  -d '{"chapter_id":"Chapter_16","paragraph_id":18}'
```

Example full chapter summary lookup:

```bash
curl -sS -X POST http://localhost:8001/retrieve/summary-chapter \
  -H 'Content-Type: application/json' \
  -d '{"chapter_id":"Chapter_16"}'
```

Example full chapter text lookup:

```bash
curl -sS -X POST http://localhost:8001/retrieve/raw-chapter \
  -H 'Content-Type: application/json' \
  -d '{"chapter_id":"Chapter_16"}'
```

## Legacy `agent-api`

Start the legacy chat service only if you explicitly need it:

```bash
docker compose --profile legacy-agent up --build
```

Legacy health checks:

```bash
curl -sS http://localhost:8002/healthz
curl -sS http://localhost:8002/readyz
```

## Notes

- The recommended writing workflow is Codex + MCP, not `agent-api`.
- Query embeddings are generated outside `rag-api` by the shared client used by the MCP server.
- Full chapter raw text is reconstructed from overlapping stored chunks before being returned.
- `agent-api` is suspended by default and should not start unless the `legacy-agent` profile is requested.
