# AI Writing Assistance MCP RAG

A retrieval-augmented generation system for AI writing assistance.

## Architecture

- `postgres`: PostgreSQL with `pgvector` for structured summary embeddings and raw-text embeddings.
- `rag-api`: FastAPI retrieval service with vendor-neutral HTTP contracts.

## Preparation
- `scripts/ingest_data.py`: Local ingestion entry point for summary markdown and raw episode text.
- See [Ingest Data](#ingest-data)

## Usage

### For Local Client
- `services/codex_mcp/server.py`: MCP protocol handler used by the local STDIO server and the HTTP MCP endpoint.
- See [Codex Writing Workspace](#codex-writing-workspace)

### For Remote AI Services
- Run `make funnel-up` to start the Tailscale Funnel service. Then set up the MCP integration on your remote AI provider.
- See [Remote MCP Over HTTPS](#remote-mcp-over-https)

## Python Setup

Use the project virtual environment for every local Python command.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

## Codex Writing Workspace

That workspace is intentionally documentation-only:

- `AGENTS.md`
- `PROMPTS.md`
- `.codex/config.toml`

It does not contain Python or shell launcher files, so Codex is less likely to inspect implementation code by accident during writing sessions.

The workspace MCP config starts the server from the parent repo.

### Remote MCP Over HTTPS

The preferred public transport is Tailscale Funnel in front of `rag-api`.

The existing `rag-api` service exposes MCP JSON-RPC over HTTP at:

- `POST /mcp`
- `POST /mcp/{work}`

With Funnel, the public MCP URL becomes:

```text
https://<device-name>.<tailnet>.ts.net/mcp
```

```bash
make funnel-up
make funnel-status
make funnel-url
make funnel-down
```

Default behavior:

- proxies to local `http://127.0.0.1:${RAG_API_PORT:-8001}`
- publishes it on Funnel HTTPS port `443`
- prints the fixed `*.ts.net` URL and the `/mcp` endpoint

Prerequisites:

- Tailscale is installed and logged in on this machine
- MagicDNS and HTTPS are enabled for the tailnet
- Funnel is allowed for the tailnet and this device
- `rag-api` is healthy on the local target port

Example initialize request:

```bash
curl -s https://<device-name>.<tailnet>.ts.net/mcp \
  -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"remote-client","version":"0.1.0"}}}'
```

## Ingest Data

TODO: Use different databases for different works. Support multiple works.

Current implementation direction:

- keep the default database on the existing endpoints
- named work endpoints reuse the same PostgreSQL server/credentials and only switch the database name
- `/mcp/work_id_1` uses the `work_id_1` database, `/mcp/work_id_2` uses the `work_id_2` database
- `AI_WRITING_WORK=work_id_1` binds the local STDIO MCP server to the `work_id_1` database
- use `POST /mcp/{work}` to bind MCP to one work database without exposing work selection to the model
- use `POST /works/{work}/...` retrieval endpoints for direct HTTP access
- use `python scripts/ingest_data.py --work <work>` to ingest into a named work database

Production rule:

- `data/sample` is test/demo data only
- do not use `data/sample` for production ingestion
- for production, always ingest from the real OCR roots explicitly, or set `.env` to those real OCR roots

If sample/demo records were already imported into PostgreSQL, remove them before production import:

```bash
source venv/bin/activate
python scripts/cleanup_sample_data.py
```

Or via `make`:

```bash
make cleanup-sample-data
```

```bash
source venv/bin/activate
python scripts/ingest_data.py --summary-dir data/sample/summaries --raw-dir data/sample/raw
python scripts/ingest_data.py --work sample --summary-dir data/sample/summaries --raw-dir data/sample/raw
```

The ingestion pipeline:

- parses structured summary markdown
- validates required fields
- builds retrieval-friendly `embedding_text`
- chunks raw text while preserving chapter and paragraph linkage
- computes embeddings with `sentence-transformers`
- upserts records by stable external IDs, with `source_hash` stored for inspection

Default embedding choice: `BAAI/bge-m3`. It is multilingual and practical for Traditional Chinese (Taiwan) text in a Mac-centric local setup. The embedding layer is abstracted so a different provider can be added later without changing the retrieval API.

Query embeddings are generated outside `rag-api`. The shared vectorized client is used by the local MCP server, so `rag-api` stays retrieval-only and does not depend on PyTorch.

## Testing

```bash
source venv/bin/activate
pytest
```

## Reset and Rebuild

```bash
docker compose down -v
docker compose up --build
```
