# Local-First Fanfiction RAG

Local-first retrieval-augmented generation for fanfiction writing assistance on macOS. The stack keeps retrieval in a plain HTTP `rag-api`, isolates model/tool-calling behavior inside `agent-api`, and uses Ollama on the host machine instead of in Docker.

## Architecture

- `postgres`: PostgreSQL with `pgvector` for structured summary embeddings and raw-text embeddings.
- `rag-api`: FastAPI retrieval service with vendor-neutral HTTP contracts.
- `agent-api`: FastAPI chat service with a provider abstraction. `OllamaProvider` is implemented for v1.
- `scripts/ingest_data.py`: local ingestion entrypoint for summary markdown and raw episode text.

The retrieval policy is summary-first:

1. search episode summaries
2. if needed, fetch linked original text or search raw text directly
3. answer only from retrieved evidence

This keeps canon lookup compact and fast while still allowing fallback to scene-level wording and nuance.

## Python Setup

Use the project virtual environment for every local Python command.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

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

## Start the Stack

Apply migrations locally if you want to manage schema outside containers:

```bash
source venv/bin/activate
alembic upgrade head
```

Or let the `rag-api` container run migrations at startup.

Start everything except Ollama:

```bash
docker compose up --build
```

Startup order:

1. `postgres` becomes healthy
2. `rag-api` runs migrations and starts
3. `agent-api` starts after `rag-api` is healthy

## Ingest Data

Sample data is included under [data/sample](/Users/jeanycyang/Documents/fanfiction-rag/data/sample).

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

Query embeddings are generated outside `rag-api`. In the current implementation, `agent-api` computes query embeddings and sends them to `rag-api`, so `rag-api` stays retrieval-only and does not depend on PyTorch.

## API Overview

`rag-api`

- `POST /search/summaries`
- `POST /search/raw`
- `POST /retrieve/linked-raw`
- `GET /healthz`

`agent-api`

- `POST /chat`
- `GET /healthz`

Both APIs use explicit Pydantic request/response models and standardized citations containing chapter, paragraph, chunk, source path, and score metadata.

## Example Chat Call

```bash
curl -X POST http://localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "任隊長第一次被提到、但人還沒出現，是在哪一段？"
  }'
```

Other example prompts:

- `任隊長和林妍公開對峙之前，先發生了什麼事？`
- `把這段摘要背後對應的原文也找出來。`

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

## Known Limitations

- v1 uses vector search plus metadata filters only. There is no reranker or BM25 hybrid layer yet.
- chat history is request-scoped and not persisted.
- `OpenAIProvider` is not implemented yet, but the provider abstraction and provider-neutral tool specs are in place.
- local embedding model download can take time on first run.
- the sample data and parsing defaults now assume Traditional Chinese (Taiwan) source material.

## Future Extension Path

- add `OpenAIProvider` or other LLM backends behind the same internal tool spec layer
- add alternative embedding providers through `EmbeddingProvider`
- add a public context-bundle endpoint if external clients need preassembled context
- add MCP only later if it becomes operationally useful
