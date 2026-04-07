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

Current `agent-api /chat` behavior:

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

Health semantics:

- `/healthz` means the API process is up
- `/readyz` means the service and its dependencies are actually usable
- `rag-api /readyz` checks PostgreSQL connectivity
- `agent-api /readyz` checks both `rag-api` and Ollama, and also verifies that the configured `OLLAMA_MODEL` is available

This means `agent-api` can be live but not ready if Ollama is down or the configured model has not been pulled yet.

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
- `GET /readyz`

`agent-api`

- `POST /chat`
- `GET /healthz`
- `GET /readyz`

Both APIs use explicit Pydantic request/response models and standardized citations containing chapter, paragraph, chunk, source path, and score metadata.

## Example Chat Call

```bash
curl -X POST http://localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "任隊長第一次被提到、但人還沒出現，是在哪一段？"
  }'
```

If you want timing/debug profiling in the response:

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
  - `search_episode_summaries`
  - `get_linked_original_text`
  - `search_original_text` when fallback is used
  - `final_generation`

Current profiling finding:

- the original first-call bottleneck was `search_episode_summaries`
- after startup preload and split timing, the main summary-search cost is now dominated by query embedding generation rather than PostgreSQL retrieval
- `search_episode_summaries_embed_query` is the useful timing to watch for further optimization
- `get_linked_original_text` is fast in comparison, so raw linked retrieval is not the current hotspot

Other example prompts:

- `任隊長和林妍公開對峙之前，先發生了什麼事？`
- `把這段摘要背後對應的原文也找出來。`

## Health Checks

Check liveness:

```bash
curl -s http://localhost:8001/healthz
curl -s http://localhost:8002/healthz
```

Check readiness:

```bash
curl -s http://localhost:8001/readyz
curl -s http://localhost:8002/readyz
```

Typical `agent-api /readyz` outcomes:

- `status: ok`: `rag-api` is reachable and the configured Ollama model is available
- `status: degraded`: `rag-api` is down, Ollama is down, or the configured model is missing

If `model_available` is `false`, pull or switch the model before expecting `/chat` to work.

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

## Current TODOs

- broaden live end-to-end validation of the rewritten `/chat` beyond the initial successful prompt set
- trim citations so direct factual answers return only the most relevant evidence instead of a long citation list
- tighten answer style for direct lookup questions so replies are shorter and less repetitive
- migrate `agent-api` startup wiring away from FastAPI `on_event("startup")` to lifespan
- refresh README/spec wording anywhere that still implies the old model-driven multi-turn `/chat` loop
- run the full test suite after the latest `search_episode_summaries` optimization work

## Future Extension Path

- add `OpenAIProvider` or other LLM backends behind the same internal tool spec layer
- add alternative embedding providers through `EmbeddingProvider`
- add a public context-bundle endpoint if external clients need preassembled context
- add MCP only later if it becomes operationally useful
