Build a complete local-first RAG system for fanfiction writing assistance on my MacBook Pro M4.

Use these decisions and constraints exactly unless there is a strong technical reason not to:

# Goal

I want a local chat app / agent wrapper that uses a local Ollama model and retrieves from my own RAG server.

Retrieval flow must be:

1. Search structured episode summaries first
2. If the summaries are insufficient, ambiguous, or too compressed, retrieve linked original text chunks
3. Then answer using the retrieved context
4. Do not answer from model memory when retrieval is needed

This is for fanfiction writing assistance based on episode-by-episode plot text.

Important source-language requirement:

* both the structured summaries and the raw/original text are in Traditional Chinese (Taiwan)
* preserve Traditional Chinese throughout ingestion, storage, retrieval, and answers
* do not convert source text to Simplified Chinese
* do not normalize away Taiwan-specific wording unless explicitly requested

# Core stack

Use:

* Ollama for local model serving
* model name: `hauhau-gemma4-e4b-q4km`
* Python `3.12.9`
* FastAPI for the local app / API services
* PostgreSQL with pgvector
* Docker Compose for PostgreSQL and any other containerized services
* plain HTTP between services
* no MCP for v1
* tool calling from the local app to the RAG server
* uv for Python package management if convenient; otherwise pip is okay

Do NOT use heavyweight frameworks unless they clearly simplify things. Prefer straightforward, explicit code over magic.

# Important local execution rule

When running Python commands locally for testing, migrations, ingestion, or development tasks, always activate the project virtual environment first:

`source venv/bin/activate`

Requirements:

* Prefer using the project virtual environment rather than system Python
* Do not use system Python unless there is a clear documented reason
* If the virtual environment does not exist yet, create it first with the project’s documented setup, then activate it before running Python commands
* Prefer commands like:

  * `source venv/bin/activate && python ...`
  * `source venv/bin/activate && alembic ...`
  * `source venv/bin/activate && uvicorn ...`
* Example commands in README should assume the user may activate the environment with:

  * `source venv/bin/activate`

# Architecture

```text
+----------------------------- macOS host ------------------------------+
|                                                                      |
|  +------------------------+                                          |
|  | user / local app       |                                          |
|  | curl / UI / client     |                                          |
|  +------------------------+                                          |
|              |                                                       |
|              | HTTP to published agent-api port                      |
|              v                                                       |
|  +------------------------+                                          |
|  | Ollama                 |                                          |
|  | hauhau-gemma4-e4b-q4km |                                          |
|  +------------------------+                                          |
|                                                                      |
|  +------------------------+                                          |
|  | local ingestion script |                                          |
|  | parse summaries/raw    |                                          |
|  | build embeddings       |                                          |
|  +-----------+------------+                                          |
|              |                                                       |
|              | SQL to published postgres port                        |
+--------------|-------------------------------------------------------+
               |
               | containers -> host via host.docker.internal
               | host -> containers via published ports
               v
+--------------------------- Docker Compose ----------------------------+
|                                                                       |
|  +------------------------+        HTTP        +--------------------+ |
|  | agent-api              | -----------------> | rag-api            | |
|  | FastAPI chat orchestration                        | FastAPI retrieval  | |
|  +-----------+------------+                    +---------+----------+ |
|              |                                             |          |
|              | HTTP via host.docker.internal               | SQL      |
|              v                                             v          |
|  +------------------------+                    +--------------------+ |
|  | Ollama on macOS host   |                    | postgres + pgvector| |
|  +------------------------+                    | summary_chunks     | |
|                                                | raw_chunks         | |
|                                                +--------------------+ |
|                                                                       |
+----------------------------------------------------------------------+

Deployment intent:

- `agent-api`, `rag-api`, and `postgres` run in Docker Compose
- Ollama runs on the macOS host
- the local ingestion script runs on the macOS host and writes to PostgreSQL
- the user/client calls the Dockerized `agent-api`

Summary-first retrieval flow:

user query
  -> agent-api
  -> search_episode_summaries
  -> if insufficient: get_linked_original_text or search_original_text
  -> agent-api composes cited answer
  -> user
```

Current implementation note:

- the original model-driven multi-turn tool loop in `/chat` was replaced
- `/chat` now uses a deterministic summary-first retrieval pipeline:
  1. `search_episode_summaries`
  2. `get_linked_original_text`
  3. optional `search_original_text` fallback
  4. one final Ollama generation call with retrieved context
- this change was made because the earlier loop was unstable and too CPU-expensive in local use

Implement these components:

1. `postgres`

   * PostgreSQL container
   * pgvector enabled
   * persistent volume
   * initialized automatically

2. `rag-api`

   * FastAPI service
   * connects to PostgreSQL
   * stores and retrieves:

     * structured summary chunks
     * raw/original text chunks
   * provides retrieval endpoints

3. `agent-api` or `chat-api`

   * FastAPI service
   * talks to local Ollama over HTTP on host machine
   * exposes a simple chat endpoint
   * owns retrieval orchestration for `/chat`
   * calls `rag-api` directly in a deterministic summary-first order
   * then performs one final model call to produce the answer

Current `/chat` debug contract:

* request supports `include_timing: true`
* response debug includes:
  * `elapsed_ms`
  * `step_timings`
  * tool-call style retrieval trace for the deterministic steps

Current performance finding:

* the original first-call bottleneck was `search_episode_summaries`
* after startup preload and split timing, the meaningful substep is `search_episode_summaries_embed_query`
* PostgreSQL retrieval itself is comparatively fast
* linked raw retrieval is comparatively fast

Next optimization target:

* continue optimizing `search_episode_summaries_embed_query`
* if needed, evaluate a smaller embedding model or additional warmup/caching strategies

Current TODOs worth tracking:

* broaden live end-to-end validation of the rewritten `/chat` beyond the first successful prompt
* continue reducing citation noise for direct factual answers
* continue tightening answer style for concise lookup questions
* refresh docs that may still imply the old tool-driven `/chat` loop
* run the full test suite after the latest performance changes

4. local file-based ingestion script

   * reads input episode text files and summary markdown files
   * parses them
   * generates embeddings
   * inserts records into PostgreSQL

Important:

* Ollama runs locally on macOS host, not inside Docker, unless there is a compelling reason otherwise
* Docker Compose should set up PostgreSQL and the Python services
* Python containers should be able to call Ollama running on the host; use the correct host URL strategy for Docker Desktop on macOS
* all Python code, Dockerfiles, and tooling must target Python `3.12.9`

# LLM provider flexibility

Design this RAG system so it can be used not only by the local Ollama model, but also by other LLM backends in the future, including OpenAI API clients.

Requirements:

* Keep the `rag-api` provider-agnostic
* Do not couple retrieval logic to Ollama-specific request/response shapes
* Keep tool definitions conceptually generic so they can be mapped to:

  * Ollama tool calling
  * OpenAI tool/function calling
  * other future LLM providers

Implement a thin LLM provider abstraction for the `agent-api`, for example:

* `LLMProvider` interface / base class
* `OllamaProvider` implementation for v1
* code structure that makes it easy to add:

  * `OpenAIProvider`
  * other providers later

The retrieval tools and their schemas should be defined once in an internal provider-neutral format, then adapted by each provider integration layer.

At minimum, design the code so these tool operations remain stable across providers:

* `search_episode_summaries`
* `get_linked_original_text`
* `search_original_text`
* optional `build_context_bundle`

Important design constraints:

* `rag-api` must remain a plain HTTP service that can be called by any client
* retrieval request/response models should be clean, explicit, and independent of any LLM vendor
* citations / provenance returned by `rag-api` should be standardized and reusable across different LLM backends
* the `agent-api` should contain provider-specific chat/tool-calling logic, while `rag-api` should not

For v1:

* fully implement `OllamaProvider`
* structure the code so adding `OpenAIProvider` later is straightforward
* do not fully implement OpenAI unless it is easy, but do prepare the architecture for it

If implementing provider flexibility is simple enough, also include:

* environment-variable-based provider selection, for example:

  * `LLM_PROVIDER=ollama`
  * future: `LLM_PROVIDER=openai`
* a provider config section in `.env.example`
* README notes explaining how the architecture can later support OpenAI Chat Completions / Responses style tool calling

The key point:

* RAG retrieval and data storage must be reusable regardless of which LLM is calling it
* provider-specific logic must be isolated in the chat/agent layer

# Retrieval design

I want two linked storage layers.

## A. Structured summary layer

Each summary record should store at least:

* id
* chapter_id
* paragraph_id
* priority_score
* timeline_layer
* scene
* characters (array)
* mentioned_characters (array)
* tags (array)
* key_events (JSON array or text array)
* plot
* embedding_text
* source_path
* source_hash
* embedding
* linked_raw_chunk_ids or enough metadata to retrieve corresponding raw chunks

## B. Raw/original text layer

Each raw chunk should store at least:

* id
* chapter_id
* paragraph_id
* chunk_id
* original_text
* embedding_text
* source_path
* source_hash
* embedding
* linked_summary_id or enough metadata to relate it back

Make the schema explicit and clean. Use Alembic migrations.

# Retrieval behavior

Implement these tool-like operations in the chat service:

1. `search_episode_summaries`

   * input:

     * query
     * optional filters:

       * chapter_id
       * timeline_layer
       * character
       * mentioned_character
       * min_priority_score
       * tags
       * top_k
   * search the summary layer
   * return compact, citation-friendly results

2. `get_linked_original_text`

   * input:

     * summary hit ids
     * top_k_per_hit
   * fetch raw chunks linked to summary hits

3. `search_original_text`

   * input:

     * query
     * optional filters
     * top_k
   * semantic search raw chunks directly

4. optional `build_context_bundle`

   * takes summary hits and optional raw hits
   * returns a compact context package for the model

The agent should prefer summary retrieval first, and only use raw text when needed.

# Model behavior

The system prompt for the local model should enforce:

* Use `search_episode_summaries` first for canon lookup
* If summaries are insufficient, call `get_linked_original_text` or `search_original_text`
* Prefer original text when user asks for:

  * exact evidence
  * exact wording
  * scene nuance
  * dialogue details
* Do not invent facts not supported by retrieved context
* If retrieved evidence is weak or conflicting, say so
* Cite which chapter / paragraph / chunk the answer is based on

# Ingestion

Implement an ingestion pipeline that can load:

* structured summary markdown files
* original episode text files

Assume I already have summary files produced by an LLM prompt similar to this structure:

* `## <paragraph number>`
* `priority_score`
* `timeline_layer`
* `scene`
* `characters`
* `mentioned_characters`
* `tags`
* `key_events`
* `plot`

Requirements:

* parse these safely
* validate required fields
* normalize arrays
* fail loudly on malformed input
* compute embeddings for both summary records and raw chunks
* chunk raw text in a reasonable way while preserving chapter_id / paragraph_id linkage where possible
* use batch insertion where sensible
* make re-import idempotent where practical, preferably using `source_hash`
* store `embedding_text` explicitly for debugging and inspection

For summary records, do not blindly embed the raw markdown. Build a retrieval-friendly `embedding_text` from fields such as:

* chapter_id
* paragraph_id
* timeline_layer
* scene
* characters
* mentioned_characters
* tags
* key_events
* plot

For raw text records, embed the raw chunk text, optionally with a small identifying prefix such as chapter / paragraph / chunk metadata.

# Embeddings

Pick a practical embedding approach that works well for Chinese text on a Mac-centric setup.

Guidelines:

* prefer a local embedding model if reasonably practical
* if local embeddings on the Mac are too weak or inconvenient, structure the code so the embedding provider is pluggable
* create a clear embedding interface / abstraction
* default to one provider, but make it easy to swap later
* do not assume the chat model and embedding model are the same
* keep the embedding provider configurable via environment variables
* optimize for Traditional Chinese (Taiwan) retrieval quality, not generic mixed-language defaults

Explain your choice briefly in README.

# Docker Compose requirements

Write a `docker-compose.yml` that brings up everything needed except Ollama, which should run on the host.

Include at least:

* `postgres`
* `rag-api`
* `agent-api`

Requirements:

* healthchecks
* environment variables via `.env`
* named volumes
* sensible restart policies
* init SQL or migration flow for pgvector
* documentation for startup order

# Project structure

Create a clean repo structure, for example:

* `docker-compose.yml`
* `.env.example`
* `README.md`
* `services/rag_api/...`
* `services/agent_api/...`
* `shared/...`
* `scripts/...`
* `alembic/...`
* `data/...` for local sample input
* `Makefile` if useful

# API design

Implement:

## rag-api

* `POST /search/summaries`
* `POST /search/raw`
* `POST /retrieve/linked-raw`
* `GET /healthz`

## agent-api

* `POST /chat`
* `GET /healthz`

The `/chat` endpoint should:

* accept a user message
* run deterministic summary-first retrieval
* call Ollama once with the selected evidence context
* return:

  * assistant answer
  * retrieved citations
  * debug info such as which tools were called

# Ollama integration

Use Ollama’s HTTP chat API correctly.

Requirements:

* model configurable via env var
* default model `hauhau-gemma4-e4b-q4km`
* support plain chat generation for the current deterministic `/chat` path
* keep the provider abstraction compatible with tool calling if a future workflow needs it
* keep message history in request scope for now; no DB persistence needed unless easy
* assume the model is already installed locally in Ollama, but document how to change the model name in `.env`

# Sample data and demo

Include:

* a tiny sample dataset with a few fake episode summaries and fake raw text files
* a script or make target to ingest the sample data
* an example curl command for `/chat`
* one or two example questions, such as:

  * “When was X first mentioned but not yet present?”
  * “What happened before the confrontation with Y?”
  * “Show the deeper original text behind this summary.”

# Quality requirements

Code quality:

* typed Python where practical
* clear docstrings
* explicit error handling
* no giant files
* avoid unnecessary abstraction
* keep the retrieval logic readable

Operational quality:

* robust healthchecks
* clear logs
* easy local startup
* easy reset / rebuild commands

# README requirements

Write a good README that explains:

* overall architecture
* why summary-first then raw-text fallback is used
* how to run Ollama on macOS host
* how Docker services connect to Ollama on host
* how to start the stack
* how to ingest data
* how to test chat
* known limitations
* future extension path, including optional MCP later

README and example commands should reflect the local virtual environment workflow and show commands with:

`source venv/bin/activate`

where applicable.

# Important design preferences

* Prefer Postgres + pgvector over a separate vector DB
* Keep the design friendly to future extension
* Do not introduce MCP in v1
* Do not over-engineer auth or multi-user features
* Keep citations / provenance explicit
* Keep retrieval deterministic and debuggable
* The RAG backend and retrieval contracts must be reusable by other LLMs in the future, including OpenAI API-based clients, so isolate provider-specific tool-calling/chat logic inside the agent layer and keep the RAG service provider-neutral

# Deliverables

Please generate all necessary files, not just an outline.

At minimum I want:

* `docker-compose.yml`
* `.env.example`
* SQLAlchemy models
* Alembic migrations
* FastAPI apps for `rag-api` and `agent-api`
* ingestion scripts
* sample data
* README
* example curl commands

Also include:

* a clear embedding provider abstraction
* DB models with `embedding_text`
* batch insertion logic where appropriate
* idempotent or deduplicated re-import behavior using `source_hash` where practical

If you need to make implementation choices, choose the simplest robust option and explain them briefly in README.

Also:

* make the system runnable end-to-end locally
* make sure the Python services can reach Ollama running on the macOS host
* verify that the Docker Compose setup is internally consistent
* include comments where setup is tricky
* pin Python tooling and Dockerfiles appropriately for Python `3.12.9`
* treat the Ollama model name as a configurable string and do not assume the embedding model is the same as the chat model
