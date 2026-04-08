# Session

## Goal

Add interactive in-memory chat sessions so the local app behaves more like `ollama run <model>` or a normal chat panel with follow-up turns.

This phase does **not** require persistent conversation storage.

## Desired Behavior

- user starts a session
- user sends multiple follow-up turns
- `agent-api` keeps session history in memory
- retrieval still runs fresh on every new turn
- if `agent-api` restarts, the session is gone
- a small local web UI can use the same session backend and feel like a normal AI chat app

## Non-Goals

- no database-backed chat history
- no long-term memory
- no vector memory over old chat transcripts
- no autonomous multi-step tool loop

## Local Web UI

Phase 2 should also build a small local web UI on top of the session backend.

Target experience:

- looks and behaves like a simple AI chat interface
- closer to ChatGPT than to a raw API playground
- user can create a session and continue follow-up turns in one panel
- messages appear as a normal chat transcript
- citations and debug details may be collapsible instead of always visible

Recommended scope:

- local-only UI
- minimal implementation
- no auth
- no persistence beyond the in-memory session API

Recommended features:

- create a new session
- list or switch active sessions
- send a message to the current session
- render assistant replies in a chat transcript
- optional expandable section for citations and debug/timing
- loading/error states

Recommended implementation approach:

- keep it small and pragmatic
- use the existing `agent-api` session endpoints as the backend
- do not duplicate retrieval logic in the frontend

## Endpoint Rule

- keep existing `POST /chat` behavior unchanged
- do not add hidden server-side session behavior to `POST /chat`
- add session-aware chat alongside it
- reuse the same internal chat engine for both paths

## Recommended Endpoints

### `POST /chat`

Keep this as the current stateless endpoint.

Behavior:

- caller may pass `history`
- one-shot request/response
- no server-side memory

### `POST /sessions`

Create a new in-memory session.

Response:

- `session_id`
- `created_at`
- `updated_at`

### `GET /sessions`

Optional but useful for local UI/debugging.

Response:

- active in-memory sessions

### `GET /sessions/{id}`

Optional inspection endpoint.

Response:

- session metadata
- recent messages

### `POST /sessions/{id}/chat`

Primary interactive endpoint.

Request:

- `message`
- optional timing/debug flags

Response:

- `answer`
- `citations`
- `debug`
- optional session metadata

### `DELETE /sessions/{id}`

Explicitly clear one session.

## Session Model

Use an in-memory session manager inside `agent-api`.

Each session should contain:

- `session_id`
- `created_at`
- `updated_at`
- `messages`
- optional `title`
- optional `last_debug`

Each message should contain:

- `role`
- `content`
- `created_at`

Recommended message roles:

- `system`
- `user`
- `assistant`

Do **not** persist tool-call transcripts as session memory by default.
Do **not** persist retrieved raw context as canonical session state.

## Session Lifecycle

### Create

Client creates a new session and receives a `session_id`.

### Chat Turn

For each session-backed turn:

1. load session from memory
2. append new user turn
3. derive recent history window
4. run deterministic retrieval for the new user turn
5. run one final model generation using:
   - recent history
   - fresh retrieval context
6. append assistant reply
7. return answer, citations, and debug

### Expire

Sessions should expire automatically after idle timeout.

Recommended defaults:

- idle TTL: `30-60` minutes
- max sessions: bounded, e.g. `100`
- max stored turns per session: bounded, e.g. last `20-40` messages

### Delete

Support explicit session deletion.

## History Handling

The model should not receive unlimited history.

Recommended strategy:

- include only recent turns in the final prompt
- keep retrieval based on the newest user turn
- treat prior turns as conversational context, not as canon evidence

Suggested first implementation:

- keep last `6-10` user/assistant messages
- drop older turns silently for v1

Optional later enhancement:

- summarize older turns into one compact system note

## Chat Flow

Phase 2 should preserve the current deterministic retrieval pattern.

Implementation note:

- refactor the current `/chat` logic into one reusable internal turn function
- keep `POST /chat` calling that function with caller-supplied `history`
- let `POST /sessions/{id}/chat` load in-memory session history and call the same function
- keep one RAG path for both stateless and session-backed chat

Per turn:

1. accept new user message
2. load recent history
3. call `search_episode_summaries`
4. call `get_linked_original_text`
5. call `search_original_text` only if necessary
6. build final prompt from:
   - recent session history
   - strongest retrieved evidence
7. generate one answer
8. append assistant reply to session memory

Important:

- retrieval is run per turn
- prior retrieved citations are not reused as memory
- “not stated / insufficient evidence” rules remain in force

## Recommended Data Structures

### In-memory session manager

Responsibilities:

- create session
- fetch session
- append message
- evict expired sessions
- clear session

Recommended implementation:

- plain Python dict protected by a lock
- session objects stored by `session_id`
- lazy TTL cleanup on read/write

That is enough for local single-process use.

## Debug Behavior

Keep current debug behavior and extend it for session chat responses.

Return:

- provider/model
- elapsed time
- step timings
- deterministic retrieval trace
- session id
- optional turn index

Do not expose full internal session memory by default in normal `/chat` responses.

## Testing Plan

### Unit tests

- create session
- append turns
- TTL expiry
- delete session
- history window truncation
- thread-safe session manager behavior

### API tests

- `POST /sessions`
- `POST /sessions/{id}/chat`
- `GET /sessions/{id}`
- `DELETE /sessions/{id}`
- invalid session id returns `404`

### End-to-end checks

- start a session
- ask one question
- ask a follow-up that depends on prior turn wording
- confirm session history is used
- confirm retrieval is still run on the new turn

## Minimal Successful Outcome

Phase 2 session work is successful if:

- a user can create a session and send follow-up messages without resending full history
- session state is in memory only
- current deterministic retrieval quality is preserved


# Provider-Agnostic

## Goal

Finish the provider boundary so the system is not effectively Ollama-only in its internal shapes, while keeping `rag-api` retrieval contracts vendor-neutral and exposing one stable local conversation backend for editor/chat clients.

Primary user-facing target:

- the Codex panel can use the RAG backend

Secondary target:

- a VS Code chat panel can use the same backend

Important clarification:

- this section is not asking for multiple UIs to be implemented in Phase 2
- it is asking for one backend design that those clients can call without depending on Ollama-specific request/response shapes
- `agent-api` is that backend

## Desired Outcome

- `rag-api` remains plain HTTP and provider-neutral
- `agent-api` can swap model backends without rewriting chat orchestration
- Codex panel integration is possible through one stable local API surface
- a VS Code chat panel can also call the same backend
- tool support remains optional instead of defining the whole architecture

## Current State

Already present:

- `LLMProvider` base class
- `OllamaProvider`
- provider selection through settings
- provider-neutral retrieval APIs in `rag-api`

Still incomplete:

- message model is still effectively Ollama-first
- response extraction helpers are Ollama-shaped
- there is no second provider implementation proving the abstraction
- the client-facing backend contract is not yet described clearly enough for Codex panel / VS Code chat panel use

## Scope

Finish provider-agnostic design at two distinct levels:

1. model backend abstraction inside `agent-api`
2. client-facing conversation API so editor/chat clients can call the same local RAG service

This section is **not** the same as “implement OpenAI now”.
It is about making sure:

- model-provider internals are normalized
- client-facing chat endpoints stay stable
- Codex panel is the primary integration target
- VS Code chat panel can reuse the same backend contract

Implementation note:

- do not guess how Codex panel integration or VS Code-side integration should work
- search online official documentation before implementation
- prefer official Codex, VS Code, extension, or chat integration docs over informal examples

## Internal Provider Contract

Define a cleaner internal provider contract around:

- input messages
- final text generation
- health check
- optional structured tool support

Recommended internal types:

- `ProviderMessage`
- `ProviderResponse`
- `ProviderHealth`

Recommended provider methods:

- `complete(messages, *, max_output_tokens?, temperature?, stop?)`
- `healthcheck()`
- optional future:
  - `supports_tools()`
  - `build_tool_definitions()`

Important:

- current stateless chat and future session-based chat should still use deterministic retrieval and one final completion
- tool support stays optional in the provider interface
- do not design this phase around model-driven tool loops again

## Message Normalization

Normalize internal message shape so it is not Ollama-specific.

Recommended internal message model:

```text
role: system | user | assistant
content: string
```

Optional future fields:

- `name`
- `metadata`

Provider adapters should translate this internal format into:

- Ollama chat payloads
- future OpenAI Responses / Chat Completions payloads
- any other backend-specific message shape

## Response Normalization

Normalize provider output into a simple internal response model:

```text
content: string
raw_payload: dict
provider_name: string
model_name: string
```

This keeps chat orchestration code from depending on Ollama-specific response fields.

## Provider Configuration

Make provider configuration explicit and complete in settings and docs.

Keep:

- `LLM_PROVIDER=ollama`

Prepare for:

- `LLM_PROVIDER=openai`
- `LLM_PROVIDER=<other>`

Recommended config groups by provider:

### Ollama

- base URL
- model name
- timeout
- keep-alive
- generation cap

### OpenAI-ready placeholders

- API key
- model name
- base URL override if needed
- request timeout

Do not fully implement OpenAI if it is not needed yet, but make adding it straightforward.

## Client-Facing Backend Contract

The Codex panel and any future VS Code chat panel integration should talk to `agent-api`, not directly to provider-specific model APIs.

Recommended principle:

- clients talk to `agent-api`
- `agent-api` owns session memory and retrieval orchestration
- `rag-api` remains retrieval-only

The intended consumers here include:

- the Codex panel
- a VS Code chat panel
- a small local web UI
- a CLI REPL

The important requirement is:

- these clients should be able to use the RAG system through one stable local API surface
- they should not need to reimplement retrieval orchestration
- they should not depend on Ollama-specific request/response shapes

This is provider-agnostic at the client layer because:

- the client does not need to know whether the backend model is Ollama or something else
- the client does not need to know how retrieval was orchestrated internally
- the client only needs the stable `agent-api` contract

## Testing Plan

### Provider unit tests

- provider-independent message normalization
- provider response normalization
- healthcheck contract normalization

### Adapter tests

- Ollama adapter remains green against the normalized provider contract
- adding a second provider should require a new adapter, not a rewrite of chat orchestration

### Client-contract checks

- validate that the exposed conversation API is stable enough for Codex panel usage
- validate that the same backend contract can also support a VS Code chat panel
- confirm that retrieval remains internal to `agent-api` and is not pushed into the client

## Minimal Successful Outcome

Provider-agnostic work is successful if:

- the provider boundary is cleaner and no longer tightly coupled to Ollama-shaped messages/responses
- the Codex panel can use `agent-api` as the stable local RAG conversation backend
- a VS Code chat panel can also use the same backend contract
- adding a future provider is straightforward and does not affect `rag-api`
