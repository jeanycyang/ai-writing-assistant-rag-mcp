# Phase 2 Quick Guide

## Start

Start the local stack:

```bash
source venv/bin/activate
docker compose up --build
```

Open the local UI:

```text
http://localhost:8002/
```

## Web UI

Use the local chat UI like this:

1. Click `New Session` for a fresh chat.
2. Type a question in the composer.
3. Click `Send`.
4. Ask follow-up questions in the same session.
5. Switch sessions from the left sidebar.
6. Expand `Citations` or `Debug` under assistant replies when needed.

## API

Create a session:

```bash
curl -sS -X POST http://localhost:8002/sessions
```

List sessions:

```bash
curl -sS http://localhost:8002/sessions
```

Get one session transcript:

```bash
curl -sS http://localhost:8002/sessions/<session_id>
```

Send a message to a session:

```bash
curl -sS -X POST http://localhost:8002/sessions/<session_id>/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"任隊長第一次被提到是在哪裡？","include_timing":true}'
```

Delete a session:

```bash
curl -sS -X DELETE http://localhost:8002/sessions/<session_id>
```

Stateless chat still works:

```bash
curl -sS -X POST http://localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"任隊長第一次被提到是在哪裡？","history":[],"include_timing":true}'
```

## Notes

- Session history is in memory only.
- If `agent-api` restarts, sessions disappear.
- Retrieval still runs fresh on every turn.
- Follow-up context comes from recent session history, not persisted retrieval memory.
