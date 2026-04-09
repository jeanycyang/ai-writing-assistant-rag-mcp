# Findings

## Summary

The main problems are in chat orchestration and tool use, not in the vector DB itself.

The recurring failure pattern is:
- the model answers before calling a retrieval tool
- the model chooses the wrong tool or invents a nonexistent one
- the backend tries to repair the turn
- verifier or tool-format fragility turns a recoverable mistake into a bad user result

## Main Findings

### 1. The core weakness is the LLM control loop

The vector DB was often blamed by symptoms, but the actual failures were usually upstream:
- no tool call at all
- wrong tool family selected
- malformed tool output
- unsupported invented tool name
- verifier rejecting or derailing a turn after retrieval had already succeeded

### 2. Exact chapter/paragraph lookup originally failed because of ID mismatch

The model often generated chapter labels like `Chapter 16`, while the stored canonical metadata uses `Chapter_16`.

That caused exact retrieval to miss even when the model was conceptually asking for the right chapter. This was an orchestration/input normalization issue, not a retrieval-index issue.

### 3. The verifier is a major instability source

The verifier depends on the same model to produce valid structured JSON. That creates a fragile loop:
- retrieval may succeed
- draft answer may be usable
- verifier emits incomplete or malformed JSON
- the backend returns a failure message like `verifier returned incomplete JSON`

This means the system can fail even after finding the correct evidence.

### 4. Successful scoped retrieval should not be blocked by verifier gating

There are two especially important scoped success cases:
- exact location lookup such as `get_raw_paragraph(chapter_id, paragraph_id)`
- chapter-scoped retrieval where the request and returned evidence clearly align on the same chapter

If those succeed, the turn should generally proceed instead of being vetoed by a flaky verifier pass.

### 5. The model is weak at disciplined tool use

Even with tools exposed correctly, the model may:
- answer from its own text before retrieval
- invent fake references like `Summary hit ID: 987654321`
- emit unsupported tool names such as `google:search`
- produce messy or half-structured tool chatter

This is a model/tool-discipline problem. The backend can mitigate it, but cannot fully eliminate it if the model is not reliable.

### 6. Unsupported tool names were treated too harshly

When the model hallucinated a tool like `google:search`, the backend surfaced `Unsupported tool call: google:search`.

This does **not** mean online search was actually available or intended. It means the model invented a tool name and the executor treated that as fatal instead of repairable.

### 7. Retrieval breadth was originally too narrow

Earlier settings constrained search too aggressively:
- small `top_k`
- very small final context
- too few linked/raw follow-ups

That caused premature “unknown” answers, especially on follow-up detail questions like occupation, relationship, or scene-specific facts.

Widening retrieval helped, but broader search alone does not solve tool-orchestration failures.

### 8. Session history handling was a separate real bug

The earlier prompt construction flattened prior turns into a synthetic text block instead of passing them as real prior chat messages. That made conversation continuity weak and caused failures like:
- user: `my name is Jean`
- later: `what's my name?`
- model acting as if it had no usable session context

Passing prior turns as actual chat messages is the correct fix for session continuity.

### 9. Debugging visibility was insufficient

Troubleshooting was slowed down because it was hard to see what the model actually received.

Useful improvements include:
- copyable HTTP response payloads per assistant turn
- the actual model inputs recorded in debug output
- display of those inputs on the conversation page

Without that, it is too easy to confuse retrieval failure with orchestration failure.

## What The Evidence Shows

The most common bad sequence has been:
1. user asks a canon question
2. model answers directly without a tool
3. backend forces retry
4. model then calls a tool
5. retrieval succeeds
6. verifier or malformed output still causes a bad final result, or wastes another round

This means the system is spending significant effort compensating for the model instead of simply retrieving and answering.

## Highest-Priority Remaining Fixes

1. Treat unsupported tool names as repairable, not fatal.
2. Do not keep pre-tool bluff answers in the running conversation state before retrying.
3. Reduce or remove verifier hard-gating where scoped retrieval already succeeded.
4. Keep deterministic acceptance for exact-location and clearly chapter-scoped successful retrieval.
5. Continue exposing real model inputs and raw HTTP responses for debugging.
6. If reliable tool use is a hard requirement, consider using a stronger model. The current model is the biggest remaining source of instability.

## Bottom Line

The system’s main problem is not the DB. It is the reliability of model-driven tool orchestration and the fragility of the repair/verifier loop around it.
