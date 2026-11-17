# Pinecone Neural Agent Harness (v2.6)

Production-oriented **agentic chatbot** with:

- GPT-6 Luna Chat Completions by default (override with `LLM_MODEL`)

- Pinecone vector memory (session-isolated namespaces)
- Mitigated memory (summarization-ready, soft-delete pruning, near-duplicate detection)
- Structured agent loop (tool_call / final_answer)
- Allow-listed structured tool execution
- FastAPI REST API

---

## Quick Start

### 1. Clone / unzip & install

```bash
cd pinecone-agent-harness
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set:
#   PINECONE_API_KEY=...
#   OPENAI_API_KEY=...
```

### 3. Configure authentication and run locally

Generate a unique API key (at least 32 random characters) and set `API_KEYS=<key>:operator` in `.env`. Do not reuse the sample values from other systems; no demo key is shipped.

```bash
./run.sh
```

The local runner binds to `127.0.0.1` by default. Open the interactive docs at http://127.0.0.1:8000/docs. For Docker, publish the container port only on host loopback, e.g. `-p 127.0.0.1:8000:8000`.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service information |
| `GET` | `/health` | Unauthenticated liveness check |
| `GET` | `/health/neural` | Authenticated neural-layer health |
| `POST` | `/chat` | Authenticated agent endpoint |
| `DELETE` | `/sessions/{session_id}` | Delete the caller's session history |

All routes except `/` and `/health` require `X-API-Key` or `Authorization: Bearer`. Session IDs are scoped to the authenticated API-key user.### Example – Chat

```bash
export API_KEY='<your-random-api-key>'
curl -X POST http://127.0.0.1:8000/chat \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message":"Summarize the operational plan and store the approved summary."}'
```

Response shape:

```json
{
  "session_id": "session-a1b2c3d4e5",
  "answer": "...",
  "status": "completed",
  "steps": 3,
  "tokens_used": 1200,
  "embedding_tokens_used": 320,
  "cost_usd": 0.0004
}
```

Reuse the returned `session_id` with the same API-key identity to continue the session.

## Resource and security controls

- Prior conversation history is replayed only for the authenticated user's session and is bounded to recent user/assistant turns.
- Tool execution uses the configured `TOOL_TIMEOUT_S` and `TOOL_OUTPUT_MAX_BYTES` limits; OpenAI requests have bounded `PROVIDER_TIMEOUT_S` and `PROVIDER_MAX_RETRIES`; Redis connections use `REDIS_CONNECT_TIMEOUT_S` and `REDIS_SOCKET_TIMEOUT_S`.
- Each run tracks chat-completion and embedding tokens. `MAX_RUN_USD` uses configurable estimated rates: `LLM_INPUT_COST_PER_1M`, `LLM_OUTPUT_COST_PER_1M`, and `EMBEDDING_COST_PER_1M`. Keep these rates aligned with the active provider/model pricing; provider-side billing can differ from estimates. This estimate does not include Pinecone read/write units, Redis, or server costs.
- Neural storage paths are configurable through `SIDECAR_DB`, `SYNAPTIC_DB`, and `UTILITY_MODEL_PATH`; the health endpoint reads those same paths. Soft-deleted vectors are purged on a subsequent maintenance run after `SOFT_DELETE_RETENTION_DAYS`. Pruning/purge queries are bounded to 100 candidates per invocation.
- The local runner defaults to loopback. Docker binds inside the container; publish it only to host loopback unless an authenticated reverse-proxy/TLS setup is deliberately configured.
- Browser CORS is disabled by default. Set `CORS_ORIGINS` only to explicit trusted origins; wildcard origins are rejected.
- GPT-6 Chat Completions requests use `reasoning_effort=none` and `max_completion_tokens`; legacy model overrides retain the legacy parameter path. Set the token-price environment variables to match the selected model.
