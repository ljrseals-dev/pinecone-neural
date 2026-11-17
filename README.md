# Pinecone Neural Agent Harness (v2.5)

Production-oriented **agentic chatbot** with:

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
  "steps": 3
}
```

Reuse the returned `session_id` with the same API-key identity to continue the session.
