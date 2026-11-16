# Pinecone Agent Harness (v2. 2)

Production-oriented **agentic chatbot** with:

- Pinecone vector memory (session-isolated namespaces)
- Mitigated memory (summarization-ready, soft-delete pruning, near-duplicate detection)
- Structured agent loop (tool_call / final_answer)
- Restricted tool execution
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

### 3. Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open the interactive docs: http://localhost:8000/docs

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Health check |
| `POST` | `/sessions` | Create a new session |
| `GET`  | `/sessions/{id}` | Check session exists |
| `POST` | `/chat` | Main chatbot endpoint |
| `GET`  | `/sessions/{id}/memory` | Inspect retrieved memories |
| `DELETE` | `/sessions/{id}` | Drop in-memory session |

### Example – Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the key differences between Pinecone serverless and pod-based indexes? Store the comparison and summarize it."
  }'
```

Response shape:

```json
{
  "session_id": "session-a1b2c3d4e5",
  "answer": "...",
  "status": "completed",
  "steps": 3,
  "tokens_used": 1842
}
```

Reuse the same `session_id` on subsequent calls to continue the conversation with the same memory namespace.

---

## Architecture Overview

```
Client
  │
  ▼
FastAPI  (/chat, /sessions, /memory)
  │
  ▼
AgentHarness  (structured loop: tool_call | final_answer)
  │
  ├── restricted tool runtime (allow-listed builtins)
  │
  └── PineconeMemory
        ├── content-addressable IDs + near-dupe detection
        ├── soft-delete + prune
        └── token-budgeted context assembly
              │
              ▼
         Pinecone Serverless (per-session namespace)
```

---

## Key Design Decisions

| Concern | Approach |
|---------|----------|
| Memory isolation | One Pinecone namespace per `session_id` |
| Idempotent writes | Content-hash vector IDs |
| Near-duplicates | Cosine ≥ 0.95 → bump access instead of insert |
| Deletion | Soft-delete (`deleted_at`) first |
| Context control | Hard token budget + composition logging |
| Tool safety | Restricted builtins + output size cap + sanitization |
| Termination | JSON schema: `tool_call` or `final_answer` only |
| Cost control | `max_steps` + `max_run_tokens` circuit breakers |

---

## Project Layout

```
pinecone-agent-harness/
├── app/
│   ├── __init__.py
│   ├── config.py          # pydantic-settings
│   ├── memory.py          # PineconeMemory
│   ├── harness.py         # AgentHarness + restricted executor
│   └── main.py            # FastAPI routes
├── tests/
├── .env.example
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Production Notes

- Replace the in-memory `SESSIONS` dict with Redis or a database.
- Run tool execution inside a real sandbox (gVisor / Firecracker / WASM) for untrusted code.
- Add authentication (API keys / JWT) in front of `/chat`.
- Use Pinecone `fetch_by_metadata` with pagination for large-scale pruning.
- Enable OpenTelemetry for traces and the metrics defined in the design review.
- Set `CORS_ORIGINS` to your frontend domain(s).

---

## License

MIT

---

## Neural Memory Layers (v2.1)

The harness now includes the first two neural layers from the full neural memory stack:

| Layer | Name | Status | Training required? |
|-------|------|--------|--------------------|
| 2 | GNN Score Diffusion Reranking | **Implemented** | No (parameter-free) |
| 3 | Synaptic Associative Memory | **Implemented** | No (online Hebbian) |
| 4 | Learned Utility (MemRL) | Interface ready | Yes (MLP) |
| 5 | Neuro-Symbolic KG | Interface ready | No (extractor needed) |
| 1 | Neural Compression | Planned | Optional (PCA or AE) |
| 6 | Differentiable Hippocampal | Frontier | Yes |

### How it works

```
Pinecone ANN  →  GNN diffusion (Layer 2)  →  Synaptic spreading (Layer 3)  →  ranked memories
```

- **GNN diffusion** recovers rank inversions caused by HNSW/PQ noise using 1-hop message passing over the candidate subgraph (~1 ms).
- **Synaptic layer** builds Hebbian edges between co-activated memories and surfaces associative context that pure vector similarity misses.

Both layers degrade gracefully: if the sidecar or graph is unavailable, the system falls back to raw Pinecone scores.

### Files

```
app/neural/
├── gnn_rerank.py      # Layer 2
├── synaptic.py        # Layer 3 (SQLite)
├── sidecar.py         # Full-precision vector store for GNN
└── pipeline.py        # Orchestrates the enhanced recall path
```

Local data is written under `data/synapses.db` and `data/vectors.db`.

---

## v2.2 Hardening + Layer 4

### Critical fixes from the assessment

| Issue | Fix |
|-------|-----|
| `on_coactivation` never called | Wired into agent loop termination with success/failure signal |
| SQLite thread safety | Connection-per-thread for both `vectors.db` and `synapses.db` |
| Sidecar storage bloat | Binary float32 blobs via `struct` (~5× smaller than JSON) |
| Utility scoring missing | Layer 4 MemRL-style MLP implemented with cold-start imitation |

### Layer status (v2.2)

| Layer | Status |
|-------|--------|
| 2 GNN Score Diffusion | ✅ Production-shaped, binary sidecar |
| 3 Synaptic Associative Memory | ✅ Thread-safe, Hebbian feedback wired |
| 4 Learned Utility (MemRL) | ✅ MLP + hand-tuned cold-start + online updates |
| 5 Neuro-Symbolic KG | Interface ready |
| 1 Neural Compression | Planned |
| 6 Differentiable Hippocampal | Frontier |

### Neural feedback loop

At the end of every agent run:

1. All memory IDs that entered context are collected.
2. `on_coactivation(ids, success)` strengthens/weakens Hebbian edges.
3. `on_episode_reward(metas, reward)` updates the utility MLP (reward = 1.0 on clean completion).
