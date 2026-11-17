import uuid
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.auth import require_api_key
from app.harness import AgentHarness
from app.memory import PineconeMemory
from app.neural.health import neural_health
from app.sessions import make_session_store
from app.sessions.base import SessionStore
from app.tools import build_default_registry
from app.config import settings
from openai import OpenAI
from pinecone import Pinecone

app = FastAPI(title="Pinecone Neural Agent", version="2.5.0")
session_store: SessionStore = make_session_store()
pc = Pinecone(api_key=settings.pinecone_api_key)
openai_client = OpenAI(api_key=settings.openai_api_key)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    session_id: Optional[str] = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    status: str = "ok"
    steps: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "version": "2.5.0"}


@app.get("/health/neural")
def health_neural(_user: Dict[str, Any] = Depends(require_api_key)) -> Dict[str, Any]:
    return neural_health()


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    user: Dict[str, Any] = Depends(require_api_key),
) -> ChatResponse:
    # Session IDs are scoped to the authenticated principal; one API-key user
    # cannot address another user's Redis history or Pinecone namespace.
    principal = str(user["user_id"])
    session_id = req.session_id or str(uuid.uuid4())
    scoped_id = f"{principal}:{session_id}"
    history = await session_store.get(scoped_id) or []
    history.append({"role": "user", "content": req.message})

    memory = PineconeMemory(
        namespace=f"user:{principal}:session:{session_id}",
        pc=pc,
        openai_client=openai_client,
    )
    tools = build_default_registry(memory)
    harness = AgentHarness(
        session_id=session_id,
        pc=pc,
        openai_client=openai_client,
        memory=memory,
        tools=tools,
        conversation_history=history[:-1],
    )
    try:
        result = await harness.run(req.message)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        # Do not disclose API credentials, provider errors, or stack details.
        raise HTTPException(status_code=500, detail="Agent request failed; inspect protected server logs.")

    answer = result.get("answer", "")
    await session_store.set(
        scoped_id,
        history + [{"role": "assistant", "content": answer}],
        ttl=86400,
    )
    return ChatResponse(
        session_id=session_id,
        answer=answer,
        status=result.get("status", "ok"),
        steps=result.get("steps", 0),
        tokens_used=result.get("tokens_used", 0),
        cost_usd=result.get("cost_usd", 0.0),
    )


@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    user: Dict[str, Any] = Depends(require_api_key),
) -> Dict[str, str]:
    if not session_id or len(session_id) > 64 or not all(c.isalnum() or c in "_-" for c in session_id):
        raise HTTPException(status_code=422, detail="Invalid session ID")
    scoped_id = f"{user['user_id']}:{session_id}"
    await session_store.delete(scoped_id)
    return {"deleted": session_id}


@app.get("/")
def root():
    return {
        "name": "Pinecone Neural Agent",
        "version": "2.5.0",
        "docs": "/docs",
        "health": "/health",
        "auth": "X-API-Key or Authorization: Bearer required on agent endpoints",
    }
