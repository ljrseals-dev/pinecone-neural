"""
FastAPI REST API for the Pinecone Agent Harness chatbot.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel, Field

from app.config import settings
from app.harness import AgentHarness
from app.memory import PineconeMemory

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("harness.api")

app = FastAPI(
    title="Pinecone Agent Harness",
    description="Agentic chatbot with mitigated Pinecone memory and structured tool loop",
    version="2.0.0",
)

origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared clients (created once)
pc = Pinecone(api_key=settings.pinecone_api_key)
openai_client = OpenAI(api_key=settings.openai_api_key)

# In-memory session registry (for demo; replace with Redis/DB in production)
SESSIONS: Dict[str, AgentHarness] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: Optional[str] = Field(
        None, description="Reuse an existing session. If omitted a new one is created."
    )
    max_steps: Optional[int] = Field(None, ge=1, le=12)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    status: str
    steps: int
    tokens_used: int


class SessionInfo(BaseModel):
    session_id: str
    created: bool


class MemoryItem(BaseModel):
    id: str
    score: float
    text: str
    role: str
    timestamp: str
    pinned: bool


class HealthResponse(BaseModel):
    status: str
    pinecone_index: str
    llm_model: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        pinecone_index=settings.pinecone_index,
        llm_model=settings.llm_model,
    )


@app.post("/sessions", response_model=SessionInfo)
def create_session():
    session_id = f"session-{uuid.uuid4().hex[:10]}"
    harness = AgentHarness(
        session_id=session_id, pc=pc, openai_client=openai_client
    )
    SESSIONS[session_id] = harness
    return SessionInfo(session_id=session_id, created=True)


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "exists": True}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main chatbot endpoint.
    - If session_id is provided and exists → continue that session
    - Otherwise create a new session
    """
    if req.session_id and req.session_id in SESSIONS:
        harness = SESSIONS[req.session_id]
    else:
        session_id = req.session_id or f"session-{uuid.uuid4().hex[:10]}"
        harness = AgentHarness(
            session_id=session_id, pc=pc, openai_client=openai_client
        )
        SESSIONS[session_id] = harness

    try:
        result = harness.run(req.message, max_steps=req.max_steps)
    except Exception as e:
        logger.exception("Agent run failed")
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        session_id=result["session_id"],
        answer=result["answer"],
        status=result["status"],
        steps=result["steps"],
        tokens_used=result["tokens_used"],
    )


@app.get("/sessions/{session_id}/memory", response_model=List[MemoryItem])
def inspect_memory(session_id: str, q: str = "recent", top_k: int = 8):
    """Retrieve relevant memories for a session (debug / inspection)."""
    if session_id not in SESSIONS:
        # Still allow inspection even if the in-memory harness is gone
        memory = PineconeMemory(
            namespace=session_id, pc=pc, openai_client=openai_client
        )
    else:
        memory = SESSIONS[session_id].memory

    try:
        items = memory.recall(q, top_k=min(top_k, 20))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return [
        MemoryItem(
            id=i["id"],
            score=i["score"],
            text=i["text"],
            role=i["role"],
            timestamp=i["timestamp"],
            pinned=i["pinned"],
        )
        for i in items
    ]


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id in SESSIONS:
        del SESSIONS[session_id]
    return {"session_id": session_id, "deleted": True}


@app.get("/")
def root():
    return {
        "name": "Pinecone Agent Harness",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "chat": "POST /chat",
    }
