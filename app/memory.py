"""
Mitigated Pinecone memory layer with:
- content-addressable IDs
- near-duplicate detection
- soft-delete
- token-budgeted context assembly
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import tiktoken
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from app.config import settings

logger = logging.getLogger("harness.memory")

# Optional neural layers (graceful if import fails)
try:
    from app.neural.pipeline import NeuralRecallPipeline
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    NeuralRecallPipeline = None  # type: ignore

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text or ""))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN .*PRIVATE KEY-----"),
]

INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore (all |previous |above )*instructions"),
    re.compile(r"(?i)you are now"),
    re.compile(r"(?i)system\s*:"),
    re.compile(r"(?i)assistant\s*:"),
]


def normalize_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text or "")
    return text.strip()


def redact_secrets(text: str) -> str:
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def sanitize_untrusted(text: str) -> str:
    text = normalize_text(text)
    text = redact_secrets(text)
    for pat in INJECTION_PATTERNS:
        text = pat.sub("[FILTERED]", text)
    return text


def content_id(session_id: str, role: str, text: str) -> str:
    normalized = normalize_text(text)
    payload = f"{session_id}|{role}|{normalized}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


class PineconeMemory:
    def __init__(
        self,
        namespace: str,
        pc: Pinecone,
        openai_client: OpenAI,
        neural: Optional["NeuralRecallPipeline"] = None,
        enable_neural: bool = True,
    ):
        self.namespace = namespace
        self.pc = pc
        self.client = openai_client
        self.embedding_tokens_used = 0
        self.index = self._ensure_index()
        self.neural = neural
        if enable_neural and NEURAL_AVAILABLE and self.neural is None:
            self.neural = NeuralRecallPipeline()

    def _ensure_index(self):
        existing = [idx.name for idx in self.pc.list_indexes()]
        if settings.pinecone_index not in existing:
            logger.info("Creating Pinecone index %s", settings.pinecone_index)
            try:
                self.pc.create_index(
                    name=settings.pinecone_index,
                    dimension=settings.embedding_dim,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=settings.pinecone_cloud,
                        region=settings.pinecone_region,
                    ),
                )
            except Exception:
                # Another worker may have won the create race. Only suppress
                # the error if the target index now exists.
                refreshed = [idx.name for idx in self.pc.list_indexes()]
                if settings.pinecone_index not in refreshed:
                    raise
        deadline = time.monotonic() + settings.pinecone_index_ready_timeout_s
        description = self.pc.describe_index(settings.pinecone_index)
        while not description.status["ready"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("Pinecone index did not become ready before the configured timeout")
            time.sleep(2)
            description = self.pc.describe_index(settings.pinecone_index)
        dimension = getattr(description, "dimension", None)
        if dimension is not None and int(dimension) != settings.embedding_dim:
            raise ValueError(
                f"Pinecone index dimension {dimension} does not match configured embedding dimension {settings.embedding_dim}"
            )
        metric = getattr(description, "metric", None)
        if metric is not None and metric != "cosine":
            raise ValueError(f"Pinecone index metric must be cosine, not {metric}")
        return self.pc.Index(settings.pinecone_index)

    def _embed(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(
            model=settings.embedding_model, input=text
        )
        usage = getattr(resp, "usage", None)
        used = getattr(usage, "total_tokens", None)
        self.embedding_tokens_used = int(getattr(self, "embedding_tokens_used", 0)) + int(used if used is not None else count_tokens(text))
        vector = list(resp.data[0].embedding)
        if len(vector) != settings.embedding_dim:
            raise ValueError(
                f"Embedding dimension {len(vector)} does not match configured dimension {settings.embedding_dim}"
            )
        if not all(math.isfinite(float(value)) for value in vector):
            raise ValueError("Embedding vector contains non-finite values")
        return vector

    def remember(
        self,
        text: str,
        role: str = "assistant",
        importance: float = 0.5,
        pinned: bool = False,
        provenance: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("memory text must be a non-empty string")
        if not isinstance(role, str) or not role.strip() or len(role) > 64:
            raise ValueError("memory role must be a non-empty string of at most 64 characters")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        if not isinstance(importance, (int, float)) or isinstance(importance, bool):
            raise ValueError("importance must be a number between 0 and 1")
        if not 0.0 <= float(importance) <= 1.0:
            raise ValueError("importance must be between 0 and 1")
        importance = float(importance)
        clean = sanitize_untrusted(text)
        if not clean:
            raise ValueError("memory text is empty after sanitization")
        if len(clean) > 8000:
            clean = clean[:8000] + "…[truncated]"

        vid = content_id(self.namespace, role, clean)
        now = utc_now()

        # Idempotent path
        existing = self.index.fetch(ids=[vid], namespace=self.namespace)
        if existing.vectors:
            existing_vector = existing.vectors[vid]
            meta = existing_vector.metadata or {}
            if not meta.get("deleted_at"):
                try:
                    access_count = max(0, int(meta.get("access_count", 0))) + 1
                except (TypeError, ValueError):
                    access_count = 1
                try:
                    created_epoch = float(meta.get("created_at_epoch") or datetime.fromisoformat(
                        str(meta.get("created_at", now)).replace("Z", "+00:00")
                    ).timestamp())
                except (TypeError, ValueError, OverflowError):
                    created_epoch = time.time()
                effective_importance = max(float(meta.get("importance", 0.5)), importance)
                effective_pinned = bool(meta.get("pinned", False) or pinned)
                self.index.update(
                    id=vid,
                    namespace=self.namespace,
                    set_metadata={
                        "schema_version": "2.1",
                        "created_at_epoch": created_epoch,
                        "deleted_at_epoch": 0.0,
                        "importance": effective_importance,
                        "pinned": effective_pinned,
                        "access_count": access_count,
                        "last_accessed_at": now,
                    },
                )
                values = getattr(existing_vector, "values", None)
                if self.neural is not None and values:
                    neural_meta = {
                        **meta,
                        "importance": effective_importance,
                        "pinned": effective_pinned,
                    }
                    self.neural.on_remember(
                        vid, list(values), meta.get("text", clean), neural_meta
                    )
                return vid
            # Re-remembering a soft-deleted item must revive it through upsert.

        emb = self._embed(clean)

        # Near-duplicate check
        matches = self.index.query(
            vector=emb,
            top_k=3,
            include_metadata=True,
            namespace=self.namespace,
            filter={"deleted_at": {"$eq": ""}},
        )
        for m in matches.matches:
            if m.score and m.score >= 0.95 and not (m.metadata or {}).get("is_summary"):
                self.index.update(
                    id=m.id,
                    namespace=self.namespace,
                    set_metadata={
                        "access_count": int((m.metadata or {}).get("access_count", 0)) + 1,
                        "last_accessed_at": now,
                    },
                )
                return m.id

        meta = {
            "schema_version": "2.1",
            "session_id": self.namespace,
            "role": role,
            "text": clean,
            "text_hash": hashlib.sha256(clean.encode()).hexdigest(),
            "created_at": now,
            "created_at_epoch": time.time(),
            "observed_at": now,
            "observed_at_epoch": time.time(),
            "importance": float(importance),
            "pinned": bool(pinned),
            "is_summary": False,
            "parent_id": "",
            "summary_version": 0,
            "access_count": 0,
            "last_accessed_at": now,
            "deleted_at": "",
            "deleted_at_epoch": 0.0,
            "provenance": provenance or "",
            **{
                key: value for key, value in (metadata or {}).items()
                if key not in {
                    "schema_version", "session_id", "role", "text", "text_hash",
                    "created_at", "created_at_epoch", "observed_at", "observed_at_epoch",
                    "importance", "pinned",
                    "is_summary", "parent_id", "summary_version", "access_count",
                    "last_accessed_at", "deleted_at", "deleted_at_epoch",
                    "provenance",
                }
            },
        }
        self.index.upsert(
            vectors=[{"id": vid, "values": emb, "metadata": meta}],
            namespace=self.namespace,
        )

        # Neural write path (sidecar + synaptic node)
        if self.neural is not None:
            try:
                self.neural.on_remember(vid, emb, clean, meta)
            except Exception as e:
                logger.warning("Neural on_remember failed: %s", e)

        return vid

    def recall(self, query: str, top_k: int = 8, use_neural: bool = True) -> List[Dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        # Over-fetch when neural reranking is active
        fetch_k = min(100, top_k * 3) if (use_neural and self.neural) else top_k

        query_vector = self._embed(query)
        results = self.index.query(
            vector=query_vector,
            top_k=fetch_k,
            include_metadata=True,
            namespace=self.namespace,
            filter={"deleted_at": {"$eq": ""}},
        )
        out = []
        legacy_epoch_updates = []
        for m in results.matches:
            meta = m.metadata or {}
            if (
                not meta.get("created_at_epoch")
                or "deleted_at_epoch" not in meta
            ) and meta.get("created_at"):
                try:
                    created_epoch = datetime.fromisoformat(
                        str(meta["created_at"]).replace("Z", "+00:00")
                    ).timestamp()
                    legacy_epoch_updates.append((m.id, created_epoch))
                except (TypeError, ValueError, OverflowError):
                    pass
            score = float(m.score or 0.0)
            if not math.isfinite(score):
                continue
            try:
                importance = float(meta.get("importance", 0.5))
                if not math.isfinite(importance):
                    importance = 0.5
            except (TypeError, ValueError):
                importance = 0.5
            try:
                access_count = max(0, int(meta.get("access_count", 0)))
            except (TypeError, ValueError):
                access_count = 0
            out.append(
                {
                    "id": m.id,
                    "score": score,
                    "text": meta.get("text", ""),
                    "role": meta.get("role", ""),
                    "timestamp": meta.get("created_at", ""),
                    "pinned": bool(meta.get("pinned", False)),
                    "is_summary": bool(meta.get("is_summary", False)),
                    "importance": max(0.0, min(1.0, importance)),
                    "access_count": access_count,
                }
            )

        # Neural enhancement (GNN diffusion + synaptic spreading)
        for memory_id, created_epoch in legacy_epoch_updates:
            try:
                self.index.update(
                    id=memory_id, namespace=self.namespace,
                    set_metadata={
                        "schema_version": "2.1",
                        "created_at_epoch": created_epoch,
                        "deleted_at_epoch": 0.0,
                    },
                )
            except Exception as exc:
                logger.warning("Legacy memory timestamp migration failed: %s", type(exc).__name__)

        if use_neural and self.neural is not None and out:
            try:
                out = self.neural.enhance(query_vector, out, top_k=top_k)
                # Synaptic spreading can return IDs outside the ANN candidate
                # set. Hydrate those IDs before exposing them as memory results.
                missing = [item["id"] for item in out if not item.get("text")]
                if missing:
                    fetched = self.index.fetch(ids=missing, namespace=self.namespace)
                    for item in out:
                        vector = fetched.vectors.get(item["id"])
                        if vector:
                            meta = vector.metadata or {}
                            if meta.get("deleted_at", ""):
                                continue
                            item.update({
                                "text": meta.get("text", ""),
                                "role": meta.get("role", ""),
                                "timestamp": meta.get("created_at", ""),
                                "pinned": bool(meta.get("pinned", False)),
                                "is_summary": bool(meta.get("is_summary", False)),
                                "importance": float(meta.get("importance", 0.5)),
                                "metadata": meta,
                            })
                    out = [item for item in out if item.get("text")]
            except Exception as e:
                logger.warning("Neural enhance failed, using raw ANN: %s", type(e).__name__)
                out = out[:top_k]
        else:
            out = out[:top_k]

        return out

    def build_context(
        self,
        query: str,
        system_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, int]]:
        max_tokens = max_tokens or settings.max_context_tokens
        memories = self.recall(query, top_k=12)

        pinned = [m for m in memories if m.get("pinned")]
        others = [m for m in memories if not m.get("pinned")]
        others.sort(key=lambda x: x["score"], reverse=True)

        blocks: List[Tuple[str, str]] = [
            ("system", system_prompt),
            ("pinned", "\n".join(f"[{m['role']}] {m['text']}" for m in pinned)),
            ("evidence", "\n".join(f"[{m['role']}] {m['text']}" for m in others)),
            ("query", query),
        ]

        selected = []
        budget = max_tokens
        composition = {"system": 0, "pinned": 0, "evidence": 0, "query": 0, "total": 0}

        for name, text in blocks:
            if not text.strip():
                continue
            n = count_tokens(text)
            if n <= budget:
                selected.append(f"[{name.upper()}]\n{text}")
                composition[name] = n
                budget -= n
            else:
                ratio = budget / max(n, 1)
                cut = int(len(text) * ratio * 0.9)
                truncated = text[:cut] + "…[truncated]"
                selected.append(f"[{name.upper()}]\n{truncated}")
                composition[name] = count_tokens(truncated)
                budget = 0
                break

        prompt = "\n\n".join(selected)
        composition["total"] = count_tokens(prompt)
        if composition["total"] > max_tokens:
            raise RuntimeError(
                f"Context overflow: {composition['total']} > {max_tokens}"
            )

        logger.info(
            "context.built composition=%s hard_limit=%s",
            composition,
            max_tokens,
        )
        return prompt, composition

    def soft_delete(self, vector_id: str, reason: str = "prune") -> None:
        self.index.update(
            id=vector_id,
            namespace=self.namespace,
            set_metadata={
                "deleted_at": utc_now(),
                "deleted_at_epoch": time.time(),
                "deleted_reason": reason,
            },
        )

    def purge_deleted(self) -> int:
        """Permanently remove soft-deleted records older than the retention window."""
        retention_days = settings.soft_delete_retention_days
        if retention_days == 0:
            cutoff = utc_now()
        else:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=retention_days)
            ).isoformat()
        results = self.index.query(
            vector=[1.0] + [0.0] * (settings.embedding_dim - 1),
            top_k=100,
            include_metadata=True,
            namespace=self.namespace,
            filter={
                "deleted_at_epoch": {"$gt": 0.0, "$lt": datetime.fromisoformat(cutoff).timestamp()}
            },
        )
        cutoff_epoch = datetime.fromisoformat(cutoff).timestamp()
        ids = [
            match.id for match in results.matches
            if (match.metadata or {}).get("deleted_at_epoch", 0) > 0
            and (match.metadata or {}).get("deleted_at_epoch", 0) < cutoff_epoch
        ]
        # Migrate pre-epoch soft deletes as well. Removing each bounded batch
        # lets later prune runs advance to the next group of legacy records.
        legacy = self.index.query(
            vector=[1.0] + [0.0] * (settings.embedding_dim - 1),
            top_k=100,
            include_metadata=True,
            namespace=self.namespace,
            filter={"deleted_at": {"$ne": ""}},
        )
        for match in legacy.matches:
            meta = match.metadata or {}
            deleted_at = meta.get("deleted_at")
            if not deleted_at or meta.get("deleted_at_epoch"):
                continue
            try:
                deleted_epoch = datetime.fromisoformat(
                    str(deleted_at).replace("Z", "+00:00")
                ).timestamp()
            except (TypeError, ValueError, OverflowError):
                continue
            if deleted_epoch < cutoff_epoch and match.id not in ids:
                ids.append(match.id)
        if not ids:
            return 0
        self.index.delete(ids=ids, namespace=self.namespace)
        if self.neural is not None:
            for memory_id in ids:
                try:
                    self.neural.forget(memory_id)
                except Exception as exc:
                    logger.warning("Neural purge failed: %s", type(exc).__name__)
        return len(ids)

    def prune(
        self,
        age_days: Optional[int] = None,
        utility_floor: Optional[float] = None,
    ) -> int:
        age_days = settings.prune_age_days if age_days is None else age_days
        utility_floor = settings.prune_utility_floor if utility_floor is None else utility_floor
        if not isinstance(age_days, int) or isinstance(age_days, bool) or age_days < 0:
            raise ValueError("age_days must be a non-negative integer")
        if not isinstance(utility_floor, (int, float)) or not 0.0 <= utility_floor <= 1.0:
            raise ValueError("utility_floor must be between 0 and 1")
        purged = self.purge_deleted()
        cutoff_epoch = time.time() - (age_days * 86400)

        # Practical bounded query (production should paginate with fetch_by_metadata)
        results = self.index.query(
            vector=[1.0] + [0.0] * (settings.embedding_dim - 1),
            top_k=100,
            include_metadata=True,
            namespace=self.namespace,
            filter={"created_at_epoch": {"$lt": cutoff_epoch}, "deleted_at_epoch": {"$eq": 0.0}},
        )
        deleted = purged
        for m in results.matches:
            meta = m.metadata or {}
            if meta.get("pinned") or meta.get("role") == "decision":
                continue
            importance = float(meta.get("importance", 0.5))
            access = int(meta.get("access_count", 0))
            utility = importance * (1 + 0.1 * access)
            if utility < utility_floor:
                self.soft_delete(m.id, reason="low_utility_age")
                deleted += 1
        return deleted
