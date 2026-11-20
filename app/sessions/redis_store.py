# Copyright (c) 2026 Lewis James Seals Jr.
# Author: Lewis James Seals Jr. (@ljrseals-dev)
# Project: Pinecone Neural Agent Harness
# SPDX-License-Identifier: MIT

import json
from typing import Optional, List, Dict, Any

import redis.asyncio as redis

from app.config import settings
from app.sessions.base import SessionStore


class RedisSessionStore(SessionStore):
    def __init__(self, url: str, prefix: str = "sess:", pool_size: int = 20) -> None:
        self.client = redis.from_url(
            url,
            decode_responses=True,
            max_connections=pool_size,
            socket_connect_timeout=settings.redis_connect_timeout_s,
            socket_timeout=settings.redis_socket_timeout_s,
        )
        self.prefix = prefix

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"

    async def get(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        raw = await self.client.get(self._key(session_id))
        return json.loads(raw) if raw is not None else None

    async def set(self, session_id: str, history: List[Dict[str, Any]],
                  ttl: int = 86400) -> None:
        await self.client.set(
            self._key(session_id),
            json.dumps(history),
            ex=ttl if ttl > 0 else None,
        )

    async def delete(self, session_id: str) -> None:
        await self.client.delete(self._key(session_id))

    async def exists(self, session_id: str) -> bool:
        return bool(await self.client.exists(self._key(session_id)))

    async def close(self) -> None:
        await self.client.aclose()
