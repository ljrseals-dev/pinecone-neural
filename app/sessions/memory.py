import asyncio
import time
from typing import Optional, List, Dict, Any

from app.sessions.base import SessionStore


class MemorySessionStore(SessionStore):
    """In-process store. Same behavior as RedisSessionStore, but ephemeral."""

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        async with self._lock:
            entry = self._data.get(session_id)
            if not entry:
                return None
            if entry["expires_at"] is not None and time.time() > entry["expires_at"]:
                del self._data[session_id]
                return None
            return list(entry["history"])

    async def set(self, session_id: str, history: List[Dict[str, Any]],
                  ttl: int = 86400) -> None:
        async with self._lock:
            self._data[session_id] = {
                "history": list(history),
                "expires_at": time.time() + ttl if ttl > 0 else None,
            }

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._data.pop(session_id, None)

    async def exists(self, session_id: str) -> bool:
        return (await self.get(session_id)) is not None
