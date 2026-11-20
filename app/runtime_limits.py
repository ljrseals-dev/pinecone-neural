# Copyright (c) 2026 Lewis James Seals Jr.
# Author: Lewis James Seals Jr. (@ljrseals-dev)
# Project: Pinecone Neural Agent Harness
# SPDX-License-Identifier: MIT

"""Per-process admission control and session-level request serialization."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Tuple

from app.config import settings


class SessionLockPool:
    """Reference-counted per-session locks; idle entries are removed safely."""

    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._entries: Dict[str, Tuple[asyncio.Lock, int]] = {}

    @property
    def size(self) -> int:
        return len(self._entries)

    @asynccontextmanager
    async def hold(self, key: str) -> AsyncIterator[None]:
        if not isinstance(key, str) or not key:
            raise ValueError("session lock key must be a non-empty string")

        async with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = (asyncio.Lock(), 0)
            lock, users = entry
            self._entries[key] = (lock, users + 1)

        acquired = False
        try:
            await lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                lock.release()
            async with self._guard:
                current = self._entries.get(key)
                if current is not None and current[0] is lock:
                    remaining = current[1] - 1
                    if remaining <= 0:
                        self._entries.pop(key, None)
                    else:
                        self._entries[key] = (lock, remaining)


# One process-wide cap. This does not replace host/container CPU and memory limits.
run_semaphore = asyncio.Semaphore(settings.max_concurrent_runs)
