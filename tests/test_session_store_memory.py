# Copyright (c) 2026 Lewis James Seals Jr.
# Author: Lewis James Seals Jr. (@ljrseals-dev)
# Project: Pinecone Neural Agent Harness
# SPDX-License-Identifier: MIT

import asyncio

from app.sessions.memory import MemorySessionStore


def test_memory_store_copies_nested_history_on_set_and_get():
    async def exercise():
        store = MemorySessionStore()
        original = [{"role": "user", "content": {"parts": ["hello"]}}]
        await store.set("session-1", original)

        # Mutating the caller's input after set must not alter stored history.
        original[0]["content"]["parts"].append("caller mutation")
        first_read = await store.get("session-1")
        assert first_read == [{"role": "user", "content": {"parts": ["hello"]}}]

        # Mutating a returned nested object must not alter subsequent reads.
        first_read[0]["content"]["parts"].append("read mutation")
        second_read = await store.get("session-1")
        assert second_read == [{"role": "user", "content": {"parts": ["hello"]}}]

    asyncio.run(exercise())
