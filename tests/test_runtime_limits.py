# Copyright (c) 2026 Lewis James Seals Jr.
# Author: Lewis James Seals Jr. (@ljrseals-dev)
# Project: Pinecone Neural Agent Harness
# SPDX-License-Identifier: MIT

import asyncio

from app.runtime_limits import SessionLockPool


def test_same_session_requests_are_serialized_and_lock_entries_are_reclaimed():
    async def exercise():
        pool = SessionLockPool()
        active = 0
        maximum_active = 0

        async def worker():
            nonlocal active, maximum_active
            async with pool.hold("user-a:session-1"):
                active += 1
                maximum_active = max(maximum_active, active)
                await asyncio.sleep(0.005)
                active -= 1

        await asyncio.gather(*(worker() for _ in range(8)))
        return maximum_active, pool.size

    maximum_active, remaining_entries = asyncio.run(exercise())
    assert maximum_active == 1
    assert remaining_entries == 0


def test_different_sessions_can_progress_concurrently():
    async def exercise():
        pool = SessionLockPool()
        entered = set()
        both_entered = asyncio.Event()

        async def worker(key):
            async with pool.hold(key):
                entered.add(key)
                if len(entered) == 2:
                    both_entered.set()
                await asyncio.wait_for(both_entered.wait(), timeout=0.5)

        await asyncio.gather(worker("user-a:session-1"), worker("user-a:session-2"))
        return pool.size

    assert asyncio.run(exercise()) == 0


def test_cancelled_waiter_does_not_leak_session_lock_entry():
    async def exercise():
        pool = SessionLockPool()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def holder():
            async with pool.hold("user-a:session-1"):
                entered.set()
                await release.wait()

        first = asyncio.create_task(holder())
        await entered.wait()

        async def waiter():
            async with pool.hold("user-a:session-1"):
                return

        second = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        second.cancel()
        try:
            await second
        except asyncio.CancelledError:
            pass
        release.set()
        await first
        return pool.size

    assert asyncio.run(exercise()) == 0
