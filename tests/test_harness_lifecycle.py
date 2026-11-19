import asyncio

import pytest

from app.harness import AgentHarness


class AsyncClient:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    async def close(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("async close failed")


class SyncClient:
    def __init__(self):
        self.calls = 0

    def close(self):
        self.calls += 1


def make_harness(*, owns_async=True, owns_embedding=True, async_fail=False):
    harness = AgentHarness.__new__(AgentHarness)
    harness.client = AsyncClient(fail=async_fail)
    harness.embedding_client = SyncClient()
    harness._owns_async_client = owns_async
    harness._owns_embedding_client = owns_embedding
    harness._closed = False
    return harness


def test_aclose_closes_owned_clients_once():
    harness = make_harness()

    asyncio.run(harness.aclose())
    asyncio.run(harness.aclose())

    assert harness.client.calls == 1
    assert harness.embedding_client.calls == 1
    assert harness._closed is True


def test_aclose_does_not_close_injected_clients():
    harness = make_harness(owns_async=False, owns_embedding=False)

    asyncio.run(harness.aclose())

    assert harness.client.calls == 0
    assert harness.embedding_client.calls == 0


def test_aclose_attempts_sync_cleanup_when_async_close_fails():
    harness = make_harness(async_fail=True)

    with pytest.raises(RuntimeError, match="async close failed"):
        asyncio.run(harness.aclose())

    assert harness.client.calls == 1
    assert harness.embedding_client.calls == 1
    assert harness._closed is True


def test_aclose_closes_only_the_owned_client():
    harness = make_harness(owns_async=True, owns_embedding=False)

    asyncio.run(harness.aclose())

    assert harness.client.calls == 1
    assert harness.embedding_client.calls == 0
