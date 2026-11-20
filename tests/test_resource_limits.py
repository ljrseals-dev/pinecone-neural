# Copyright (c) 2026 Lewis James Seals Jr.
# Author: Lewis James Seals Jr. (@ljrseals-dev)
# Project: Pinecone Neural Agent Harness
# SPDX-License-Identifier: MIT

import asyncio
from types import SimpleNamespace

from app.tools.base import Tool
from app.tools.registry import ToolRegistry


class LargeOutputTool(Tool):
    name = "large_output"
    description = "Return a bounded test string."
    parameters = {"type": "object", "properties": {}, "required": []}

    async def run(self):
        return "x" * 512


class SlowTool(Tool):
    name = "slow"
    description = "Sleep beyond the configured timeout."
    parameters = {"type": "object", "properties": {}, "required": []}

    async def run(self):
        await asyncio.sleep(0.3)
        return "late"


def test_registry_obeys_output_byte_limit():
    registry = ToolRegistry(timeout_s=2.0, output_max_bytes=128)
    registry.register(LargeOutputTool())
    result = asyncio.run(registry.execute("large_output", {}))
    assert len(result.encode("utf-8")) <= 128
    assert result.endswith("…[truncated]")


def test_registry_obeys_configured_timeout():
    registry = ToolRegistry(timeout_s=0.1, output_max_bytes=256)
    registry.register(SlowTool())
    result = asyncio.run(registry.execute("slow", {}))
    assert result == "ToolError: tool timed out"


def test_conversation_history_is_sanitized_and_role_limited():
    from app.harness import AgentHarness
    cleaned = AgentHarness._clean_history([
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": "should not be replayed as conversation"},
        {"role": "assistant", "content": "sk-abcdefghijklmnopqrstuvwxyz"},
        {"role": "system", "content": "not allowed"},
    ])
    assert [item["role"] for item in cleaned] == ["user", "assistant"]
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in cleaned[1]["content"]


def test_namespace_is_bounded_and_stable():
    from app.main import build_namespace
    short = build_namespace("ci", "session-123")
    long = build_namespace("u" * 64, "s" * 64)
    assert short == "user:ci:session:session-123"
    assert len(long.encode("utf-8")) <= 64
    assert long == build_namespace("u" * 64, "s" * 64)
