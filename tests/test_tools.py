# Copyright (c) 2026 Lewis James Seals Jr.
# Author: Lewis James Seals Jr. (@ljrseals-dev)
# Project: Pinecone Neural Agent Harness
# SPDX-License-Identifier: MIT

import asyncio

from app.tools.builtin import CalculatorTool
from app.tools.registry import ToolRegistry
from app.tools.base import Tool
from app.harness import execute_restricted


class EchoTool(Tool):
    name = "echo"
    description = "Echo a string."
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string", "minLength": 1, "maxLength": 10}},
        "required": ["value"],
    }

    async def run(self, value: str) -> str:
        return value


def test_model_python_execution_is_disabled():
    result = execute_restricted("__import__('os').getcwd()", object())
    assert result.startswith("ToolError:")
    assert "disabled" in result


def test_registry_accepts_only_registered_tools_and_schema():
    registry = ToolRegistry()
    registry.register(EchoTool())
    assert asyncio.run(registry.execute("echo", {"value": "safe"})) == "safe"
    assert "Unknown tool" in asyncio.run(registry.execute("not_registered", {}))
    assert "missing required" in asyncio.run(registry.execute("echo", {}))
    assert "unknown fields" in asyncio.run(registry.execute("echo", {"value": "x", "extra": True}))
    assert "must be string" in asyncio.run(registry.execute("echo", {"value": 7}))
    assert "too long" in asyncio.run(registry.execute("echo", {"value": "01234567890"}))


def test_calculator_uses_bounded_ast_not_eval():
    tool = CalculatorTool()
    assert asyncio.run(tool.run("2 + 3 * 4")) == "14"
    assert asyncio.run(tool.run("2 ** 9")).startswith("ToolError:")
    assert asyncio.run(tool.run("__import__('os').getcwd()")).startswith("ToolError:")
    assert asyncio.run(tool.run("8 ** 8")) == "16777216"


def test_tool_names_must_be_unique():
    registry = ToolRegistry()
    registry.register(EchoTool())
    try:
        registry.register(EchoTool())
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate tool registration must fail")
