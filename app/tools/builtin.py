import ast
import asyncio
import json
import math
import operator
from typing import Any

from app.tools.base import Tool


class MemorySearchTool(Tool):
    name = "memory_search"
    description = "Search this authenticated session's Pinecone memory."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 2000},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": ["query"],
    }

    def __init__(self, memory: Any) -> None:
        self.memory = memory

    async def run(self, query: str, top_k: int = 5) -> str:
        results = await asyncio.to_thread(self.memory.recall, query, top_k=top_k)
        return json.dumps([
            {
                "id": r.get("id"),
                "score": r.get("score"),
                "text": r.get("text") or r.get("metadata", {}).get("text", ""),
            }
            for r in results
        ])


class MemoryStoreTool(Tool):
    name = "memory_store"
    description = "Store a fact in this authenticated session's memory."
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1, "maxLength": 4000},
            "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["text"],
    }

    def __init__(self, memory: Any) -> None:
        self.memory = memory

    async def run(self, text: str, importance: float = 0.5) -> str:
        mid = await asyncio.to_thread(self.memory.remember, text, role="fact", importance=importance)
        return f"Stored as {mid}"


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate bounded arithmetic without Python eval or code execution."
    parameters = {
        "type": "object",
        "properties": {"expression": {"type": "string", "minLength": 1, "maxLength": 200}},
        "required": ["expression"],
    }

    _binary = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    @classmethod
    def _evaluate(cls, node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return cls._evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and type(node.op) in cls._unary:
            value = cls._unary[type(node.op)](cls._evaluate(node.operand))
        elif isinstance(node, ast.BinOp) and type(node.op) in cls._binary:
            left = cls._evaluate(node.left)
            right = cls._evaluate(node.right)
            if type(node.op) is ast.Pow and (abs(right) > 8 or abs(left) > 1000):
                raise ValueError("exponentiation exceeds calculator limits")
            value = cls._binary[type(node.op)](left, right)
        else:
            raise ValueError("unsupported expression")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError("result exceeds calculator limits")
        return value

    async def run(self, expression: str) -> str:
        try:
            tree = ast.parse(expression, mode="eval")
            if sum(1 for _ in ast.walk(tree)) > 32:
                return "ToolError: expression is too complex"
            return str(self._evaluate(tree))
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
            return f"ToolError: {type(exc).__name__}: invalid arithmetic expression"


def build_default_registry(memory: Any) -> "ToolRegistry":
    from app.tools.registry import ToolRegistry
    from app.config import settings
    registry = ToolRegistry(timeout_s=settings.tool_timeout_s, output_max_bytes=settings.tool_output_max_bytes)
    registry.register(MemorySearchTool(memory))
    registry.register(MemoryStoreTool(memory))
    registry.register(CalculatorTool())
    return registry
