import json
import re
from typing import Any

from app.tools.base import Tool


class MemorySearchTool(Tool):
    name = "memory_search"
    description = "Search Pinecone memory. Returns top-k matches with id, score, text."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer"},
        },
        "required": ["query"],
    }

    def __init__(self, memory: Any) -> None:
        self.memory = memory

    async def run(self, query: str, top_k: int = 5) -> str:
        top_k = max(1, min(int(top_k), 20))
        results = self.memory.recall(query, top_k=top_k)
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
    description = "Store a fact in memory for later retrieval."
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "importance": {"type": "number"},
        },
        "required": ["text"],
    }

    def __init__(self, memory: Any) -> None:
        self.memory = memory

    async def run(self, text: str, importance: float = 0.5) -> str:
        importance = max(0.0, min(float(importance), 1.0))
        mid = self.memory.remember(
            text, role="fact", metadata={"importance": importance}
        )
        return f"Stored as {mid}"


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate a simple arithmetic expression (digits, +-*/(). only)."
    parameters = {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    }

    async def run(self, expression: str) -> str:
        if not re.fullmatch(r"[\d\s+\-*/().]{1,200}", expression):
            return "ToolError: expression contains forbidden characters"
        try:
            result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
            return str(result)
        except Exception as e:
            return f"ToolError: {e}"


def build_default_registry(memory: Any) -> "ToolRegistry":
    from app.tools.registry import ToolRegistry
    reg = ToolRegistry()
    reg.register(MemorySearchTool(memory))
    reg.register(MemoryStoreTool(memory))
    reg.register(CalculatorTool())
    return reg
