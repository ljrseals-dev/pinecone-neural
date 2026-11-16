import json
from typing import Any, Dict, List

from app.tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("Tool must define a name")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    async def execute(self, name: str, args: Dict[str, Any]) -> str:
        try:
            tool = self.get(name)
        except KeyError as e:
            return f"ToolError: {e}"

        required = tool.parameters.get("required", [])
        for field in required:
            if field not in args:
                return f"ToolError: missing required field '{field}'"

        allowed = set(tool.parameters.get("properties", {}).keys())
        unknown = set(args.keys()) - allowed
        if unknown:
            return f"ToolError: unknown fields {sorted(unknown)}"

        try:
            result = await tool.run(**args)
        except Exception as e:
            return f"ToolError: {type(e).__name__}: {e}"

        if not isinstance(result, str):
            result = json.dumps(result, default=str)
        return result[:4096]
