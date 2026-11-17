import asyncio
import json
import math
from typing import Any, Dict, List

from app.tools.base import Tool


class ToolRegistry:
    def __init__(self, timeout_s: float = 2.0, output_max_bytes: int = 4096) -> None:
        self._tools: Dict[str, Tool] = {}
        self.timeout_s = max(0.1, min(float(timeout_s), 60.0))
        self.output_max_bytes = max(128, min(int(output_max_bytes), 1_048_576))

    def register(self, tool: Tool) -> None:
        if not isinstance(tool.name, str) or not tool.name or tool.name in self._tools:
            raise ValueError("Tool name must be non-empty and unique")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    @staticmethod
    def _matches_type(value: Any, expected: str) -> bool:
        if expected == "string":
            return isinstance(value, str)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "object":
            return isinstance(value, dict)
        if expected == "array":
            return isinstance(value, list)
        return False

    async def execute(self, name: str, args: Dict[str, Any]) -> str:
        try:
            tool = self.get(name)
        except KeyError as exc:
            return f"ToolError: {exc}"
        if not isinstance(args, dict):
            return "ToolError: arguments must be a JSON object"

        parameters = tool.parameters
        required = parameters.get("required", [])
        properties = parameters.get("properties", {})
        missing = [field for field in required if field not in args]
        if missing:
            return f"ToolError: missing required field(s) {sorted(missing)}"
        unknown = set(args) - set(properties)
        if unknown:
            return f"ToolError: unknown fields {sorted(unknown)}"
        for field, value in args.items():
            expected = properties[field].get("type")
            if expected and not self._matches_type(value, expected):
                return f"ToolError: field '{field}' must be {expected}"
            if isinstance(value, str):
                if len(value) > int(properties[field].get("maxLength", 12000)):
                    return f"ToolError: field '{field}' is too long"
                if len(value) < int(properties[field].get("minLength", 0)):
                    return f"ToolError: field '{field}' is too short"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if "minimum" in properties[field] and value < properties[field]["minimum"]:
                    return f"ToolError: field '{field}' is below minimum"
                if "maximum" in properties[field] and value > properties[field]["maximum"]:
                    return f"ToolError: field '{field}' exceeds maximum"

        try:
            result = await asyncio.wait_for(tool.run(**args), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            return "ToolError: tool timed out"
        except Exception as exc:
            return f"ToolError: {type(exc).__name__}"
        if not isinstance(result, str):
            result = json.dumps(result, default=str)
        encoded = result.encode("utf-8")
        if len(encoded) > self.output_max_bytes:
            result = encoded[:self.output_max_bytes].decode("utf-8", errors="ignore") + "…[truncated]"
        return result
