"""
Agentic loop with structured termination and restricted tool execution.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pinecone import Pinecone

from app.config import settings
from app.memory import PineconeMemory, sanitize_untrusted

logger = logging.getLogger("harness.agent")

ALLOWED_BUILTINS = {
    "len": len,
    "range": range,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "sum": sum,
    "min": min,
    "max": max,
    "sorted": sorted,
}


def execute_restricted(code: str, memory: PineconeMemory) -> str:
    """Minimal restricted execution environment."""
    env: Dict[str, Any] = {
        "__builtins__": ALLOWED_BUILTINS,
        "memory": memory,
    }
    try:
        result = eval(compile(code, "<tool>", "eval"), env)
        out = repr(result) if result is not None else ""
    except SyntaxError:
        try:
            exec(compile(code, "<tool>", "exec"), env)
            out = str(env.get("__result__", ""))
        except Exception as e:
            return f"ToolError: {type(e).__name__}: {e}"
    except Exception as e:
        return f"ToolError: {type(e).__name__}: {e}"

    out = sanitize_untrusted(out)
    if len(out.encode()) > settings.tool_output_max_bytes:
        out = out[: settings.tool_output_max_bytes] + "…[truncated]"
    return out


class AgentHarness:
    def __init__(
        self,
        session_id: Optional[str] = None,
        pc: Optional[Pinecone] = None,
        openai_client: Optional[OpenAI] = None,
    ):
        self.session_id = session_id or f"session-{uuid.uuid4().hex[:8]}"
        self.pc = pc or Pinecone(api_key=settings.pinecone_api_key)
        self.client = openai_client or OpenAI(api_key=settings.openai_api_key)
        self.memory = PineconeMemory(
            namespace=self.session_id,
            pc=self.pc,
            openai_client=self.client,
        )
        self.seen_hashes: set[str] = set()
        self.tokens_used = 0
        self.steps_taken = 0

    def run(self, task: str, max_steps: Optional[int] = None) -> Dict[str, Any]:
        max_steps = max_steps or settings.max_steps
        try:
            self.memory.prune()
        except Exception as e:
            logger.warning("Prune failed (non-fatal): %s", e)

        system = (
            "You are an autonomous agent with a restricted memory object named `memory`.\n"
            "Respond ONLY with valid JSON of one of these two forms:\n"
            '  {"type": "tool_call", "code": "<python expression using memory.recall or memory.remember>"}\n'
            '  {"type": "final_answer", "answer": "<your final answer>"}\n'
            "Never mix both. Use memory.recall(query) and memory.remember(text, role=...) only.\n"
            "Do not invent facts; answer only from retrieved evidence."
        )

        history: List[Dict[str, str]] = []
        final_answer: Optional[str] = None
        status = "max_steps"

        for step in range(max_steps):
            self.steps_taken = step + 1

            if self.tokens_used > settings.max_run_tokens:
                status = "budget_exceeded"
                break

            try:
                prompt, composition = self.memory.build_context(
                    query=task, system_prompt=system
                )
            except RuntimeError as e:
                logger.error("Context build failed: %s", e)
                status = "context_overflow"
                break

            self.tokens_used += composition.get("total", 0)

            try:
                response = self.client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=600,
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                logger.exception("LLM call failed")
                status = "llm_error"
                final_answer = f"LLM error: {e}"
                break

            raw = (response.choices[0].message.content or "").strip()
            if response.usage:
                self.tokens_used += response.usage.total_tokens

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                self.memory.remember(
                    f"Malformed model output: {raw[:300]}", role="system"
                )
                continue

            if payload.get("type") == "final_answer":
                final_answer = (payload.get("answer") or "").strip()
                self.memory.remember(
                    final_answer,
                    role="assistant",
                    importance=0.9,
                    pinned=True,
                )
                status = "completed"
                break

            if payload.get("type") != "tool_call":
                continue

            code = (payload.get("code") or "").strip()
            if not code:
                continue

            code_hash = hashlib.sha256(code.encode()).hexdigest()
            if code_hash in self.seen_hashes:
                status = "repeated_tool_call"
                final_answer = "Stopped: repeated identical tool call detected."
                break
            self.seen_hashes.add(code_hash)

            output = execute_restricted(code, self.memory)
            self.memory.remember(
                f"Tool output:\n{output}",
                role="tool",
                provenance="tool_execution",
            )
            history.append({"role": "tool", "content": output[:1500]})

        if final_answer is None:
            final_answer = "Agent reached max steps without a final answer."

        return {
            "session_id": self.session_id,
            "answer": final_answer,
            "status": status,
            "steps": self.steps_taken,
            "tokens_used": self.tokens_used,
            "history_length": len(history),
        }
