"""
Agent loop with explicit allow-listed tool calls and neural memory feedback.

Model output is data only. It is never compiled, eval'd, or exec'd.
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
from app.tools import build_default_registry
from app.tools.registry import ToolRegistry

logger = logging.getLogger("harness.agent")


def execute_restricted(code: str, memory: PineconeMemory) -> str:
    """Compatibility shim: arbitrary model-generated Python is intentionally disabled."""
    del code, memory
    return "ToolError: arbitrary code execution is disabled; use a registered tool name and JSON arguments."


class AgentHarness:
    def __init__(
        self,
        session_id: Optional[str] = None,
        pc: Optional[Pinecone] = None,
        openai_client: Optional[OpenAI] = None,
        memory: Optional[PineconeMemory] = None,
        tools: Optional[ToolRegistry] = None,
    ):
        self.session_id = session_id or f"session-{uuid.uuid4().hex[:8]}"
        self.pc = pc or Pinecone(api_key=settings.pinecone_api_key)
        self.client = openai_client or OpenAI(api_key=settings.openai_api_key)
        self.memory = memory or PineconeMemory(
            namespace=self.session_id,
            pc=self.pc,
            openai_client=self.client,
        )
        self.tools = tools or build_default_registry(self.memory)
        self.seen_hashes: set[str] = set()
        self.tokens_used = 0
        self.steps_taken = 0
        self._used_memory_ids: List[str] = []
        self._used_metas: List[Dict[str, Any]] = []

    def _step_coactivation(self, step_ids: List[str], success: bool) -> None:
        neural = getattr(self.memory, "neural", None)
        if neural is None or len(step_ids) < 2:
            return
        try:
            neural.on_coactivation(step_ids, success=success)
        except Exception as exc:
            logger.warning("Per-step coactivation failed: %s", type(exc).__name__)

    async def run(self, task: str, max_steps: Optional[int] = None) -> Dict[str, Any]:
        if not isinstance(task, str) or not task.strip() or len(task) > 12000:
            raise ValueError("task must be 1-12000 characters")
        max_steps = max_steps or settings.max_steps
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or not 1 <= max_steps <= 12:
            raise ValueError("max_steps must be between 1 and 12")
        try:
            self.memory.prune()
        except Exception as exc:
            logger.warning("Prune failed (non-fatal): %s", type(exc).__name__)

        tool_schemas = json.dumps(self.tools.schemas(), separators=(",", ":"), sort_keys=True)
        system = (
            "You are an assistant with access only to the explicitly registered tools below.\n"
            "Return ONLY one JSON object per step. Allowed forms:\n"
            '{"type":"tool_call","name":"registered_tool_name","arguments":{}}\n'
            '{"type":"final_answer","answer":"..."}\n'
            "Never return Python, shell, SQL, source code, or a command string as a tool call. "
            "Tool arguments are JSON data and are validated by the host. "
            "Unknown tools and invalid arguments are rejected. Do not invent facts.\n"
            f"Registered tools: {tool_schemas}"
        )

        history: List[Dict[str, str]] = []
        final_answer: Optional[str] = None
        status = "max_steps"
        self._used_memory_ids = []
        self._used_metas = []
        self.seen_hashes = set()
        self.tokens_used = 0

        for step in range(max_steps):
            self.steps_taken = step + 1
            if self.tokens_used > settings.max_run_tokens:
                status = "budget_exceeded"
                break
            try:
                prompt, composition = self.memory.build_context(query=task, system_prompt=system)
            except RuntimeError as exc:
                logger.error("Context build failed: %s", type(exc).__name__)
                status = "context_overflow"
                break
            self.tokens_used += composition.get("total", 0)

            step_memory_ids: List[str] = []
            try:
                for item in self.memory.recall(task, top_k=8):
                    memory_id = item.get("id")
                    if memory_id:
                        step_memory_ids.append(memory_id)
                        if memory_id not in self._used_memory_ids:
                            self._used_memory_ids.append(memory_id)
                            self._used_metas.append(item)
            except Exception as exc:
                logger.warning("Memory recall for feedback failed: %s", type(exc).__name__)

            try:
                response = self.client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=600,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                logger.exception("LLM call failed")
                status = "llm_error"
                final_answer = "The language-model request failed. Check server logs."
                break

            raw = (response.choices[0].message.content or "").strip()
            if response.usage:
                self.tokens_used += response.usage.total_tokens
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                self._step_coactivation(step_memory_ids, success=False)
                continue
            if not isinstance(payload, dict):
                self._step_coactivation(step_memory_ids, success=False)
                continue

            if payload.get("type") == "final_answer":
                answer = payload.get("answer")
                if not isinstance(answer, str):
                    self._step_coactivation(step_memory_ids, success=False)
                    continue
                final_answer = sanitize_untrusted(answer[:12000])
                self.memory.remember(final_answer, role="assistant", importance=0.9, pinned=True)
                status = "completed"
                self._step_coactivation(step_memory_ids, success=True)
                break

            if payload.get("type") != "tool_call":
                self._step_coactivation(step_memory_ids, success=False)
                continue

            name = payload.get("name")
            arguments = payload.get("arguments", {})
            if not isinstance(name, str) or not isinstance(arguments, dict):
                self._step_coactivation(step_memory_ids, success=False)
                continue
            call_key = hashlib.sha256(
                (name + "\0" + json.dumps(arguments, sort_keys=True, separators=(",", ":"))).encode()
            ).hexdigest()
            if call_key in self.seen_hashes:
                status = "repeated_tool_call"
                final_answer = "Stopped: repeated identical tool call detected."
                self._step_coactivation(step_memory_ids, success=False)
                break
            self.seen_hashes.add(call_key)

            output = await self.tools.execute(name, arguments)
            succeeded = not output.startswith("ToolError:")
            safe_output = sanitize_untrusted(output)
            self.memory.remember(
                f"Tool output ({name}):\n{safe_output}",
                role="tool",
                provenance="registered_tool",
            )
            history.append({"role": "tool", "content": safe_output[:1500]})
            self._step_coactivation(step_memory_ids, success=succeeded)

        if final_answer is None:
            final_answer = "Agent reached its step or token limit without a final answer."
        success = status == "completed"
        neural = getattr(self.memory, "neural", None)
        if neural is not None and self._used_memory_ids:
            try:
                neural.on_coactivation(self._used_memory_ids, success=success)
                reward = 1.0 if success else (0.3 if status == "max_steps" else 0.0)
                neural.on_episode_reward(self._used_metas, reward=reward)
            except Exception as exc:
                logger.warning("Neural feedback failed: %s", type(exc).__name__)
        return {
            "session_id": self.session_id,
            "answer": final_answer,
            "status": status,
            "steps": self.steps_taken,
            "tokens_used": self.tokens_used,
            "history_length": len(history),
            "memories_used": len(self._used_memory_ids),
        }
