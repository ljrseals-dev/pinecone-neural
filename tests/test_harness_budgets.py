import asyncio
from types import SimpleNamespace

from app.config import settings
from app.harness import AgentHarness


class FakeMemory:
    neural = None

    def __init__(self):
        self.remembered = []
        self.embedding_tokens_used = 0

    def prune(self):
        return 0

    def build_context(self, query, system_prompt):
        self.embedding_tokens_used += 100
        return "[QUERY]\n" + query, {"total": 4}

    def recall(self, query, top_k=8):
        self.embedding_tokens_used += 20
        return []

    def remember(self, text, **kwargs):
        self.embedding_tokens_used += 10
        self.remembered.append((text, kwargs))
        return "memory-id"


class FakeTools:
    def schemas(self):
        return []

    async def execute(self, name, arguments):
        return "unused"


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.last_messages = None
        self.last_kwargs = None
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        self.last_messages = kwargs["messages"]
        self.last_kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
        )


def make_harness(content):
    harness = AgentHarness.__new__(AgentHarness)
    harness.session_id = "session-test"
    harness.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions(content))
    )
    harness.memory = FakeMemory()
    harness.tools = FakeTools()
    harness.conversation_history = [{"role": "user", "content": "earlier context"}]
    harness.seen_hashes = set()
    harness.tokens_used = 0
    harness.cost_usd = 0.0
    harness.steps_taken = 0
    harness._used_memory_ids = []
    harness._used_metas = []
    return harness


def test_harness_replays_prior_history_and_enforces_cost_accounting(monkeypatch):
    monkeypatch.setattr(settings, "max_run_tokens", 10000)
    monkeypatch.setattr(settings, "max_run_usd", 1.0)
    monkeypatch.setattr(settings, "llm_input_cost_per_1m", 0.15)
    monkeypatch.setattr(settings, "llm_output_cost_per_1m", 0.60)
    harness = make_harness('{"type":"final_answer","answer":"done"}')
    result = asyncio.run(harness.run("current question"))
    assert result["status"] == "completed"
    assert result["tokens_used"] == 20
    assert result["embedding_tokens_used"] == 130
    assert result["cost_usd"] > 0
    assert harness.client.chat.completions.last_kwargs["reasoning_effort"] == "none"
    assert "max_completion_tokens" in harness.client.chat.completions.last_kwargs
    assert "temperature" not in harness.client.chat.completions.last_kwargs
    assert any(m["content"] == "earlier context" for m in harness.client.chat.completions.last_messages)


def test_harness_stops_before_provider_call_when_cost_budget_is_exhausted(monkeypatch):
    monkeypatch.setattr(settings, "max_run_tokens", 10000)
    monkeypatch.setattr(settings, "max_run_usd", 0.0000000001)
    monkeypatch.setattr(settings, "llm_input_cost_per_1m", 100.0)
    monkeypatch.setattr(settings, "llm_output_cost_per_1m", 100.0)
    harness = make_harness('{"type":"final_answer","answer":"should not run"}')
    result = asyncio.run(harness.run("current question"))
    assert result["status"] == "budget_exceeded"
    assert harness.client.chat.completions.calls == 0
