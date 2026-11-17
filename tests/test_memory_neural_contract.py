from types import SimpleNamespace
import asyncio

from app.memory import PineconeMemory


class FakeIndex:
    def __init__(self):
        self.saved = None

    def fetch(self, ids, namespace):
        return SimpleNamespace(vectors={})

    def query(self, **kwargs):
        return SimpleNamespace(matches=[])

    def upsert(self, **kwargs):
        self.saved = kwargs["vectors"][0]


class FakeClient:
    class Embeddings:
        @staticmethod
        def create(**kwargs):
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])

    embeddings = Embeddings()


class FakeNeural:
    def __init__(self):
        self.args = None

    def on_remember(self, memory_id, vector, text, metadata):
        self.args = (memory_id, vector, text, metadata)


def test_remember_preserves_namespace_fields_and_calls_neural_contract():
    memory = PineconeMemory.__new__(PineconeMemory)
    memory.namespace = "user:operator:session:s1"
    memory.index = FakeIndex()
    memory.client = FakeClient()
    memory.neural = FakeNeural()

    memory_id = memory.remember(
        "safe content",
        role="fact",
        importance=0.8,
        metadata={"role": "attacker", "session_id": "other-user",
                  "importance": 0.0, "deleted_at": "deleted"},
    )

    saved = memory.index.saved
    assert saved["id"] == memory_id
    assert saved["metadata"]["role"] == "fact"
    assert saved["metadata"]["session_id"] == memory.namespace
    assert saved["metadata"]["importance"] == 0.8
    assert saved["metadata"]["deleted_at"] == ""
    assert memory.neural.args[0] == memory_id
    assert memory.neural.args[2] == "safe content"


def test_remember_rejects_invalid_importance():
    memory = PineconeMemory.__new__(PineconeMemory)
    try:
        memory.remember("content", importance=float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite importance must be rejected")
