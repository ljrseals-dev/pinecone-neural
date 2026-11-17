from types import SimpleNamespace
import asyncio

from app.memory import PineconeMemory
from app.config import settings


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
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])], usage=SimpleNamespace(total_tokens=3))

    embeddings = Embeddings()


class FakeNeural:
    def __init__(self):
        self.args = None

    def on_remember(self, memory_id, vector, text, metadata):
        self.args = (memory_id, vector, text, metadata)


def test_remember_preserves_namespace_fields_and_calls_neural_contract(monkeypatch):
    monkeypatch.setattr(settings, "embedding_dim", 3)
    memory = PineconeMemory.__new__(PineconeMemory)
    memory.namespace = "user:operator:session:s1"
    memory.index = FakeIndex()
    memory.client = FakeClient()
    memory.neural = FakeNeural()
    memory.embedding_tokens_used = 0

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
    assert memory.embedding_tokens_used == 3


def test_remember_rejects_invalid_importance():
    memory = PineconeMemory.__new__(PineconeMemory)
    try:
        memory.remember("content", importance=float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite importance must be rejected")


class PreviouslyDeletedIndex(FakeIndex):
    def fetch(self, ids, namespace):
        return SimpleNamespace(vectors={
            ids[0]: SimpleNamespace(metadata={
                "deleted_at": "2026-01-01T00:00:00+00:00",
                "access_count": 4,
            })
        })


def test_remember_revives_previously_soft_deleted_content(monkeypatch):
    monkeypatch.setattr(settings, "embedding_dim", 3)
    memory = PineconeMemory.__new__(PineconeMemory)
    memory.namespace = "user:operator:session:s1"
    memory.index = PreviouslyDeletedIndex()
    memory.client = FakeClient()
    memory.neural = FakeNeural()
    memory.embedding_tokens_used = 0

    memory_id = memory.remember("safe content", role="fact", importance=0.8)
    assert memory.index.saved["id"] == memory_id
    assert memory.index.saved["metadata"]["deleted_at"] == ""
    assert memory.index.saved["metadata"]["deleted_at_epoch"] == 0


class ExistingActiveIndex(FakeIndex):
    def __init__(self):
        super().__init__()
        self.updated = None

    def fetch(self, ids, namespace):
        return SimpleNamespace(vectors={
            ids[0]: SimpleNamespace(
                metadata={
                    "text": "safe content",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "access_count": 3,
                    "importance": 0.8,
                    "deleted_at": "",
                },
                values=[0.1, 0.2, 0.3],
            )
        })

    def update(self, **kwargs):
        self.updated = kwargs


def test_idempotent_remember_repairs_neural_sidecar_and_legacy_metadata(monkeypatch):
    monkeypatch.setattr(settings, "embedding_dim", 3)
    memory = PineconeMemory.__new__(PineconeMemory)
    memory.namespace = "user:operator:session:s1"
    memory.index = ExistingActiveIndex()
    memory.client = FakeClient()
    memory.neural = FakeNeural()
    memory.embedding_tokens_used = 0

    memory_id = memory.remember("safe content", role="fact", importance=0.8)
    assert memory.index.updated["set_metadata"]["schema_version"] == "2.1"
    assert memory.index.updated["set_metadata"]["deleted_at_epoch"] == 0.0
    assert memory.neural.args[0] == memory_id
    assert memory.neural.args[1] == [0.1, 0.2, 0.3]
    assert memory.embedding_tokens_used == 0
