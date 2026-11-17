from types import SimpleNamespace

from app.config import settings
from app.memory import PineconeMemory
from app.neural.synaptic import SynapticLayer


class PurgeIndex:
    def __init__(self):
        self.deleted = []

    def query(self, **kwargs):
        match = SimpleNamespace(
            id="old-memory",
            metadata={"deleted_at": "1970-01-01T00:00:01+00:00", "deleted_at_epoch": 1.0},
        )
        return SimpleNamespace(matches=[match])

    def delete(self, ids, namespace):
        self.deleted.extend(ids)


class NeuralForget:
    def __init__(self):
        self.forgotten = []

    def forget(self, memory_id):
        self.forgotten.append(memory_id)


def test_retention_purge_deletes_pinecone_and_neural_records(monkeypatch):
    monkeypatch.setattr(settings, "soft_delete_retention_days", 0)
    memory = PineconeMemory.__new__(PineconeMemory)
    memory.namespace = "user:operator:session:s1"
    memory.index = PurgeIndex()
    memory.neural = NeuralForget()

    assert memory.purge_deleted() == 1
    assert memory.index.deleted == ["old-memory"]
    assert memory.neural.forgotten == ["old-memory"]


def test_synaptic_forget_removes_nodes_and_edges(tmp_path):
    layer = SynapticLayer(db_path=str(tmp_path / "synapses.db"))
    layer.on_remember("a", "first", {"importance": 0.8})
    layer.on_remember("b", "second", {"importance": 0.7})
    layer.on_coactivation(["a", "b"], success=True)
    layer.forget("a")
    assert layer.spread(["b"], hops=1) == [{"id": "b", "score": 0.7, "metadata": {}}]
