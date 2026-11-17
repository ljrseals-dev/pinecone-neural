import asyncio

from app.neural.gnn_rerank import gnn_score_diffusion
from app.neural.pipeline import NeuralRecallPipeline


class FakeSidecar:
    def __init__(self, vectors):
        self.vectors = vectors

    def get_many(self, ids):
        return {key: self.vectors[key] for key in ids if key in self.vectors}

    def put(self, key, vector):
        self.vectors[key] = vector


class FakeSynaptic:
    def __init__(self):
        self.remembered = None

    def on_remember(self, memory_id, text, metadata):
        self.remembered = (memory_id, text, metadata)

    def spread(self, seed_ids, hops=1, decay=0.5):
        return [
            {"id": seed_ids[0], "score": 0.4, "metadata": {}},
            {"id": "associated-memory", "score": 0.7, "metadata": {}},
        ]


class FakeUtility:
    def score(self, similarity, metadata):
        return 0.9

    def update(self, retrieved, reward, similarities=None):
        return 0.0


def test_neural_remember_uses_consistent_layer_contracts():
    pipeline = NeuralRecallPipeline(enable_gnn=False, enable_synaptic=False, enable_utility=False)
    pipeline.sidecar = FakeSidecar({})
    pipeline.synaptic = FakeSynaptic()
    pipeline.on_remember("memory-1", [0.1, 0.2], "remember this", {"importance": 0.8})
    assert pipeline.sidecar.vectors["memory-1"] == [0.1, 0.2]
    assert pipeline.synaptic.remembered == ("memory-1", "remember this", {"importance": 0.8})


def test_pipeline_enhance_accepts_normalized_memory_dicts_and_applies_utility():
    pipeline = NeuralRecallPipeline(enable_gnn=False, enable_synaptic=False, enable_utility=False)
    pipeline.enable_utility = True
    pipeline.utility = FakeUtility()
    result = pipeline.enhance(
        [1.0, 0.0],
        [{"id": "m1", "score": 0.7, "text": "evidence", "timestamp": "",
          "importance": 0.8, "metadata": {}}],
        top_k=3,
    )
    assert len(result) == 1
    assert result[0]["utility_score"] == 0.9
    assert result[0]["rank_score"] > 0.7


def test_synaptic_spread_surfaces_associated_ids_for_memory_hydration():
    pipeline = NeuralRecallPipeline(enable_gnn=False, enable_synaptic=False, enable_utility=False)
    pipeline.enable_synaptic = True
    pipeline.synaptic = FakeSynaptic()
    result = pipeline.enhance(
        [1.0, 0.0],
        [{"id": "m1", "score": 0.8, "text": "candidate", "metadata": {}}],
        top_k=5,
    )
    assert {item["id"] for item in result} == {"m1", "associated-memory"}


def test_gnn_diffusion_handles_all_negative_cosine_edges():
    candidates = [
        {"id": "a", "score": 0.9, "vector": [1.0, 0.0]},
        {"id": "b", "score": 0.8, "vector": [-1.0, 0.0]},
        {"id": "c", "score": 0.7, "vector": [0.0, 1.0]},
    ]
    result = gnn_score_diffusion(candidates)
    assert all(0.0 <= item["diffused_score"] <= 1.0 for item in result)
    assert len(result) == 3


def test_utility_feedback_rejects_mismatched_similarity_lengths():
    from app.neural.utility import UtilityModel
    model = UtilityModel(use_torch=False)
    try:
        model.update([{"score": 0.5}], reward=1.0, similarities=[])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched utility feedback must fail")
