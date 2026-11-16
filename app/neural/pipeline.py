"""Neural recall pipeline: GNN rerank + synaptic spreading + MemRL utility."""
import logging
from typing import Any, Dict, List, Optional

from app.neural.gnn_rerank import gnn_score_diffusion
from app.neural.sidecar import VectorSidecar
from app.neural.synaptic import SynapticLayer
from app.neural.utility import UtilityModel

logger = logging.getLogger(__name__)


class NeuralRecallPipeline:
    def __init__(self, enable_gnn: bool = True, enable_synaptic: bool = True,
                 enable_utility: bool = True):
        self.enable_gnn = enable_gnn
        self.enable_synaptic = enable_synaptic
        self.enable_utility = enable_utility
        self.sidecar = VectorSidecar() if enable_gnn else None
        self.synaptic = SynapticLayer() if enable_synaptic else None
        self.utility = UtilityModel() if enable_utility else None

    def on_remember(self, memory_id: str, vector: List[float],
                    text: str, metadata: Dict[str, Any]) -> None:
        if self.sidecar:
            try:
                self.sidecar.put(memory_id, vector)
            except Exception as e:
                logger.warning("Sidecar put failed: %s", e)
        if self.synaptic:
            try:
                self.synaptic.on_remember(memory_id, text, metadata)
            except Exception as e:
                logger.warning("Synaptic remember failed: %s", e)

    def on_coactivation(self, memory_ids: List[str], success: bool) -> None:
        if not self.synaptic or len(memory_ids) < 2:
            return
        try:
            self.synaptic.on_coactivation(memory_ids, success)
        except Exception as e:
            logger.warning("Synaptic coactivation failed: %s", e)

    def on_episode_reward(
        self,
        retrieved_metas: List[Dict[str, Any]],
        reward: float,
        similarities: Optional[List[float]] = None,
    ) -> None:
        """Call at the end of an agent run with the memories that were used."""
        if not self.enable_utility or not retrieved_metas:
            return
        try:
            loss = self.utility.update(retrieved_metas, reward, similarities)
            if loss is not None:
                logger.debug("Utility update loss=%.4f reward=%.2f", loss, reward)
        except Exception as e:
            logger.warning("Utility update failed: %s", e)

    def enhance(self, query_vec: List[float], matches: List[Any],
                top_k: int = 10) -> List[Dict[str, Any]]:
        candidates = [
            {"id": m.id, "score": m.score, "metadata": m.metadata}
            for m in matches
        ]
        if self.enable_gnn and self.sidecar and len(candidates) >= 3:
            try:
                ids = [c["id"] for c in candidates]
                vectors = self.sidecar.get_many(ids)
                for c in candidates:
                    c["vector"] = vectors.get(c["id"])
                candidates = [c for c in candidates if c.get("vector")]
                if len(candidates) >= 3:
                    candidates = gnn_score_diffusion(query_vec, candidates)
            except Exception as e:
                logger.warning("GNN diffusion failed: %s", e)

        if self.enable_synaptic and self.synaptic and candidates:
            try:
                seeds = [c["id"] for c in candidates[:5]]
                associated = self.synaptic.spread(seeds, hops=1, decay=0.5)
                seen = {c["id"] for c in candidates}
                for node in associated:
                    if node["id"] not in seen:
                        candidates.append({
                            "id": node["id"],
                            "score": node["score"],
                            "metadata": node.get("metadata", {}),
                        })
            except Exception as e:
                logger.warning("Synaptic spread failed: %s", e)

        candidates.sort(key=lambda c: c.get("diffused_score", c["score"]),
                        reverse=True)
        return candidates[:top_k]
