"""
Neural Recall Pipeline – Layers 2 (GNN) + 3 (Synaptic) + 4 (Utility)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.neural.gnn_rerank import gnn_score_diffusion
from app.neural.sidecar import VectorSidecar
from app.neural.synaptic import SynapticLayer
from app.neural.utility import UtilityModel

logger = logging.getLogger("harness.neural")


class NeuralRecallPipeline:
    def __init__(
        self,
        sidecar: Optional[VectorSidecar] = None,
        synaptic: Optional[SynapticLayer] = None,
        utility: Optional[UtilityModel] = None,
        enable_gnn: bool = True,
        enable_synaptic: bool = True,
        enable_utility: bool = True,
        gnn_alpha: float = 0.60,
        synaptic_hops: int = 1,
        utility_lambda: float = 0.5,
    ):
        self.sidecar = sidecar or VectorSidecar()
        self.synaptic = synaptic or SynapticLayer()
        self.utility = utility or UtilityModel()
        self.enable_gnn = enable_gnn
        self.enable_synaptic = enable_synaptic
        self.enable_utility = enable_utility
        self.gnn_alpha = gnn_alpha
        self.synaptic_hops = synaptic_hops
        self.utility_lambda = utility_lambda  # blend of similarity vs utility

    # ------------------------------------------------------------------
    # Write hooks
    # ------------------------------------------------------------------
    def on_remember(
        self,
        memory_id: str,
        vector: List[float],
        importance: float = 0.5,
    ) -> None:
        self.sidecar.put(memory_id, vector)
        if self.enable_synaptic:
            self.synaptic.on_remember(memory_id, importance=importance)

    def on_coactivation(self, ids: List[str], success: bool) -> None:
        if not self.enable_synaptic or len(ids) < 2:
            return
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                self.synaptic.on_coactivation(a, b, success=success)

    def on_episode_reward(
        self,
        retrieved_metas: List[Dict[str, Any]],
        reward: float,
        similarities: Optional[List[float]] = None,
    ) -> None:
        """Call at the end of an agent run with the memories that were used."""
        if self.enable_utility and retrieved_metas:
            try:
                loss = self.utility.update(retrieved_metas, reward, similarities)
                if loss is not None:
                    logger.debug("Utility update loss=%.4f reward=%.2f", loss, reward)
            except Exception as e:
                logger.warning("Utility update failed: %s", e)

    # ------------------------------------------------------------------
    # Enhanced recall
    # ------------------------------------------------------------------
    def enhance(
        self,
        raw_matches: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if not raw_matches:
            return []

        candidates = list(raw_matches)

        # Layer 2 – GNN diffusion
        if self.enable_gnn:
            ids = [c["id"] for c in candidates]
            vectors = self.sidecar.get_many(ids)
            gnn_input = []
            for c in candidates:
                vec = vectors.get(c["id"])
                if vec is not None:
                    gnn_input.append(
                        {
                            "id": c["id"],
                            "vector": vec,
                            "score": float(c.get("score", 0.0)),
                            "_orig": c,
                        }
                    )
            if len(gnn_input) >= 3:
                try:
                    reranked = gnn_score_diffusion(gnn_input, alpha=self.gnn_alpha)
                    candidates = []
                    for item in reranked:
                        orig = item["_orig"]
                        orig["diffused_score"] = item["diffused_score"]
                        candidates.append(orig)
                except Exception as e:
                    logger.warning("GNN diffusion failed: %s", e)

        # Layer 3 – Synaptic spreading
        if self.enable_synaptic and candidates:
            seed_ids = [c["id"] for c in candidates[:5]]
            try:
                activated = self.synaptic.spread(
                    seed_ids, hops=self.synaptic_hops, top_n=top_k
                )
                known = {c["id"] for c in candidates}
                for node in activated:
                    if node.id not in known:
                        candidates.append(
                            {
                                "id": node.id,
                                "score": node.activation,
                                "diffused_score": node.activation,
                                "text": "",
                                "role": "associated",
                                "synaptic": True,
                                "hops": node.hops,
                                "importance": 0.5,
                                "is_summary": False,
                                "access_count": 0,
                            }
                        )
                        known.add(node.id)
            except Exception as e:
                logger.warning("Synaptic spread failed: %s", e)

        # Layer 4 – Learned utility blend
        if self.enable_utility and candidates:
            try:
                for c in candidates:
                    sim = float(c.get("diffused_score", c.get("score", 0.0)))
                    util = self.utility.score(sim, c)
                    # rank = (1-λ)·sim + λ·utility   (both roughly in [0,1])
                    c["utility"] = util
                    c["final_score"] = (
                        (1.0 - self.utility_lambda) * sim
                        + self.utility_lambda * util
                    )
            except Exception as e:
                logger.warning("Utility scoring failed: %s", e)
                for c in candidates:
                    c["final_score"] = float(c.get("diffused_score", c.get("score", 0.0)))
        else:
            for c in candidates:
                c["final_score"] = float(c.get("diffused_score", c.get("score", 0.0)))

        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return candidates[:top_k]
