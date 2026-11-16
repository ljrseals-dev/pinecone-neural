"""
Neural Recall Pipeline – wires Layer 2 (GNN) + Layer 3 (Synaptic)
on top of Pinecone vector search.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.neural.gnn_rerank import gnn_score_diffusion
from app.neural.sidecar import VectorSidecar
from app.neural.synaptic import SynapticLayer

logger = logging.getLogger("harness.neural")


class NeuralRecallPipeline:
    """
    Orchestrates the neural-enhanced recall path:

        Pinecone ANN  →  GNN diffusion  →  Synaptic spreading  →  final ranked list
    """

    def __init__(
        self,
        sidecar: Optional[VectorSidecar] = None,
        synaptic: Optional[SynapticLayer] = None,
        enable_gnn: bool = True,
        enable_synaptic: bool = True,
        gnn_alpha: float = 0.60,
        synaptic_hops: int = 1,
    ):
        self.sidecar = sidecar or VectorSidecar()
        self.synaptic = synaptic or SynapticLayer()
        self.enable_gnn = enable_gnn
        self.enable_synaptic = enable_synaptic
        self.gnn_alpha = gnn_alpha
        self.synaptic_hops = synaptic_hops

    # ------------------------------------------------------------------
    # Write hooks (call from memory.remember)
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
        """Call after an agent step with the set of memory IDs that were used."""
        if not self.enable_synaptic or len(ids) < 2:
            return
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                self.synaptic.on_coactivation(a, b, success=success)

    # ------------------------------------------------------------------
    # Enhanced recall
    # ------------------------------------------------------------------
    def enhance(
        self,
        raw_matches: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        raw_matches: list of dicts from Pinecone-style recall
                     (must contain at least "id", "score", "text", ...)
        Returns a new ranked list, possibly with extra associated memories.
        """
        if not raw_matches:
            return []

        candidates = list(raw_matches)

        # ----- Layer 2: GNN score diffusion -----
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
                    reranked = gnn_score_diffusion(
                        gnn_input, alpha=self.gnn_alpha
                    )
                    # Preserve original metadata, attach diffused_score
                    candidates = []
                    for item in reranked:
                        orig = item["_orig"]
                        orig["diffused_score"] = item["diffused_score"]
                        candidates.append(orig)
                except Exception as e:
                    logger.warning("GNN diffusion failed, falling back: %s", e)

        # ----- Layer 3: Synaptic spreading -----
        if self.enable_synaptic and candidates:
            seed_ids = [c["id"] for c in candidates[:5]]
            try:
                activated = self.synaptic.spread(
                    seed_ids, hops=self.synaptic_hops, top_n=top_k
                )
                known = {c["id"] for c in candidates}
                # Append associated nodes that were not already in the vector result
                for node in activated:
                    if node.id not in known:
                        candidates.append(
                            {
                                "id": node.id,
                                "score": node.activation,
                                "diffused_score": node.activation,
                                "text": "",  # caller can hydrate if needed
                                "role": "associated",
                                "synaptic": True,
                                "hops": node.hops,
                            }
                        )
                        known.add(node.id)
            except Exception as e:
                logger.warning("Synaptic spread failed, falling back: %s", e)

        # Final sort: prefer diffused_score when present
        def sort_key(c: Dict[str, Any]) -> float:
            return float(c.get("diffused_score", c.get("score", 0.0)))

        candidates.sort(key=sort_key, reverse=True)
        return candidates[:top_k]
