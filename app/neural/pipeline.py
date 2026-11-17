"""Neural recall pipeline: GNN rerank + synaptic spreading + utility scoring."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.neural.gnn_rerank import gnn_score_diffusion
from app.neural.sidecar import VectorSidecar
from app.neural.synaptic import SynapticLayer
from app.neural.utility import UtilityModel

logger = logging.getLogger(__name__)


def _recency_days(timestamp: Any) -> float:
    if not timestamp:
        return 0.0
    try:
        value = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - value).total_seconds() / 86400.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0


class NeuralRecallPipeline:
    def __init__(self, enable_gnn: bool = True, enable_synaptic: bool = True,
                 enable_utility: bool = True):
        self.enable_gnn = enable_gnn
        self.enable_synaptic = enable_synaptic
        self.enable_utility = enable_utility
        self.sidecar = VectorSidecar() if enable_gnn else None
        self.synaptic = SynapticLayer() if enable_synaptic else None
        self.utility = UtilityModel() if enable_utility else None

    def on_remember(self, memory_id: str, vector: List[float], text: str,
                    metadata: Dict[str, Any]) -> None:
        if self.sidecar:
            try:
                self.sidecar.put(memory_id, vector)
            except Exception as exc:
                logger.warning("Sidecar put failed: %s", type(exc).__name__)
        if self.synaptic:
            try:
                self.synaptic.on_remember(memory_id, text, metadata)
            except Exception as exc:
                logger.warning("Synaptic remember failed: %s", type(exc).__name__)

    def forget(self, memory_id: str) -> None:
        if self.sidecar:
            try:
                self.sidecar.delete(memory_id)
            except Exception as exc:
                logger.warning("Sidecar delete failed: %s", type(exc).__name__)
        if self.synaptic:
            try:
                self.synaptic.forget(memory_id)
            except Exception as exc:
                logger.warning("Synaptic delete failed: %s", type(exc).__name__)

    def on_coactivation(self, memory_ids: List[str], success: bool) -> None:
        if not self.synaptic or len(memory_ids) < 2:
            return
        try:
            self.synaptic.on_coactivation(memory_ids, success)
        except Exception as exc:
            logger.warning("Synaptic coactivation failed: %s", type(exc).__name__)

    def on_episode_reward(
        self,
        retrieved_metas: List[Dict[str, Any]],
        reward: float,
        similarities: Optional[List[float]] = None,
    ) -> None:
        if not self.enable_utility or not retrieved_metas or self.utility is None:
            return
        try:
            loss = self.utility.update(retrieved_metas, reward, similarities)
            if loss is not None:
                logger.debug("Utility update loss=%.4f reward=%.2f", loss, reward)
        except Exception as exc:
            logger.warning("Utility update failed: %s", type(exc).__name__)

    def enhance(self, query_vec: List[float], matches: List[Dict[str, Any]],
                top_k: int = 10) -> List[Dict[str, Any]]:
        """Enhance normalized Pinecone result dictionaries; never assumes SDK objects."""
        if not matches or top_k <= 0:
            return []
        candidates: List[Dict[str, Any]] = []
        for match in matches:
            candidate = dict(match)
            candidate["score"] = float(candidate.get("score") or 0.0)
            candidate["metadata"] = dict(candidate.get("metadata") or {})
            candidates.append(candidate)

        if self.enable_gnn and self.sidecar and len(candidates) >= 3:
            try:
                vectors = self.sidecar.get_many([c["id"] for c in candidates if c.get("id")])
                for candidate in candidates:
                    vector = vectors.get(candidate.get("id"))
                    if vector:
                        candidate["vector"] = vector
                vector_candidates = [c for c in candidates if c.get("vector")]
                if len(vector_candidates) >= 3:
                    reranked = gnn_score_diffusion(vector_candidates)
                    by_id = {c["id"]: c for c in reranked}
                    candidates = [by_id.get(c["id"], c) for c in candidates]
            except Exception as exc:
                logger.warning("GNN diffusion failed: %s", type(exc).__name__)

        if self.enable_synaptic and self.synaptic and candidates:
            try:
                seeds = [c["id"] for c in candidates[: min(5, len(candidates))] if c.get("id")]
                associated = self.synaptic.spread(seeds, hops=1, decay=0.5)
                scores = {item["id"]: float(item["score"]) for item in associated}
                for candidate in candidates:
                    candidate["synaptic_score"] = scores.get(candidate.get("id"), 0.0)
                seen = {c.get("id") for c in candidates}
                for item in associated:
                    if item.get("id") not in seen:
                        candidates.append({
                            "id": item["id"], "score": float(item.get("score", 0.0)),
                            "text": "", "role": "", "timestamp": "", "pinned": False,
                            "is_summary": False, "importance": 0.5,
                            "metadata": {}, "synaptic_score": float(item.get("score", 0.0)),
                        })
            except Exception as exc:
                logger.warning("Synaptic spread failed: %s", type(exc).__name__)

        for candidate in candidates:
            meta = candidate["metadata"]
            meta.setdefault("recency_days", _recency_days(
                candidate.get("timestamp") or meta.get("created_at")
            ))
            meta.setdefault("importance", candidate.get("importance", 0.5))
            meta.setdefault("access_count", candidate.get("access_count", 0))
            meta.setdefault("is_summary", candidate.get("is_summary", False))
            candidate["recency_days"] = meta["recency_days"]
            candidate["access_count"] = meta["access_count"]
            utility_score = candidate["score"]
            if self.enable_utility and self.utility is not None:
                try:
                    utility_score = self.utility.score(candidate["score"], meta)
                except Exception as exc:
                    logger.warning("Utility scoring failed: %s", type(exc).__name__)
            diffused = float(candidate.get("diffused_score", candidate["score"]))
            synaptic = float(candidate.get("synaptic_score", 0.0))
            candidate["utility_score"] = max(0.0, min(1.0, float(utility_score)))
            candidate["rank_score"] = (
                0.65 * diffused + 0.30 * candidate["utility_score"]
                + 0.05 * max(0.0, min(1.0, synaptic))
            )
            candidate.pop("vector", None)

        return sorted(candidates, key=lambda item: item["rank_score"], reverse=True)[:top_k]
