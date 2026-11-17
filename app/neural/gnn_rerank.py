"""
Layer 2: GNN Score Diffusion Reranking

Parameter-free 1-hop message passing over the candidate k-NN subgraph.
Recovers rank inversions introduced by HNSW / product quantization.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def gnn_score_diffusion(
    candidates: List[Dict[str, Any]],
    alpha: float = 0.60,
    k_graph: int = 10,
) -> List[Dict[str, Any]]:
    """
    Rerank ANN candidates via 1-hop score diffusion on their mutual similarity graph.

    Parameters
    ----------
    candidates : list of dicts
        Each must contain:
          - "id": str
          - "vector": list[float] | np.ndarray  (full-precision)
          - "score": float                      (noisy ANN score)
    alpha : float
        Diffusion strength. 0 = pure original scores, 1 = pure neighbor average.
    k_graph : int
        Number of neighbors used to build the local adjacency.

    Returns
    -------
    Reranked list (same dicts, with added "diffused_score"), sorted descending.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not isinstance(k_graph, int) or isinstance(k_graph, bool) or k_graph < 1:
        raise ValueError("k_graph must be a positive integer")
    n = len(candidates)
    if n < 3:
        for c in candidates:
            c["diffused_score"] = float(c.get("score", 0.0))
        return candidates

    # 1. Stack full-precision vectors
    V = np.asarray([c["vector"] for c in candidates], dtype=np.float32)
    if V.ndim != 2 or V.shape[1] == 0 or not np.isfinite(V).all():
        raise ValueError("candidate vectors must be finite and have matching dimensions")
    scores = np.asarray([float(c.get("score", 0.0)) for c in candidates], dtype=np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("candidate scores must be finite")
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
    V_norm = V / norms

    # 2. Pairwise cosine similarity
    sim_matrix = V_norm @ V_norm.T  # (n, n)

    # 3. Build sparse adjacency: top-k neighbors per node (exclude self)
    adj = np.zeros((n, n), dtype=np.float32)
    k = min(k_graph, n - 1)
    for i in range(n):
        row = sim_matrix[i].copy()
        row[i] = -np.inf
        # Negative cosine edges destabilize row normalization. Only positive
        # similarity contributes; zero-neighbor rows retain their original score.
        positive = np.maximum(row, 0.0)
        if not np.any(positive > 0):
            continue
        top_idx = np.argpartition(positive, -k)[-k:]
        top_idx = top_idx[positive[top_idx] > 0]
        adj[i, top_idx] = positive[top_idx]

    # 4. Row-normalize → transition matrix
    row_sums = adj.sum(axis=1, keepdims=True)
    P = np.divide(adj, row_sums, out=np.zeros_like(adj), where=row_sums > 1e-8)

    # 5. Initial scores (min-max normalized)
    s0 = scores
    s_min, s_max = s0.min(), s0.max()
    if s_max - s_min > 1e-8:
        s0 = (s0 - s_min) / (s_max - s_min)
    else:
        s0 = np.ones_like(s0) * 0.5

    # 6. One-hop diffusion
    s1 = alpha * (P @ s0) + (1.0 - alpha) * s0

    # 7. Write back and sort
    for i, c in enumerate(candidates):
        c["diffused_score"] = float(s1[i])

    return sorted(candidates, key=lambda x: x["diffused_score"], reverse=True)
