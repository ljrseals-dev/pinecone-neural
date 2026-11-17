"""
Layer 4: Learned Utility Scoring (MemRL-style)

Small MLP that predicts how useful a memory will be for the current query.
Cold-start: imitates the classic hand-tuned formula.
Online: Q-learning style updates from episode rewards.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("harness.neural.utility")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def hand_tuned_utility(
    similarity: float,
    recency_days: float = 0.0,
    access_count: int = 0,
    importance: float = 0.5,
    is_summary: bool = False,
    half_life_days: float = 30.0,
) -> float:
    """Classic formula used as teacher for cold-start."""
    values = (similarity, recency_days, float(access_count), importance, half_life_days)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("utility inputs must be finite")
    if half_life_days <= 0 or access_count < 0:
        raise ValueError("half-life must be positive and access count non-negative")
    similarity = max(0.0, min(1.0, float(similarity)))
    recency_days = max(0.0, float(recency_days))
    importance = max(0.0, min(1.0, float(importance)))
    age_decay = math.exp(-math.log(2) * recency_days / half_life_days)
    access_boost = 1.0 + 0.15 * math.log1p(access_count)
    summary_penalty = 0.6 if is_summary else 1.0
    raw = similarity * age_decay * access_boost * summary_penalty * (0.5 + importance)
    return max(0.0, min(1.0, raw))


class UtilityModel:
    """
    Lightweight utility scorer.

    - If torch is available → small MLP
    - Otherwise → pure hand-tuned formula (still useful)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        use_torch: bool = True,
    ):
        if model_path is None:
            from app.config import settings
            model_path = settings.utility_model_path
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._torch = None
        self._net = None
        self._opt = None
        self._updates = 0

        if use_torch:
            try:
                import torch
                import torch.nn as nn
                self._torch = torch
                self._nn = nn
                self._net = nn.Sequential(
                    nn.Linear(5, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),
                )
                self._opt = torch.optim.Adam(self._net.parameters(), lr=1e-3)
                self._load()
            except ImportError:
                logger.info("torch not available – using hand-tuned utility only")

    def _features(
        self,
        similarity: float,
        meta: Dict[str, Any],
    ) -> List[float]:
        values = [
            float(similarity),
            float(meta.get("recency_days", 0.0)),
            float(meta.get("access_count", 0)),
            float(meta.get("importance", 0.5)),
            1.0 if meta.get("is_summary") else 0.0,
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("utility features must be finite")
        values[0] = max(0.0, min(1.0, values[0]))
        values[1] = max(0.0, values[1])
        values[2] = max(0.0, values[2])
        values[3] = max(0.0, min(1.0, values[3]))
        return values

    def score(self, similarity: float, meta: Dict[str, Any]) -> float:
        features = self._features(similarity, meta)
        # Do not expose random initial MLP weights as a useful score. Use the
        # deterministic teacher until enough online feedback has accumulated.
        if self._net is None or self._updates < 50:
            return hand_tuned_utility(
                features[0], recency_days=features[1],
                access_count=int(features[2]), importance=features[3],
                is_summary=bool(features[4]),
            )
        import torch
        feats = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            return float(self._net(feats).item())

    def update(
        self,
        retrieved: List[Dict[str, Any]],
        reward: float,
        similarities: Optional[List[float]] = None,
    ) -> float:
        """
        Move predicted utilities toward the observed episode reward.
        Returns the MSE loss (0.0 if no torch).
        """
        if not retrieved:
            return 0.0
        if not math.isfinite(float(reward)):
            raise ValueError("reward must be finite")
        reward = max(0.0, min(1.0, float(reward)))
        sims = list(similarities) if similarities is not None else [
            float(item.get("score", 0.5)) for item in retrieved
        ]
        if len(sims) != len(retrieved):
            raise ValueError("similarities and retrieved memories must have equal length")
        if self._net is None:
            return 0.0

        import torch
        import torch.nn as nn
        feats = torch.tensor(
            [self._features(s, m) for s, m in zip(sims, retrieved)],
            dtype=torch.float32,
        )
        preds = self._net(feats).squeeze(-1)
        target = torch.full_like(preds, float(reward))
        loss = nn.MSELoss()(preds, target)
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        self._updates += 1
        if self._updates % 50 == 0:
            self._save()
        return float(loss.item())

    def imitate_hand_tuned(self, batch: List[Dict[str, Any]], epochs: int = 5) -> float:
        """
        Cold-start: train the MLP to imitate the hand-tuned formula.
        batch items must contain 'similarity' and the usual meta fields.
        """
        if self._net is None or not batch:
            return 0.0

        import torch
        import torch.nn as nn

        total_loss = 0.0
        for _ in range(epochs):
            feats, targets = [], []
            for item in batch:
                sim = float(item.get("similarity", 0.5))
                meta = item
                features = self._features(sim, meta)
                feats.append(features)
                targets.append(hand_tuned_utility(
                    features[0], recency_days=features[1],
                    access_count=int(features[2]), importance=features[3],
                    is_summary=bool(features[4]),
                ))
            x = torch.tensor(feats, dtype=torch.float32)
            y = torch.tensor(targets, dtype=torch.float32)
            pred = self._net(x).squeeze(-1)
            loss = nn.MSELoss()(pred, y)
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()
            total_loss += float(loss.item())
        self._save()
        return total_loss / max(epochs, 1)

    def _save(self) -> None:
        if self._net is None:
            return
        state = {
            "updates": self._updates,
            "state_dict": {k: v.tolist() for k, v in self._net.state_dict().items()},
        }
        # Simple JSON dump of weights (good enough for a tiny MLP)
        try:
            with open(self.model_path, "w") as f:
                json.dump(state, f)
        except Exception as e:
            logger.warning("Could not save utility model: %s", e)

    def _load(self) -> None:
        if not self.model_path.exists() or self._net is None:
            return
        try:
            import torch
            with open(self.model_path) as f:
                state = json.load(f)
            sd = {k: torch.tensor(v) for k, v in state["state_dict"].items()}
            self._net.load_state_dict(sd)
            self._updates = int(state.get("updates", 0))
            logger.info("Loaded utility model (%d updates)", self._updates)
        except Exception as e:
            logger.warning("Could not load utility model: %s", e)
