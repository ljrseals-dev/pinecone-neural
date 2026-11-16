"""
Layer 3: Synaptic Associative Memory

Hebbian edge weights + spreading activation.
Thread-safe SQLite (connection-per-thread).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


@dataclass
class ActivatedNode:
    id: str
    activation: float
    hops: int


class SynapticLayer:
    """
    Lightweight associative graph stored in SQLite.

    - Nodes = memory IDs (importance, vitality, access_count)
    - Edges = Hebbian weights in [0, 1]
    - Spreading activation surfaces neighbors pure vector similarity misses
    """

    def __init__(self, db_path: str | Path = "data/synapses.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=True,
                timeout=30.0,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id          TEXT PRIMARY KEY,
                importance  REAL DEFAULT 0.5,
                vitality    REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                last_access REAL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS edges (
                src     TEXT NOT NULL,
                dst     TEXT NOT NULL,
                weight  REAL DEFAULT 0.1,
                PRIMARY KEY (src, dst)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
            """
        )
        conn.commit()

    def on_remember(
        self,
        memory_id: str,
        importance: float = 0.5,
        vitality: float = 1.0,
    ) -> None:
        now = time.time()
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO nodes (id, importance, vitality, access_count, last_access)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(id) DO UPDATE SET
                importance   = excluded.importance,
                vitality     = excluded.vitality,
                access_count = access_count + 1,
                last_access  = excluded.last_access
            """,
            (memory_id, importance, vitality, now),
        )
        conn.commit()

    def on_coactivation(
        self,
        id_a: str,
        id_b: str,
        success: bool,
        delta_success: float = 0.10,
        delta_failure: float = -0.15,
    ) -> None:
        if id_a == id_b:
            return
        delta = delta_success if success else delta_failure
        conn = self._conn()
        for src, dst in ((id_a, id_b), (id_b, id_a)):
            conn.execute(
                """
                INSERT INTO edges (src, dst, weight) VALUES (?, ?, ?)
                ON CONFLICT(src, dst) DO UPDATE SET
                    weight = MAX(0.0, MIN(1.0, weight + ?))
                """,
                (src, dst, max(0.0, min(1.0, 0.1 + delta)), delta),
            )
        conn.commit()

    def spread(
        self,
        seed_ids: List[str],
        hops: int = 1,
        decay: float = 0.5,
        top_n: int = 20,
    ) -> List[ActivatedNode]:
        if not seed_ids:
            return []

        activation: Dict[str, float] = {sid: 1.0 for sid in seed_ids}
        hop_of: Dict[str, int] = {sid: 0 for sid in seed_ids}
        frontier: Set[str] = set(seed_ids)
        conn = self._conn()

        for h in range(1, hops + 1):
            next_frontier: Set[str] = set()
            for node in frontier:
                parent_act = activation.get(node, 0.0)
                cur = conn.execute(
                    "SELECT dst, weight FROM edges WHERE src = ?", (node,)
                )
                for dst, weight in cur.fetchall():
                    contrib = parent_act * float(weight) * (decay ** h)
                    if contrib < 0.01:
                        continue
                    prev = activation.get(dst, 0.0)
                    if contrib > prev:
                        activation[dst] = contrib
                        hop_of[dst] = h
                        next_frontier.add(dst)
            frontier = next_frontier
            if not frontier:
                break

        ranked: List[Tuple[str, float, int]] = []
        for mid, act in activation.items():
            row = conn.execute(
                "SELECT importance, vitality FROM nodes WHERE id = ?", (mid,)
            ).fetchone()
            imp, vit = (row[0], row[1]) if row else (0.5, 1.0)
            score = act * float(imp) * float(vit)
            ranked.append((mid, score, hop_of.get(mid, 0)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return [
            ActivatedNode(id=mid, activation=score, hops=h)
            for mid, score, h in ranked[:top_n]
        ]

    def close(self) -> None:
        if getattr(self._local, "conn", None):
            self._local.conn.close()
            self._local.conn = None
