"""Synaptic associative memory: Hebbian edges + spreading activation."""
import math
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class SynapticLayer:
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            from app.config import settings
            db_path = settings.synaptic_db
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(str(self.db_path),
                                   check_same_thread=True, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    importance REAL DEFAULT 0.5,
                    vitality REAL DEFAULT 1.0,
                    access_count INTEGER DEFAULT 0,
                    last_access TEXT
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    src TEXT, dst TEXT, weight REAL DEFAULT 0.1,
                    PRIMARY KEY (src, dst)
                )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)")

    def on_remember(self, memory_id: str, text: str,
                    metadata: Dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        importance = float(metadata.get("importance", 0.5))
        if not math.isfinite(importance):
            raise ValueError("importance must be finite")
        importance = max(0.0, min(1.0, importance))
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO nodes(id, importance, last_access, access_count)
                VALUES (?,?,?,1)
                ON CONFLICT(id) DO UPDATE SET
                    importance = excluded.importance,
                    access_count = access_count + 1,
                    last_access = excluded.last_access""",
                (memory_id, importance, now))

    def on_coactivation(self, memory_ids: List[str], success: bool) -> None:
        if len(memory_ids) < 2:
            return
        delta = 0.10 if success else -0.15
        with self._conn() as conn:
            for i, a in enumerate(memory_ids):
                for b in memory_ids[i + 1:]:
                    for src, dst in ((a, b), (b, a)):
                        cur = conn.execute(
                            "SELECT weight FROM edges WHERE src=? AND dst=?",
                            (src, dst))
                        row = cur.fetchone()
                        w = (row[0] if row else 0.1) + delta
                        w = max(0.0, min(1.0, w))
                        conn.execute(
                            "INSERT INTO edges(src, dst, weight) VALUES (?,?,?) "
                            "ON CONFLICT(src, dst) DO UPDATE SET weight=excluded.weight",
                            (src, dst, w))

    def forget(self, memory_id: str) -> None:
        """Remove a purged memory node and every edge that references it."""
        with self._conn() as conn:
            conn.execute("DELETE FROM edges WHERE src=? OR dst=?", (memory_id, memory_id))
            conn.execute("DELETE FROM nodes WHERE id=?", (memory_id,))

    def spread(self, seed_ids: List[str], hops: int = 1,
               decay: float = 0.5) -> List[Dict[str, Any]]:
        if not seed_ids:
            return []
        if not isinstance(hops, int) or isinstance(hops, bool) or not 0 <= hops <= 3:
            raise ValueError("hops must be between 0 and 3")
        if not math.isfinite(decay) or not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be between 0 and 1")
        activation: Dict[str, float] = {sid: 1.0 for sid in set(seed_ids)}
        frontier = dict(activation)
        for _ in range(hops):
            next_frontier: Dict[str, float] = {}
            with self._conn() as conn:
                for src, act in frontier.items():
                    cur = conn.execute(
                        "SELECT dst, weight FROM edges WHERE src=?", (src,))
                    for dst, w in cur.fetchall():
                        contrib = act * w * decay
                        if contrib > activation.get(dst, 0.0):
                            activation[dst] = contrib
                            next_frontier[dst] = contrib
            frontier = next_frontier
            if not frontier:
                break
        results = []
        with self._conn() as conn:
            for mid, act in activation.items():
                cur = conn.execute(
                    "SELECT importance, vitality FROM nodes WHERE id=?", (mid,))
                row = cur.fetchone()
                if row:
                    imp, vit = row
                    results.append({
                        "id": mid,
                        "score": act * imp * vit,
                        "metadata": {},
                    })
        results.sort(key=lambda r: r["score"], reverse=True)
        return results
