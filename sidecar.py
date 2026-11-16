"""
Full-precision vector sidecar for GNN diffusion.

Uses binary float32 storage (struct) instead of JSON for ~5× smaller footprint
and faster reads. Thread-safe via connection-per-thread pattern.
"""

from __future__ import annotations

import sqlite3
import struct
import threading
from pathlib import Path
from typing import Dict, List, Optional


class VectorSidecar:
    """
    Local store of full-precision vectors keyed by Pinecone vector ID.

    - Binary float32 blobs (not JSON)
    - One SQLite connection per thread
    """

    def __init__(self, db_path: str | Path = "data/vectors.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Ensure schema exists on construction
        self._conn().execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                id     TEXT PRIMARY KEY,
                dim    INTEGER NOT NULL,
                blob   BLOB NOT NULL
            )
            """
        )
        self._conn().commit()

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

    @staticmethod
    def _pack(vector: List[float]) -> bytes:
        return struct.pack(f"{len(vector)}f", *vector)

    @staticmethod
    def _unpack(blob: bytes, dim: int) -> List[float]:
        return list(struct.unpack(f"{dim}f", blob))

    def put(self, vector_id: str, vector: List[float]) -> None:
        blob = self._pack(vector)
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO vectors (id, dim, blob) VALUES (?, ?, ?)",
            (vector_id, len(vector), blob),
        )
        conn.commit()

    def get(self, vector_id: str) -> Optional[List[float]]:
        row = self._conn().execute(
            "SELECT dim, blob FROM vectors WHERE id = ?", (vector_id,)
        ).fetchone()
        if row is None:
            return None
        return self._unpack(row[1], row[0])

    def get_many(self, ids: List[str]) -> Dict[str, List[float]]:
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self._conn().execute(
            f"SELECT id, dim, blob FROM vectors WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return {r[0]: self._unpack(r[2], r[1]) for r in rows}

    def delete(self, vector_id: str) -> None:
        conn = self._conn()
        conn.execute("DELETE FROM vectors WHERE id = ?", (vector_id,))
        conn.commit()

    def ids(self) -> List[str]:
        rows = self._conn().execute("SELECT id FROM vectors").fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        if getattr(self._local, "conn", None):
            self._local.conn.close()
            self._local.conn = None
