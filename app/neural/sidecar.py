"""Binary float32 sidecar for full-precision vectors."""
import os
import math
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Dict, List, Optional


class VectorSidecar:
    def __init__(self, db_path: str | None = None):
        self.db_path = Path(db_path or os.getenv("SIDECAR_DB", "data/vectors.db"))
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
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    dim INTEGER NOT NULL,
                    blob BLOB NOT NULL
                )""")

    @staticmethod
    def _pack(vector: List[float]) -> bytes:
        if not vector or len(vector) > 65536:
            raise ValueError("vector dimension is outside supported bounds")
        values = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("vector values must be finite")
        return struct.pack(f"{len(values)}f", *values)

    @staticmethod
    def _unpack(blob: bytes, dim: int) -> List[float]:
        if dim <= 0 or dim > 65536 or len(blob) != dim * 4:
            raise ValueError("stored vector has invalid dimensions")
        return list(struct.unpack(f"{dim}f", blob))

    def put(self, vector_id: str, vector: List[float]) -> None:
        blob = self._pack(vector)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO vectors(id, dim, blob) VALUES (?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET dim=excluded.dim, blob=excluded.blob",
                (vector_id, len(vector), blob),
            )

    def get(self, vector_id: str) -> Optional[List[float]]:
        cur = self._conn().execute(
            "SELECT dim, blob FROM vectors WHERE id=?", (vector_id,))
        row = cur.fetchone()
        return self._unpack(row[1], row[0]) if row else None

    def get_many(self, ids: List[str]) -> Dict[str, List[float]]:
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        cur = self._conn().execute(
            f"SELECT id, dim, blob FROM vectors WHERE id IN ({placeholders})",
            ids,
        )
        return {r[0]: self._unpack(r[2], r[1]) for r in cur.fetchall()}

    def delete(self, vector_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM vectors WHERE id=?", (vector_id,))
