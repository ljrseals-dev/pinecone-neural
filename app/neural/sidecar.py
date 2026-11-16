"""
Full-precision vector sidecar for GNN diffusion.

Pinecone query() returns only IDs + metadata. GNN diffusion needs the actual
vectors. This lightweight SQLite store keeps them locally, keyed by vector ID.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional


class VectorSidecar:
    def __init__(self, db_path: str | Path = "data/vectors.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                id     TEXT PRIMARY KEY,
                vector TEXT NOT NULL   -- JSON array of floats
            )
            """
        )
        self.conn.commit()

    def put(self, vector_id: str, vector: List[float]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO vectors (id, vector) VALUES (?, ?)",
            (vector_id, json.dumps(vector)),
        )
        self.conn.commit()

    def get(self, vector_id: str) -> Optional[List[float]]:
        row = self.conn.execute(
            "SELECT vector FROM vectors WHERE id = ?", (vector_id,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def get_many(self, ids: List[str]) -> Dict[str, List[float]]:
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, vector FROM vectors WHERE id IN ({placeholders})", ids
        ).fetchall()
        return {r[0]: json.loads(r[1]) for r in rows}

    def delete(self, vector_id: str) -> None:
        self.conn.execute("DELETE FROM vectors WHERE id = ?", (vector_id,))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
