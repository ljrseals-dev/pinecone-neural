"""Neural layer health and size metrics."""
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _safe_count(db: Path, table: str) -> int:
    try:
        conn = sqlite3.connect(str(db), timeout=2.0)
        try:
            return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def neural_health() -> Dict[str, Any]:
    sidecar = Path(os.getenv("SIDECAR_DB", "data/vectors.db"))
    syn = Path(os.getenv("SYNAPTIC_DB", "data/synapses.db"))
    util = Path(os.getenv("UTILITY_MODEL", "data/utility.pt"))

    return {
        "gnn": {
            "enabled": sidecar.exists(),
            "sidecar_bytes": _safe_size(sidecar),
            "sidecar_rows": _safe_count(sidecar, "vectors") if sidecar.exists() else 0,
        },
        "synaptic": {
            "enabled": syn.exists(),
            "db_bytes": _safe_size(syn),
            "nodes": _safe_count(syn, "nodes") if syn.exists() else 0,
            "edges": _safe_count(syn, "edges") if syn.exists() else 0,
        },
        "utility": {
            "enabled": util.exists(),
            "model_bytes": _safe_size(util),
        },
    }
