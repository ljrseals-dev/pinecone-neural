# Copyright (c) 2026 Lewis James Seals Jr.
# Author: Lewis James Seals Jr. (@ljrseals-dev)
# Project: Pinecone Neural Agent Harness
# SPDX-License-Identifier: MIT

"""One-shot migration: JSON TEXT sidecar -> binary float32 BLOB."""
import json
import shutil
import sqlite3
import struct
import sys
from pathlib import Path

SRC = Path("data/vectors.db")
BACKUP = Path("data/vectors.db.pre_v23")


def main() -> int:
    if not SRC.exists():
        print(f"No sidecar at {SRC}; nothing to migrate.")
        return 0

    shutil.copy2(SRC, BACKUP)
    print(f"Backed up {SRC} -> {BACKUP}")

    conn = sqlite3.connect(str(SRC))
    cols = [r[1] for r in conn.execute("PRAGMA table_info(vectors)").fetchall()]
    if "blob" in cols and "vector" not in cols:
        print("Already migrated (blob column present, vector absent).")
        return 0

    print(f"Current columns: {cols}")
    conn.execute("ALTER TABLE vectors RENAME TO vectors_old")
    conn.execute("""
        CREATE TABLE vectors (
            id TEXT PRIMARY KEY,
            dim INTEGER NOT NULL,
            blob BLOB NOT NULL
        )""")

    cur = conn.execute("SELECT id, vector FROM vectors_old")
    n = 0
    for vid, raw in cur.fetchall():
        vec = json.loads(raw)
        blob = struct.pack(f"{len(vec)}f", *vec)
        conn.execute("INSERT INTO vectors(id, dim, blob) VALUES (?,?,?)",
                     (vid, len(vec), blob))
        n += 1
    conn.commit()
    conn.execute("DROP TABLE vectors_old")
    conn.commit()
    conn.close()
    print(f"Migrated {n} vectors to binary BLOB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
