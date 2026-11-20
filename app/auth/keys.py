# Copyright (c) 2026 Lewis James Seals Jr.
# Author: Lewis James Seals Jr. (@ljrseals-dev)
# Project: Pinecone Neural Agent Harness
# SPDX-License-Identifier: MIT

"""API-key authentication. Configure API_KEYS as key:user_id comma-separated pairs."""
import hmac
import os
import re
from typing import Any, Dict, Optional


class APIKeyStore:
    def __init__(self, raw: Optional[str] = None) -> None:
        raw = (os.getenv("API_KEYS", "") if raw is None else raw).strip()
        self._keys: Dict[str, Dict[str, Any]] = {}
        if not raw:
            return
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry or ":" not in entry:
                raise ValueError("API_KEYS contains a malformed entry")
            key, user_id = entry.split(":", 1)
            key = key.strip()
            user_id = user_id.strip()
            if len(key) < 32 or len(key) > 256 or not key.isascii():
                raise ValueError("API_KEYS entries must use 32-256 character ASCII keys")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", user_id):
                raise ValueError("API_KEYS user IDs must be 1-64 safe characters")
            if key in self._keys:
                raise ValueError("API_KEYS contains a duplicate key")
            self._keys[key] = {"user_id": user_id}

    def validate(self, key: str) -> Optional[Dict[str, Any]]:
        if not isinstance(key, str) or not key:
            return None
        if len(key) > 256 or not key.isascii():
            return None
        matched: Optional[Dict[str, Any]] = None
        for stored, info in self._keys.items():
            if hmac.compare_digest(stored, key):
                matched = info
        return dict(matched) if matched is not None else None

    def __len__(self) -> int:
        return len(self._keys)
