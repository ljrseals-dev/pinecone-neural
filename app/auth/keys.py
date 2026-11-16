"""API key store. Format env var: API_KEYS=key1:user1,key2:user2"""
import os
import hmac
from typing import Optional, Dict, Any


class APIKeyStore:
    def __init__(self) -> None:
        raw = os.getenv("API_KEYS", "").strip()
        self._keys: Dict[str, Dict[str, Any]] = {}
        if raw:
            for entry in raw.split(","):
                entry = entry.strip()
                if not entry or ":" not in entry:
                    continue
                key, user_id = entry.split(":", 1)
                self._keys[key.strip()] = {"user_id": user_id.strip()}

    def validate(self, key: str) -> Optional[Dict[str, Any]]:
        if not key:
            return None
        for stored, info in self._keys.items():
            if hmac.compare_digest(stored, key):
                return info
        return None

    def __len__(self) -> int:
        return len(self._keys)
