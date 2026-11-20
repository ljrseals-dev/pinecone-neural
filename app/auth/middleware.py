# Copyright (c) 2026 Lewis James Seals Jr.
# Author: Lewis James Seals Jr. (@ljrseals-dev)
# Project: Pinecone Neural Agent Harness
# SPDX-License-Identifier: MIT

"""FastAPI dependency: requires a configured X-API-Key or Bearer token."""
from typing import Optional, Dict, Any

from fastapi import HTTPException, Header

from app.auth.keys import APIKeyStore
from app.config import settings

# Pydantic Settings reads .env without mutating os.environ; pass the validated
# setting explicitly so API_KEYS from .env is not silently ignored.
_store = APIKeyStore(settings.api_keys)


async def require_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> Dict[str, Any]:
    key = x_api_key
    if not key and authorization and authorization.startswith("Bearer "):
        key = authorization[7:]
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key")
    info = _store.validate(key)
    if not info:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return info
