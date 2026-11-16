"""FastAPI dependency: requires X-API-Key or Authorization: Bearer <key>."""
from typing import Optional, Dict, Any

from fastapi import HTTPException, Header

from app.auth.keys import APIKeyStore

_store = APIKeyStore()


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
