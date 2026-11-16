import os
from app.sessions.base import SessionStore
from app.sessions.memory import MemorySessionStore


def make_session_store() -> SessionStore:
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        from app.sessions.redis_store import RedisSessionStore
        return RedisSessionStore(redis_url)
    return MemorySessionStore()
