from app.config import settings
from app.sessions.base import SessionStore
from app.sessions.memory import MemorySessionStore


def make_session_store() -> SessionStore:
    if settings.redis_url:
        from app.sessions.redis_store import RedisSessionStore
        return RedisSessionStore(settings.redis_url)
    return MemorySessionStore()
