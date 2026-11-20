# Copyright (c) 2026 Lewis James Seals Jr.
# Author: Lewis James Seals Jr. (@ljrseals-dev)
# Project: Pinecone Neural Agent Harness
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class SessionStore(ABC):
    @abstractmethod
    async def get(self, session_id: str) -> Optional[List[Dict[str, Any]]]: ...

    @abstractmethod
    async def set(self, session_id: str, history: List[Dict[str, Any]],
                  ttl: int = 86400) -> None: ...

    @abstractmethod
    async def delete(self, session_id: str) -> None: ...

    @abstractmethod
    async def exists(self, session_id: str) -> bool: ...
