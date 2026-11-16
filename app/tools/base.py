from abc import ABC, abstractmethod
from typing import Any, Dict


class Tool(ABC):
    name: str
    description: str
    parameters: Dict[str, Any]

    @abstractmethod
    async def run(self, **kwargs: Any) -> str: ...

    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
