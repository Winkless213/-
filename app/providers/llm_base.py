from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMBase(ABC):
    @abstractmethod
    async def chat_sync(
        self, messages: list[dict], temperature: float = 0.2
    ) -> str:
        """Synchronous chat, returns complete response."""
        ...

    @abstractmethod
    async def chat_stream(
        self, messages: list[dict], temperature: float = 0.2
    ) -> AsyncIterator[str]:
        """Streaming chat, yields delta string fragments (not token boundaries)."""
        ...
