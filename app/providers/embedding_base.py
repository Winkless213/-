from abc import ABC, abstractmethod


class EmbeddingBase(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Batch encode texts for ingestion."""
        ...

    @abstractmethod
    def encode_query(self, text: str) -> list[float]:
        """Encode a single query text."""
        ...
