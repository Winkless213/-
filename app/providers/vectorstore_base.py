from abc import ABC, abstractmethod

from app.chunking.models import Chunk


class VectorStoreBase(ABC):
    @abstractmethod
    def add(
        self,
        document_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks with embeddings. document_id is injected into metadata."""
        ...

    @abstractmethod
    def search(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        """Search similar chunks. Returns: [{"text", "metadata", "score"}]."""
        ...

    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunks belonging to a document."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total chunk count."""
        ...

    @abstractmethod
    def list_documents(self) -> list[str]:
        """Return list of unique filenames in the store."""
        ...
