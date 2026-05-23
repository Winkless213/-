import hashlib
import time
from pathlib import Path

from loguru import logger

from app.chunking.splitter import split_document
from app.providers.embedding_base import EmbeddingBase
from app.providers.vectorstore_base import VectorStoreBase

SUPPORTED_EXTENSIONS = {".md", ".txt"}


class IngestService:
    def __init__(
        self,
        embedding_provider: EmbeddingBase,
        vectorstore_provider: VectorStoreBase,
        max_chunk_size: int = 400,
        chunk_overlap: int = 50,
    ):
        self.embedding = embedding_provider
        self.vectorstore = vectorstore_provider
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def compute_document_id(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def ingest_text(self, content: str, filename: str) -> dict:
        start = time.time()
        content = content.strip()

        if not content:
            raise ValueError("empty_document")

        document_id = self.compute_document_id(content)
        logger.info("Ingesting document: filename={}, document_id={}", filename, document_id)

        # Idempotent: delete existing chunks for this document
        self.vectorstore.delete_by_document_id(document_id)

        # Chunk
        chunks = split_document(
            content, filename, self.max_chunk_size, self.chunk_overlap
        )
        if not chunks:
            raise ValueError("empty_document")

        logger.debug("Split into {} chunks", len(chunks))

        # Embed
        texts = [c.text for c in chunks]
        embeddings = self.embedding.encode(texts)
        logger.debug("Generated {} embeddings", len(embeddings))

        # Store
        self.vectorstore.add(document_id, chunks, embeddings)

        latency_ms = (time.time() - start) * 1000
        logger.info(
            "Ingest completed: filename={}, chunk_count={}, latency_ms={}",
            filename,
            len(chunks),
            round(latency_ms, 1),
        )

        return {
            "status": "ok",
            "document_id": document_id,
            "filename": filename,
            "chunk_count": len(chunks),
        }

    def ingest_file(self, file_path: str) -> dict:
        path = Path(file_path)

        if not path.exists():
            raise ValueError("file_not_found")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("unsupported_file_type")

        content = path.read_text(encoding="utf-8")
        return self.ingest_text(content, filename=path.name)
