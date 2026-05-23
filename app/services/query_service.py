import time

from loguru import logger

from app.prompts.templates import build_rag_messages
from app.providers.embedding_base import EmbeddingBase
from app.providers.llm_base import LLMBase
from app.providers.vectorstore_base import VectorStoreBase


class QueryService:
    def __init__(
        self,
        llm_provider: LLMBase,
        embedding_provider: EmbeddingBase,
        vectorstore_provider: VectorStoreBase,
    ):
        self.llm = llm_provider
        self.embedding = embedding_provider
        self.vectorstore = vectorstore_provider

    def _retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.embedding.encode_query(question)
        return self.vectorstore.search(query_embedding, top_k=top_k)

    def _build_sources(self, chunks: list[dict]) -> list[dict]:
        return [
            {
                "document_id": c.get("metadata", {}).get("document_id", ""),
                "filename": c.get("metadata", {}).get("filename", ""),
                "chunk_text": c["text"],
                "score": c.get("score", 0.0),
            }
            for c in chunks
        ]

    async def query_sync(self, question: str, top_k: int = 5) -> dict:
        start = time.time()

        chunks = self._retrieve(question, top_k)
        logger.info("Retrieved {} chunks", len(chunks))

        if not chunks:
            return {
                "answer": "根据现有资料，我无法回答这个问题。",
                "sources": [],
            }

        messages = build_rag_messages(question, chunks)
        answer = await self.llm.chat_sync(messages)

        latency_ms = (time.time() - start) * 1000
        logger.info("Query completed: latency_ms={}", round(latency_ms, 1))

        return {
            "answer": answer,
            "sources": self._build_sources(chunks),
        }

    async def query_stream(self, question: str, top_k: int = 5):
        start = time.time()

        chunks = self._retrieve(question, top_k)
        logger.info("Retrieved {} chunks for stream", len(chunks))

        if not chunks:
            yield {"delta": "根据现有资料，我无法回答这个问题。"}
            yield {"done": True, "sources": []}
            return

        messages = build_rag_messages(question, chunks)
        sources = self._build_sources(chunks)

        async for delta in self.llm.chat_stream(messages):
            yield {"delta": delta}

        latency_ms = (time.time() - start) * 1000
        logger.info("Query stream completed: latency_ms={}", round(latency_ms, 1))

        yield {"done": True, "sources": sources}
