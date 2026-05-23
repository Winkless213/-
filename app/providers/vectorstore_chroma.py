import chromadb
from loguru import logger

from app.chunking.models import Chunk
from app.providers.vectorstore_base import VectorStoreBase


class ChromaVectorStore(VectorStoreBase):
    def __init__(self, persist_dir: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB initialized: persist_dir={}, collection={}",
            persist_dir,
            collection_name,
        )

    def add(
        self,
        document_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        metadatas = []
        documents = []
        for chunk in chunks:
            meta = {**chunk.metadata, "document_id": document_id}
            metadatas.append(meta)
            documents.append(chunk.text)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        logger.debug("Added chunks: document_id={}, count={}", document_id, len(chunks))

    def search(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

        items = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # cosine distance to similarity
                items.append({"text": doc, "metadata": meta, "score": round(score, 4)})

        return items

    def delete_by_document_id(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})
        logger.debug("Deleted chunks: document_id={}", document_id)

    def count(self) -> int:
        return self.collection.count()

    def list_documents(self) -> list[str]:
        all_meta = self.collection.get()["metadatas"]
        filenames = set()
        for meta in all_meta:
            if "filename" in meta:
                filenames.add(meta["filename"])
        return sorted(filenames)
