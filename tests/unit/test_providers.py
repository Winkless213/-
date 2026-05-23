from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.chunking.models import Chunk


class TestBGEEmbedding:
    @patch("app.providers.embedding_bge.SentenceTransformer")
    def test_encode_returns_correct_shape(self, mock_st_cls):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])
        mock_st_cls.return_value = mock_model

        from app.providers.embedding_bge import BGEEmbedding

        emb = BGEEmbedding(model_name="test-model")
        result = emb.encode(["hello", "world"])

        assert len(result) == 2
        assert len(result[0]) == 384
        mock_model.encode.assert_called_once_with(
            ["hello", "world"], normalize_embeddings=True
        )

    @patch("app.providers.embedding_bge.SentenceTransformer")
    def test_encode_query_returns_single_vector(self, mock_st_cls):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384])
        mock_st_cls.return_value = mock_model

        from app.providers.embedding_bge import BGEEmbedding

        emb = BGEEmbedding(model_name="test-model")
        result = emb.encode_query("hello")

        assert len(result) == 384
        mock_model.encode.assert_called_once_with(
            ["hello"], normalize_embeddings=True
        )


class TestChromaVectorStore:
    def test_add_and_search(self, tmp_path):
        from app.providers.vectorstore_chroma import ChromaVectorStore

        store = ChromaVectorStore(
            persist_dir=str(tmp_path), collection_name="test"
        )

        chunks = [
            Chunk(text="hello", metadata={"filename": "a.md", "heading": "H1", "position": 0}),
            Chunk(text="world", metadata={"filename": "a.md", "heading": "H1", "position": 1}),
        ]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        store.add("doc1", chunks, embeddings)
        assert store.count() == 2

        results = store.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "hello"
        assert "score" in results[0]

    def test_delete_by_document_id(self, tmp_path):
        from app.providers.vectorstore_chroma import ChromaVectorStore

        store = ChromaVectorStore(
            persist_dir=str(tmp_path), collection_name="test"
        )

        chunks = [Chunk(text="hi", metadata={"filename": "a.md", "heading": "", "position": 0})]
        store.add("doc1", chunks, [[1.0, 0.0]])
        assert store.count() == 1

        store.delete_by_document_id("doc1")
        assert store.count() == 0

    def test_list_documents(self, tmp_path):
        from app.providers.vectorstore_chroma import ChromaVectorStore

        store = ChromaVectorStore(
            persist_dir=str(tmp_path), collection_name="test"
        )

        chunks_a = [Chunk(text="a", metadata={"filename": "a.md", "heading": "", "position": 0})]
        chunks_b = [Chunk(text="b", metadata={"filename": "b.md", "heading": "", "position": 0})]
        store.add("doc1", chunks_a, [[1.0]])
        store.add("doc2", chunks_b, [[2.0]])

        docs = store.list_documents()
        assert sorted(docs) == ["a.md", "b.md"]

    def test_idempotent_add(self, tmp_path):
        """Re-adding same document_id should replace old chunks."""
        from app.providers.vectorstore_chroma import ChromaVectorStore

        store = ChromaVectorStore(
            persist_dir=str(tmp_path), collection_name="test"
        )

        chunks1 = [Chunk(text="old", metadata={"filename": "a.md", "heading": "", "position": 0})]
        store.add("doc1", chunks1, [[1.0]])
        assert store.count() == 1

        chunks2 = [Chunk(text="new", metadata={"filename": "a.md", "heading": "", "position": 0})]
        store.add("doc1", chunks2, [[2.0]])
        assert store.count() == 1

        results = store.search([2.0], top_k=1)
        assert results[0]["text"] == "new"


class TestMIMOLLM:
    @pytest.mark.asyncio
    @patch("app.providers.llm_mimo.openai.AsyncOpenAI")
    async def test_chat_sync_returns_text(self, mock_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello world"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_cls.return_value = mock_client

        from app.providers.llm_mimo import MIMOLLM

        llm = MIMOLLM(api_key="test", endpoint="https://test.com/v1", model="test")
        llm.client = mock_client

        result = await llm.chat_sync([{"role": "user", "content": "hi"}])
        assert result == "Hello world"

    @pytest.mark.asyncio
    @patch("app.providers.llm_mimo.openai.AsyncOpenAI")
    async def test_chat_stream_yields_deltas(self, mock_cls):
        async def fake_stream():
            chunks = [
                MagicMock(choices=[MagicMock(delta=MagicMock(content="Hel"))], usage=None),
                MagicMock(choices=[MagicMock(delta=MagicMock(content="lo"))], usage=None),
                MagicMock(choices=[MagicMock(delta=MagicMock(content=None))], usage=None),
            ]
            for c in chunks:
                yield c

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=fake_stream())
        mock_cls.return_value = mock_client

        from app.providers.llm_mimo import MIMOLLM

        llm = MIMOLLM(api_key="test", endpoint="https://test.com/v1", model="test")
        llm.client = mock_client

        tokens = []
        async for delta in llm.chat_stream([{"role": "user", "content": "hi"}]):
            tokens.append(delta)

        assert tokens == ["Hel", "lo"]
