"""Integration tests: real ChromaDB + mock LLM/Embedding."""
from unittest.mock import MagicMock

import pytest

from app.chunking.splitter import split_document
from app.providers.vectorstore_chroma import ChromaVectorStore
from app.services.ingest_service import IngestService
from app.services.query_service import QueryService


@pytest.fixture
def chroma_store(tmp_path):
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_integration",
    )
    yield store


@pytest.fixture
def mock_embedding():
    mock = MagicMock()

    def fake_encode(texts):
        return [[float(i + 1)] * 3 for i in range(len(texts))]

    def fake_encode_query(text):
        return [1.0, 0.0, 0.0]

    mock.encode.side_effect = fake_encode
    mock.encode_query.side_effect = fake_encode_query
    return mock


@pytest.fixture
def mock_llm():
    mock = MagicMock()

    async def fake_chat_sync(messages, temperature=0.2):
        return "This is a test answer about the enterprise."

    async def fake_chat_stream(messages, temperature=0.2):
        for token in ["This ", "is ", "a ", "test ", "answer."]:
            yield token

    mock.chat_sync = fake_chat_sync
    mock.chat_stream = fake_chat_stream
    return mock


class TestIngestPipeline:
    def test_ingest_and_query(self, chroma_store, mock_embedding, mock_llm):
        ingest_svc = IngestService(
            embedding_provider=mock_embedding,
            vectorstore_provider=chroma_store,
        )

        result = ingest_svc.ingest_text(
            "# 采购流程\nSINOBA 的采购流程包括申请、审批、执行三个步骤。",
            filename="采购流程.md",
        )

        assert result["status"] == "ok"
        assert result["chunk_count"] >= 1
        assert chroma_store.count() >= 1

    def test_ingest_idempotent(self, chroma_store, mock_embedding):
        ingest_svc = IngestService(
            embedding_provider=mock_embedding,
            vectorstore_provider=chroma_store,
        )

        content = "# Test\nHello world"
        ingest_svc.ingest_text(content, filename="test.md")
        count_before = chroma_store.count()

        ingest_svc.ingest_text(content, filename="test.md")
        count_after = chroma_store.count()

        assert count_before == count_after

    def test_delete_and_reingest(self, chroma_store, mock_embedding):
        ingest_svc = IngestService(
            embedding_provider=mock_embedding,
            vectorstore_provider=chroma_store,
        )

        result1 = ingest_svc.ingest_text("# Old\nOld content", filename="test.md")
        doc_id = result1["document_id"]

        chroma_store.delete_by_document_id(doc_id)
        assert chroma_store.count() == 0

        result2 = ingest_svc.ingest_text("# New\nNew content", filename="test.md")
        assert result2["document_id"] != doc_id
        assert chroma_store.count() >= 1


class TestQueryPipeline:
    @pytest.mark.asyncio
    async def test_full_query_flow(self, chroma_store, mock_embedding, mock_llm):
        ingest_svc = IngestService(
            embedding_provider=mock_embedding,
            vectorstore_provider=chroma_store,
        )
        ingest_svc.ingest_text(
            "# 采购流程\nSINOBA 的采购流程包括申请、审批、执行。",
            filename="采购流程.md",
        )

        query_svc = QueryService(
            llm_provider=mock_llm,
            embedding_provider=mock_embedding,
            vectorstore_provider=chroma_store,
        )

        result = await query_svc.query_sync("采购流程是什么？")

        assert result["answer"] == "This is a test answer about the enterprise."
        assert len(result["sources"]) >= 1
        assert result["sources"][0]["filename"] == "采购流程.md"

    @pytest.mark.asyncio
    async def test_query_empty_store(self, chroma_store, mock_embedding, mock_llm):
        query_svc = QueryService(
            llm_provider=mock_llm,
            embedding_provider=mock_embedding,
            vectorstore_provider=chroma_store,
        )

        result = await query_svc.query_sync("What is X?")

        assert "无法回答" in result["answer"]
        assert result["sources"] == []


class TestChunkingIntegration:
    def test_split_sample_docs(self):
        text = """# 采购流程

## 申请
各部门填写采购申请单。

## 审批
5000 元以下部门经理审批。

## 执行
采购部执行采购。"""

        chunks = split_document(text, filename="采购流程.md")
        assert len(chunks) == 3
        assert all(c.metadata["filename"] == "采购流程.md" for c in chunks)
        headings = [c.metadata["heading"] for c in chunks]
        assert "申请" in headings
        assert "审批" in headings
        assert "执行" in headings
