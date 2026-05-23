import hashlib
from unittest.mock import MagicMock, patch

import pytest

from app.chunking.models import Chunk
from app.services.ingest_service import IngestService


class TestIngestService:
    def _make_service(self):
        mock_embedding = MagicMock()
        mock_embedding.encode.return_value = [[0.1] * 384, [0.2] * 384]

        mock_vectorstore = MagicMock()
        mock_vectorstore.count.return_value = 0

        return IngestService(
            embedding_provider=mock_embedding,
            vectorstore_provider=mock_vectorstore,
            max_chunk_size=400,
            chunk_overlap=50,
        ), mock_embedding, mock_vectorstore

    def test_ingest_text_success(self):
        service, mock_emb, mock_vs = self._make_service()

        result = service.ingest_text("# Title\nHello world", filename="test.md")

        assert result["status"] == "ok"
        assert result["filename"] == "test.md"
        assert result["chunk_count"] >= 1
        assert len(result["document_id"]) == 64  # SHA256 hex

        mock_emb.encode.assert_called_once()
        mock_vs.add.assert_called_once()

    def test_ingest_text_empty_raises(self):
        service, _, _ = self._make_service()

        with pytest.raises(ValueError, match="empty_document"):
            service.ingest_text("", filename="empty.md")

    def test_ingest_text_whitespace_only_raises(self):
        service, _, _ = self._make_service()

        with pytest.raises(ValueError, match="empty_document"):
            service.ingest_text("   \n\n  ", filename="blank.md")

    def test_ingest_text_idempotent(self):
        """Same content should produce same document_id and trigger delete."""
        service, mock_emb, mock_vs = self._make_service()

        content = "# Title\nHello world"
        service.ingest_text(content, filename="test.md")

        # Second ingest of same content
        service.ingest_text(content, filename="test.md")

        # delete_by_document_id should be called on second ingest
        assert mock_vs.delete_by_document_id.call_count >= 1

    def test_ingest_file_success(self, tmp_path):
        service, mock_emb, mock_vs = self._make_service()

        test_file = tmp_path / "test.md"
        test_file.write_text("# Title\nContent here", encoding="utf-8")

        result = service.ingest_file(str(test_file))

        assert result["status"] == "ok"
        assert result["filename"] == "test.md"

    def test_ingest_file_not_found(self, tmp_path):
        service, _, _ = self._make_service()

        with pytest.raises(ValueError, match="file_not_found"):
            service.ingest_file(str(tmp_path / "nonexistent.md"))

    def test_ingest_file_unsupported_type(self, tmp_path):
        service, _, _ = self._make_service()

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf")

        with pytest.raises(ValueError, match="unsupported_file_type"):
            service.ingest_file(str(test_file))

    def test_compute_document_id_deterministic(self):
        service, _, _ = self._make_service()
        content = "hello world"
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert service.compute_document_id(content) == expected


from app.services.query_service import QueryService


class TestQueryService:
    def _make_service(self):
        mock_embedding = MagicMock()
        mock_embedding.encode_query.return_value = [0.1] * 384

        mock_vectorstore = MagicMock()
        mock_vectorstore.search.return_value = [
            {"text": "chunk1", "metadata": {"filename": "a.md"}, "score": 0.9},
            {"text": "chunk2", "metadata": {"filename": "b.md"}, "score": 0.8},
        ]

        mock_llm = MagicMock()

        async def fake_chat_sync(messages, temperature=0.2):
            return "test answer"

        mock_llm.chat_sync = fake_chat_sync

        return QueryService(
            llm_provider=mock_llm,
            embedding_provider=mock_embedding,
            vectorstore_provider=mock_vectorstore,
        ), mock_llm, mock_embedding, mock_vectorstore

    @pytest.mark.asyncio
    async def test_query_sync(self):
        service, mock_llm, mock_emb, mock_vs = self._make_service()

        result = await service.query_sync("What is X?")

        assert result["answer"] == "test answer"
        assert len(result["sources"]) == 2
        assert result["sources"][0]["filename"] == "a.md"

        mock_emb.encode_query.assert_called_once_with("What is X?")
        mock_vs.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_stream(self):
        service, mock_llm, mock_emb, mock_vs = self._make_service()

        async def fake_stream(*args, **kwargs):
            for token in ["Hello", " world"]:
                yield token

        mock_llm.chat_stream = fake_stream

        tokens = []
        sources = None
        async for event in service.query_stream("What is X?"):
            if "delta" in event:
                tokens.append(event["delta"])
            if event.get("done"):
                sources = event["sources"]

        assert tokens == ["Hello", " world"]
        assert sources is not None
        assert len(sources) == 2

    @pytest.mark.asyncio
    async def test_query_empty_vectorstore(self):
        service, _, mock_emb, mock_vs = self._make_service()
        mock_vs.search.return_value = []

        result = await service.query_sync("What is X?")

        assert "无法回答" in result["answer"] or "没有找到" in result["answer"]
        assert result["sources"] == []
