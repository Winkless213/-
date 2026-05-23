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
