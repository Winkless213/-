import pytest
from app.config import Settings


@pytest.fixture
def test_settings():
    return Settings(
        mimo_api_key="test_key",
        mimo_endpoint="https://test.example.com/v1",
        mimo_model="test-model",
        embedding_model="test-embedding",
        chroma_persist_dir="./test_chroma_db",
        chroma_collection_name="test_collection",
        sample_docs_dir="./test_sample_docs",
    )
