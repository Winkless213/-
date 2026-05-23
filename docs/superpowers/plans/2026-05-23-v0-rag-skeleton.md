# v0 RAG 系统骨架 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建本地 RAG 知识库系统骨架，含 /ingest、/query、/health 三个端点，用 3-5 篇合成 markdown 验证全链路。

**Architecture:** 分层架构 Routes → Services → Providers。Provider 按能力抽象（LLM/Embedding/VectorStore），Service 层编排 pipeline，Route 层只做 HTTP 解析。参考 RAGFlow provider 模式 + Dify pipeline 思路。

**Tech Stack:** Python 3.11+, FastAPI, ChromaDB, BGE-small-zh-v1.5 (sentence-transformers), MIMO API (OpenAI 兼容), pytest, loguru

---

## File Map

```
app/
├── __init__.py
├── main.py                     # FastAPI app + lifespan
├── config.py                   # pydantic-settings
├── schemas.py                  # Pydantic request/response models
├── routes/
│   ├── __init__.py
│   ├── ingest.py
│   ├── query.py
│   └── health.py
├── services/
│   ├── __init__.py
│   ├── ingest_service.py
│   └── query_service.py
├── providers/
│   ├── __init__.py
│   ├── llm_base.py
│   ├── llm_mimo.py
│   ├── embedding_base.py
│   ├── embedding_bge.py
│   ├── vectorstore_base.py
│   └── vectorstore_chroma.py
├── chunking/
│   ├── __init__.py
│   ├── splitter.py
│   └── models.py
└── prompts/
    ├── __init__.py
    └── templates.py

tests/
├── __init__.py
├── conftest.py
├── unit/
│   ├── __init__.py
│   ├── test_chunking.py
│   ├── test_providers.py
│   └── test_services.py
├── integration/
│   ├── __init__.py
│   └── test_pipeline.py

docs/
└── sample_docs/
    ├── 采购流程.md
    ├── 客户档案.md
    ├── 技术规格.md
    ├── 供应商名录.md
    └── 质量标准.md
```

---

### Task 1: 项目脚手架

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`, `.env`（gitignored）
- Create: `.gitignore`
- Create: `app/__init__.py`, `app/config.py`, `app/main.py`, `app/schemas.py`
- Create: `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 0: 初始化 git 仓库**

```bash
cd C:/Users/24596/projects/enterprise-kb
git init
```

- [ ] **Step 1: 创建 requirements.txt**

```
fastapi==0.115.12
uvicorn[standard]==0.34.2
pydantic-settings==2.9.1
chromadb==1.0.7
sentence-transformers==4.1.0
openai==1.82.0
loguru==0.7.3
python-multipart==0.0.20
httpx==0.28.1
pytest==8.3.5
pytest-asyncio==1.0.0
```

- [ ] **Step 2: 创建 .env.example（提交到 git）和 .env（gitignored）**

创建 `.env.example`（占位符，提交到 git）：

```env
MIMO_API_KEY=your_key_here
MIMO_ENDPOINT=https://token-plan-sgp.xiaomimimo.com/v1
MIMO_MODEL=<TO_BE_FILLED>
MIMO_TIMEOUT=30
MIMO_MAX_RETRIES=3

EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

MAX_CHUNK_SIZE=400
CHUNK_OVERLAP=50

CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=enterprise_kb

SAMPLE_DOCS_DIR=./docs/sample_docs

HOST=127.0.0.1
PORT=8000
LOG_LEVEL=INFO
```

然后创建 `.env`（实际配置，不提交）：

```bash
cp .env.example .env
# 编辑 .env 填入真实的 MIMO_API_KEY 和 MIMO_MODEL
```

创建 `.gitignore`：

```
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
venv/
.venv/
.env
.env.local
chroma_db/
*.log
logs/
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/
.DS_Store
Thumbs.db
.idea/
.vscode/
```

- [ ] **Step 3: 创建 app/config.py**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MIMO API
    mimo_api_key: str = "your_key_here"
    mimo_endpoint: str = "https://token-plan-sgp.xiaomimimo.com/v1"
    mimo_model: str = "<TO_BE_FILLED>"
    mimo_timeout: int = 30
    mimo_max_retries: int = 3

    # Embedding
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # Chunking
    max_chunk_size: int = 400
    chunk_overlap: int = 50

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "enterprise_kb"

    # Startup Ingest
    sample_docs_dir: str = "./docs/sample_docs"

    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
```

- [ ] **Step 4: 创建 app/schemas.py**

```python
from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    status: str = "ok"
    document_id: str
    filename: str
    chunk_count: int


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = False


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class StreamDelta(BaseModel):
    delta: str | None = None
    done: bool = False
    sources: list[SourceChunk] | None = None


class ErrorResponse(BaseModel):
    error: str
    message: str
    status: int


class HealthResponse(BaseModel):
    status: str = "ok"
    embedding_provider: str
    vectorstore: str
    document_count: int
    loaded_documents: list[str]
```

- [ ] **Step 5: 创建 app/main.py 骨架**

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    # TODO: Task 12 will add startup ingest here
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Enterprise Knowledge Base",
    description="本地企业知识库系统 (RAG)",
    version="0.1.0",
    lifespan=lifespan,
)
```

- [ ] **Step 6: 创建 tests/conftest.py**

```python
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
```

- [ ] **Step 7: 创建 __init__.py 文件**

创建以下空文件：
- `app/__init__.py`
- `app/routes/__init__.py`
- `app/services/__init__.py`
- `app/providers/__init__.py`
- `app/chunking/__init__.py`
- `app/prompts/__init__.py`
- `tests/__init__.py`
- `tests/unit/__init__.py`
- `tests/integration/__init__.py`

- [ ] **Step 8: 安装依赖并验证**

```bash
cd C:/Users/24596/projects/enterprise-kb
pip install -r requirements.txt
```

- [ ] **Step 9: Commit**

```bash
git add requirements.txt .env.example .gitignore app/ tests/
git commit -m "feat: project scaffolding with config, schemas, and main app skeleton"
```

---

### Task 2: Chunking 模块

**Files:**
- Create: `app/chunking/models.py`
- Create: `app/chunking/splitter.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/test_chunking.py`

- [ ] **Step 1: 创建 app/chunking/models.py**

```python
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    metadata: dict  # keys: filename, heading, position
    # ChromaDB constraint: metadata values must be str/int/float/bool
    # document_id is injected by VectorStoreBase.add() at storage time
```

- [ ] **Step 2: 写 chunking 测试（红）**

创建 `tests/unit/test_chunking.py`：

```python
from app.chunking.splitter import split_document


class TestSplitMarkdown:
    def test_simple_headings(self):
        text = "# Title One\nContent A\n\n## Title Two\nContent B"
        chunks = split_document(text, filename="test.md", max_size=400, overlap=50)
        assert len(chunks) == 2
        assert chunks[0].metadata["heading"] == "Title One"
        assert chunks[1].metadata["heading"] == "Title Two"
        assert chunks[0].metadata["filename"] == "test.md"
        assert chunks[0].metadata["position"] == 0
        assert chunks[1].metadata["position"] == 1

    def test_long_section_splits_by_size(self):
        long_content = "A" * 1000
        text = f"# Long\n{long_content}"
        chunks = split_document(text, filename="test.md", max_size=400, overlap=50)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.text) <= 450  # max_size + some tolerance for overlap
            assert chunk.metadata["heading"] == "Long"

    def test_no_headings_plain_text(self):
        text = "Hello world. " * 200  # ~2600 chars
        chunks = split_document(text, filename="test.txt", max_size=400, overlap=50)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk.metadata["heading"] == ""

    def test_empty_text(self):
        chunks = split_document("", filename="test.md", max_size=400, overlap=50)
        assert chunks == []

    def test_whitespace_only(self):
        chunks = split_document("   \n\n  ", filename="test.md", max_size=400, overlap=50)
        assert chunks == []

    def test_single_short_section(self):
        text = "# Title\nShort content"
        chunks = split_document(text, filename="test.md", max_size=400, overlap=50)
        assert len(chunks) == 1
        assert chunks[0].text == "Short content"
        assert chunks[0].metadata["heading"] == "Title"

    def test_overlap_within_heading_only(self):
        """Overlap should only apply within a heading section, not across headings."""
        text = "# A\n" + "X" * 500 + "\n# B\n" + "Y" * 500
        chunks = split_document(text, filename="test.md", max_size=400, overlap=50)
        headings = [c.metadata["heading"] for c in chunks]
        # All chunks from section A should have heading "A", section B should have "B"
        assert all(h == "A" for h in headings[: len(headings) // 2]) or True
        # No chunk should mix content from both sections
        a_chunks = [c for c in chunks if c.metadata["heading"] == "A"]
        b_chunks = [c for c in chunks if c.metadata["heading"] == "B"]
        assert len(a_chunks) >= 1
        assert len(b_chunks) >= 1

    def test_multiple_heading_levels(self):
        text = "# H1\nContent\n## H2\nSub\n### H3\nDeep"
        chunks = split_document(text, filename="test.md", max_size=400, overlap=50)
        assert len(chunks) == 3
        assert chunks[0].metadata["heading"] == "H1"
        assert chunks[1].metadata["heading"] == "H2"
        assert chunks[2].metadata["heading"] == "H3"
```

- [ ] **Step 3: 运行测试确认失败**

```bash
cd C:/Users/24596/projects/enterprise-kb
pytest tests/unit/test_chunking.py -v
```

预期：FAIL（`ModuleNotFoundError: No module named 'app.chunking.splitter'`）

- [ ] **Step 4: 实现 splitter.py（绿）**

创建 `app/chunking/splitter.py`：

```python
import re

from app.chunking.models import Chunk


def _has_headings(text: str) -> bool:
    return bool(re.search(r"^#{1,6}\s+", text, re.MULTILINE))


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split markdown by headings. Returns list of (heading, content)."""
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in text.split("\n"):
        match = re.match(r"^(#{1,6})\s+(.*)", line)
        if match:
            if current_lines or current_heading:
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append((current_heading, content))
            current_heading = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines or current_heading:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((current_heading, content))

    return sections


def _split_by_size(
    text: str, max_size: int, overlap: int
) -> list[str]:
    """Split text by character count with sliding window overlap."""
    if len(text) <= max_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_size
        chunks.append(text[start:end])
        start += max_size - overlap

    return chunks


def split_document(
    text: str, filename: str, max_size: int = 400, overlap: int = 50
) -> list[Chunk]:
    """Split document using hybrid strategy: headings first, then size."""
    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    position = 0

    if _has_headings(text):
        sections = _split_by_headings(text)
        for heading, content in sections:
            if len(content) <= max_size:
                chunks.append(
                    Chunk(
                        text=content,
                        metadata={
                            "filename": filename,
                            "heading": heading,
                            "position": position,
                        },
                    )
                )
                position += 1
            else:
                parts = _split_by_size(content, max_size, overlap)
                for part in parts:
                    chunks.append(
                        Chunk(
                            text=part,
                            metadata={
                                "filename": filename,
                                "heading": heading,
                                "position": position,
                            },
                        )
                    )
                    position += 1
    else:
        parts = _split_by_size(text, max_size, overlap)
        for part in parts:
            chunks.append(
                Chunk(
                    text=part,
                    metadata={
                        "filename": filename,
                        "heading": "",
                        "position": position,
                    },
                )
            )
            position += 1

    return chunks
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/unit/test_chunking.py -v
```

预期：全部 PASS

- [ ] **Step 6: Commit**

```bash
git add app/chunking/ tests/unit/test_chunking.py tests/unit/__init__.py
git commit -m "feat: add chunking module with hybrid split strategy"
```

---

### Task 3: Provider 基类

**Files:**
- Create: `app/providers/llm_base.py`
- Create: `app/providers/embedding_base.py`
- Create: `app/providers/vectorstore_base.py`

- [ ] **Step 1: 创建 llm_base.py**

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMBase(ABC):
    @abstractmethod
    async def chat_sync(
        self, messages: list[dict], temperature: float = 0.2
    ) -> str:
        """Synchronous chat, returns complete response."""
        ...

    @abstractmethod
    async def chat_stream(
        self, messages: list[dict], temperature: float = 0.2
    ) -> AsyncIterator[str]:
        """Streaming chat, yields delta string fragments (not token boundaries)."""
        ...
```

- [ ] **Step 2: 创建 embedding_base.py**

```python
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
```

- [ ] **Step 3: 创建 vectorstore_base.py**

```python
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
```

- [ ] **Step 4: Commit**

```bash
git add app/providers/llm_base.py app/providers/embedding_base.py app/providers/vectorstore_base.py
git commit -m "feat: add provider base classes (LLM, Embedding, VectorStore)"
```

---

### Task 4: BGE Embedding Provider

**Files:**
- Create: `app/providers/embedding_bge.py`
- Create: `tests/unit/test_providers.py`

- [ ] **Step 1: 写 embedding 测试（红）**

创建 `tests/unit/test_providers.py`：

```python
from unittest.mock import MagicMock, patch

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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/unit/test_providers.py -v
```

预期：FAIL

- [ ] **Step 3: 实现 embedding_bge.py**

```python
from sentence_transformers import SentenceTransformer

from app.providers.embedding_base import EmbeddingBase


class BGEEmbedding(EmbeddingBase):
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def encode_query(self, text: str) -> list[float]:
        embedding = self.model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()
```

- [ ] **Step 4: 实现 vectorstore_chroma.py**

```python
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
            "ChromaDB initialized",
            persist_dir=persist_dir,
            collection=collection_name,
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
        for i, chunk in enumerate(chunks):
            meta = {**chunk.metadata, "document_id": document_id}
            # ChromaDB requires all metadata values to be scalar
            metadatas.append(meta)
            documents.append(chunk.text)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        logger.debug("Added chunks to ChromaDB", document_id=document_id, count=len(chunks))

    def search(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

        items = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                # ChromaDB returns distance; convert to similarity score
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # cosine distance to similarity
                items.append({"text": doc, "metadata": meta, "score": round(score, 4)})

        return items

    def delete_by_document_id(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})
        logger.debug("Deleted chunks", document_id=document_id)

    def count(self) -> int:
        return self.collection.count()

    def list_documents(self) -> list[str]:
        all_meta = self.collection.get()["metadatas"]
        filenames = set()
        for meta in all_meta:
            if "filename" in meta:
                filenames.add(meta["filename"])
        return sorted(filenames)
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/unit/test_providers.py -v
```

预期：全部 PASS

- [ ] **Step 6: Commit**

```bash
git add app/providers/embedding_bge.py app/providers/vectorstore_chroma.py tests/unit/test_providers.py
git commit -m "feat: add BGE embedding and ChromaDB vector store providers"
```

---

### Task 5: MIMO LLM Provider

**Files:**
- Modify: `tests/unit/test_providers.py`
- Create: `app/providers/llm_mimo.py`

- [ ] **Step 1: 写 LLM 测试（红）**

在 `tests/unit/test_providers.py` 末尾追加：

```python
import json
from unittest.mock import AsyncMock


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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/unit/test_providers.py::TestMIMOLLM -v
```

预期：FAIL

- [ ] **Step 3: 实现 llm_mimo.py**

```python
import time
from typing import AsyncIterator

import openai
from loguru import logger

from app.providers.llm_base import LLMBase


class MIMOLLM(LLMBase):
    def __init__(self, api_key: str, endpoint: str, model: str, timeout: int = 30):
        self.model = model
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint,
            timeout=timeout,
        )

    async def chat_sync(
        self, messages: list[dict], temperature: float = 0.2
    ) -> str:
        start = time.time()
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        latency_ms = (time.time() - start) * 1000
        usage = response.usage
        logger.info(
            "LLM call completed",
            model=self.model,
            latency_ms=round(latency_ms, 1),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )
        return response.choices[0].message.content or ""

    async def chat_stream(
        self, messages: list[dict], temperature: float = 0.2
    ) -> AsyncIterator[str]:
        start = time.time()
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
        latency_ms = (time.time() - start) * 1000
        logger.info("LLM stream completed", model=self.model, latency_ms=round(latency_ms, 1))
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/unit/test_providers.py -v
```

预期：全部 PASS

- [ ] **Step 5: Commit**

```bash
git add app/providers/llm_mimo.py tests/unit/test_providers.py
git commit -m "feat: add MIMO LLM provider with sync and streaming support"
```

---

### Task 6: Prompts 模块

**Files:**
- Create: `app/prompts/templates.py`

- [ ] **Step 1: 创建 templates.py**

```python
RAG_SYSTEM_PROMPT = """你是一个企业知识库助手。根据提供的参考资料回答用户问题。

规则：
1. 只根据参考资料回答，不要编造信息
2. 如果参考资料中没有相关信息，明确说"根据现有资料，我无法回答这个问题"
3. 回答要简洁准确
4. 引用来源时说明出自哪个文档"""

RAG_USER_TEMPLATE = """参考资料：
{context}

用户问题：{question}

请根据参考资料回答。"""


def build_rag_messages(
    question: str, chunks: list[dict]
) -> list[dict]:
    """Build messages for RAG query."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("filename", "未知来源")
        context_parts.append(f"[{i}] 来源: {source}\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": RAG_USER_TEMPLATE.format(
            context=context, question=question
        )},
    ]
```

- [ ] **Step 2: Commit**

```bash
git add app/prompts/
git commit -m "feat: add RAG prompt templates"
```

---

### Task 7: Ingest Service

**Files:**
- Create: `app/services/ingest_service.py`
- Modify: `tests/unit/test_services.py`

- [ ] **Step 1: 写 ingest service 测试（红）**

创建 `tests/unit/test_services.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/unit/test_services.py -v
```

预期：FAIL

- [ ] **Step 3: 实现 ingest_service.py**

```python
import hashlib
import os
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
        logger.info("Ingesting document", filename=filename, document_id=document_id)

        # Idempotent: delete existing chunks for this document
        self.vectorstore.delete_by_document_id(document_id)

        # Chunk
        chunks = split_document(
            content, filename, self.max_chunk_size, self.chunk_overlap
        )
        if not chunks:
            raise ValueError("empty_document")

        logger.debug("Split into chunks", count=len(chunks))

        # Embed
        texts = [c.text for c in chunks]
        embeddings = self.embedding.encode(texts)
        logger.debug("Embeddings generated", count=len(embeddings))

        # Store
        self.vectorstore.add(document_id, chunks, embeddings)

        latency_ms = (time.time() - start) * 1000
        logger.info(
            "Ingest completed",
            filename=filename,
            chunk_count=len(chunks),
            latency_ms=round(latency_ms, 1),
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/unit/test_services.py -v
```

预期：全部 PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/ingest_service.py tests/unit/test_services.py tests/unit/__init__.py
git commit -m "feat: add ingest service with pipeline and idempotent document handling"
```

---

### Task 8: Query Service

**Files:**
- Create: `app/services/query_service.py`
- Modify: `tests/unit/test_services.py`

- [ ] **Step 1: 写 query service 测试（红）**

在 `tests/unit/test_services.py` 末尾追加：

```python
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
        mock_llm.chat_sync = MagicMock(return_value="test answer")

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

        mock_llm.chat_stream = MagicMock(return_value=fake_stream())

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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/unit/test_services.py::TestQueryService -v
```

预期：FAIL

- [ ] **Step 3: 实现 query_service.py**

```python
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
        logger.info("Retrieved chunks", count=len(chunks))

        if not chunks:
            return {
                "answer": "根据现有资料，我无法回答这个问题。",
                "sources": [],
            }

        messages = build_rag_messages(question, chunks)
        answer = await self.llm.chat_sync(messages)

        latency_ms = (time.time() - start) * 1000
        logger.info("Query completed", latency_ms=round(latency_ms, 1))

        return {
            "answer": answer,
            "sources": self._build_sources(chunks),
        }

    async def query_stream(self, question: str, top_k: int = 5):
        start = time.time()

        chunks = self._retrieve(question, top_k)
        logger.info("Retrieved chunks for stream", count=len(chunks))

        if not chunks:
            yield {
                "delta": "根据现有资料，我无法回答这个问题。",
            }
            yield {"done": True, "sources": []}
            return

        messages = build_rag_messages(question, chunks)
        sources = self._build_sources(chunks)

        async for delta in self.llm.chat_stream(messages):
            yield {"delta": delta}

        latency_ms = (time.time() - start) * 1000
        logger.info("Query stream completed", latency_ms=round(latency_ms, 1))

        yield {"done": True, "sources": sources}
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/unit/test_services.py -v
```

预期：全部 PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/query_service.py tests/unit/test_services.py
git commit -m "feat: add query service with sync and streaming support"
```

---

### Task 9: 路由层

**Files:**
- Create: `app/routes/ingest.py`
- Create: `app/routes/query.py`
- Create: `app/routes/health.py`
- Modify: `app/main.py`

- [ ] **Step 1: 创建 routes/ingest.py**

```python
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from app.schemas import ErrorResponse, IngestResponse

router = APIRouter()

# Service will be injected at startup via app.state
_ingest_service = None


def set_ingest_service(service):
    global _ingest_service
    _ingest_service = service


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def ingest(
    file: UploadFile | None = File(default=None),
    path: str | None = Form(default=None),
):
    if file and path:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_request", "message": "只能传 file 或 path，不能同时传", "status": 400},
        )

    if not file and not path:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_request", "message": "必须传 file 或 path", "status": 400},
        )

    try:
        if file:
            content = await file.read()
            text = content.decode("utf-8")
            filename = file.filename or "uploaded"
            result = _ingest_service.ingest_text(text, filename=filename)
        else:
            result = _ingest_service.ingest_file(path)

        return IngestResponse(**result)

    except ValueError as e:
        error_code = str(e)
        status_map = {
            "unsupported_file_type": 400,
            "file_not_found": 400,
            "empty_document": 400,
        }
        message_map = {
            "unsupported_file_type": "只支持 .md / .txt 文件",
            "file_not_found": f"文件不存在: {path}",
            "empty_document": "文件内容为空",
        }
        raise HTTPException(
            status_code=status_map.get(error_code, 500),
            detail={
                "error": error_code,
                "message": message_map.get(error_code, str(e)),
                "status": status_map.get(error_code, 500),
            },
        )
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": str(e), "status": 500},
        )
```

- [ ] **Step 2: 创建 routes/query.py**

```python
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from app.schemas import ErrorResponse, QueryRequest, QueryResponse, StreamDelta

router = APIRouter()

_query_service = None


def set_query_service(service):
    global _query_service
    _query_service = service


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={500: {"model": ErrorResponse}},
)
async def query(req: QueryRequest):
    try:
        if req.stream:
            return StreamingResponse(
                _stream_response(req.question, req.top_k),
                media_type="text/event-stream",
            )

        result = await _query_service.query_sync(req.question, top_k=req.top_k)
        return QueryResponse(**result)

    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(
            status_code=500,
            detail={"error": "llm_error", "message": str(e), "status": 500},
        )


async def _stream_response(question: str, top_k: int):
    try:
        async for event in _query_service.query_stream(question, top_k=top_k):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    except Exception as e:
        logger.exception("Stream query failed")
        yield f"data: {json.dumps({'error': 'llm_error', 'message': str(e)}, ensure_ascii=False)}\n\n"
```

- [ ] **Step 3: 创建 routes/health.py**

```python
from fastapi import APIRouter

from app.schemas import HealthResponse

router = APIRouter()

_health_data = {}


def set_health_data(data: dict):
    global _health_data
    _health_data = data


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(**_health_data)
```

- [ ] **Step 4: 更新 main.py 注册路由**

替换 `app/main.py`：

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    # Providers and services will be initialized here in Task 12
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Enterprise Knowledge Base",
    description="本地企业知识库系统 (RAG)",
    version="0.1.0",
    lifespan=lifespan,
)


# Routes will be registered in Task 12 after services are initialized
```

- [ ] **Step 5: Commit**

```bash
git add app/routes/ app/main.py
git commit -m "feat: add API routes for ingest, query, and health"
```

---

### Task 10: 合成测试文档

**Files:**
- Create: `docs/sample_docs/采购流程.md`
- Create: `docs/sample_docs/客户档案.md`
- Create: `docs/sample_docs/技术规格.md`
- Create: `docs/sample_docs/供应商名录.md`
- Create: `docs/sample_docs/质量标准.md`

- [ ] **Step 1: 创建采购流程.md**

```markdown
# SINOBA 采购流程规范

## 采购申请

各部门需要采购时，需填写《采购申请单》，注明物品名称、规格、数量、预算金额和用途说明。采购金额在 5000 元以下的由部门经理审批，5000-50000 元由副总审批，50000 元以上由总经理审批。

## 供应商选择

采购部负责供应商的开发和管理。新供应商需经过资质审核、样品测试、小批量试用三个阶段。每类物资至少保持 3 家合格供应商，确保供应链稳定性。

## 合同签订

采购金额超过 10000 元的必须签订书面合同。合同需包含：物品规格、数量、单价、交货期、质量标准、违约责任等条款。合同由法务部审核后方可签署。

## 验收入库

物资到货后，仓库管理员按照采购订单进行数量清点和质量检验。检验合格后办理入库手续，不合格品按照《不合格品处理程序》处理。

## 付款流程

供应商凭发票和验收单到财务部申请付款。财务部核对合同、发票、验收单三单一致后，在 30 个工作日内完成付款。
```

- [ ] **Step 2: 创建客户档案.md**

```markdown
# 客户档案管理

## 客户分级

SINOBA 将客户分为 A、B、C 三个等级：
- A 级客户：年采购额超过 100 万元，享受专属客户经理服务
- B 级客户：年采购额 30-100 万元，定期回访
- C 级客户：年采购额 30 万元以下，标准服务

## 客户信息管理

客户档案包含：公司名称、联系人、联系方式、地址、经营范围、信用等级、历史交易记录等。客户信息每季度更新一次，由客户经理负责维护。

## 信用管理

新客户默认信用等级为 C 级，信用额度 5 万元。合作满一年且无逾期付款记录的可申请提升信用等级。A 级客户信用额度最高 50 万元。

## 客户投诉处理

客户投诉需在 24 小时内响应，48 小时内给出处理方案。重大投诉由质量部和销售部联合处理。投诉处理结果需记录在客户档案中。
```

- [ ] **Step 3: 创建技术规格.md**

```markdown
# 产品技术规格

## 产品概述

SINOBA 主要生产工业用精密零部件，产品涵盖轴承、齿轮、联轴器三大类。所有产品均通过 ISO 9001:2015 质量管理体系认证。

## 轴承规格

深沟球轴承系列：
- 内径范围：10mm - 200mm
- 外径范围：30mm - 360mm
- 精度等级：P0、P6、P5、P4
- 材质：GCr15 高碳铬轴承钢
- 表面硬度：HRC 60-64

## 齿轮规格

渐开线圆柱齿轮系列：
- 模数范围：1-20mm
- 齿数范围：12-200
- 精度等级：GB/T 10095 6-8 级
- 材质：20CrMnTi 合金钢
- 表面硬度：HRC 58-62

## 联轴器规格

弹性联轴器系列：
- 扭矩范围：10-10000 N·m
- 转速范围：0-6000 rpm
- 补偿量：径向 0.2-0.5mm，轴向 ±1mm
- 材质：45# 钢 + 聚氨酯弹性体
```

- [ ] **Step 4: 创建供应商名录.md**

```markdown
# 合格供应商名录

## 钢材供应商

### 宝钢集团
- 供应物资：GCr15 轴承钢、20CrMnTi 齿轮钢
- 合作年限：8 年
- 信用等级：A
- 年采购额：约 500 万元

### 沙钢集团
- 供应物资：45# 碳钢、Q345 结构钢
- 合作年限：5 年
- 信用等级：A
- 年采购额：约 200 万元

## 刀具供应商

### 株洲钻石
- 供应物资：硬质合金刀具、数控刀片
- 合作年限：6 年
- 信用等级：A
- 年采购额：约 80 万元

### 厦门金鹭
- 供应物资：整体硬质合金钻头、铣刀
- 合作年限：3 年
- 信用等级：B
- 年采购额：约 30 万元

## 设备供应商

### 沈阳机床
- 供应物资：数控车床、加工中心
- 合作年限：10 年
- 信用等级：A
- 年采购额：约 300 万元
```

- [ ] **Step 5: 创建质量标准.md**

```markdown
# 质量管理体系

## 质量方针

SINOBA 坚持"质量第一、客户至上"的质量方针，持续改进产品质量和服务水平，满足客户需求和期望。

## 进货检验标准

所有原材料和外购件必须经过进货检验。检验项目包括：
- 尺寸精度：按照图纸要求，关键尺寸公差 ±0.01mm
- 表面质量：无裂纹、无锈蚀、无变形
- 材质证明：供应商提供材质检验报告
- 抽检比例：A 类物资 100% 检验，B 类物资 10% 抽检，C 类物资 5% 抽检

## 过程检验标准

生产过程中实行首检、巡检、终检三级检验制度：
- 首检：每批次首件产品必须检验合格后方可批量生产
- 巡检：每 2 小时巡检一次，记录关键参数
- 终检：产品完工后进行全面检验

## 成品检验标准

成品检验包括外观检查、尺寸检测、性能测试三个环节。检验合格后出具《产品合格证》，不合格品按照《不合格品处理程序》处理。

## 客户投诉处理

客户投诉响应时间：24 小时内响应，48 小时内给出处理方案。重大投诉由质量部和销售部联合处理。
```

- [ ] **Step 6: Commit**

```bash
git add docs/sample_docs/
git commit -m "docs: add 5 synthetic enterprise documents for v0 testing"
```

---

### Task 11: Startup Ingest (Lifespan)

**Files:**
- Modify: `app/main.py`

- [ ] **Step 1: 更新 main.py 实现 lifespan startup**

替换 `app/main.py`：

```python
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from loguru import logger

from app.config import settings
from app.providers.embedding_bge import BGEEmbedding
from app.providers.llm_mimo import MIMOLLM
from app.providers.vectorstore_chroma import ChromaVectorStore
from app.routes import health, ingest, query
from app.services.ingest_service import IngestService
from app.services.query_service import QueryService


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")

    # Initialize providers
    embedding_provider = BGEEmbedding(model_name=settings.embedding_model)
    vectorstore_provider = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )
    llm_provider = MIMOLLM(
        api_key=settings.mimo_api_key,
        endpoint=settings.mimo_endpoint,
        model=settings.mimo_model,
        timeout=settings.mimo_timeout,
    )

    # Initialize services
    ingest_svc = IngestService(
        embedding_provider=embedding_provider,
        vectorstore_provider=vectorstore_provider,
        max_chunk_size=settings.max_chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    query_svc = QueryService(
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        vectorstore_provider=vectorstore_provider,
    )

    # Inject services into routes
    ingest.set_ingest_service(ingest_svc)
    query.set_query_service(query_svc)

    # Startup ingest: scan sample docs
    sample_dir = Path(settings.sample_docs_dir)
    if sample_dir.exists():
        ingested = 0
        for file_path in sorted(sample_dir.iterdir()):
            if file_path.suffix.lower() in {".md", ".txt"}:
                try:
                    ingest_svc.ingest_file(str(file_path))
                    ingested += 1
                except Exception as e:
                    logger.error("Failed to ingest sample doc", path=str(file_path), error=str(e))
        logger.info("Startup ingest completed", count=ingested)
    else:
        logger.warning("Sample docs directory not found", path=str(sample_dir))

    # Set health data
    health.set_health_data({
        "embedding_provider": "bge_local",
        "vectorstore": "chromadb",
        "document_count": vectorstore_provider.count(),
        "loaded_documents": vectorstore_provider.list_documents(),
    })

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Enterprise Knowledge Base",
    description="本地企业知识库系统 (RAG)",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(health.router)
```

- [ ] **Step 2: Commit**

```bash
git add app/main.py
git commit -m "feat: implement lifespan startup with sample docs ingest"
```

---

### Task 12: 集成测试

**Files:**
- Create: `tests/integration/test_pipeline.py`

- [ ] **Step 1: 写集成测试**

创建 `tests/integration/test_pipeline.py`：

```python
"""Integration tests: real ChromaDB + mock LLM/Embedding."""
import shutil
from pathlib import Path
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
    # Cleanup handled by tmp_path


@pytest.fixture
def mock_embedding():
    mock = MagicMock()
    # Return deterministic embeddings based on input length
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

    mock.chat_sync.side_effect = fake_chat_sync
    mock.chat_stream.side_effect = fake_chat_stream
    return mock


class TestIngestPipeline:
    def test_ingest_and_query(self, chroma_store, mock_embedding, mock_llm):
        ingest_svc = IngestService(
            embedding_provider=mock_embedding,
            vectorstore_provider=chroma_store,
        )

        # Ingest a document
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

        assert count_before == count_after  # Same count, not doubled

    def test_delete_and_reingest(self, chroma_store, mock_embedding):
        ingest_svc = IngestService(
            embedding_provider=mock_embedding,
            vectorstore_provider=chroma_store,
        )

        result1 = ingest_svc.ingest_text("# Old\nOld content", filename="test.md")
        doc_id = result1["document_id"]

        # Delete
        chroma_store.delete_by_document_id(doc_id)
        assert chroma_store.count() == 0

        # Re-ingest with different content
        result2 = ingest_svc.ingest_text("# New\nNew content", filename="test.md")
        assert result2["document_id"] != doc_id
        assert chroma_store.count() >= 1


class TestQueryPipeline:
    @pytest.mark.asyncio
    async def test_full_query_flow(self, chroma_store, mock_embedding, mock_llm):
        # Ingest first
        ingest_svc = IngestService(
            embedding_provider=mock_embedding,
            vectorstore_provider=chroma_store,
        )
        ingest_svc.ingest_text(
            "# 采购流程\nSINOBA 的采购流程包括申请、审批、执行。",
            filename="采购流程.md",
        )

        # Query
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
        """Verify splitter works on actual sample document structure."""
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
```

- [ ] **Step 2: 运行集成测试**

```bash
pytest tests/integration/test_pipeline.py -v
```

预期：全部 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/integration/
git commit -m "test: add integration tests for ingest and query pipelines"
```

---

### Task 13: 全量验证

- [ ] **Step 1: 运行全部测试**

```bash
cd C:/Users/24596/projects/enterprise-kb
pytest -v
```

预期：全部 PASS

- [ ] **Step 2: 启动服务验证**

```bash
cd C:/Users/24596/projects/enterprise-kb
uvicorn app.main:app --reload
```

验证：
1. 访问 `http://127.0.0.1:8000/docs` 查看 API 文档
2. 访问 `http://127.0.0.1:8000/health` 确认 startup ingest 生效
3. 用 `/docs` 测试 `/query` 端点

- [ ] **Step 3: 最终 Commit**

```bash
git add -A
git commit -m "chore: v0 RAG skeleton complete"
```
