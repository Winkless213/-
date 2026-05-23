# v0 RAG 系统骨架设计

## 概述

搭建本地企业知识库系统（RAG 架构）的 v0 骨架。v0 目标是系统完整可运行，用 3-5 篇合成 markdown 文档验证全链路，不接真实数据。

**性能假设**: v0 单用户单进程，文档量 < 10 篇，无并发要求
**安全假设**: v0 本地运行无鉴权，不接触敏感数据，不暴露公网

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 后端 | Python 3.11+ / FastAPI | REST API |
| 向量库 | ChromaDB | 本地、零配置 |
| LLM | MIMO API | OpenAI 兼容端点 (`/v1`) |
| Embedding | BGE-small-zh-v1.5 | 本地 sentence-transformers |
| 测试 | pytest | 单元 + 集成 |
| 部署 | uvicorn 本地 | 不做 Docker |

## 架构

```
┌─────────────────────────────────────────┐
│            FastAPI Routes               │
│  /ingest  /query  /health               │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│           Service Layer                 │
│  IngestService    QueryService          │
│  (chunking +      (retrieval +          │
│   embedding +      LLM call +           │
│   storage)         response build)      │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│          Provider Layer                 │
│  LLMProvider   EmbeddingProvider        │
│  (MIMO API)    (BGE local)             │
│  VectorStoreProvider                    │
│  (ChromaDB)                             │
└─────────────────────────────────────────┘
```

设计原则：参考 RAGFlow 的 provider 按能力抽象 + Dify 的 pipeline 化 ingest。

## 目录结构

```
app/
├── main.py                     # FastAPI app + lifespan startup ingest
├── config.py                   # pydantic-settings from .env
├── schemas.py                  # Pydantic 请求/响应模型（API 边界）
├── routes/
│   ├── ingest.py               # POST /ingest
│   ├── query.py                # POST /query
│   └── health.py               # GET /health
├── services/
│   ├── ingest_service.py       # Pipeline: extract → hash → check → chunk → embed → store
│   └── query_service.py        # Pipeline: embed query → retrieve → prompt → LLM
├── providers/
│   ├── llm_base.py             # ABC: chat(stream) / chat_sync()
│   ├── llm_mimo.py             # MIMO 实现 (OpenAI 兼容)
│   ├── embedding_base.py       # ABC: encode(texts) / encode_query(text)
│   ├── embedding_bge.py        # BGE-small-zh-v1.5 实现
│   ├── vectorstore_base.py     # ABC: add / search / delete / count / list_documents
│   └── vectorstore_chroma.py   # ChromaDB 实现
├── chunking/
│   ├── splitter.py             # 混合策略: 按标题 → 按字数
│   └── models.py               # Chunk 数据结构
└── prompts/
    └── templates.py            # RAG prompt 模板

tests/
├── unit/
│   ├── test_chunking.py
│   ├── test_providers.py
│   └── test_services.py
├── integration/
│   └── test_pipeline.py        # real Chroma + mock LLM/Embedding
└── conftest.py

docs/
└── sample_docs/                # 3-5 篇合成 markdown 测试文档
    ├── 采购流程.md
    ├── 客户档案.md
    ├── 技术规格.md
    ├── 供应商名录.md
    └── 质量标准.md
```

## Provider 抽象

### LLM Provider (llm_base.py)

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator

class LLMBase(ABC):
    @abstractmethod
    async def chat_sync(self, messages: list[dict], temperature: float = 0.2) -> str:
        """同步调用，返回完整回答"""
        ...

    @abstractmethod
    async def chat_stream(self, messages: list[dict], temperature: float = 0.2) -> AsyncIterator[str]:
        """流式调用，每个 yield 是模型返回的 delta 字符串片段，不保证是 token 边界"""
        ...
```

### Embedding Provider (embedding_base.py)

```python
from abc import ABC, abstractmethod

class EmbeddingBase(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        """批量编码，用于 ingest"""
        ...

    @abstractmethod
    def encode_query(self, text: str) -> list[float]:
        """单条编码，用于 query"""
        ...
```

### VectorStore Provider (vectorstore_base.py)

```python
from abc import ABC, abstractmethod

class VectorStoreBase(ABC):
    @abstractmethod
    def add(self, document_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        # document_id 是 source of truth，由 add 内部注入到每个 chunk 的 metadata 中
        # Chunk.metadata 不需要包含 document_id
        ...

    @abstractmethod
    def search(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        # 返回: [{"text": str, "metadata": dict, "score": float}, ...]
        ...

    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> None:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def list_documents(self) -> list[str]:
        # 返回已入库的 document filename 列表（用于 /health）
        ...
```

## API 契约

### POST /ingest

**Request** (multipart/form-data):
- `file`: 文件上传（.md / .txt）
- `path`: 文件路径
- 二选一，同时传返回 400 `invalid_request`

**Response 200**:
```json
{
  "status": "ok",
  "document_id": "sha256_hash",
  "filename": "采购流程.md",
  "chunk_count": 5
}
```

### POST /query

**Request**:
```json
{
  "question": "SINOBA 的采购流程是什么？",
  "top_k": 5,
  "stream": false
}
```

**Response 200 (非流式)**:
```json
{
  "answer": "SINOBA 的采购流程...",
  "sources": [
    {
      "document_id": "sha256_hash",
      "filename": "采购流程.md",
      "chunk_text": "...",
      "score": 0.87
    }
  ]
}
```

**Response 200 (流式 SSE)**:
```
data: {"delta": "SINOBA"}
data: {"delta": "的"}
...
data: {"done": true, "sources": [...]}
```

### GET /health

```json
{
  "status": "ok",
  "embedding_provider": "bge_local",
  "vectorstore": "chromadb",
  "document_count": 5,
  "loaded_documents": ["采购流程.md", "客户档案.md", ...]
}
```

### 错误契约

所有错误统一格式：
```json
{
  "error": "error_code",
  "message": "人类可读描述",
  "status": 400
}
```

| 场景 | HTTP | error code |
|------|------|------------|
| 同时传 file 和 path | 400 | `invalid_request` |
| 文件格式不支持 | 400 | `unsupported_file_type` |
| 文件路径不存在 | 400 | `file_not_found` |
| 文件为空或 chunk 后无内容 | 400 | `empty_document` |
| Embedding 失败 | 500 | `embedding_error` |
| LLM 调用失败 | 500 | `llm_error` |
| 向量库操作失败 | 500 | `vectorstore_error` |

## Startup Ingest

FastAPI `lifespan` 事件中执行：

1. 初始化 providers（ChromaDB collection、BGE model、MIMO client）
2. 扫描 `SAMPLE_DOCS_DIR` 目录下所有 `.md` / `.txt` 文件
3. 对每个文件执行完整 ingest pipeline（含幂等检查：同 document_id 先删旧再存新）
4. 日志输出已 ingest 文档数

目的：启动即有数据可查询，验证全链路。

## 结构化日志

选型：**loguru**（简单、开箱即用）

关键日志点：
- Provider 层：`{provider, model, latency_ms, token_count, error}` 记录每次 LLM/Embedding 调用
- Service 层：pipeline 每阶段进入/退出记日志（extract/chunk/embed/store/search/llm）
- Route 层：请求方法、路径、状态码、耗时
- 错误：异常栈 + 结构化上下文

## 数据流

### Ingest Pipeline

```
文件输入 (upload/path)
  │
  ▼
[extract] 读取文件内容 (.md/.txt)
  │
  ▼
[compute_id] SHA256(content) → document_id
  │
  ▼
[check_existing] 如果 document_id 已存在 → delete_by_document_id
  │
  ▼
[chunk] 混合切片策略
  │
  ▼
[embed] BGE encode(texts) → List[List[float]]
  │
  ▼
[store] ChromaDB add(document_id, embeddings, metadatas, documents)
```

### Query Pipeline

```
用户问题
  │
  ▼
[embed] BGE encode_query(text) → List[float]
  │
  ▼
[search] ChromaDB query(embedding, top_k) → List[Chunk]
  │
  ▼
[prompt] 拼装 RAG prompt: system + context chunks + question
  │
  ▼
[llm] MIMO chat(prompt) → answer
  │
  ▼
[response] 组装: answer + sources
```

## Chunking 策略

混合策略：先按 markdown 标题分大块，再按字数切小块。

### 流程

```
原始文本
  │
  ▼ 检测是否有 markdown 标题（# 开头行）
  │
  ├── 有标题 → 按 #/##/### 标题拆分为 HeadingChunk[]
  │             │
  │             ▼ 对每个 HeadingChunk: 如果 len > MAX_CHUNK_SIZE
  │             按字数滑动窗口切分，overlap 仅在 HeadingChunk 内部
  │             │
  │             ▼ FinalChunk[]
  │
  └── 无标题（纯 .txt）→ 直接按字数滑动窗口切分
                        │
                        ▼ FinalChunk[]
```

### 参数

- `MAX_CHUNK_SIZE = 400` 字（留 BGE 512 token 安全边距）
- `OVERLAP = 50` 字（仅在字数切分时滑动，跨 heading 不 overlap）

### Chunk 数据结构

```python
@dataclass
class Chunk:
    text: str
    metadata: dict  # keys: filename, heading, position, document_id

# metadata key 定义（chunking 阶段产出）:
# - filename: str  原始文件名
# - heading: str   所属标题（无标题时为 ""）
# - position: int  在文档中的 chunk 序号（0-based）
# 注: document_id 由 VectorStoreBase.add() 在存储时注入，chunking 阶段不产出

# ChromaDB 约束: metadata value 只支持 str/int/float/bool，不支持 list/dict
# 所有 metadata value 必须是标量类型
```

## 配置 (.env)

```env
# MIMO API
MIMO_API_KEY=your_key_here
MIMO_ENDPOINT=https://token-plan-sgp.xiaomimimo.com/v1
MIMO_MODEL=<TO_BE_FILLED>  # 需填入真实模型名
MIMO_TIMEOUT=30
MIMO_MAX_RETRIES=3

# Embedding
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

# Chunking
MAX_CHUNK_SIZE=400
CHUNK_OVERLAP=50

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=enterprise_kb

# Startup Ingest
SAMPLE_DOCS_DIR=./docs/sample_docs

# Server
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=INFO
```

## 测试策略

三层测试：

### 单元测试 (unit/)
- **test_chunking.py**: 纯函数测试，给 markdown 输入验证输出结构、边界 case
- **test_providers.py**: mock 外部依赖，测接口契约
- **test_services.py**: mock providers，测 pipeline 编排逻辑

### 集成测试 (integration/)
- **test_pipeline.py**: 真 ChromaDB + mock LLM/Embedding，验证存储/检索行为

### 端到端 (可选)
- TestClient + 全真实组件，验证完整链路

## 边界 Case 清单

- [ ] 空文件 → 返回 `empty_document` 错误
- [ ] 文件格式不支持（非 .md/.txt）→ 返回 `unsupported_file_type`
- [ ] 文件路径不存在 → 返回 `file_not_found`
- [ ] 同文件重复 ingest → 幂等（先删旧再存新）
- [ ] 超长单行文本（无换行符）→ 按字数强制切分
- [ ] Embedding 调用超时 → 重试 + 结构化日志
- [ ] LLM 调用超时 → 重试 + 结构化日志
- [ ] ChromaDB 持久化目录不存在 → 自动创建
- [ ] Query 时向量库为空 → 返回友好提示
- [ ] 流式响应中断 → 客户端收到部分 token + 错误事件

## v0 不做

- 不接真实数据
- 不做权限 / 多用户
- 不做前端 UI
- 不支持 PDF / Word / Excel
- 不做对话历史 / 多轮上下文
- 不做监控、日志聚合
- 不做 Docker 化
- 不做 rerank（v0 直接用 ChromaDB 距离排序）
