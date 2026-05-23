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
