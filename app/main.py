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
                    logger.error("Failed to ingest sample doc: path={}, error={}", str(file_path), str(e))
        logger.info("Startup ingest completed: count={}", ingested)
    else:
        logger.warning("Sample docs directory not found: path={}", str(sample_dir))

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
