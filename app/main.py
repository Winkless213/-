from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from app.config import settings
from app.routes import health, ingest, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    # TODO: Task 11 will add startup ingest here
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
