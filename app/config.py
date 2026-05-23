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
