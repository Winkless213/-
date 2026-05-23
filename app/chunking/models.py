from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    metadata: dict  # keys: filename, heading, position
    # ChromaDB constraint: metadata values must be str/int/float/bool
    # document_id is injected by VectorStoreBase.add() at storage time
