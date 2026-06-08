"""Storage layer — KnowledgeStore, EventStore, StateStore, TraceStore."""

from typing import Protocol, runtime_checkable
from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    """A single retrieval candidate with metadata."""
    chunk_id: str
    source_ref: str
    text_preview: str
    score: float = 0.0
    trust_level: str = "external_untrusted"
    strategy: str = "unknown"
    location: dict | None = None
    keywords: list[str] = field(default_factory=list)


@runtime_checkable
class KnowledgeStoreProtocol(Protocol):
    """Protocol for knowledge storage and retrieval."""

    def search(self, query: str, strategy: str = "vector", top_k: int = 20) -> list[RetrievalResult]:
        ...

    def get_chunks(self, source_id: str) -> list[dict]:
        ...

    def index_chunks(self, chunks: list[dict], source_id: str) -> int:
        ...

    def list_sources(self) -> list[dict]:
        ...

    def count_chunks(self) -> int:
        ...

    def get_chunk(self, chunk_id: str) -> dict | None:
        ...
