"""Retrieval strategies for multi-strategy hybrid retrieval."""

from ard.store import RetrievalResult
from ard.store.knowledge_store import KnowledgeStore


class VectorStrategy:
    """FAISS vector similarity search."""

    def __init__(self, store: KnowledgeStore):
        self.store = store

    def search(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        return self.store._vector_search(query, top_k)


class KeywordStrategy:
    """FTS5 full-text keyword search."""

    def __init__(self, store: KnowledgeStore):
        self.store = store

    def search(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        return self.store._keyword_search(query, top_k)
