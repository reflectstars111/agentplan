"""Relevance-scoped retrieval for working and long-term memories."""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.index.keyword_index import KeywordIndex
from src.models.memory import MemoryItem, MemoryStatus, MemoryType
from src.storage.memory_store import MemoryStore


WORKING_MEMORY_TYPES = {
    MemoryType.PROJECT_STATE,
    MemoryType.CONVERSATION_SUMMARY,
    MemoryType.INTERMEDIATE_RESULT,
}


@dataclass
class MemorySelection:
    """Relevant memories split by L2 working and L3 long-term levels."""

    working: list[MemoryItem] = field(default_factory=list)
    long_term: list[MemoryItem] = field(default_factory=list)

    @property
    def all(self) -> list[MemoryItem]:
        return [*self.working, *self.long_term]


class MemoryRetriever:
    """Select active memories by relevance, scope, quality, and recency."""

    def __init__(
        self,
        memory_store: MemoryStore,
        keyword_index: KeywordIndex,
    ):
        self.memory_store = memory_store
        self.keyword_index = keyword_index

    def retrieve(
        self,
        query: str,
        scopes: list[str] | None = None,
        limit: int = 8,
    ) -> MemorySelection:
        if not query.strip() or limit <= 0:
            return MemorySelection()

        ranked = self.keyword_index.search_memories(query, k=max(limit * 3, 10))
        now = datetime.now(timezone.utc)
        candidates = []
        allowed_scopes = set(scopes or [])

        for memory_id, keyword_score in ranked:
            item = self.memory_store.get(memory_id)
            if item is None or item.status != MemoryStatus.ACTIVE:
                continue
            if allowed_scopes and item.scope not in allowed_scopes:
                continue
            score = (
                0.55 * keyword_score
                + 0.20 * item.importance
                + 0.15 * item.confidence
                + 0.10 * self._recency_score(item, now)
            )
            candidates.append((score, item))

        candidates.sort(key=lambda pair: pair[0], reverse=True)
        selected = [item for _, item in candidates[:limit]]
        for item in selected:
            self.memory_store.touch(item.memory_id)

        return MemorySelection(
            working=[
                item for item in selected if item.type in WORKING_MEMORY_TYPES
            ],
            long_term=[
                item for item in selected if item.type not in WORKING_MEMORY_TYPES
            ],
        )

    @staticmethod
    def _recency_score(item: MemoryItem, now: datetime) -> float:
        timestamp = item.last_used_at or item.updated_at or item.created_at
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - timestamp).total_seconds() / 86400)
        return max(0.0, min(1.0, 2.0 ** (-age_days / 30.0)))
