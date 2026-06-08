"""Multi-strategy retrieval layer."""

from typing import Protocol, runtime_checkable
from dataclasses import dataclass, field
from ard.store import RetrievalResult


@dataclass
class RetrievalPlan:
    """Output of QueryPlanner: which strategies to use with what weights."""
    strategies: list[str] = field(default_factory=lambda: ["vector", "keyword"])
    weights: dict[str, float] = field(default_factory=dict)
    filters: dict = field(default_factory=dict)
    top_k: int = 20

    def __post_init__(self):
        if not self.weights:
            n = len(self.strategies)
            self.weights = {s: 1.0 / n for s in self.strategies}


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Protocol for any retriever."""

    def retrieve(self, query: str, plan: RetrievalPlan | None = None) -> list[RetrievalResult]:
        ...


@runtime_checkable
class QueryPlannerProtocol(Protocol):
    """Protocol for query planning."""

    def plan(self, query: str) -> RetrievalPlan:
        ...


@runtime_checkable
class RerankerProtocol(Protocol):
    """Protocol for reranking."""

    def rerank(self, query: str, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
        ...
