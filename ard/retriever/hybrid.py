"""HybridRetriever — orchestrates multi-strategy retrieval with reranking."""

from collections import defaultdict

from ard.infra.logging import log
from ard.retriever import RetrievalPlan, RetrievalResult, RetrieverProtocol, QueryPlannerProtocol, RerankerProtocol
from ard.retriever.strategies import VectorStrategy, KeywordStrategy
from ard.store.knowledge_store import KnowledgeStore


class HybridRetriever(RetrieverProtocol):
    """Multi-strategy hybrid retrieval with query planning and reranking.

    Orchestrates: QueryPlanner → [Vector, Keyword, ...] → Merge → Reranker → Results
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        query_planner: QueryPlannerProtocol,
        reranker: RerankerProtocol,
        vector_strategy: VectorStrategy | None = None,
        keyword_strategy: KeywordStrategy | None = None,
    ):
        self.knowledge_store = knowledge_store
        self.query_planner = query_planner
        self.reranker = reranker
        self.vector = vector_strategy or VectorStrategy(knowledge_store)
        self.keyword = keyword_strategy or KeywordStrategy(knowledge_store)
        self._strategy_map = {
            "vector": self.vector,
            "keyword": self.keyword,
        }

    def retrieve(self, query: str, plan: RetrievalPlan | None = None) -> list[RetrievalResult]:
        """Execute multi-strategy retrieval with reranking.

        Args:
            query: Natural language query.
            plan: Optional RetrievalPlan. Auto-generated if not provided.

        Returns:
            Ranked list of RetrievalResult.
        """
        if plan is None:
            plan = self.query_planner.plan(query)

        # Phase 1: Collect candidates from all strategies
        all_candidates: list[RetrievalResult] = []
        seen_ids: set[str] = set()

        for strategy_name in plan.strategies:
            strategy = self._strategy_map.get(strategy_name)
            if strategy is None:
                log.warn("unknown_strategy", strategy=strategy_name)
                continue

            results = strategy.search(query, top_k=plan.top_k)
            weight = plan.weights.get(strategy_name, 1.0 / len(plan.strategies))

            for r in results:
                if r.chunk_id in seen_ids:
                    continue
                seen_ids.add(r.chunk_id)
                r.score *= weight  # apply strategy weight
                all_candidates.append(r)

        # Phase 2: Rerank
        ranked = self.reranker.rerank(query, all_candidates)

        log.info("hybrid_retrieval_complete",
                 query_len=len(query),
                 candidates=len(all_candidates),
                 strategies=plan.strategies)

        return ranked
