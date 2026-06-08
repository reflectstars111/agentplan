"""Reranker — applies the 8-factor ARD scoring formula to retrieval candidates."""

import re

from ard.infra.config import Config
from ard.retriever import RetrievalResult, RerankerProtocol


class Reranker(RerankerProtocol):
    """Re-ranks retrieval candidates using the ARD scoring formula.

    Score = 0.35*semantic + 0.20*keyword + 0.15*entity + 0.10*recency
          + 0.10*importance + 0.10*structure - 0.10*token_cost - 0.20*trust_penalty
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def rerank(self, query: str, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
        if not candidates:
            return []

        w = self.config
        query_terms = set(self._tokenize(query))

        for result in candidates:
            text_terms = set(self._tokenize(result.text_preview))

            semantic = result.score  # passed through from retriever
            keyword = self._keyword_match(query_terms, text_terms)
            entity = self._entity_score(query_terms, result)
            recency = 0.5  # Phase 1: no recency data; neutral
            importance = self._importance_score(result)
            structure = self._structure_score(query, result)
            token_cost_penalty = self._token_cost_penalty(result)
            trust_penalty = self._trust_penalty(result)

            result.score = (
                w.weight_semantic * semantic
                + w.weight_keyword * keyword
                + w.weight_entity * entity
                + w.weight_recency * recency
                + w.weight_importance * importance
                + w.weight_structural * structure
                - w.penalty_token_cost * token_cost_penalty
                - w.penalty_trust * trust_penalty
            )

        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())

    @staticmethod
    def _keyword_match(query_terms: set[str], text_terms: set[str]) -> float:
        if not query_terms:
            return 0.0
        overlap = query_terms & text_terms
        return len(overlap) / len(query_terms)

    @staticmethod
    def _entity_score(query_terms: set[str], result: RetrievalResult) -> float:
        """Score based on keyword overlap between query and chunk keywords."""
        if not result.keywords:
            return 0.0
        kw_terms = set()
        for kw in result.keywords:
            kw_terms.update(Reranker._tokenize(kw))
        if not kw_terms:
            return 0.0
        overlap = query_terms & kw_terms
        return len(overlap) / max(len(kw_terms), 1)

    @staticmethod
    def _importance_score(result: RetrievalResult) -> float:
        """Heuristic: longer chunks from structured sources are more important."""
        length = len(result.text_preview)
        if length > 2000:
            return 0.9
        elif length > 1000:
            return 0.7
        elif length > 500:
            return 0.5
        elif length > 200:
            return 0.3
        return 0.1

    @staticmethod
    def _structure_score(query: str, result: RetrievalResult) -> float:
        """Higher score if result has structural metadata (page, section)."""
        score = 0.0
        if result.location:
            if result.location.get("section"):
                score += 0.5
            if result.location.get("page"):
                score += 0.3
            if result.location.get("line_start"):
                score += 0.2
        return min(score, 1.0)

    @staticmethod
    def _token_cost_penalty(result: RetrievalResult) -> float:
        """Penalize very long chunks (expensive in tokens)."""
        length = len(result.text_preview)
        if length > 2000:
            return 1.0
        elif length > 1000:
            return 0.5
        elif length > 500:
            return 0.2
        return 0.0

    @staticmethod
    def _trust_penalty(result: RetrievalResult) -> float:
        """Penalize untrusted sources."""
        trust_scores = {
            "internal_memory": 0.0,
            "user_provided_data": 0.1,
            "agent_generated": 0.3,
            "tool_observation": 0.4,
            "external_untrusted": 0.6,
        }
        return trust_scores.get(result.trust_level, 0.6)
