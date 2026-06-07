"""Reranker — post-retrieval re-scoring to improve precision.

Maps to agent_os_initial_plan.md §5.3 (retrieval flow), §5.4 (mandatory rerank).
"""

from src.index.hybrid_retriever import RetrievalResult


class Reranker:
    """Re-score retrieval results using query-chunk relevance heuristics.

    After the initial hybrid retrieval and scoring, apply additional
    heuristics to boost genuinely relevant chunks and suppress noise.
    """

    def rerank(
        self,
        results: list[RetrievalResult],
        query: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Rerank results and return top_k.

        Scoring adjustments:
          - Exact phrase match in chunk text → +0.10
          - Title/heading keyword match → +0.08
          - Source diversity bonus → +0.05 (prefer different files)
          - Very short chunks → -0.05 (likely noise)
          - Very long chunks → -0.03 (less focused)
        """
        if not results:
            return []

        query_lower = query.lower()
        query_terms = query_lower.split()
        seen_sources = set()

        adjusted = []
        for r in results:
            boost = 0.0
            text_lower = r.text_preview.lower()

            # Exact phrase match
            if query_lower in text_lower:
                boost += 0.10
            else:
                # Partial term overlap
                matches = sum(1 for t in query_terms if t in text_lower)
                if matches >= len(query_terms) * 0.5:
                    boost += 0.05

            # Title/heading match (text starts with # or contains section-like patterns)
            if text_lower.startswith("#") or "section" in text_lower.lower():
                if any(t in text_lower[:100] for t in query_terms):
                    boost += 0.08

            # Source diversity: first result from each source gets a bonus
            src = r.source_ref
            if src not in seen_sources:
                boost += 0.05
                seen_sources.add(src)

            # Length penalties
            if len(r.text_preview) < 30:
                boost -= 0.05
            if len(r.text_preview) > 500:
                boost -= 0.03

            adjusted.append(RetrievalResult(
                chunk_id=r.chunk_id,
                score=min(1.0, max(0.0, r.score + boost)),
                source_ref=r.source_ref,
                trust_level=r.trust_level,
                text_preview=r.text_preview,
                score_breakdown={**r.score_breakdown, "rerank_boost": round(boost, 4)},
            ))

        adjusted.sort(key=lambda r: r.score, reverse=True)
        return adjusted[:top_k]
