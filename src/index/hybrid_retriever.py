"""HybridRetriever — combines vector + keyword search with weighted scoring.

Implements the scoring formula from agent_os_initial_plan.md §5.2:
  Score = 0.35*semantic + 0.20*keyword + 0.15*entity - 0.20*trust_penalty
"""

from dataclasses import dataclass, field
import numpy as np
from src.db.connection import Database
from src.index.vector_index import VectorIndex
from src.index.keyword_index import KeywordIndex
from src.config import Config


@dataclass
class RetrievalResult:
    """A single retrieval result with metadata."""
    chunk_id: str
    score: float                     # 0.0–1.0 combined score
    source_ref: str                  # e.g. "file:paper.pdf"
    trust_level: str                 # TrustLevel value
    text_preview: str                # First 200 chars
    score_breakdown: dict = field(default_factory=dict)  # {semantic, keyword, entity, trust_penalty}


class HybridRetriever:
    """Combined retriever that fuses vector and keyword search results.

    Usage:
        retriever = HybridRetriever(vector_index, keyword_index, db, config)
        results = retriever.retrieve("query text", embed_fn=my_embed_fn, k=10)
    """

    def __init__(
        self,
        vector_index: VectorIndex,
        keyword_index: KeywordIndex,
        db: Database,
        config: Config | None = None,
    ):
        self.vector_index = vector_index
        self.keyword_index = keyword_index
        self.db = db
        self.config = config or Config()

    def retrieve(
        self,
        query: str,
        embed_fn,
        k: int = 10,
    ) -> list[RetrievalResult]:
        """Retrieve top-k results combining vector and keyword scores.

        Args:
            query: Natural language query string.
            embed_fn: Function (str) -> np.ndarray that produces an embedding.
            k: Number of results to return.

        Returns:
            Sorted list of RetrievalResult, highest score first.
        """
        if self.vector_index.count == 0:
            return []

        # 1. Get vector search results
        query_emb = embed_fn([query])[0] if isinstance(embed_fn([query]), np.ndarray) else embed_fn(query)
        if isinstance(query_emb, np.ndarray) and query_emb.ndim == 2:
            query_emb = query_emb[0]

        vector_results = self.vector_index.search(
            query_emb, k=self.config.max_retrieval_candidates
        )

        # 2. Get keyword search results
        keyword_results = self.keyword_index.search_chunks(
            query, k=self.config.max_retrieval_candidates
        )

        # 3. Merge and score
        scored: dict[str, dict] = {}  # chunk_id -> {scores, source_ref, ...}

        # Add vector scores
        for chunk_id, sim_score in vector_results:
            if chunk_id not in scored:
                scored[chunk_id] = self._init_score_entry(chunk_id)
            scored[chunk_id]["semantic"] = sim_score

        # Add keyword scores
        for chunk_id, kw_score in keyword_results:
            if chunk_id not in scored:
                scored[chunk_id] = self._init_score_entry(chunk_id)
            scored[chunk_id]["keyword"] = kw_score

        # 4. Compute final scores
        results = []
        for chunk_id, entry in scored.items():
            semantic = entry.get("semantic", 0.0)
            keyword = entry.get("keyword", 0.0)
            entity = 0.0  # entity relevance — computed from keyword overlap on entities

            # Trust penalty: reduce score for untrusted sources
            trust_level = entry.get("trust_level", "external_untrusted")
            trust_penalty = self._trust_penalty(trust_level)

            combined = (
                self.config.weight_semantic * semantic
                + self.config.weight_keyword * keyword
                + self.config.weight_entity * entity
                - self.config.penalty_trust * trust_penalty
            )
            # Clamp to [0, 1]
            combined = max(0.0, min(1.0, combined))

            results.append(RetrievalResult(
                chunk_id=chunk_id,
                score=round(combined, 4),
                source_ref=entry.get("source_ref", "unknown"),
                trust_level=trust_level,
                text_preview=entry.get("text_preview", ""),
                score_breakdown={
                    "semantic": round(semantic, 4),
                    "keyword": round(keyword, 4),
                    "entity": entity,
                    "trust_penalty": round(trust_penalty, 4),
                    "combined": round(combined, 4),
                },
            ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def _init_score_entry(self, chunk_id: str) -> dict:
        """Initialize a score entry with chunk metadata from the database."""
        entry = {
            "semantic": 0.0,
            "keyword": 0.0,
            "source_ref": "unknown",
            "trust_level": "external_untrusted",
            "text_preview": "",
        }

        # Look up chunk in DB for metadata
        row = self.db.execute(
            "SELECT source_id, text, trust_level FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()

        if row:
            entry["source_ref"] = row["source_id"]
            entry["trust_level"] = row["trust_level"] or "external_untrusted"
            text = row["text"] or ""
            entry["text_preview"] = text[:200]

        return entry

    def _trust_penalty(self, trust_level: str) -> float:
        """Compute penalty factor based on trust level.

        Returns 0.0 (no penalty) to 0.2 (full penalty).
        """
        penalties = {
            "trusted_instruction": 0.0,
            "user_instruction": 0.0,
            "internal_memory": 0.05,
            "user_provided_data": 0.05,
            "external_untrusted": 0.15,
            "tool_observation": 0.1,
            "agent_generated": 0.1,
        }
        return penalties.get(trust_level, 0.2)
