"""HybridRetriever — combines vector + keyword search with weighted scoring.

Implements the scoring formula from agent_os_initial_plan.md §5.2:
  Score = 0.35*semantic + 0.20*keyword + 0.15*entity + 0.10*recency
        + 0.10*importance + 0.10*structural - 0.10*token_cost - 0.20*trust_penalty
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
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
    score_breakdown: dict = field(default_factory=dict)


@dataclass
class RetrievalFilters:
    """Optional filters for retrieval queries."""
    chunk_type: Optional[str] = None       # "paragraph", "code", "table"
    source_type: Optional[str] = None      # "pdf", "markdown", "code"
    source_id: Optional[str] = None        # specific file
    section: Optional[str] = None          # e.g. "3.2"
    min_recency: Optional[str] = None      # ISO date, only chunks newer than this
    language: Optional[str] = None         # "python", "javascript"


class HybridRetriever:
    """Combined retriever that fuses vector and keyword search results."""

    def __init__(
        self,
        vector_index: VectorIndex,
        keyword_index: KeywordIndex,
        db: Database,
        config: Config | None = None,
        structure_index=None,
        entity_index=None,
    ):
        self.vector_index = vector_index
        self.keyword_index = keyword_index
        self.db = db
        self.config = config or Config()
        self.structure_index = structure_index
        self.entity_index = entity_index

    def retrieve(
        self,
        query: str,
        embed_fn,
        k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve top-k results combining vector and keyword scores."""
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
        scored: dict[str, dict] = {}
        query_terms = query.lower().split()

        for chunk_id, sim_score in vector_results:
            if not self._passes_filters(chunk_id, filters):
                continue
            scored.setdefault(chunk_id, self._init_score_entry(chunk_id))["semantic"] = sim_score

        for chunk_id, kw_score in keyword_results:
            if not self._passes_filters(chunk_id, filters):
                continue
            scored.setdefault(chunk_id, self._init_score_entry(chunk_id))["keyword"] = kw_score

        # 4. Compute final scores with all 8 components
        results = []
        now = datetime.now(timezone.utc)
        for chunk_id, entry in scored.items():
            semantic = entry.get("semantic", 0.0)
            keyword = entry.get("keyword", 0.0)
            entity = self._entity_score(chunk_id, query_terms)
            recency = self._recency_score(entry.get("created_at"), now)
            importance = self._importance_score(entry.get("importance", 0.5))
            structural = self._structural_score(chunk_id, query_terms)
            token_cost = self._token_cost_penalty(entry.get("text_preview", ""))
            trust_penalty = self._trust_penalty(entry.get("trust_level", "external_untrusted"))

            combined = (
                self.config.weight_semantic * semantic
                + self.config.weight_keyword * keyword
                + self.config.weight_entity * entity
                + self.config.weight_recency * recency
                + self.config.weight_importance * importance
                + self.config.weight_structural * structural
                - self.config.penalty_token_cost * token_cost
                - self.config.penalty_trust * trust_penalty
            )
            combined = max(0.0, min(1.0, combined))

            results.append(RetrievalResult(
                chunk_id=chunk_id,
                score=round(combined, 4),
                source_ref=entry.get("source_ref", "unknown"),
                trust_level=entry.get("trust_level", "external_untrusted"),
                text_preview=entry.get("text_preview", ""),
                score_breakdown={
                    "semantic": round(semantic, 4),
                    "keyword": round(keyword, 4),
                    "entity": round(entity, 4),
                    "recency": round(recency, 4),
                    "importance": round(importance, 4),
                    "structural": round(structural, 4),
                    "token_cost": round(token_cost, 4),
                    "trust_penalty": round(trust_penalty, 4),
                    "combined": round(combined, 4),
                },
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    # ── Scoring Components ─────────────────────────────────────

    def _entity_score(self, chunk_id: str, query_terms: list[str]) -> float:
        """Entity relevance: overlap between query terms and chunk entities."""
        if not self.entity_index or not query_terms:
            return 0.0
        try:
            entities = self.entity_index.get_entities_for_chunk(chunk_id)
            if not entities:
                return 0.0
            hits = sum(1 for t in query_terms if any(t in e.lower() for e in entities))
            return min(1.0, hits / max(1, len(query_terms)))
        except Exception:
            return 0.0

    def _recency_score(self, created_at, now) -> float:
        """Exponential decay: newer chunks score higher."""
        if not created_at:
            return 0.5
        try:
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            age_days = (now - created_at).total_seconds() / 86400
            # Half-life of 30 days
            return max(0.0, min(1.0, 2.0 ** (-age_days / 30)))
        except Exception:
            return 0.5

    def _importance_score(self, importance: float) -> float:
        """Pass through importance value (0.0-1.0)."""
        return max(0.0, min(1.0, importance))

    def _structural_score(self, chunk_id: str, query_terms: list[str]) -> float:
        """Structural relevance via StructureIndex."""
        if not self.structure_index or not query_terms:
            return 0.0
        try:
            return self.structure_index.structural_relevance(chunk_id, query_terms)
        except Exception:
            return 0.0

    def _token_cost_penalty(self, text: str) -> float:
        """Penalty proportional to text length / budget."""
        if not text or self.config.default_token_budget <= 0:
            return 0.0
        est_tokens = len(text) // 4
        ratio = est_tokens / self.config.default_token_budget
        return min(0.5, ratio * 0.5)

    def _trust_penalty(self, trust_level: str) -> float:
        """Compute penalty factor based on trust level."""
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

    # ── Helpers ────────────────────────────────────────────────

    def _init_score_entry(self, chunk_id: str) -> dict:
        """Initialize a score entry with chunk metadata from the database."""
        entry = {
            "semantic": 0.0, "keyword": 0.0,
            "source_ref": "unknown", "trust_level": "external_untrusted",
            "text_preview": "", "created_at": "", "importance": 0.5,
        }
        row = self.db.execute(
            "SELECT source_id, text, trust_level, created_at FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row:
            entry["source_ref"] = row["source_id"]
            entry["trust_level"] = row["trust_level"] or "external_untrusted"
            text = row["text"] or ""
            entry["text_preview"] = text[:200]
            entry["created_at"] = row["created_at"]
        return entry

    def _passes_filters(self, chunk_id: str, filters: RetrievalFilters | None) -> bool:
        """Check if a chunk matches the given filters."""
        if filters is None:
            return True
        row = self.db.execute(
            """SELECT source_id, source_type, chunk_type,
                      location_section, created_at
               FROM chunks WHERE chunk_id = ?""",
            (chunk_id,),
        ).fetchone()
        if not row:
            return True
        row = dict(row)
        if filters.chunk_type and row.get("chunk_type") != filters.chunk_type:
            return False
        if filters.source_type and row.get("source_type") != filters.source_type:
            return False
        if filters.source_id and row.get("source_id") != filters.source_id:
            return False
        if filters.section:
            section = row.get("location_section") or ""
            if filters.section.lower() not in section.lower():
                return False
        if filters.min_recency and row.get("created_at"):
            if row["created_at"] < filters.min_recency:
                return False
        return True
