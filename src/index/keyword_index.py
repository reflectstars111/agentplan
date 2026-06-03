"""KeywordIndex — BM25-like keyword search via SQLite FTS5."""

from src.db.connection import Database


class KeywordIndex:
    """Keyword search wrapper over SQLite FTS5 virtual tables.

    Provides BM25-like relevance scoring for chunks and memories.
    Scores are normalized to [0.0, 1.0] range.
    """

    def __init__(self, db: Database):
        self.db = db

    def search_chunks(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Search chunks via FTS5. Returns [(chunk_id, normalized_score), ...]."""
        clean_query = self._escape_fts5(query)
        rows = self.db.execute(
            """SELECT c.chunk_id, c.text, fts.rank
               FROM chunks c
               INNER JOIN chunks_fts fts ON c.rowid = fts.rowid
               WHERE chunks_fts MATCH ?
               ORDER BY fts.rank
               LIMIT ?""",
            (clean_query, k),
        ).fetchall()

        if not rows:
            return []

        # FTS5 rank is negative (more negative = better match)
        # Normalize to [0, 1] where 1 = best match
        ranks = [r["rank"] for r in rows]
        min_rank = min(ranks)
        max_rank = max(ranks)

        if max_rank == min_rank:
            return [(r["chunk_id"], 1.0) for r in rows]

        results = []
        for r in rows:
            normalized = (r["rank"] - min_rank) / (max_rank - min_rank)
            results.append((r["chunk_id"], round(normalized, 4)))
        return results

    def search_memories(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Search memories via FTS5. Returns [(memory_id, normalized_score), ...]."""
        clean_query = self._escape_fts5(query)
        rows = self.db.execute(
            """SELECT m.memory_id, m.content, fts.rank
               FROM memories m
               INNER JOIN memories_fts fts ON m.rowid = fts.rowid
               WHERE memories_fts MATCH ?
               ORDER BY fts.rank
               LIMIT ?""",
            (clean_query, k),
        ).fetchall()

        if not rows:
            return []

        ranks = [r["rank"] for r in rows]
        min_rank = min(ranks)
        max_rank = max(ranks)

        if max_rank == min_rank:
            return [(r["memory_id"], 1.0) for r in rows]

        results = []
        for r in rows:
            normalized = (r["rank"] - min_rank) / (max_rank - min_rank)
            results.append((r["memory_id"], round(normalized, 4)))
        return results

    def _escape_fts5(self, query: str) -> str:
        """Escape FTS5 special characters and format for phrase matching."""
        # Remove or escape characters that break FTS5 query syntax
        clean = query.replace('"', '').replace("'", "")
        # Wrap each word for prefix-like matching
        terms = clean.split()
        if not terms:
            return '""'
        # Use OR between terms for broad matching; FTS5 rank still works
        return " OR ".join(terms)
