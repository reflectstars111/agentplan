"""KnowledgeStore — manages external knowledge (chunks, sources, FAISS index).

Phase 1: Direct writes to SQLite + FAISS. No Event Store.
Phase 2+: Will wrap writes in Transaction → EventStore.
"""

import json
import os
import uuid
from datetime import datetime, timezone

import numpy as np

from ard.infra.db import Database
from ard.infra.logging import log
from ard.store import RetrievalResult, KnowledgeStoreProtocol


class KnowledgeStore(KnowledgeStoreProtocol):
    """Storage for external knowledge: chunks, sources, and vector index."""

    def __init__(
        self,
        db: Database,
        vector_index,          # FAISS-based VectorIndex
        embed_fn: callable,    # text list → np.ndarray
        data_dir: str = "data",
    ):
        self.db = db
        self.vector_index = vector_index
        self.embed_fn = embed_fn
        self.data_dir = data_dir

    # ── write path ────────────────────────────────────────────

    def index_chunks(self, chunks: list[dict], source_id: str) -> int:
        """Insert chunks into SQLite, build embeddings, add to FAISS.

        Args:
            chunks: List of chunk dicts with keys: text, summary, location, keywords.
            source_id: The source these chunks belong to.

        Returns:
            Number of chunks indexed.
        """
        if not chunks:
            return 0

        # Insert source record FIRST (FK constraint on chunks.source_id)
        self.db.execute(
            """INSERT INTO sources (source_id, source_type, file_name, chunk_count)
               VALUES (?, ?, ?, 0)
               ON CONFLICT(source_id) DO NOTHING""",
            (source_id, chunks[0].get("source_type", "text"),
             chunks[0].get("file_name", "")),
        )

        texts = [c["text"] for c in chunks]
        embeddings = self.embed_fn(texts)

        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
            embedding_id = f"emb_{chunk_id}"

            self.db.execute(
                """INSERT INTO chunks (chunk_id, source_id, source_type, text, summary,
                   location, keywords, embedding_id, trust_level, char_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk_id,
                    source_id,
                    chunk.get("source_type", "text"),
                    chunk["text"],
                    chunk.get("summary", ""),
                    json.dumps(chunk.get("location")) if chunk.get("location") else None,
                    json.dumps(chunk.get("keywords", [])) if chunk.get("keywords") else None,
                    embedding_id,
                    chunk.get("trust_level", "external_untrusted"),
                    len(chunk["text"]),
                ),
            )

            # Add to FAISS index
            vec = embeddings[i].reshape(1, -1).astype(np.float32)
            self.vector_index.add(vec, [chunk_id])

        # Update source chunk count
        self.db.execute(
            """UPDATE sources SET chunk_count = chunk_count + ? WHERE source_id = ?""",
            (len(chunks), source_id),
        )
        self.db.commit()

        # Persist FAISS
        if hasattr(self.vector_index, 'persist'):
            self.vector_index.persist()

        log.info("chunks_indexed", source_id=source_id, count=len(chunks))
        return len(chunks)

    # ── read path ─────────────────────────────────────────────

    def search(self, query: str, strategy: str = "vector", top_k: int = 20) -> list[RetrievalResult]:
        """Search for chunks matching the query.

        Args:
            query: Search query text.
            strategy: Which strategy to use (delegated to HybridRetriever).
            top_k: Maximum results.

        Returns:
            Ranked list of RetrievalResult.
        """
        # Single-strategy search — typically called by HybridRetriever
        if strategy == "vector":
            return self._vector_search(query, top_k)
        elif strategy == "keyword":
            return self._keyword_search(query, top_k)
        else:
            log.warn("unknown_search_strategy", strategy=strategy)
            return []

    def _vector_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Pure vector search via FAISS."""
        query_vec = self.embed_fn([query])
        distances, indices = self.vector_index.search(query_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            chunk_id = self.vector_index.get_id(int(idx))
            if not chunk_id:
                continue
            row = self.db.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
            if row:
                results.append(self._row_to_result(row, score=float(1.0 / (1.0 + dist)), strategy="vector"))
        return results

    def _keyword_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """FTS5 keyword search."""
        # Escape FTS5 special characters and build safe query
        safe_query = self._fts5_escape(query)
        sql = f"""SELECT c.* FROM chunks c
                  JOIN chunks_fts f ON c.rowid = f.rowid
                  WHERE chunks_fts MATCH ?
                  ORDER BY rank
                  LIMIT {top_k}"""
        rows = self.db.execute(sql, (safe_query,)).fetchall()
        return [self._row_to_result(row, score=0.7, strategy="keyword") for row in rows]

    @staticmethod
    def _fts5_escape(query: str) -> str:
        """Escape FTS5 special characters and format as OR query with prefix matching."""
        import re
        safe = re.sub(r'[^\w\s]', ' ', query)
        terms = safe.split()
        if not terms:
            return '""'
        # Use OR with prefix matching (term*) for robust partial matching
        # Filter short/common words (<3 chars) to avoid noise
        meaningful = [t for t in terms[:20] if len(t) >= 3]
        if not meaningful:
            meaningful = terms[:5]
        # OR query: "term1*" OR "term2*" OR ...
        return " OR ".join(f'"{t}"*' for t in meaningful)


    def get_chunks(self, source_id: str) -> list[dict]:
        """Get all chunks for a source."""
        rows = self.db.execute(
            "SELECT * FROM chunks WHERE source_id = ? ORDER BY created_at",
            (source_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_chunk(self, chunk_id: str) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_sources(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM sources WHERE status = 'active' ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def count_chunks(self) -> int:
        row = self.db.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
        return row["cnt"] if row else 0

    @staticmethod
    def _row_to_result(row, score: float = 0.0, strategy: str = "unknown") -> RetrievalResult:
        location = None
        loc_raw = row["location"]
        if loc_raw:
            try:
                location = json.loads(loc_raw) if isinstance(loc_raw, str) else loc_raw
            except (json.JSONDecodeError, TypeError):
                pass
        keywords = []
        kw_raw = row["keywords"]
        if kw_raw:
            try:
                keywords = json.loads(kw_raw) if isinstance(kw_raw, str) else kw_raw
            except (json.JSONDecodeError, TypeError):
                pass
        tl = row["trust_level"]
        trust_level = tl if tl else "external_untrusted"
        return RetrievalResult(
            chunk_id=row["chunk_id"],
            source_ref=f"source:{row['source_id']}",
            text_preview=row["text"],
            score=score,
            trust_level=trust_level,
            strategy=strategy,
            location=location,
            keywords=keywords,
        )
