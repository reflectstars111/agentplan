"""EntityIndex — entity extraction and indexing for graph retrieval.

Maps to agent_os_initial_plan.md §5.1 (Graph Index).
"""

import json
import re
import uuid
from datetime import datetime, timezone
from src.db.connection import Database
from src.models.chunk import DocumentChunk


# Entity extraction patterns
ENTITY_PATTERNS = [
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',      # Multi-word capitalized: "Apache Kafka"
    r'\b[A-Z][A-Za-z0-9_]{2,}\b',                # CamelCase: "FastAPI", "PostgreSQL"
    r'\b[a-z]+(?:_[a-z]+){2,}\b',                 # snake_case identifiers
]


class EntityIndex:
    """Keyword-based entity extraction and SQLite-backed index."""

    def __init__(self, db: Database):
        self.db = db

    def extract_and_index(self, chunks: list[DocumentChunk]) -> int:
        """Extract entities from chunk text and store in entity_graph table."""
        now = datetime.now(timezone.utc).isoformat()
        count = 0

        for chunk in chunks:
            entities = self._extract_entities(chunk.text)
            for entity in entities:
                entity_id = f"ent_{uuid.uuid4().hex[:12]}"
                # Check if entity already exists
                existing = self.db.execute(
                    "SELECT entity_id, source_chunk_ids, mention_count FROM entity_graph WHERE entity_name = ?",
                    (entity,),
                ).fetchone()
                if existing:
                    ids = json.loads(existing["source_chunk_ids"])
                    if chunk.chunk_id not in ids:
                        ids.append(chunk.chunk_id)
                    self.db.execute(
                        "UPDATE entity_graph SET source_chunk_ids = ?, mention_count = ? WHERE entity_id = ?",
                        (json.dumps(ids), existing["mention_count"] + 1, existing["entity_id"]),
                    )
                else:
                    self.db.execute(
                        "INSERT INTO entity_graph (entity_id, entity_name, entity_type, source_chunk_ids, mention_count, created_at) VALUES (?,?,?,?,?,?)",
                        (entity_id, entity, "auto", json.dumps([chunk.chunk_id]), 1, now),
                    )
                count += 1

        self.db.commit()
        return count

    def get_entities_for_chunk(self, chunk_id: str) -> list[str]:
        rows = self.db.execute(
            "SELECT entity_name FROM entity_graph WHERE source_chunk_ids LIKE ?",
            (f"%{chunk_id}%",),
        ).fetchall()
        return [r["entity_name"] for r in rows]

    def get_chunks_for_entity(self, entity_name: str) -> list[str]:
        rows = self.db.execute(
            "SELECT source_chunk_ids FROM entity_graph WHERE entity_name LIKE ?",
            (f"%{entity_name}%",),
        ).fetchall()
        return [r["source_chunk_ids"] for r in rows]

    def entity_relevance(self, chunk_id: str, query_terms: list[str]) -> float:
        """Jaccard-like overlap between query terms and chunk entities."""
        if not query_terms:
            return 0.0
        entities = self.get_entities_for_chunk(chunk_id)
        if not entities:
            return 0.0
        hits = sum(1 for t in query_terms if any(t.lower() in e.lower() for e in entities))
        return min(1.0, hits / max(1, len(query_terms)))

    def _extract_entities(self, text: str) -> list[str]:
        entities = set()
        for pattern in ENTITY_PATTERNS:
            for match in re.findall(pattern, text):
                if len(match) > 3 and not match.isdigit():
                    entities.add(match)
        return list(entities)
