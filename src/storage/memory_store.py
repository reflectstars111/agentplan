"""MemoryStore — CRUD operations for MemoryItem records (L2/L3 storage)."""

import json
from datetime import datetime, timezone
from typing import Optional
from src.db.connection import Database
from src.models.memory import MemoryItem, MemoryType, MemoryStatus


class MemoryStore:
    """Manages persistent storage of MemoryItem records in SQLite."""

    def __init__(self, db: Database):
        self.db = db

    def insert(self, item: MemoryItem) -> None:
        """Insert or replace a memory item."""
        now = datetime.now(timezone.utc).isoformat()
        last_used = (
            item.last_used_at.isoformat()
            if item.last_used_at and hasattr(item.last_used_at, 'isoformat')
            else None
        )
        sql = """
        INSERT OR REPLACE INTO memories
            (memory_id, type, content, summary, entities, importance, confidence,
             source, scope, status, version, source_ref, last_used_at, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.db.execute(sql, (
            item.memory_id,
            item.type.value,
            item.content,
            item.summary,
            json.dumps(item.entities),
            item.importance,
            item.confidence,
            item.source,
            item.scope,
            item.status.value,
            item.version,
            item.source_ref,
            last_used,
            item.created_at.isoformat() if hasattr(item.created_at, 'isoformat') else str(item.created_at),
            now,
        ))
        self.db.commit()

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a single memory by ID."""
        row = self.db.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_item(dict(row))

    def list_by_scope(self, scope: str) -> list[MemoryItem]:
        """List all memories in a given scope."""
        rows = self.db.execute(
            "SELECT * FROM memories WHERE scope = ? ORDER BY updated_at DESC", (scope,)
        ).fetchall()
        return [self._row_to_item(dict(r)) for r in rows]

    def list_active(self) -> list[MemoryItem]:
        """List all active (non-archived, non-superseded) memories."""
        rows = self.db.execute(
            "SELECT * FROM memories WHERE status = 'active' ORDER BY updated_at DESC"
        ).fetchall()
        return [self._row_to_item(dict(r)) for r in rows]

    def list_by_type(self, mem_type: MemoryType) -> list[MemoryItem]:
        """List memories of a specific type."""
        rows = self.db.execute(
            "SELECT * FROM memories WHERE type = ? AND status = 'active' ORDER BY updated_at DESC",
            (mem_type.value,),
        ).fetchall()
        return [self._row_to_item(dict(r)) for r in rows]

    def update_status(self, memory_id: str, status: MemoryStatus) -> None:
        """Update the status of a memory."""
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            "UPDATE memories SET status = ?, updated_at = ? WHERE memory_id = ?",
            (status.value, now, memory_id),
        )
        self.db.commit()

    def delete(self, memory_id: str) -> None:
        """Delete a memory record."""
        self.db.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        self.db.commit()

    def search_keyword(self, query: str, limit: int = 20) -> list[MemoryItem]:
        """Keyword search using FTS5 on memories."""
        # Escape FTS5 special characters
        clean_query = query.replace('"', '""')
        rows = self.db.execute(
            """SELECT m.* FROM memories m
               INNER JOIN memories_fts fts ON m.rowid = fts.rowid
               WHERE memories_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (f'"{clean_query}"', limit),
        ).fetchall()
        if not rows:
            # Fallback to LIKE search if FTS returns nothing
            rows = self.db.execute(
                "SELECT * FROM memories WHERE content LIKE ? OR summary LIKE ? LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        return [self._row_to_item(dict(r)) for r in rows]

    def count(self) -> int:
        """Total number of memories."""
        row = self.db.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
        return row["cnt"] if row else 0

    def archive_old(self, days: int = 90) -> int:
        """Archive memories not touched in N days. Returns count archived."""
        from datetime import datetime, timezone, timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self.db.execute(
            "SELECT memory_id FROM memories WHERE status = 'active' "
            "AND (last_used_at IS NULL OR last_used_at < ?)",
            (cutoff,),
        ).fetchall()
        count = len(rows)
        for r in rows:
            self.update_status(r["memory_id"], MemoryStatus.ARCHIVED)
        return count

    def list_archived(self) -> list[MemoryItem]:
        """List all archived (L5 cold storage) memories."""
        rows = self.db.execute(
            "SELECT * FROM memories WHERE status = 'archived' ORDER BY updated_at DESC"
        ).fetchall()
        return [self._row_to_item(dict(r)) for r in rows]

    def restore(self, memory_id: str) -> None:
        """Restore an archived memory back to active."""
        self.update_status(memory_id, MemoryStatus.ACTIVE)

    def touch(self, memory_id: str) -> None:
        """Update last_used_at to now for a memory record."""
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            "UPDATE memories SET last_used_at = ? WHERE memory_id = ?",
            (now, memory_id),
        )
        self.db.commit()

    def _row_to_item(self, row: dict) -> MemoryItem:
        """Convert a database row dict to a MemoryItem."""
        last_used = None
        if row.get("last_used_at"):
            try:
                last_used = datetime.fromisoformat(row["last_used_at"])
            except (ValueError, TypeError):
                pass

        return MemoryItem(
            memory_id=row["memory_id"],
            type=MemoryType(row["type"]),
            content=row["content"],
            summary=row.get("summary") or "",
            entities=json.loads(row.get("entities", "[]")) if row.get("entities") else [],
            importance=row.get("importance") or 0.5,
            confidence=row.get("confidence") or 0.5,
            source=row.get("source", "conversation"),
            scope=row.get("scope", "project"),
            status=MemoryStatus(row["status"]) if row.get("status") else MemoryStatus.ACTIVE,
            version=row.get("version", 1),
            source_ref=row.get("source_ref"),
            last_used_at=last_used,
            created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row.get("updated_at") else datetime.now(timezone.utc),
        )
