"""SQLite connection management and schema initialization."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from src.db.migrations import (
    ALL_MIGRATIONS,
    CHUNKS_FTS_TRIGGERS,
    MEMORIES_FTS_TRIGGERS,
    SCHEMA_MIGRATIONS_TABLE,
    VERSIONED_MIGRATIONS,
)


class Database:
    """Thin wrapper around SQLite connection with schema management."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        """Get or create a connection."""
        if self._conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                self.db_path, check_same_thread=False
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def init_schema(self) -> None:
        """Create current tables and apply pending versioned migrations."""
        conn = self.connect()
        conn.execute(SCHEMA_MIGRATIONS_TABLE)
        for name, ddl in ALL_MIGRATIONS:
            conn.execute(ddl)
        self._apply_versioned_migrations(conn)
        conn.executescript(MEMORIES_FTS_TRIGGERS)
        conn.executescript(CHUNKS_FTS_TRIGGERS)
        conn.commit()

    def _apply_versioned_migrations(self, conn: sqlite3.Connection) -> None:
        applied = {
            row["version"]
            for row in conn.execute(
                "SELECT version FROM schema_migrations"
            ).fetchall()
        }
        for version, name, table, column, ddl in VERSIONED_MIGRATIONS:
            if version in applied:
                continue
            columns = {
                row["name"]
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if column not in columns:
                conn.execute(ddl)
            conn.execute(
                """
                INSERT INTO schema_migrations (version, name, applied_at)
                VALUES (?, ?, ?)
                """,
                (version, name, datetime.now(timezone.utc).isoformat()),
            )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        return self.connect().execute(sql, params)

    def commit(self) -> None:
        if self._conn:
            self._conn.commit()
