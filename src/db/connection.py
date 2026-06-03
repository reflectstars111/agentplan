"""SQLite connection management and schema initialization."""

import sqlite3
from pathlib import Path
from src.db.migrations import ALL_MIGRATIONS, MEMORIES_FTS_TRIGGERS, CHUNKS_FTS_TRIGGERS


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
        """Create all tables if they don't exist."""
        conn = self.connect()
        for name, ddl in ALL_MIGRATIONS:
            conn.execute(ddl)
        conn.executescript(MEMORIES_FTS_TRIGGERS)
        conn.executescript(CHUNKS_FTS_TRIGGERS)
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        return self.connect().execute(sql, params)

    def commit(self) -> None:
        if self._conn:
            self._conn.commit()
