"""SQLite connection management for ARD Phase 2."""

from contextlib import contextmanager
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Iterator


class Database:
    """SQLite connection wrapper with WAL mode and schema init."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._transaction_depth = 0
        self._lock = RLock()

    def connect(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA foreign_keys=ON")
            return self._conn

    def init_schema(self) -> None:
        """Create all tables (Phase 1 + Phase 2)."""
        conn = self.connect()
        conn.executescript(SCHEMA_DDL)
        conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
                self._transaction_depth = 0

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        with self._lock:
            return self.connect().execute(sql, params)

    def commit(self) -> None:
        with self._lock:
            if self._conn and self._transaction_depth == 0:
                self._conn.commit()

    def executemany(self, sql: str, params_list: list) -> sqlite3.Cursor:
        with self._lock:
            return self.connect().executemany(sql, params_list)

    @property
    def in_transaction(self) -> bool:
        return self._transaction_depth > 0

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """Run a group of writes in one SQLite transaction.

        Nested callers share the outer transaction. Existing store and projection
        code may still call ``commit()``; those calls are intentionally deferred
        until the outermost transaction exits.
        """
        self._lock.acquire()
        conn = self.connect()
        outermost = self._transaction_depth == 0
        try:
            if outermost:
                conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            self._transaction_depth += 1
            try:
                yield conn
            except Exception:
                self._transaction_depth -= 1
                if outermost:
                    conn.rollback()
                raise
            else:
                self._transaction_depth -= 1
                if outermost:
                    conn.commit()
        finally:
            self._lock.release()


SCHEMA_DDL = """
-- ===== Phase 1: Knowledge Store =====
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    file_name TEXT,
    file_path TEXT,
    chunk_count INTEGER DEFAULT 0,
    total_size INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    text TEXT NOT NULL,
    summary TEXT,
    location TEXT,
    keywords TEXT,
    embedding_id TEXT,
    trust_level TEXT DEFAULT 'external_untrusted',
    char_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (source_id) REFERENCES sources(source_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id, text, summary, keywords,
    content='chunks', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, chunk_id, text, summary, keywords)
    VALUES (new.rowid, new.chunk_id, new.text, new.summary, new.keywords);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, text, summary, keywords)
    VALUES ('delete', old.rowid, old.chunk_id, old.text, old.summary, old.keywords);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, text, summary, keywords)
    VALUES ('delete', old.rowid, old.chunk_id, old.text, old.summary, old.keywords);
    INSERT INTO chunks_fts(rowid, chunk_id, text, summary, keywords)
    VALUES (new.rowid, new.chunk_id, new.text, new.summary, new.keywords);
END;

-- ===== Phase 2: Event Store + State + Transaction + Trace =====

CREATE TABLE IF NOT EXISTS events (
    seq_num INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    stream TEXT NOT NULL,
    stream_key TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload JSON NOT NULL,
    txn_id TEXT NOT NULL,
    causation_seq INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_events_stream ON events(stream, stream_key);
CREATE INDEX IF NOT EXISTS idx_events_txn ON events(txn_id);

CREATE TABLE IF NOT EXISTS transactions (
    txn_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    read_set JSON,
    event_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS state_snapshots (
    stream_key TEXT PRIMARY KEY,
    value JSON NOT NULL,
    version INTEGER NOT NULL,
    stream TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ===== SemState research projections =====

CREATE TABLE IF NOT EXISTS semstate_nodes (
    state_key TEXT PRIMARY KEY,
    value JSON NOT NULL,
    version INTEGER NOT NULL,
    node_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'valid',
    producer_task TEXT,
    source_refs JSON,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS semstate_dependencies (
    source_key TEXT NOT NULL,
    target_key TEXT NOT NULL,
    source_version INTEGER NOT NULL,
    origin TEXT NOT NULL,
    confidence REAL NOT NULL,
    edge_kind TEXT NOT NULL,
    PRIMARY KEY (source_key, target_key)
);
CREATE INDEX IF NOT EXISTS idx_semstate_dep_target
    ON semstate_dependencies(target_key);

CREATE TABLE IF NOT EXISTS semstate_conflicts (
    conflict_id TEXT PRIMARY KEY,
    txn_id TEXT NOT NULL,
    anomaly_type TEXT,
    decision JSON NOT NULL,
    envelope JSON NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    step_type TEXT NOT NULL,
    input JSON,
    output JSON,
    status TEXT DEFAULT 'success',
    error TEXT,
    seq_num INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (trace_id, step_id)
);
"""
