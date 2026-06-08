"""DDL statements for Agent-OS MVP. Maps to agent_os_initial_plan.md §21."""

SCHEMA_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL
);
"""

MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT DEFAULT '',
    entities TEXT DEFAULT '[]',       -- JSON array
    importance REAL DEFAULT 0.5,
    confidence REAL DEFAULT 0.5,
    source TEXT DEFAULT 'conversation',
    scope TEXT DEFAULT 'project',
    status TEXT DEFAULT 'active',      -- active | superseded | archived
    version INTEGER DEFAULT 1,
    source_ref TEXT,
    last_used_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    text TEXT NOT NULL,
    summary TEXT DEFAULT '',
    keywords TEXT DEFAULT '[]',        -- JSON array
    location_page INTEGER,
    location_section TEXT,
    location_line_start INTEGER,
    location_line_end INTEGER,
    chunk_type TEXT DEFAULT 'paragraph',
    embedding_id TEXT,
    trust_level TEXT DEFAULT 'external_untrusted',
    created_at TEXT NOT NULL
);
"""

TRACES_TABLE = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    parent_trace_id TEXT,
    steps TEXT DEFAULT '[]',           -- JSON array of TraceStep
    created_at TEXT NOT NULL
);
"""

AGENTS_TABLE = """
CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    status TEXT DEFAULT 'created',
    priority INTEGER DEFAULT 5,
    prompt_id TEXT,
    memory_scope TEXT DEFAULT '{}',    -- JSON
    permissions TEXT DEFAULT '{}',     -- JSON
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

TASKS_TABLE = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    agent_id TEXT,
    parent_task_id TEXT,
    task_type TEXT NOT NULL,
    status TEXT DEFAULT 'created',
    dependencies TEXT DEFAULT '[]',    -- JSON array
    input_refs TEXT DEFAULT '[]',      -- JSON array
    output_ref TEXT,
    priority INTEGER DEFAULT 5,
    created_at TEXT NOT NULL
);
"""

# FTS5 virtual table for keyword search on memories
MEMORIES_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id,
    content,
    summary,
    entities,
    content='memories',
    content_rowid='rowid'
);
"""

# FTS5 virtual table for keyword search on chunks
CHUNKS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id,
    text,
    summary,
    keywords,
    content='chunks',
    content_rowid='rowid'
);
"""

# Phase 4: Deep indexing tables
STRUCTURE_NODES_TABLE = """
CREATE TABLE IF NOT EXISTS structure_nodes (
    node_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    name TEXT NOT NULL,
    parent_id TEXT,
    depth INTEGER DEFAULT 0,
    location_page INTEGER,
    location_section TEXT,
    location_line_start INTEGER,
    location_line_end INTEGER,
    chunk_ids TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);
"""

CODE_SYMBOLS_TABLE = """
CREATE TABLE IF NOT EXISTS code_symbols (
    symbol_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    name TEXT NOT NULL,
    symbol_type TEXT NOT NULL,
    language TEXT NOT NULL,
    signature TEXT DEFAULT '',
    body TEXT NOT NULL,
    docstring TEXT DEFAULT '',
    location_line_start INTEGER,
    location_line_end INTEGER,
    parent_symbol_id TEXT,
    chunk_id TEXT,
    created_at TEXT NOT NULL
);
"""

ENTITY_GRAPH_TABLE = """
CREATE TABLE IF NOT EXISTS entity_graph (
    entity_id TEXT PRIMARY KEY,
    entity_name TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    source_chunk_ids TEXT DEFAULT '[]',
    mention_count INTEGER DEFAULT 1,
    created_at TEXT NOT NULL
);
"""

DEPENDENCY_EDGES_TABLE = """
CREATE TABLE IF NOT EXISTS dependency_edges (
    edge_id TEXT PRIMARY KEY,
    source_symbol_id TEXT NOT NULL,
    target_symbol_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    source_file TEXT,
    created_at TEXT NOT NULL
);
"""

ALL_MIGRATIONS = [
    ("memories", MEMORIES_TABLE),
    ("chunks", CHUNKS_TABLE),
    ("traces", TRACES_TABLE),
    ("agents", AGENTS_TABLE),
    ("tasks", TASKS_TABLE),
    ("memories_fts", MEMORIES_FTS),
    ("chunks_fts", CHUNKS_FTS),
    ("structure_nodes", STRUCTURE_NODES_TABLE),
    ("code_symbols", CODE_SYMBOLS_TABLE),
    ("entity_graph", ENTITY_GRAPH_TABLE),
    ("dependency_edges", DEPENDENCY_EDGES_TABLE),
]

VERSIONED_MIGRATIONS = [
    (
        1,
        "add_memories_last_used_at",
        "memories",
        "last_used_at",
        "ALTER TABLE memories ADD COLUMN last_used_at TEXT",
    ),
    (
        2,
        "add_traces_parent_trace_id",
        "traces",
        "parent_trace_id",
        "ALTER TABLE traces ADD COLUMN parent_trace_id TEXT",
    ),
]

# Triggers to keep FTS in sync with base tables
MEMORIES_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, memory_id, content, summary, entities)
    VALUES (new.rowid, new.memory_id, new.content, new.summary, new.entities);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, memory_id, content, summary, entities)
    VALUES ('delete', old.rowid, old.memory_id, old.content, old.summary, old.entities);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, memory_id, content, summary, entities)
    VALUES ('delete', old.rowid, old.memory_id, old.content, old.summary, old.entities);
    INSERT INTO memories_fts(rowid, memory_id, content, summary, entities)
    VALUES (new.rowid, new.memory_id, new.content, new.summary, new.entities);
END;
"""

CHUNKS_FTS_TRIGGERS = """
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
"""
