"""Regression tests for database and retrieval persistence."""

import sqlite3

from src.config import Config
from src.db.connection import Database
from src.embedding import create_mock_embed_fn
from src.index.hybrid_retriever import HybridRetriever
from src.index.keyword_index import KeywordIndex
from src.index.vector_index import VectorIndex
from src.models.memory import MemoryItem, MemoryType
from src.storage.file_store import FileStore
from src.storage.memory_store import MemoryStore


def test_init_schema_migrates_legacy_memories_table(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE memories (
            memory_id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            summary TEXT DEFAULT '',
            entities TEXT DEFAULT '[]',
            importance REAL DEFAULT 0.5,
            confidence REAL DEFAULT 0.5,
            source TEXT DEFAULT 'conversation',
            scope TEXT DEFAULT 'project',
            status TEXT DEFAULT 'active',
            version INTEGER DEFAULT 1,
            source_ref TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    db = Database(str(db_path))
    db.init_schema()
    columns = {
        row["name"] for row in db.execute("PRAGMA table_info(memories)").fetchall()
    }
    assert "last_used_at" in columns

    store = MemoryStore(db)
    store.insert(
        MemoryItem(
            memory_id="legacy-upgrade",
            type=MemoryType.PROJECT_STATE,
            content="legacy database remains writable",
        )
    )
    assert store.get("legacy-upgrade") is not None
    applied = db.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()
    assert [row["version"] for row in applied] == [1, 2]


def test_keyword_retrieval_works_with_empty_vector_index(tmp_path):
    db = Database(str(tmp_path / "keyword.db"))
    db.init_schema()
    store = FileStore(db, str(tmp_path / "files"))
    store.ingest_text(
        "Restart persistence keeps historical documents searchable.",
        "restart.txt",
    )
    retriever = HybridRetriever(
        VectorIndex(dim=64),
        KeywordIndex(db),
        db,
        Config(
            db_path=str(tmp_path / "keyword.db"),
            file_store_path=str(tmp_path / "files"),
            vector_index_path=str(tmp_path / "missing.faiss"),
            embedding_dim=64,
        ),
    )

    def fail_if_embedded(_texts):
        raise AssertionError("empty vector indexes should not require embedding")

    results = retriever.retrieve("restart persistence", fail_if_embedded, k=5)
    assert results
    assert results[0].source_ref == "file:restart.txt"
    assert results[0].score_breakdown["keyword"] > 0


def test_vector_index_rebuilds_from_chunks_then_loads_on_restart(tmp_path):
    from src.__main__ import initialize_vector_index

    config = Config(
        db_path=str(tmp_path / "agent.db"),
        file_store_path=str(tmp_path / "files"),
        vector_index_path=str(tmp_path / "vector.faiss"),
        embedding_dim=64,
    )
    db = Database(config.db_path)
    db.init_schema()
    file_store = FileStore(db, config.file_store_path)
    file_store.ingest_text(
        "Historical chunks survive an application restart.",
        "history.txt",
    )
    embed_fn = create_mock_embed_fn(dim=64)

    rebuilt = initialize_vector_index(config, file_store, embed_fn)
    assert rebuilt.count == file_store.count_chunks()
    assert (tmp_path / "vector.faiss").exists()
    assert (tmp_path / "vector.faiss.meta.json").exists()

    def fail_if_reembedded(_texts):
        raise AssertionError("persisted indexes should load without rebuilding")

    loaded = initialize_vector_index(config, file_store, fail_if_reembedded)
    assert loaded.count == rebuilt.count


def test_build_runtime_wires_retrieval_components_and_indexes_structure(
    tmp_path, monkeypatch
):
    from src.__main__ import build_runtime
    from src.index.query_planner import QueryPlanner
    from src.index.reranker import Reranker
    from src.index.structure_index import StructureIndex

    monkeypatch.chdir(tmp_path)
    runtime = build_runtime(embed_provider="mock")

    assert isinstance(runtime.retriever.query_planner, QueryPlanner)
    assert isinstance(runtime.retriever.reranker, Reranker)
    assert isinstance(runtime.retriever.structure_index, StructureIndex)

    source_id = runtime.upload_text(
        "# Architecture\nThe scheduler executes a task graph.",
        "architecture.md",
    )
    nodes = runtime.retriever.structure_index.get_by_source(source_id)
    assert nodes
    assert source_id in {node.source_id for node in nodes}


def test_build_controller_registers_distinct_worker_and_verifier(
    tmp_path, monkeypatch
):
    from src.__main__ import build_controller, build_runtime

    monkeypatch.chdir(tmp_path)
    runtime = build_runtime(embed_provider="mock")
    controller = build_controller(runtime)

    registry = controller.agent_registry
    worker = registry.get_runtime("worker")
    verifier = registry.get_runtime("verifier")
    assert worker is not None
    assert verifier is not None
    assert worker is not verifier
    assert worker.agent_id != verifier.agent_id
    assert controller.scheduler.agent_registry is registry
    assert controller.scheduler.blackboard is controller.blackboard


def test_build_runtime_wires_security_and_audit_components(
    tmp_path, monkeypatch
):
    from src.__main__ import build_runtime
    from src.runtime.audit_log import AuditLog
    from src.runtime.input_sanitizer import InputSanitizer
    from src.runtime.tool_router import ToolRouter

    monkeypatch.chdir(tmp_path)
    runtime = build_runtime(embed_provider="mock")

    assert isinstance(runtime.input_sanitizer, InputSanitizer)
    assert isinstance(runtime.audit_log, AuditLog)
    assert isinstance(runtime.tool_router, ToolRouter)
