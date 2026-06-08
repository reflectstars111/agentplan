"""Tests for relevance-scoped L2/L3 memory retrieval."""

from src.db import Database
from src.index.keyword_index import KeywordIndex
from src.index.memory_retriever import MemoryRetriever
from src.models.memory import MemoryItem, MemoryStatus, MemoryType
from src.storage.memory_store import MemoryStore


def test_memory_retriever_filters_scope_status_and_splits_levels(tmp_path):
    db = Database(str(tmp_path / "memory.db"))
    db.init_schema()
    store = MemoryStore(db)
    store.insert(
        MemoryItem(
            memory_id="working",
            type=MemoryType.PROJECT_STATE,
            content="FastAPI serves the project API.",
            scope="project",
            importance=0.8,
        )
    )
    store.insert(
        MemoryItem(
            memory_id="long-term",
            type=MemoryType.DECISION,
            content="The architecture decision selected FastAPI.",
            scope="project",
            importance=0.9,
        )
    )
    store.insert(
        MemoryItem(
            memory_id="wrong-scope",
            type=MemoryType.USER_PREFERENCE,
            content="The user prefers FastAPI examples.",
            scope="user",
        )
    )
    store.insert(
        MemoryItem(
            memory_id="archived",
            type=MemoryType.DECISION,
            content="FastAPI was considered in an old draft.",
            scope="project",
            status=MemoryStatus.ARCHIVED,
        )
    )
    store.insert(
        MemoryItem(
            memory_id="irrelevant",
            type=MemoryType.PROJECT_STATE,
            content="Kafka handles event streaming.",
            scope="project",
        )
    )

    selection = MemoryRetriever(store, KeywordIndex(db)).retrieve(
        "FastAPI architecture",
        scopes=["project"],
        limit=5,
    )

    assert [item.memory_id for item in selection.working] == ["working"]
    assert [item.memory_id for item in selection.long_term] == ["long-term"]
    assert store.get("working").last_used_at is not None
    assert store.get("long-term").last_used_at is not None
    assert store.get("wrong-scope").last_used_at is None
    assert store.get("archived").last_used_at is None
    assert store.get("irrelevant").last_used_at is None


def test_agent_runtime_context_contains_only_relevant_memories(tmp_path):
    import numpy as np

    from src.config import Config
    from src.context.mmu import ContextMMU
    from src.context.token_budgeter import TokenBudgeter
    from src.index.hybrid_retriever import HybridRetriever
    from src.index.vector_index import VectorIndex
    from src.runtime.agent_runtime import AgentRuntime
    from src.runtime.trace_logger import TraceLogger
    from src.runtime.verifier import Verifier
    from src.runtime.writeback_gate import WritebackGate
    from src.storage.file_store import FileStore

    config = Config(
        db_path=str(tmp_path / "runtime.db"),
        file_store_path=str(tmp_path / "files"),
        vector_index_path=str(tmp_path / "vector.faiss"),
        embedding_dim=8,
        default_token_budget=4000,
    )
    db = Database(config.db_path)
    db.init_schema()
    store = MemoryStore(db)
    store.insert(
        MemoryItem(
            memory_id="relevant",
            type=MemoryType.PROJECT_STATE,
            content="FastAPI is the selected API framework.",
        )
    )
    store.insert(
        MemoryItem(
            memory_id="noise",
            type=MemoryType.PROJECT_STATE,
            content="Kafka retention is seven days.",
        )
    )
    captured = {}

    def llm(context_pack, query):
        captured["pack"] = context_pack
        return "FastAPI is selected."

    runtime = AgentRuntime(
        file_store=FileStore(db, config.file_store_path),
        memory_store=store,
        retriever=HybridRetriever(
            VectorIndex(dim=8),
            KeywordIndex(db),
            db,
            config,
        ),
        mmu=ContextMMU(TokenBudgeter(), config),
        verifier=Verifier(),
        writeback_gate=WritebackGate(),
        trace_logger=TraceLogger(db),
        config=config,
        embed_fn=lambda texts: np.zeros((len(texts), 8), dtype=np.float32),
        llm_fn=llm,
    )

    runtime.process_query("Which FastAPI framework did we select?")
    memory_text = " ".join(
        item["text"]
        for section in captured["pack"].sections
        if section.name in {"working_memory", "long_term_memory"}
        for item in section.items
    )
    assert "FastAPI" in memory_text
    assert "Kafka" not in memory_text
    assert captured["pack"].memory_ids == ["relevant"]
