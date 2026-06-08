"""Tests for Controller — full ARD execution cycle."""

import os
import tempfile
import uuid

import pytest

from ard.infra.config import Config
from ard.infra.db import Database
from ard.store.knowledge_store import KnowledgeStore
from ard.store.event_store import EventStore
from ard.store.projections import Projections
from ard.store.state_store import StateStore
from ard.store.trace_store import TraceStore
from ard.store.transaction import TransactionManager
from ard.retriever.vector_index import VectorIndex
from ard.retriever.reranker import Reranker
from ard.retriever.query_planner import QueryPlanner
from ard.retriever.hybrid import HybridRetriever
from ard.context.token_budgeter import TokenBudgeter
from ard.context.mmu import ContextMMU
from ard.runtime.executor import Executor
from ard.runtime.controller import Controller
from src.embedding import create_mock_embed_fn


@pytest.fixture
def controller():
    """Build a full Controller with all Phase 1+Phase 2 components."""
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test.db")
    idx_path = os.path.join(tmp, "test.faiss")
    config = Config(db_path=db_path, vector_index_path=idx_path, embedding_dim=1536)

    db = Database(db_path)
    db.init_schema()

    embed_fn = create_mock_embed_fn(dim=1536)
    vi = VectorIndex(dim=1536, index_path=idx_path)
    ks = KnowledgeStore(db, vi, embed_fn, config.file_store_path)

    # Ingest some test data
    docs = [
        {"text": "ARD uses hybrid retrieval with vector and keyword search. The scoring formula: 0.35 semantic.", "source_type": "text", "file_name": "doc.txt", "trust_level": "user_provided_data"},
    ]
    ks.index_chunks(docs, f"src_{uuid.uuid4().hex[:8]}")

    # Phase 2 stack
    proj = Projections()
    es = EventStore(db, proj)
    ss = StateStore(es)
    proj.register("state.created", ss.apply_event)
    proj.register("state.updated", ss.apply_event)
    proj.register("state.archived", ss.apply_event)
    proj.register("state.deleted", ss.apply_event)

    ts = TraceStore(es)
    txn_mgr = TransactionManager(es)

    # Phase 1 stack
    reranker = Reranker(config)
    hybrid = HybridRetriever(ks, QueryPlanner(), reranker)
    mmu = ContextMMU(TokenBudgeter(config), config)
    executor = Executor()

    ctrl = Controller(ss, ts, txn_mgr, hybrid, mmu, executor)
    yield ctrl
    db.close()


class TestController:
    def test_process_returns_response(self, controller):
        result = controller.process("What is ARD?")
        assert "response" in result
        assert "trace_id" in result
        assert "verdict" in result
        assert result["trace_id"].startswith("trace_")

    def test_trace_recorded(self, controller):
        result = controller.process("Tell me about hybrid retrieval")
        steps = controller.trace_store.query(result["trace_id"])
        step_types = [s["step_type"] for s in steps]
        assert "plan" in step_types
        assert "retrieve" in step_types
        assert "execute" in step_types
        assert "verify" in step_types
        assert "writeback" in step_types

    def test_state_written(self, controller):
        result = controller.process("What is the scoring formula?")
        assert "state_keys" in result

    def test_writeback_happens(self, controller):
        result = controller.process("Analyze the ARD system")
        assert result["writeback"]["action"] in ("committed", "rolled_back")

    def test_verdict_in_result(self, controller):
        result = controller.process("What retrieval methods does ARD use?")
        verdict = result["verdict"]
        assert "verified" in verdict
        assert "confidence" in verdict
        assert isinstance(verdict["confidence"], (float, int))

    def test_sources_in_result(self, controller):
        result = controller.process("Tell me about ARD")
        assert isinstance(result["sources"], list)
