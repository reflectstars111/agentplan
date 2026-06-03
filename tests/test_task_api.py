"""Tests for /task API endpoints."""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from src.config import Config
from src.db import Database
from src.storage.file_store import FileStore
from src.storage.memory_store import MemoryStore
from src.index.vector_index import VectorIndex
from src.index.keyword_index import KeywordIndex
from src.index.hybrid_retriever import HybridRetriever
from src.context.token_budgeter import TokenBudgeter
from src.context.mmu import ContextMMU
from src.runtime.verifier import Verifier
from src.runtime.writeback_gate import WritebackGate
from src.runtime.trace_logger import TraceLogger
from src.runtime.agent_runtime import AgentRuntime
from src.runtime.intent_decoder import IntentDecoder
from src.runtime.planner import Planner
from src.runtime.scheduler import Scheduler
from src.runtime.controller import Controller
from src.api.main import create_app
from src.api.task_routes import create_task_router


def _mock_embed_fn(texts):
    dim = 64
    result = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for j, ch in enumerate(text):
            result[i, j % dim] += (ord(ch) / 256.0)
        norm = np.linalg.norm(result[i])
        if norm > 0:
            result[i] /= norm
    return result


def _mock_llm_fn(context_pack, query):
    for s in context_pack.sections:
        if s.name == "retrieved_evidence" and s.items:
            parts = []
            for item in s.items[:2]:
                src = item.get("source_ref", "unknown")
                text = item.get("text", "")
                if text:
                    parts.append(f"According to {src}: {text}")
            if parts:
                return " ".join(parts)
    return f"No info for: {query}"


@pytest.fixture
def client(tmp_path):
    config = Config(default_token_budget=4000)
    db_path = str(tmp_path / "task_api.db")
    db = Database(db_path)
    db.init_schema()

    runtime = AgentRuntime(
        file_store=FileStore(db),
        memory_store=MemoryStore(db),
        retriever=HybridRetriever(VectorIndex(dim=64), KeywordIndex(db), db, config),
        mmu=ContextMMU(TokenBudgeter(), config),
        verifier=Verifier(),
        writeback_gate=WritebackGate(),
        trace_logger=TraceLogger(db),
        config=config,
        embed_fn=_mock_embed_fn,
        llm_fn=_mock_llm_fn,
    )
    controller = Controller(
        agent_runtime=runtime,
        intent_decoder=IntentDecoder(),
        planner=Planner(),
        scheduler=Scheduler(runtime),
        trace_logger=TraceLogger(db),
        config=config,
    )

    app = create_app(runtime)
    task_router = create_task_router(controller)
    app.include_router(task_router)
    return TestClient(app)


class TestTaskEndpoint:
    def test_post_task_returns_200(self, client):
        resp = client.post("/task", json={"query": "What is Python?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "intent" in data

    def test_post_task_includes_task_graph_summary(self, client):
        resp = client.post("/task", json={"query": "Explain the RAPTOR algorithm."})
        data = resp.json()
        assert "task_graph_summary" in data
        assert data["task_graph_summary"]["node_count"] >= 1

    def test_post_task_includes_status(self, client):
        resp = client.post("/task", json={"query": "Hello"})
        data = resp.json()
        assert data["status"] in ("completed", "partial_failure")

    def test_query_endpoint_still_works(self, client):
        """/query should still work (backward compat)."""
        resp = client.post("/query", json={"query": "Hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "trace_id" in data

    def test_health_endpoint_still_works(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
