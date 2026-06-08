"""Tests for FastAPI endpoints."""

import io
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
from src.api.main import create_app


def _mock_embed_fn(texts: list[str]) -> np.ndarray:
    dim = 64
    result = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for j, ch in enumerate(text):
            result[i, j % dim] += (ord(ch) / 256.0)
        norm = np.linalg.norm(result[i])
        if norm > 0:
            result[i] /= norm
    return result


def _mock_llm_fn(context_pack, query: str) -> str:
    for section in context_pack.sections:
        if section.name == "retrieved_evidence" and section.items:
            parts = []
            for item in section.items[:2]:
                src = item.get("source_ref", "unknown")
                text = item.get("text", "")
                if text:
                    parts.append(f"According to {src}: {text}")
            if parts:
                return " ".join(parts)
    return f"No information found for: {query}"


@pytest.fixture
def app(tmp_path):
    """Create a test FastAPI app with temp-file DB (thread-safe for FastAPI TestClient)."""
    config = Config(default_token_budget=4000)
    db_path = str(tmp_path / "test_api.db")
    db = Database(db_path)
    db.init_schema()

    file_store = FileStore(db)
    memory_store = MemoryStore(db)
    vector_index = VectorIndex(dim=64)
    keyword_index = KeywordIndex(db)
    retriever = HybridRetriever(vector_index, keyword_index, db, config)
    budgeter = TokenBudgeter()
    mmu = ContextMMU(budgeter, config)
    verifier = Verifier()
    gate = WritebackGate()
    logger = TraceLogger(db)

    runtime = AgentRuntime(
        file_store=file_store,
        memory_store=memory_store,
        retriever=retriever,
        mmu=mmu,
        verifier=verifier,
        writeback_gate=gate,
        trace_logger=logger,
        config=config,
        embed_fn=_mock_embed_fn,
        llm_fn=_mock_llm_fn,
    )

    return create_app(runtime)


@pytest.fixture
def client(app):
    return TestClient(app)


class TestUploadEndpoint:
    def test_upload_text(self, client):
        resp = client.post("/upload", json={
            "content": "FastAPI is a Python web framework for building APIs.",
            "source_name": "fastapi.txt",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "source_id" in data
        assert data["chunks_created"] > 0

    def test_upload_empty_content(self, client):
        resp = client.post("/upload", json={
            "content": "",
            "source_name": "empty.txt",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunks_created"] == 0

    def test_upload_api_builds_untrusted_vector_index(self, client, monkeypatch):
        import io

        monkeypatch.setattr(
            "src.sources.api_source.urlopen",
            lambda req, timeout=30: io.BytesIO(
                b'{"framework": "FastAPI", "purpose": "API"}'
            ),
        )
        before = client.app.state.runtime.retriever.vector_index.count
        response = client.post(
            "/upload/api",
            json={
                "url": "https://api.example.com/framework",
                "source_name": "external_api",
            },
        )

        assert response.status_code == 200
        source_id = response.json()["source_id"]
        chunks = client.app.state.runtime.file_store.get_chunks(source_id)
        assert chunks
        assert all(
            chunk.trust_level.value == "external_untrusted"
            for chunk in chunks
        )
        assert client.app.state.runtime.retriever.vector_index.count > before


class TestQueryEndpoint:
    def test_query_returns_response(self, client):
        # Upload first
        client.post("/upload", json={
            "content": "Python is a programming language for web development and data science.",
            "source_name": "python.txt",
        })

        resp = client.post("/query", json={"query": "What is Python?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "trace_id" in data
        assert "conflicting_pairs" in data
        assert "suggestions" in data
        assert "writeback" in data
        assert "writeback_confirmation_required" in data
        assert len(data["response"]) > 0

    def test_query_with_empty_knowledge_base(self, client):
        resp = client.post("/query", json={"query": "Something unknown?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data

    def test_query_returns_trace_id(self, client):
        client.post("/upload", json={
            "content": "Docker containers provide isolated application environments.",
            "source_name": "docker.txt",
        })
        resp = client.post("/query", json={"query": "Docker containers"})
        data = resp.json()
        assert data["trace_id"].startswith("trace_")


class TestTraceEndpoint:
    def test_get_trace(self, client):
        # Create a trace by querying
        client.post("/upload", json={
            "content": "Machine learning uses algorithms to learn patterns from data.",
            "source_name": "ml.txt",
        })
        query_resp = client.post("/query", json={"query": "machine learning"})
        trace_id = query_resp.json()["trace_id"]

        # Retrieve the trace
        resp = client.get(f"/trace/{trace_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trace_id"] == trace_id
        assert len(data["steps"]) > 0

    def test_get_nonexistent_trace(self, client):
        resp = client.get("/trace/nonexistent")
        assert resp.status_code == 404


class TestHealthEndpoint:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
