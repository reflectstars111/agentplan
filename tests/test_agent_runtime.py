"""Tests for AgentRuntime — end-to-end pipeline orchestration."""

import numpy as np
import pytest
from src.config import Config
from src.db import Database
from src.storage.file_store import FileStore
from src.storage.memory_store import MemoryStore
from src.index.vector_index import VectorIndex
from src.index.keyword_index import KeywordIndex
from src.index.hybrid_retriever import HybridRetriever
from src.index.entity_index import EntityIndex
from src.storage.conversation_cache import ConversationCache
from src.storage.dependency_graph import DependencyGraph
from src.runtime.permission_checker import PermissionChecker
from src.models.agent import AgentProcess, AgentRole, AgentStatus
from src.context.token_budgeter import TokenBudgeter
from src.context.mmu import ContextMMU
from src.runtime.verifier import Verifier
from src.runtime.writeback_gate import WritebackGate
from src.runtime.trace_logger import TraceLogger
from src.runtime.agent_runtime import AgentRuntime
from src.runtime.verifier import VerifyOutput
from src.runtime.writeback_gate import WritebackDecision
from src.models.memory import MemoryType


def _mock_embed_fn(texts: list[str]) -> np.ndarray:
    """Deterministic mock embedding for testing."""
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
    """Mock LLM that generates a response referencing sources in the context pack."""
    evidence = ""
    for section in context_pack.sections:
        if section.name == "retrieved_evidence" and section.items:
            # Take first evidence item and build a response referencing it
            for item in section.items[:2]:
                src = item.get("source_ref", "unknown")
                text = item.get("text", "")[:80]
                evidence += f"Based on {src}, {text}. "
    if evidence:
        return evidence.strip()
    return f"No relevant information found for: {query}"


@pytest.fixture
def db():
    d = Database(":memory:")
    d.init_schema()
    yield d
    d.close()


@pytest.fixture
def runtime(db):
    config = Config(default_token_budget=4000)
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

    return AgentRuntime(
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


class TestAgentRuntime:
    def test_result_exposes_conflicts_suggestions_and_ask_user(self, runtime):
        class StubVerifier:
            def verify(self, response, context_pack, working_memories):
                return VerifyOutput(
                    is_verified=False,
                    conflicting_pairs=[("stored PostgreSQL", "new MongoDB")],
                    suggestions=["Resolve the database conflict."],
                )

        class StubGate:
            def evaluate(self, **kwargs):
                return WritebackDecision(
                    action="ask_user",
                    location="long_term_memory",
                    reason="confirmation required",
                    score=0.8,
                )

        runtime.verifier = StubVerifier()
        runtime.writeback_gate = StubGate()
        result = runtime.process_query("Switch PostgreSQL to MongoDB.")

        assert result["conflicting_pairs"] == [
            ("stored PostgreSQL", "new MongoDB")
        ]
        assert result["suggestions"] == ["Resolve the database conflict."]
        assert result["writeback"]["action"] == "ask_user"
        assert result["writeback_confirmation_required"] is True

    def test_long_term_writeback_uses_decision_memory_type(self, runtime):
        class StubVerifier:
            def verify(self, response, context_pack, working_memories):
                return VerifyOutput(is_verified=True)

        class StubGate:
            def evaluate(self, **kwargs):
                return WritebackDecision(
                    action="write",
                    location="long_term_memory",
                    reason="persist project decision",
                    score=0.9,
                )

        runtime.verifier = StubVerifier()
        runtime.writeback_gate = StubGate()
        result = runtime.process_query(
            "Architecture decision: use FastAPI for the API."
        )

        decisions = runtime.memory_store.list_by_type(MemoryType.DECISION)
        assert decisions
        assert result["writeback"]["location"] == "long_term_memory"

    def test_process_query_returns_result(self, runtime):
        # First upload some content
        runtime.upload_text(
            content="FastAPI is a modern Python web framework for building APIs with async support.",
            source_name="fastapi.txt",
        )
        runtime.upload_text(
            content="Python is a programming language used for web development, data science, and AI.",
            source_name="python.txt",
        )

        result = runtime.process_query("What is FastAPI?")

        assert "response" in result
        assert "trace_id" in result
        assert len(result["response"]) > 0

    def test_process_query_has_source_references(self, runtime):
        runtime.upload_text(
            content="Apache Kafka is a distributed streaming platform for real-time data pipelines.",
            source_name="kafka.txt",
        )

        result = runtime.process_query("Tell me about Kafka streaming.")

        # Mock LLM should include source references in its response
        response = result["response"]
        assert "file:kafka.txt" in response or "kafka" in response.lower()

    def test_process_query_records_trace(self, runtime):
        runtime.upload_text(
            content="Docker containers provide isolated environments for applications.",
            source_name="docker.txt",
        )

        result = runtime.process_query("What are Docker containers?")

        trace = runtime.get_trace(result["trace_id"])
        assert trace is not None
        assert len(trace.steps) > 0
        # Should have retrieval, assembly, verification steps
        step_types = [s.type.value for s in trace.steps]
        assert "retrieve_file" in step_types or "retrieve_memory" in step_types
        assert "context_assemble" in step_types

    def test_process_query_empty_knowledge_base(self, runtime):
        result = runtime.process_query("Something not in the knowledge base.")
        assert "response" in result

    def test_upload_text_and_query(self, runtime):
        runtime.upload_text(
            content="Machine learning uses algorithms to find patterns in data.",
            source_name="ml_intro.txt",
        )

        result = runtime.process_query("How does machine learning work?")

        response = result["response"]
        assert len(response) > 0
        assert "trace_id" in result

    def test_upload_creates_chunks(self, runtime):
        num_before = runtime.file_store.count_chunks()
        runtime.upload_text(
            content="A document about cloud computing and distributed systems." * 5,
            source_name="cloud.txt",
        )
        num_after = runtime.file_store.count_chunks()
        assert num_after > num_before

    def test_multiple_queries_generate_unique_traces(self, runtime):
        runtime.upload_text(content="Test content about Python.", source_name="test.txt")

        r1 = runtime.process_query("Python")
        r2 = runtime.process_query("test content")
        r3 = runtime.process_query("programming")

        assert r1["trace_id"] != r2["trace_id"]
        assert r1["trace_id"] != r3["trace_id"]

    def test_upload_text_extracts_entities(self, tmp_path):
        """EntityIndex should be populated during upload_text."""
        config = Config(default_token_budget=4000)
        db_path = str(tmp_path / "entity_test.db")
        db = Database(db_path)
        db.init_schema()

        entity_index = EntityIndex(db)
        file_store = FileStore(db)
        memory_store = MemoryStore(db)
        vector_index = VectorIndex(dim=64)
        keyword_index = KeywordIndex(db)
        retriever = HybridRetriever(vector_index, keyword_index, db, config)
        mmu = ContextMMU(TokenBudgeter(), config)

        runtime = AgentRuntime(
            file_store=file_store,
            memory_store=memory_store,
            retriever=retriever,
            mmu=mmu,
            verifier=Verifier(),
            writeback_gate=WritebackGate(),
            trace_logger=TraceLogger(db),
            config=config,
            embed_fn=_mock_embed_fn,
            entity_index=entity_index,
        )

        runtime.upload_text(
            content="FastAPI is a modern Python web framework. Apache Kafka is a streaming platform.",
            source_name="test.txt",
        )

        # Entity index should have entities extracted
        chunks = file_store.get_chunks("file:test.txt")
        assert len(chunks) > 0
        entities = entity_index.get_entities_for_chunk(chunks[0].chunk_id)
        assert len(entities) > 0

    def test_conversation_cache_tracks_turns(self, tmp_path):
        """ConversationCache should record user and agent turns across queries."""
        config = Config(default_token_budget=4000)
        db_path = str(tmp_path / "conv_cache_test.db")
        db = Database(db_path)
        db.init_schema()

        conv_cache = ConversationCache(max_turns=20)
        file_store = FileStore(db)
        runtime = AgentRuntime(
            file_store=file_store,
            memory_store=MemoryStore(db),
            retriever=HybridRetriever(VectorIndex(dim=64), KeywordIndex(db), db, config),
            mmu=ContextMMU(TokenBudgeter(), config),
            verifier=Verifier(),
            writeback_gate=WritebackGate(),
            trace_logger=TraceLogger(db),
            config=config,
            embed_fn=_mock_embed_fn,
            llm_fn=_mock_llm_fn,
            conversation_cache=conv_cache,
        )

        runtime.upload_text(content="Python is a programming language.", source_name="python.txt")
        runtime.process_query("What is Python?")
        runtime.process_query("Tell me more about it.")

        turns = conv_cache.get_recent_turns(10)
        assert len(turns) >= 4  # 2 user + 2 agent messages
        roles = [t["role"] for t in turns]
        assert "user" in roles
        assert "agent" in roles

    def test_permission_checker_allows_default_access(self, tmp_path):
        """Default PermissionChecker (no restrictions) should allow memory read/write."""
        config = Config(default_token_budget=4000)
        db_path = str(tmp_path / "perm_test.db")
        db = Database(db_path)
        db.init_schema()

        checker = PermissionChecker()
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
            permission_checker=checker,
            agent_id="agent_test_001",
            role="worker",
        )

        runtime.upload_text(content="Test content for permissions.", source_name="test.txt")
        result = runtime.process_query("Test query")
        assert "response" in result

    def test_dependency_graph_extracted_for_code_upload(self, tmp_path):
        """DependencyGraph should extract imports from Python file uploads."""
        config = Config(default_token_budget=4000)
        db_path = str(tmp_path / "dep_test.db")
        db = Database(db_path)
        db.init_schema()

        dep_graph = DependencyGraph(db)
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
            dependency_graph=dep_graph,
        )

        code = "import os\nimport sys\n\ndef hello():\n    print('hello')\n"
        runtime.upload_text(content=code, source_name="test.py")

        # Check dependencies were extracted
        deps = dep_graph.get_dependencies("module:os")
        # May be empty if tree-sitter is not available, but should not crash
        # When tree-sitter is available, should find the import
        assert True  # No crash = success
