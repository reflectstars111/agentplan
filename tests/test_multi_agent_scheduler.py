"""Tests for Scheduler with multi-agent routing (AgentRegistry + Blackboard)."""

import pytest
import numpy as np
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
from src.runtime.scheduler import Scheduler
from src.runtime.agent_registry import AgentRegistry
from src.models.agent import AgentProcess, AgentRole
from src.models.blackboard import SharedBlackboard, BlackboardEntry
from src.models.task import Task, TaskGraph


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
def base_runtime(tmp_path):
    config = Config(default_token_budget=4000)
    db_path = str(tmp_path / "ma_sched.db")
    db = Database(db_path)
    db.init_schema()
    return AgentRuntime(
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
        agent_id="shared_agent",
        role="worker",
    )


class TestMultiAgentScheduler:
    def test_scheduler_with_registry_routes_correctly(self, base_runtime):
        registry = AgentRegistry()
        registry.register("worker", AgentProcess(agent_id="w1", role=AgentRole.WORKER), base_runtime)
        scheduler = Scheduler(base_runtime, agent_registry=registry)

        graph = TaskGraph(intent_id="test_route")
        graph.add_node(Task(task_id="t1", task_type="retrieve",
                           agent_type="worker", input={"query": "test"}))
        result = scheduler.execute(graph)
        assert result["status"] == "completed"

    def test_scheduler_falls_back_without_registry(self, base_runtime):
        scheduler = Scheduler(base_runtime)  # no registry
        graph = TaskGraph(intent_id="test_fallback")
        graph.add_node(Task(task_id="t1", task_type="retrieve",
                           agent_type="worker", input={"query": "test"}))
        result = scheduler.execute(graph)
        assert result["status"] == "completed"

    def test_scheduler_writes_to_blackboard(self, base_runtime):
        blackboard = SharedBlackboard()
        scheduler = Scheduler(base_runtime, blackboard=blackboard)

        graph = TaskGraph(intent_id="test_bb")
        graph.add_node(Task(task_id="t1", task_type="retrieve",
                           agent_type="worker", input={"query": "BB test"},
                           output_ref="result_key"))
        result = scheduler.execute(graph)
        assert result["status"] == "completed"
        entry = blackboard.read("result_key")
        assert entry is not None
        assert len(entry.value) > 0

    def test_unknown_agent_type_falls_back(self, base_runtime):
        registry = AgentRegistry()
        registry.register("worker", AgentProcess(agent_id="w", role=AgentRole.WORKER), base_runtime)
        scheduler = Scheduler(base_runtime, agent_registry=registry)

        graph = TaskGraph(intent_id="test_unknown")
        graph.add_node(Task(task_id="t1", task_type="retrieve",
                           agent_type="unknown_type", input={"query": "test"}))
        result = scheduler.execute(graph)
        assert result["status"] == "completed"
