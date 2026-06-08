"""Tests for Scheduler."""

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
from src.models.task import Task, TaskStatus, TaskGraph


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
def runtime(tmp_path):
    config = Config(default_token_budget=4000)
    db_path = str(tmp_path / "sched_test.db")
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
    )


@pytest.fixture
def scheduler(runtime):
    return Scheduler(runtime)


def _make_linear_graph(n: int = 3) -> TaskGraph:
    g = TaskGraph(intent_id="test")
    for i in range(n):
        g.add_node(Task(task_id=f"task_{i}", task_type="general",
                        input={"query": f"Step {i}"}))
    for i in range(n - 1):
        g.add_edge(f"task_{i}", f"task_{i+1}")
    return g


class TestScheduler:
    def test_execute_linear_graph(self, scheduler):
        graph = _make_linear_graph(3)
        result = scheduler.execute(graph)
        assert result["status"] == "completed"
        assert len(result["results"]) == 3

    def test_execute_single_node(self, scheduler):
        graph = _make_linear_graph(1)
        result = scheduler.execute(graph)
        assert result["status"] == "completed"

    def test_execute_empty_graph(self, scheduler):
        graph = TaskGraph(intent_id="empty")
        result = scheduler.execute(graph)
        assert result["status"] == "completed"
        assert len(result["results"]) == 0

    def test_nodes_marked_completed(self, scheduler):
        graph = _make_linear_graph(3)
        scheduler.execute(graph)
        for tid in ["task_0", "task_1", "task_2"]:
            assert graph.get_node(tid).status == TaskStatus.COMPLETED

    def test_trace_ids_returned(self, scheduler):
        graph = _make_linear_graph(2)
        result = scheduler.execute(graph)
        assert len(result["trace_ids"]) == 2

    def test_execute_returns_results_dict(self, scheduler):
        graph = _make_linear_graph(2)
        result = scheduler.execute(graph)
        assert "task_0" in result["results"]
        assert "response" in result["results"]["task_0"]

    def test_downstream_nodes_receive_dependency_outputs(self):
        class RecordingRuntime:
            def __init__(self):
                self.queries = []

            def process_query(self, query, request_id=None):
                self.queries.append((request_id, query))
                return {
                    "response": {
                        "retrieve": "retrieved evidence",
                        "reason": "reasoned answer",
                        "verify": "verified answer",
                    }[request_id],
                    "trace_id": f"trace_{request_id}",
                    "verified": request_id == "verify",
                }

        runtime = RecordingRuntime()
        scheduler = Scheduler(runtime)
        graph = TaskGraph(intent_id="dataflow")
        retrieve = Task(
            task_id="retrieve",
            task_type="retrieve",
            input={"query": "original question", "task": "find evidence"},
            output_ref="retrieval.output",
        )
        reason = Task(
            task_id="reason",
            task_type="reason",
            input={"query": "original question", "task": "answer from evidence"},
            input_refs=["retrieval.output"],
            output_ref="reason.output",
        )
        verify = Task(
            task_id="verify",
            task_type="verify",
            input={"query": "original question", "task": "check the answer"},
            input_refs=["reason.output"],
            output_ref="verify.output",
        )
        for task in (retrieve, reason, verify):
            graph.add_node(task)
        graph.add_edge("retrieve", "reason")
        graph.add_edge("reason", "verify")

        scheduler.execute(graph)
        queries = dict(runtime.queries)
        assert "retrieved evidence" in queries["reason"]
        assert "reasoned answer" in queries["verify"]
        assert queries["retrieve"] != queries["reason"]
        assert queries["reason"] != queries["verify"]
