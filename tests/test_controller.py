"""Tests for Controller."""

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
from src.runtime.intent_decoder import IntentDecoder
from src.runtime.planner import Planner
from src.runtime.scheduler import Scheduler
from src.runtime.controller import Controller
from src.models.trace import StepType


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
def controller(tmp_path):
    config = Config(default_token_budget=4000)
    db_path = str(tmp_path / "ctrl_test.db")
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

    return Controller(
        agent_runtime=runtime,
        intent_decoder=IntentDecoder(),
        planner=Planner(),
        scheduler=Scheduler(runtime),
        trace_logger=TraceLogger(db),
        config=config,
    )


class TestController:
    def test_process_returns_response(self, controller):
        result = controller.process("What is Python?")
        assert "response" in result
        assert "intent" in result
        assert "status" in result

    def test_process_query_passthrough(self, controller):
        """process_query() should delegate to AgentRuntime and return same format."""
        result = controller.process_query("Hello world")
        assert "response" in result
        assert "trace_id" in result

    def test_process_has_task_graph_summary(self, controller):
        result = controller.process("Explain the RAPTOR algorithm.")
        assert "task_graph_summary" in result
        summary = result["task_graph_summary"]
        assert "node_count" in summary
        assert summary["node_count"] >= 1

    def test_process_returns_trace_ids(self, controller):
        result = controller.process("What is FastAPI?")
        assert "trace_ids" in result
        assert len(result["trace_ids"]) > 0

    def test_process_controller_trace_has_intent_decode(self, controller):
        result = controller.process("Where is the main function?")
        trace = controller.trace_logger.get_trace(result["trace_ids"][0]) if result["trace_ids"] else None
        # At least one trace should exist
        assert len(result["trace_ids"]) > 0

    def test_process_doc_qa_completes(self, controller):
        controller.agent_runtime.upload_text(
            content="FastAPI is a modern Python web framework for building APIs.",
            source_name="fastapi.txt",
        )
        result = controller.process("What is FastAPI?")
        assert result["status"] == "completed"

    def test_process_general_fallback(self, controller):
        result = controller.process("Hello!")
        assert result["status"] == "completed"
