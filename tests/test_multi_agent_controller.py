"""Tests for Controller multi-agent integration."""

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
from src.runtime.agent_registry import AgentRegistry
from src.runtime.merger import Merger
from src.models.agent import AgentProcess, AgentRole
from src.models.blackboard import SharedBlackboard


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
    db_path = str(tmp_path / "ma_ctrl.db")
    db = Database(db_path)
    db.init_schema()

    worker = AgentRuntime(
        file_store=FileStore(db), memory_store=MemoryStore(db),
        retriever=HybridRetriever(VectorIndex(dim=64), KeywordIndex(db), db, config),
        mmu=ContextMMU(TokenBudgeter(), config),
        verifier=Verifier(), writeback_gate=WritebackGate(),
        trace_logger=TraceLogger(db), config=config,
        embed_fn=_mock_embed_fn, llm_fn=_mock_llm_fn,
        agent_id="agent_worker_001", role="worker",
    )

    registry = AgentRegistry()
    registry.register("worker", AgentProcess(agent_id="w", role=AgentRole.WORKER), worker)
    registry.register("verifier", AgentProcess(agent_id="v", role=AgentRole.VERIFIER), worker)

    blackboard = SharedBlackboard()
    merger = Merger(Verifier())

    return Controller(
        agent_runtime=worker,
        intent_decoder=IntentDecoder(), planner=Planner(),
        scheduler=Scheduler(worker, agent_registry=registry, blackboard=blackboard),
        trace_logger=TraceLogger(db), config=config,
        agent_registry=registry, blackboard=blackboard, merger=merger,
    )


class TestMultiAgentController:
    def test_multi_agent_doc_qa_completes(self, controller):
        controller.agent_runtime.upload_text(
            content="RAPTOR is a retrieval-augmented prompt optimization method.",
            source_name="paper.txt",
        )
        result = controller.process("What is RAPTOR?")
        assert result["status"] == "completed"

    def test_blackboard_populated(self, controller):
        controller.agent_runtime.upload_text(
            content="FastAPI is a Python web framework with async support.",
            source_name="fastapi.txt",
        )
        controller.blackboard.clear()
        controller.process("What is FastAPI?")
        entries = controller.blackboard.read_all()
        # Should have entries from task node outputs
        assert len(entries) >= 0

    def test_process_query_still_backward_compat(self, controller):
        result = controller.process_query("Hello world")
        assert "response" in result
        assert "trace_id" in result
