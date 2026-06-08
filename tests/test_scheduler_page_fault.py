"""Tests for ContextPageFault integration with AgentRuntime."""
import pytest
import numpy as np
from unittest.mock import patch
from src.db import Database
from src.storage.file_store import FileStore
from src.storage.memory_store import MemoryStore
from src.index.vector_index import VectorIndex
from src.index.keyword_index import KeywordIndex
from src.index.hybrid_retriever import HybridRetriever
from src.context.token_budgeter import TokenBudgeter
from src.context.mmu import ContextMMU
from src.context.page_fault import ContextPageFault
from src.runtime.verifier import Verifier
from src.runtime.writeback_gate import WritebackGate
from src.runtime.trace_logger import TraceLogger
from src.runtime.agent_runtime import AgentRuntime
from src.config import Config


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


@pytest.fixture
def runtime_with_page_fault(tmp_path):
    config = Config(default_token_budget=4000)
    db_path = str(tmp_path / "pf_test.db")
    db = Database(db_path)
    db.init_schema()

    file_store = FileStore(db)
    memory_store = MemoryStore(db)
    vector_index = VectorIndex(dim=64)
    keyword_index = KeywordIndex(db)
    retriever = HybridRetriever(vector_index, keyword_index, db, config)
    mmu = ContextMMU(TokenBudgeter(), config)
    trace_logger = TraceLogger(db)

    page_fault = ContextPageFault(
        retriever=retriever,
        mmu=mmu,
        max_faults=2,
    )

    runtime = AgentRuntime(
        file_store=file_store,
        memory_store=memory_store,
        retriever=retriever,
        mmu=mmu,
        verifier=Verifier(),
        writeback_gate=WritebackGate(),
        trace_logger=trace_logger,
        config=config,
        embed_fn=_mock_embed_fn,
        page_fault=page_fault,
    )
    return runtime


class TestPageFaultInAgentRuntime:

    def test_no_page_fault_falls_back_to_normal(self, runtime_with_page_fault):
        """Without page_fault configured, process_query_with_page_fault == process_query."""
        runtime_with_page_fault.page_fault = None
        result = runtime_with_page_fault.process_query_with_page_fault("what is Python?")
        assert "response" in result
        assert "trace_id" in result

    def test_page_fault_no_trigger_returns_first_response(self, runtime_with_page_fault):
        """When response is confident, page fault should not trigger retry."""
        def mock_llm(ctx, query, model_override=""):
            return "Python is a high-level programming language created by Guido van Rossum."
        runtime_with_page_fault.llm_fn = mock_llm

        result = runtime_with_page_fault.process_query_with_page_fault("what is Python?")
        assert "Python" in result["response"]
        assert "don't have" not in result["response"].lower()

    def test_page_fault_triggers_on_uncertain_response(self, runtime_with_page_fault):
        """When response is uncertain, page fault should trigger re-retrieval."""
        # Add some data so retrieval works during page fault
        runtime_with_page_fault.upload_text(
            "FastAPI is a modern Python web framework for building APIs.",
            "fastapi.txt",
        )

        call_count = [0]

        def mock_llm(ctx, query, model_override=""):
            call_count[0] += 1
            if call_count[0] == 1:
                return "I don't have enough information about this topic to answer properly."
            return "Based on the retrieved documents, FastAPI is a Python web framework."

        runtime_with_page_fault.llm_fn = mock_llm
        candidates = runtime_with_page_fault.retriever.retrieve_and_rerank(
            "what is FastAPI?",
            runtime_with_page_fault.embed_fn,
            k=5,
        )

        with patch.object(
            runtime_with_page_fault.retriever,
            "retrieve_and_rerank",
            return_value=candidates,
        ) as reranked, patch.object(
            runtime_with_page_fault.retriever,
            "retrieve",
            side_effect=AssertionError("page faults must use the reranked path"),
        ):
            result = runtime_with_page_fault.process_query_with_page_fault("what is FastAPI?")
        assert call_count[0] >= 2
        assert reranked.call_count >= 2
        assert "response" in result
