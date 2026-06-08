"""Tests for eval benchmark framework."""

import os
import tempfile
import uuid

import pytest

from ard.infra.config import Config
from ard.infra.db import Database
from ard.store.knowledge_store import KnowledgeStore
from ard.retriever.vector_index import VectorIndex
from ard.retriever.reranker import Reranker
from ard.retriever.query_planner import QueryPlanner
from ard.retriever.hybrid import HybridRetriever
from ard.context.token_budgeter import TokenBudgeter
from ard.context.mmu import ContextMMU
from ard.runtime.executor import Executor
from ard.eval.benchmark import (
    Benchmark, EvalQuery, EvalResult, BenchmarkReport, generate_sample_queries,
)


class TestEvalQuery:
    def test_create(self):
        q = EvalQuery(
            query_id="q1", query="What is X?",
            expected_keywords=["X", "definition"],
            relevant_chunk_ids=["c1"],
            category="factoid",
        )
        assert q.query_id == "q1"
        assert q.category == "factoid"


class TestBenchmarkReport:
    def test_summary_string(self):
        r = BenchmarkReport(
            condition="baseline1", total_queries=10,
            avg_latency_ms=50.0, avg_tokens_input=200.0,
            avg_precision=0.8, avg_recall=0.7, avg_mrr=0.75,
            avg_keyword_recall=0.6, avg_token_efficiency=0.3,
        )
        s = r.summary()
        assert "baseline1" in s
        assert "10" in s


class TestGenerateSampleQueries:
    def test_returns_list(self):
        queries = generate_sample_queries()
        assert len(queries) > 0
        for q in queries:
            assert isinstance(q, EvalQuery)
            assert q.query_id
            assert q.query


class TestBenchmark:
    @pytest.fixture
    def bench(self):
        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, "test.db")
        idx_path = os.path.join(tmp, "test.faiss")
        config = Config(db_path=db_path, vector_index_path=idx_path, embedding_dim=1536)
        db = Database(db_path); db.init_schema()
        from src.embedding import create_mock_embed_fn
        embed_fn = create_mock_embed_fn(dim=1536)
        store = KnowledgeStore(db, VectorIndex(dim=1536, index_path=idx_path),
                               embed_fn, config.file_store_path)

        # Ingest test data
        docs = [
            {"text": "ARD uses hybrid retrieval. Context MMU assembles results into token-budgeted context packs.", "source_type": "text", "file_name": "d1.txt", "trust_level": "user_provided_data"},
            {"text": "Traditional RAG uses vector similarity only. ARD improves with multi-strategy retrieval.", "source_type": "text", "file_name": "d2.txt", "trust_level": "user_provided_data"},
        ]
        for doc in docs:
            store.index_chunks([doc], f"src_{uuid.uuid4().hex[:8]}")

        r = Reranker(config)
        hybrid = HybridRetriever(store, QueryPlanner(), r)
        mmu = ContextMMU(TokenBudgeter(config), config)
        executor = Executor()

        b = Benchmark(store, hybrid, mmu, executor, config)
        yield b
        db.close()

    def test_run_all_conditions(self, bench):
        queries = generate_sample_queries()[:2]
        reports = bench.run_single_turn(queries)
        assert "baseline1" in reports
        assert "baseline2" in reports
        assert "proposed" in reports
        for name, report in reports.items():
            assert report.total_queries == 2
            assert report.avg_latency_ms >= 0

    def test_reports_contain_per_query_data(self, bench):
        queries = [EvalQuery(query_id="q1", query="What is ARD?",
                              expected_keywords=["ARD", "retrieval"],
                              category="conceptual")]
        reports = bench.run_single_turn(queries)
        for report in reports.values():
            assert len(report.per_query) == 1
            assert report.per_query[0]["query_id"] == "q1"
