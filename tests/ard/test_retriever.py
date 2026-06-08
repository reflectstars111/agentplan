"""Tests for retriever layer."""

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
from src.embedding import create_mock_embed_fn


@pytest.fixture
def hybrid():
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test.db")
    idx_path = os.path.join(tmp, "test.faiss")
    config = Config(db_path=db_path, vector_index_path=idx_path, embedding_dim=1536)
    db = Database(db_path); db.init_schema()
    store = KnowledgeStore(db, VectorIndex(dim=1536, index_path=idx_path),
                           create_mock_embed_fn(dim=1536), config.file_store_path)
    # Ingest test data
    docs = [
        {"text": "ARD hybrid retrieval system combines vector and keyword search. Context MMU manages token budgets.", "source_type": "text", "file_name": "d1.txt", "trust_level": "user_provided_data"},
        {"text": "Traditional RAG uses only vector similarity. This causes low precision for keyword queries.", "source_type": "text", "file_name": "d2.txt", "trust_level": "user_provided_data"},
        {"text": "The scoring formula: 0.35 semantic, 0.20 keyword, 0.15 entity.", "source_type": "text", "file_name": "d3.txt", "trust_level": "user_provided_data"},
    ]
    for doc in docs:
        store.index_chunks([doc], f"src_{uuid.uuid4().hex[:8]}")
    h = HybridRetriever(store, QueryPlanner(), Reranker(config))
    yield h
    db.close()


class TestQueryPlanner:
    def test_default_plan(self):
        qp = QueryPlanner()
        plan = qp.plan("What is machine learning?")
        assert "vector" in plan.strategies
        assert "keyword" in plan.strategies
        assert plan.top_k == 20

    def test_structural_query(self):
        qp = QueryPlanner()
        plan = qp.plan("Where is the main function located?")
        assert "structure" in plan.strategies

    def test_factoid_query(self):
        qp = QueryPlanner()
        plan = qp.plan("What is ARD?")
        assert plan.weights.get("keyword", 0) >= plan.weights.get("vector", 0)


class TestReranker:
    def test_rerank_scores_descending(self):
        from ard.store import RetrievalResult
        reranker = Reranker()
        results = [
            RetrievalResult("c1", "s1", "machine learning neural networks deep AI", score=0.8, strategy="vector"),
            RetrievalResult("c2", "s2", "cooking pasta recipe Italian food", score=0.6, strategy="vector"),
            RetrievalResult("c3", "s3", "deep learning neural network training", score=0.9, strategy="vector"),
        ]
        ranked = reranker.rerank("neural networks", results)
        assert ranked[0].score >= ranked[-1].score

    def test_empty_candidates(self):
        reranker = Reranker()
        assert reranker.rerank("query", []) == []


class TestHybridRetriever:
    def test_retrieve_returns_results(self, hybrid):
        results = hybrid.retrieve("What is hybrid retrieval?")
        assert len(results) > 0
        for r in results:
            assert r.chunk_id
            assert r.text_preview

    def test_retrieve_all_results_have_scores(self, hybrid):
        results = hybrid.retrieve("scoring formula")
        for r in results:
            assert r.score != 0.0
