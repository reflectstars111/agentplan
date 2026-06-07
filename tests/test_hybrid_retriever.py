"""Tests for HybridRetriever."""

import numpy as np
import pytest
from src.db import Database
from src.storage.file_store import FileStore
from src.index.vector_index import VectorIndex
from src.index.keyword_index import KeywordIndex
from src.index.hybrid_retriever import HybridRetriever


def _mock_embed_fn(texts: list[str]) -> np.ndarray:
    """Deterministic mock embedding: hash each word's characters to build a vector."""
    dim = 64
    result = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        # Simple hash-based embedding for testing
        for j, ch in enumerate(text):
            result[i, j % dim] += (ord(ch) / 256.0)
        # Normalize
        norm = np.linalg.norm(result[i])
        if norm > 0:
            result[i] /= norm
    return result


@pytest.fixture
def db():
    d = Database(":memory:")
    d.init_schema()
    yield d
    d.close()


@pytest.fixture
def file_store(db):
    return FileStore(db)


@pytest.fixture
def vector_index():
    return VectorIndex(dim=64)


@pytest.fixture
def keyword_index(db):
    return KeywordIndex(db)


@pytest.fixture
def populated_store(file_store, vector_index):
    """Insert chunks into both file_store and vector_index."""
    texts = [
        ("The FastAPI framework provides excellent async support for building Python web APIs.", "fastapi.txt"),
        ("Apache Kafka is a distributed streaming platform used for building real-time data pipelines.", "kafka.txt"),
        ("Machine learning models require careful feature engineering and hyperparameter tuning.", "ml.txt"),
        ("Docker containers provide isolated environments for running applications consistently.", "docker.txt"),
    ]
    for text, name in texts:
        chunks = file_store.ingest_text(content=text, source_name=name)
        # Also index in vector store
        for chunk in file_store.get_chunks(chunks):
            emb = _mock_embed_fn([chunk.text])[0]
            vector_index.add(chunk.chunk_id, emb)
    return file_store


@pytest.fixture
def retriever(vector_index, keyword_index, db):
    from src.config import config
    return HybridRetriever(vector_index, keyword_index, db, config)


class TestHybridRetriever:
    def test_retrieve_finds_results(self, populated_store, retriever):
        results = retriever.retrieve(
            query="Python async API framework",
            embed_fn=_mock_embed_fn,
            k=5,
        )
        assert len(results) > 0
        # FastAPI chunk should appear near the top
        top_chunks = [r.chunk_id for r in results]
        assert any("fastapi" in cid for cid in top_chunks)

    def test_retrieve_results_have_required_fields(self, populated_store, retriever):
        results = retriever.retrieve(
            query="streaming data pipelines",
            embed_fn=_mock_embed_fn,
            k=3,
        )
        for r in results:
            assert r.chunk_id
            assert 0.0 <= r.score <= 1.0
            assert r.source_ref.startswith("file:")
            assert r.trust_level in ("external_untrusted", "user_provided_data")
            assert len(r.text_preview) > 0

    def test_retrieve_handles_empty_indexes(self, retriever):
        results = retriever.retrieve(
            query="anything",
            embed_fn=_mock_embed_fn,
            k=5,
        )
        assert results == []

    def test_retrieve_respects_k_limit(self, populated_store, retriever):
        results = retriever.retrieve(
            query="data processing machine learning docker containers",
            embed_fn=_mock_embed_fn,
            k=2,
        )
        assert len(results) <= 2

    def test_retrieve_dedupes_same_chunk(self, populated_store, retriever):
        """The same chunk should not appear twice even if found by both indexes."""
        results = retriever.retrieve(
            query="Python web API",
            embed_fn=_mock_embed_fn,
            k=10,
        )
        chunk_ids = [r.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), f"Duplicates found: {chunk_ids}"

    def test_retrieve_sorts_by_score_descending(self, populated_store, retriever):
        results = retriever.retrieve(
            query="real-time streaming platform",
            embed_fn=_mock_embed_fn,
            k=5,
        )
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), f"Scores not sorted: {scores}"

    def test_retrieve_trust_penalty_applied(self, populated_store, retriever):
        """Results with external_untrusted trust level should have a penalty factor."""
        results = retriever.retrieve(
            query="Python API",
            embed_fn=_mock_embed_fn,
            k=5,
        )
        # All test data is external_untrusted, so all should have scores < 1.0
        # even if they matched perfectly
        for r in results:
            # The trust penalty should reduce scores somewhat
            # (exact value depends on weighting, just verify it's not > 0.95)
            assert r.score <= 0.95

    def test_retrieve_and_rerank_falls_back_without_reranker(self, populated_store, retriever):
        """retrieve_and_rerank() without a reranker should match retrieve()."""
        retriever.reranker = None
        results_a = retriever.retrieve("Python async API", _mock_embed_fn, k=5)
        results_b = retriever.retrieve_and_rerank("Python async API", _mock_embed_fn, k=5)
        assert len(results_a) == len(results_b)
        assert [r.chunk_id for r in results_a] == [r.chunk_id for r in results_b]

    def test_retrieve_and_rerank_with_reranker(self, populated_store, retriever):
        """retrieve_and_rerank() with a reranker should return ≤k results."""
        from src.index.reranker import Reranker
        retriever.reranker = Reranker()
        results = retriever.retrieve_and_rerank("Python async API", _mock_embed_fn, k=3)
        assert len(results) <= 3
        # All results should be valid
        for r in results:
            assert r.chunk_id
            assert 0.0 <= r.score <= 1.0
