"""Tests for VectorIndex (FAISS wrapper)."""

import numpy as np
import pytest
from src.index.vector_index import VectorIndex


@pytest.fixture
def index():
    """A fresh in-memory index with small dimension for fast testing."""
    return VectorIndex(dim=64)


class TestVectorIndex:
    def test_add_and_search(self, index):
        # Add embeddings with known similarity
        emb1 = np.random.RandomState(42).randn(64).astype(np.float32)
        emb2 = emb1 + np.random.RandomState(99).randn(64).astype(np.float32) * 0.01  # very similar
        emb3 = np.random.RandomState(77).randn(64).astype(np.float32)  # unrelated

        index.add("chunk_a", emb1)
        index.add("chunk_b", emb2)
        index.add("chunk_c", emb3)

        results = index.search(emb1, k=2)
        # chunk_a should be top result (exact match to query)
        assert results[0][0] == "chunk_a"
        # chunk_b should be second (nearly identical embedding)
        # chunk_c is unrelated so should score lower

    def test_search_returns_correct_count(self, index):
        for i in range(10):
            emb = np.random.RandomState(i).randn(64).astype(np.float32)
            index.add(f"chunk_{i}", emb)

        results = index.search(np.random.RandomState(99).randn(64).astype(np.float32), k=5)
        assert len(results) == 5

    def test_empty_index_search_returns_empty(self, index):
        results = index.search(np.random.RandomState(0).randn(64).astype(np.float32), k=10)
        assert results == []

    def test_count(self, index):
        assert index.count == 0
        for i in range(5):
            index.add(f"chunk_{i}", np.random.RandomState(i).randn(64).astype(np.float32))
        assert index.count == 5

    def test_save_and_load(self, index, tmp_path):
        path = str(tmp_path / "test_index.faiss")
        for i in range(3):
            index.add(f"chunk_{i}", np.random.RandomState(i).randn(64).astype(np.float32))
        index.save(path)

        # Load into new index
        index2 = VectorIndex(dim=64)
        index2.load(path)
        assert index2.count == 3

        # Search should work after load
        query = np.random.RandomState(0).randn(64).astype(np.float32)
        results = index2.search(query, k=1)
        assert len(results) == 1

    def test_add_duplicate_replaces(self, index):
        emb1 = np.random.RandomState(42).randn(64).astype(np.float32)
        emb2 = np.random.RandomState(43).randn(64).astype(np.float32)

        index.add("chunk_x", emb1)
        assert index.count == 1
        index.add("chunk_x", emb2)  # same id, different embedding
        assert index.count == 1  # should still be 1

    def test_remove(self, index):
        for i in range(5):
            index.add(f"chunk_{i}", np.random.RandomState(i).randn(64).astype(np.float32))
        assert index.count == 5

        index.remove("chunk_2")
        assert index.count == 4

    def test_similar_embeddings_rank_higher(self, index):
        """Verify that cosine-similar embeddings score higher in search."""
        base = np.random.RandomState(1).randn(64).astype(np.float32)
        base = base / np.linalg.norm(base)

        # Create embeddings at known distances from base
        similar = base + np.random.RandomState(2).randn(64).astype(np.float32) * 0.01
        similar = similar / np.linalg.norm(similar)
        medium = base + np.random.RandomState(3).randn(64).astype(np.float32) * 0.5
        medium = medium / np.linalg.norm(medium)
        far = base + np.random.RandomState(4).randn(64).astype(np.float32) * 2.0
        far = far / np.linalg.norm(far)

        index.add("similar", similar)
        index.add("medium", medium)
        index.add("far", far)

        results = index.search(base, k=3)
        ids = [r[0] for r in results]
        # similar should rank first or second (close to base embedding)
        assert ids[0] == "similar", f"Expected 'similar' first, got {ids}"
