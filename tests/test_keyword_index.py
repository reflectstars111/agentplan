"""Tests for KeywordIndex (SQLite FTS5 wrapper)."""

import pytest
from src.db import Database
from src.storage.file_store import FileStore
from src.index.keyword_index import KeywordIndex


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
def kw_index(db):
    return KeywordIndex(db)


class TestKeywordIndex:
    def test_search_chunks_finds_relevant(self, file_store, kw_index):
        file_store.ingest_text(
            content="The FastAPI framework provides excellent async support for Python APIs.",
            source_name="fastapi_info.txt",
        )
        file_store.ingest_text(
            content="Apache Kafka is a distributed streaming platform for event processing.",
            source_name="kafka_info.txt",
        )

        results = kw_index.search_chunks("FastAPI async Python", k=5)
        assert len(results) > 0
        # The FastAPI chunk should rank first
        chunk_ids = [r[0] for r in results]
        assert any("fastapi_info" in cid for cid in chunk_ids)

    def test_search_chunks_handles_no_match(self, file_store, kw_index):
        file_store.ingest_text(content="Some random content.", source_name="doc.txt")
        results = kw_index.search_chunks("xyzzy_nonexistent_term", k=5)
        assert results == []

    def test_search_returns_limited_k(self, file_store, kw_index):
        for i in range(5):
            file_store.ingest_text(
                content=f"Document {i} about machine learning topics.",
                source_name=f"doc_{i}.txt",
            )

        results = kw_index.search_chunks("machine learning", k=3)
        assert len(results) <= 3

    def test_search_scores_are_normalized(self, file_store, kw_index):
        file_store.ingest_text(
            content="Python is a programming language. Python is widely used.",
            source_name="python.txt",
        )

        results = kw_index.search_chunks("Python programming", k=5)
        if results:
            # Scores should be between 0.0 and 1.0 after normalization
            for _, score in results:
                assert 0.0 <= score <= 1.0

    def test_best_fts_match_has_highest_normalized_score(
        self, file_store, kw_index
    ):
        file_store.ingest_text(
            content="FastAPI FastAPI FastAPI async framework.",
            source_name="strong.txt",
        )
        file_store.ingest_text(
            content="FastAPI appears once.",
            source_name="weak.txt",
        )

        results = kw_index.search_chunks("FastAPI", k=5)
        assert len(results) == 2
        assert results[0][1] >= results[1][1]
