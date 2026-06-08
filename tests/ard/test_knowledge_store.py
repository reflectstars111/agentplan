"""Tests for KnowledgeStore."""

import os
import tempfile
import uuid

import pytest

from ard.infra.config import Config
from ard.infra.db import Database
from ard.store.knowledge_store import KnowledgeStore
from ard.retriever.vector_index import VectorIndex
from src.embedding import create_mock_embed_fn


@pytest.fixture
def store():
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test.db")
    idx_path = os.path.join(tmp, "test.faiss")
    config = Config(db_path=db_path, vector_index_path=idx_path, embedding_dim=1536)
    db = Database(db_path)
    db.init_schema()
    embed_fn = create_mock_embed_fn(dim=1536)
    vi = VectorIndex(dim=1536, index_path=idx_path)
    ks = KnowledgeStore(db, vi, embed_fn, config.file_store_path)
    yield ks
    db.close()


class TestKnowledgeStore:
    def test_index_chunks(self, store):
        chunks = [{
            "text": "Test document about machine learning.",
            "source_type": "text",
            "file_name": "test.txt",
            "trust_level": "user_provided_data",
        }]
        count = store.index_chunks(chunks, f"src_{uuid.uuid4().hex[:8]}")
        assert count == 1
        assert store.count_chunks() == 1
        assert store.vector_index.count == 1

    def test_index_multiple_chunks(self, store):
        chunks = [
            {"text": f"Document {i} about topic {i}.", "source_type": "text",
             "file_name": "multi.txt", "trust_level": "user_provided_data"}
            for i in range(5)
        ]
        count = store.index_chunks(chunks, f"src_{uuid.uuid4().hex[:8]}")
        assert count == 5
        assert store.count_chunks() == 5

    def test_list_sources(self, store):
        chunks = [{"text": "Content.", "source_type": "text",
                    "file_name": "src_test.txt", "trust_level": "user_provided_data"}]
        sid = f"src_{uuid.uuid4().hex[:8]}"
        store.index_chunks(chunks, sid)
        sources = store.list_sources()
        assert len(sources) == 1
        assert sources[0]["source_id"] == sid

    def test_get_chunks(self, store):
        chunks = [
            {"text": "Chunk A content.", "source_type": "text",
             "file_name": "chunks.txt", "trust_level": "user_provided_data"},
            {"text": "Chunk B content.", "source_type": "text",
             "file_name": "chunks.txt", "trust_level": "user_provided_data"},
        ]
        sid = f"src_{uuid.uuid4().hex[:8]}"
        store.index_chunks(chunks, sid)
        retrieved = store.get_chunks(sid)
        assert len(retrieved) == 2

    def test_vector_search(self, store):
        chunks = [
            {"text": "Machine learning and deep neural networks.", "source_type": "text",
             "file_name": "ml.txt", "trust_level": "user_provided_data"},
            {"text": "Cooking recipes for Italian pasta.", "source_type": "text",
             "file_name": "cooking.txt", "trust_level": "user_provided_data"},
        ]
        store.index_chunks(chunks, f"src_{uuid.uuid4().hex[:8]}")
        results = store._vector_search("neural networks AI", top_k=5)
        assert len(results) > 0
        # All results should have the correct structure
        for r in results:
            assert r.chunk_id
            assert r.text_preview
            assert r.score >= 0

    def test_keyword_search(self, store):
        chunks = [{
            "text": "The ARD system uses hybrid retrieval and context management.",
            "source_type": "text",
            "file_name": "ard.txt",
            "trust_level": "user_provided_data",
        }]
        store.index_chunks(chunks, f"src_{uuid.uuid4().hex[:8]}")
        results = store._keyword_search("hybrid retrieval", top_k=5)
        assert len(results) > 0

    def test_count_chunks_empty(self, store):
        assert store.count_chunks() == 0
