"""Tests for FileStore."""

import pytest
from pathlib import Path
from src.db import Database
from src.storage.file_store import FileStore


@pytest.fixture
def db():
    d = Database(":memory:")
    d.init_schema()
    yield d
    d.close()


@pytest.fixture
def store(db, tmp_path):
    return FileStore(db, file_store_path=str(tmp_path / "files"))


class TestFileStore:
    def test_ingest_markdown(self, store, sample_markdown_path):
        source_id = store.ingest_file(sample_markdown_path)
        assert source_id.startswith("file:")
        # Should have created chunks
        chunks = store.get_chunks(source_id)
        assert len(chunks) > 0
        # Each chunk should have source_id
        for c in chunks:
            assert c.source_id == source_id

    def test_ingest_text_content(self, store):
        source_id = store.ingest_text(
            content="This is a test document. " * 50,
            source_name="test_doc.txt",
        )
        chunks = store.get_chunks(source_id)
        assert len(chunks) > 0

    def test_get_chunks_by_source(self, store, sample_markdown_path):
        sid1 = store.ingest_file(sample_markdown_path)
        sid2 = store.ingest_text(content="Other content.", source_name="other.txt")
        chunks1 = store.get_chunks(sid1)
        chunks2 = store.get_chunks(sid2)
        assert len(chunks1) > 0
        # Each set should only contain its own source_id
        assert all(c.source_id == sid1 for c in chunks1)
        assert all(c.source_id == sid2 for c in chunks2)

    def test_list_sources(self, store, sample_markdown_path):
        store.ingest_file(sample_markdown_path)
        store.ingest_text(content="Another doc.", source_name="doc2.txt")
        sources = store.list_sources()
        assert len(sources) >= 2

    def test_delete_source(self, store, sample_markdown_path):
        sid = store.ingest_file(sample_markdown_path)
        assert len(store.get_chunks(sid)) > 0
        store.delete_source(sid)
        assert store.get_chunks(sid) == []

    def test_ingest_empty_file(self, store, tmp_path):
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("")
        sid = store.ingest_file(empty_file)
        chunks = store.get_chunks(sid)
        assert chunks == []
