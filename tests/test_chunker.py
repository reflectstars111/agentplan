"""Tests for the chunker module."""

import pytest
from src.storage.chunker import chunk_text, ChunkerConfig


class TestChunkText:
    def test_splits_long_text_into_chunks(self):
        text = "hello world. " * 200  # ~2400 chars
        config = ChunkerConfig(chunk_size=100, chunk_overlap=20)
        chunks = chunk_text(text, source_id="test", config=config)
        assert len(chunks) > 1
        # Each chunk should be roughly within chunk_size chars
        for c in chunks:
            assert len(c.text) <= config.chunk_size + config.chunk_overlap + 50  # some slack

    def test_short_text_returns_single_chunk(self):
        text = "A short piece of text."
        config = ChunkerConfig(chunk_size=1000, chunk_overlap=50)
        chunks = chunk_text(text, source_id="test", config=config)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_empty_text_returns_empty_list(self):
        chunks = chunk_text("", source_id="test")
        assert chunks == []

    def test_chunks_have_unique_ids(self):
        text = "sentence. " * 100
        chunks = chunk_text(text, source_id="doc_1")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunks_have_source_id(self):
        text = "sentence. " * 50
        chunks = chunk_text(text, source_id="my_doc.pdf")
        for c in chunks:
            assert c.source_id == "my_doc.pdf"

    def test_overlap_preserves_context(self):
        text = "AAAA BBBB CCCC DDDD EEEE FFFF GGGG HHHH IIII JJJJ"
        config = ChunkerConfig(chunk_size=20, chunk_overlap=10)
        chunks = chunk_text(text, source_id="test", config=config)
        if len(chunks) >= 2:
            # Last chars of chunk 0 should appear in chunk 1 somewhere
            assert chunks[0].text[-10:] in chunks[1].text or any(
                w in chunks[1].text for w in chunks[0].text.split()[-3:]
            )
