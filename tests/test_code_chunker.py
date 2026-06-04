"""Tests for CodeChunker."""

import pytest
from src.storage.code_chunker import chunk_code
from src.models.chunk import ChunkType


PYTHON_CODE = """
import os
import sys

def hello(name):
    \"\"\"Say hello.\"\"\"
    return f"Hello, {name}!"

class Greeter:
    \"\"\"A greeter.\"\"\"
    def greet(self, name):
        return f"Hi, {name}"

def goodbye():
    return "Bye!"
"""


class TestCodeChunker:
    def test_produces_code_chunks(self):
        chunks = chunk_code(PYTHON_CODE, "file:test.py", language="python")
        assert len(chunks) > 0

    def test_chunks_have_code_type(self):
        chunks = chunk_code(PYTHON_CODE, "file:test.py", language="python")
        code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE]
        assert len(code_chunks) >= 2  # at least hello and goodbye

    def test_chunks_have_source_id(self):
        chunks = chunk_code(PYTHON_CODE, "file:test.py", language="python")
        for c in chunks:
            assert c.source_id == "file:test.py"

    def test_empty_code(self):
        chunks = chunk_code("", "file:empty.py")
        assert chunks == []

    def test_single_function(self):
        code = "def f():\n    return 42\n"
        chunks = chunk_code(code, "file:single.py")
        code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE]
        assert len(code_chunks) == 1
