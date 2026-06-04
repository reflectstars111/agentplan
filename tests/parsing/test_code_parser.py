"""Tests for CodeParser (tree-sitter)."""

import pytest
from src.parsing.code_parser import CodeParser, ParseResult


@pytest.fixture
def parser():
    return CodeParser("python")


SIMPLE_CODE = """
def hello(name):
    \"\"\"Say hello.\"\"\"
    print(f"Hello, {name}!")

class Greeter:
    \"\"\"A greeter class.\"\"\"
    def greet(self, name):
        return f"Hi, {name}"
"""


class TestCodeParser:
    def test_parse_returns_parse_result(self, parser):
        result = parser.parse(SIMPLE_CODE, "file:test.py")
        assert isinstance(result, ParseResult)
        assert result.language == "python"

    def test_extract_functions(self, parser):
        symbols = parser.extract_symbols(SIMPLE_CODE, "file:test.py")
        func_names = {s.name for s in symbols if s.symbol_type == "function"}
        assert "hello" in func_names

    def test_extract_classes(self, parser):
        symbols = parser.extract_symbols(SIMPLE_CODE, "file:test.py")
        class_names = {s.name for s in symbols if s.symbol_type == "class"}
        assert "Greeter" in class_names

    def test_extract_methods(self, parser):
        symbols = parser.extract_symbols(SIMPLE_CODE, "file:test.py")
        method_names = {s.name for s in symbols if s.symbol_type == "method"}
        assert "greet" in method_names

    def test_extract_docstring(self, parser):
        symbols = parser.extract_symbols(SIMPLE_CODE, "file:test.py")
        hello = next(s for s in symbols if s.name == "hello")
        assert "Say hello" in hello.docstring

    def test_method_has_parent(self, parser):
        symbols = parser.extract_symbols(SIMPLE_CODE, "file:test.py")
        greet = next(s for s in symbols if s.name == "greet")
        parent = next((s for s in symbols if s.symbol_id == greet.parent_symbol_id), None)
        assert parent is not None
        assert parent.name == "Greeter"

    def test_line_numbers(self, parser):
        symbols = parser.extract_symbols(SIMPLE_CODE, "file:test.py")
        hello = next(s for s in symbols if s.name == "hello")
        assert hello.location_line_start > 0
        assert hello.location_line_end > hello.location_line_start

    def test_extract_signature(self, parser):
        symbols = parser.extract_symbols(SIMPLE_CODE, "file:test.py")
        greet = next(s for s in symbols if s.name == "greet")
        assert "self" in greet.signature
        assert "name" in greet.signature

    def test_parse_empty_file(self, parser):
        result = parser.parse("", "file:empty.py")
        assert len(result.symbols) == 0
        assert len(result.structure) >= 1  # at least root file node

    def test_detect_language(self):
        assert CodeParser.detect_language("test.py") == "python"
        assert CodeParser.detect_language("test.js") == "javascript"
        assert CodeParser.detect_language("test.ts") == "typescript"
        assert CodeParser.detect_language("test.md") is None
        assert CodeParser.detect_language("test.txt") is None
