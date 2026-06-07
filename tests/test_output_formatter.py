"""Tests for OutputFormatter."""
from src.runtime.output_formatter import OutputFormatter


class TestOutputFormatter:
    def test_mermaid(self):
        result = OutputFormatter.mermaid("A --> B", "flowchart")
        assert "```mermaid" in result
        assert "flowchart" in result
        assert "A --> B" in result

    def test_latex_display(self):
        result = OutputFormatter.latex("E = mc^2")
        assert "$$" in result
        assert "E = mc^2" in result

    def test_latex_inline(self):
        result = OutputFormatter.latex("x^2", display=False)
        assert result.startswith("$")
        assert not result.startswith("$$")

    def test_table(self):
        result = OutputFormatter.table(["Name", "Value"], [["A", "1"], ["B", "2"]])
        assert "| Name | Value |" in result
        assert "| A | 1 |" in result
        assert "| B | 2 |" in result

    def test_code_block(self):
        result = OutputFormatter.code_block("print('hello')", "python")
        assert "```python" in result
        assert "print('hello')" in result

    def test_json_output(self):
        result = OutputFormatter.json_output({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_report(self):
        result = OutputFormatter.report("Title", [("H1", "Body1"), ("H2", "Body2")])
        assert "# Title" in result
        assert "## H1" in result
        assert "Body1" in result
        assert "## H2" in result
