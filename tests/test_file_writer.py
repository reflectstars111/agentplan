"""Tests for FileWriter."""
import pytest
from src.runtime.file_writer import FileWriter


class TestFileWriter:
    def test_write_creates_file(self, tmp_path):
        writer = FileWriter(output_dir=str(tmp_path / "output"))
        path = writer.write("hello world", "test.txt")
        assert path.exists()
        assert path.read_text() == "hello world"

    def test_write_report(self, tmp_path):
        writer = FileWriter(output_dir=str(tmp_path / "output"))
        sections = [("Intro", "This is the intro."), ("Details", "More details here.")]
        path = writer.write_report("Test Report", sections, "report.md")
        assert path.exists()
        content = path.read_text()
        assert "# Test Report" in content
        assert "## Intro" in content

    def test_rejects_path_traversal(self, tmp_path):
        writer = FileWriter(output_dir=str(tmp_path / "output"))
        with pytest.raises(ValueError, match="Invalid filename"):
            writer.write("evil", "../escape.txt")

    def test_list_files(self, tmp_path):
        writer = FileWriter(output_dir=str(tmp_path / "output"))
        writer.write("a", "a.txt")
        writer.write("b", "b.txt")
        files = writer.list_files()
        assert "a.txt" in files
        assert "b.txt" in files
