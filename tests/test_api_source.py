"""Tests for ApiSource."""
import pytest
from src.db import Database
from src.storage.file_store import FileStore
from src.sources.api_source import ApiSource


@pytest.fixture
def db():
    d = Database(":memory:")
    d.init_schema()
    yield d
    d.close()


@pytest.fixture
def file_store(db):
    return FileStore(db)


class TestApiSource:
    def test_fetch_and_index_json_array(self, file_store, monkeypatch):
        """Should parse JSON array and create chunks."""
        import json
        import io

        mock_data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        mock_body = json.dumps(mock_data).encode("utf-8")

        def mock_urlopen(req, timeout=30):
            return io.BytesIO(mock_body)

        monkeypatch.setattr("src.sources.api_source.urlopen", mock_urlopen)

        src = ApiSource()
        result = src.fetch_and_index(
            url="https://api.example.com/users",
            file_store=file_store,
            source_name="test_api",
        )
        assert "source_id" in result
        assert result["item_count"] == 2

        chunks = file_store.get_chunks(result["source_id"])
        assert len(chunks) > 0

    def test_fetch_and_index_with_json_path(self, file_store, monkeypatch):
        """json_path should extract nested data."""
        import json
        import io

        mock_data = {"data": {"items": [{"x": 1}, {"x": 2}]}}
        mock_body = json.dumps(mock_data).encode("utf-8")

        def mock_urlopen(req, timeout=30):
            return io.BytesIO(mock_body)

        monkeypatch.setattr("src.sources.api_source.urlopen", mock_urlopen)

        src = ApiSource()
        result = src.fetch_and_index(
            url="https://api.example.com/data",
            file_store=file_store,
            json_path="data.items",
            source_name="test_nested",
        )
        assert result["item_count"] == 2

    def test_fetch_error_returns_error(self, file_store, monkeypatch):
        """HTTP errors should return error dict."""
        def mock_urlopen(req, timeout=30):
            raise OSError("Connection refused")

        monkeypatch.setattr("src.sources.api_source.urlopen", mock_urlopen)

        src = ApiSource()
        result = src.fetch_and_index(
            url="https://bad.example.com",
            file_store=file_store,
        )
        assert "error" in result
