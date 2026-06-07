"""Tests for DbSource."""
import sqlite3
import pytest
from src.db import Database
from src.storage.file_store import FileStore
from src.sources.db_source import DbSource


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
def test_sqlite_db(tmp_path):
    """Create a temporary SQLite database with test data."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice')")
    conn.execute("INSERT INTO users VALUES (2, 'Bob')")
    conn.commit()
    conn.close()
    return db_path


class TestDbSource:
    def test_query_sqlite_and_index(self, file_store, test_sqlite_db):
        src = DbSource()
        result = src.query_and_index(
            db_path=test_sqlite_db,
            query="SELECT * FROM users",
            file_store=file_store,
            source_name="test_db",
        )
        assert result["row_count"] == 2
        assert "source_id" in result
        assert result["columns"] == ["id", "name"]

        chunks = file_store.get_chunks(result["source_id"])
        assert len(chunks) > 0

    def test_rejects_non_select(self, file_store, test_sqlite_db):
        src = DbSource()
        result = src.query_and_index(
            db_path=test_sqlite_db,
            query="DROP TABLE users",
            file_store=file_store,
        )
        assert "error" in result

    def test_empty_result(self, file_store, test_sqlite_db):
        src = DbSource()
        result = src.query_and_index(
            db_path=test_sqlite_db,
            query="SELECT * FROM users WHERE id = 999",
            file_store=file_store,
        )
        assert result["row_count"] == 0
