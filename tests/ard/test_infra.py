"""Tests for ARD infrastructure layer."""

import os
import tempfile

import pytest

from ard.infra.config import Config
from ard.infra.db import Database


class TestConfig:
    def test_defaults(self):
        c = Config()
        assert c.embedding_dim == 1024
        assert c.default_token_budget == 8000
        assert c.weight_semantic == 0.35

    def test_data_dirs_created(self):
        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, "sub", "test.db")
        c = Config(db_path=db_path, file_store_path=os.path.join(tmp, "files"))
        assert os.path.exists(os.path.dirname(db_path))
        assert os.path.exists(os.path.join(tmp, "files"))


class TestDatabase:
    @pytest.fixture
    def db(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "test.db")
        database = Database(path)
        database.init_schema()
        yield database
        database.close()

    def test_connect_creates_file(self, db):
        assert os.path.exists(db.db_path)

    def test_sources_table(self, db):
        db.execute(
            "INSERT INTO sources (source_id, source_type) VALUES (?, ?)",
            ("src_001", "text"),
        )
        db.commit()
        row = db.execute("SELECT * FROM sources WHERE source_id = ?", ("src_001",)).fetchone()
        assert row is not None
        assert row["source_type"] == "text"

    def test_chunks_table_and_fts(self, db):
        db.execute(
            "INSERT INTO sources (source_id, source_type) VALUES (?, ?)",
            ("src_002", "text"),
        )
        db.execute(
            """INSERT INTO chunks (chunk_id, source_id, source_type, text, trust_level)
               VALUES (?, ?, ?, ?, ?)""",
            ("chunk_001", "src_002", "text", "hello world test", "user_provided_data"),
        )
        db.commit()
        row = db.execute("SELECT * FROM chunks WHERE chunk_id = ?", ("chunk_001",)).fetchone()
        assert row is not None
        assert row["text"] == "hello world test"
