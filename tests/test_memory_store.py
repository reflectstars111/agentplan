"""Tests for MemoryStore."""

import pytest
from datetime import datetime, timezone
from src.db import Database
from src.models.memory import MemoryItem, MemoryType, MemoryStatus
from src.storage.memory_store import MemoryStore


@pytest.fixture
def db():
    d = Database(":memory:")
    d.init_schema()
    yield d
    d.close()


@pytest.fixture
def store(db):
    return MemoryStore(db)


SAMPLE_MEMORY = MemoryItem(
    memory_id="mem_001",
    type=MemoryType.PROJECT_STATE,
    content="The project uses FastAPI for the API layer.",
    summary="API framework: FastAPI",
    entities=["FastAPI", "API"],
    importance=0.8,
    confidence=0.95,
    source="conversation",
    scope="project",
)


class TestMemoryStore:
    def test_insert_and_get(self, store):
        store.insert(SAMPLE_MEMORY)
        result = store.get("mem_001")
        assert result is not None
        assert result.memory_id == "mem_001"
        assert result.content == SAMPLE_MEMORY.content

    def test_get_nonexistent_returns_none(self, store):
        assert store.get("does_not_exist") is None

    def test_list_by_scope(self, store):
        m1 = MemoryItem(memory_id="m1", type=MemoryType.PROJECT_STATE,
                        content="A", scope="project")
        m2 = MemoryItem(memory_id="m2", type=MemoryType.USER_PREFERENCE,
                        content="B", scope="user")
        store.insert(m1)
        store.insert(m2)
        project_mems = store.list_by_scope("project")
        assert len(project_mems) == 1
        assert project_mems[0].memory_id == "m1"

    def test_list_active(self, store):
        active = MemoryItem(memory_id="a1", type=MemoryType.DECISION,
                            content="Active", status=MemoryStatus.ACTIVE)
        archived = MemoryItem(memory_id="a2", type=MemoryType.DECISION,
                              content="Archived", status=MemoryStatus.ARCHIVED)
        store.insert(active)
        store.insert(archived)
        result = store.list_active()
        assert len(result) == 1
        assert result[0].memory_id == "a1"

    def test_update_status(self, store):
        store.insert(SAMPLE_MEMORY)
        store.update_status("mem_001", MemoryStatus.SUPERSEDED)
        result = store.get("mem_001")
        assert result.status == MemoryStatus.SUPERSEDED

    def test_delete(self, store):
        store.insert(SAMPLE_MEMORY)
        store.delete("mem_001")
        assert store.get("mem_001") is None

    def test_search_by_keyword(self, store):
        store.insert(SAMPLE_MEMORY)
        store.insert(MemoryItem(
            memory_id="mem_002", type=MemoryType.FILE_SUMMARY,
            content="The data pipeline uses Apache Kafka for streaming.",
            entities=["Kafka", "streaming"],
        ))
        results = store.search_keyword("FastAPI")
        assert len(results) >= 1
        assert any(r.memory_id == "mem_001" for r in results)

    def test_insert_duplicate_id_updates(self, store):
        store.insert(SAMPLE_MEMORY)
        updated = MemoryItem(
            memory_id="mem_001",
            type=MemoryType.PROJECT_STATE,
            content="Updated: The project uses Flask instead.",
            version=2,
        )
        store.insert(updated)
        result = store.get("mem_001")
        assert result.content == updated.content
        assert result.version == 2

    def test_last_used_at_round_trip(self, store):
        """last_used_at field survives insert + get round-trip."""
        now = datetime.now(timezone.utc)
        item = MemoryItem(
            memory_id="mem_last_used_test",
            type=MemoryType.PROJECT_STATE,
            content="test last_used_at",
            last_used_at=now,
        )
        store.insert(item)
        retrieved = store.get("mem_last_used_test")
        assert retrieved is not None
        assert retrieved.last_used_at is not None
        diff = abs((retrieved.last_used_at - now).total_seconds())
        assert diff < 5


    def test_touch_updates_last_used_at(self, store):
        """touch() should set last_used_at to now."""
        item = MemoryItem(
            memory_id="mem_touch_test",
            type=MemoryType.PROJECT_STATE,
            content="test touch",
        )
        store.insert(item)
        store.touch("mem_touch_test")
        retrieved = store.get("mem_touch_test")
        assert retrieved is not None
        assert retrieved.last_used_at is not None
