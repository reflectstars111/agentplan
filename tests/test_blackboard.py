"""Tests for SharedBlackboard."""

import pytest
from src.models.blackboard import BlackboardEntry, SharedBlackboard


class TestBlackboardEntry:
    def test_create_entry(self):
        e = BlackboardEntry(
            key="repo_summary",
            value="This repo contains data processing modules.",
            created_by="agent_code_001",
            confidence=0.85,
            source_refs=["repo_001/README.md"],
        )
        assert e.key == "repo_summary"
        assert e.created_by == "agent_code_001"
        assert e.confidence == 0.85

    def test_serialization_roundtrip(self):
        e = BlackboardEntry(
            key="result", value="FastAPI is recommended.",
            created_by="agent_worker_001", confidence=0.9,
            source_refs=["file:fastapi.txt"],
        )
        json_str = e.to_json()
        e2 = BlackboardEntry.from_json(json_str)
        assert e2.key == "result"
        assert e2.created_by == "agent_worker_001"


class TestSharedBlackboard:
    def test_write_and_read(self):
        bb = SharedBlackboard()
        bb.write(BlackboardEntry(key="k1", value="v1", created_by="a1", confidence=0.8))
        entry = bb.read("k1")
        assert entry is not None
        assert entry.value == "v1"

    def test_read_missing_returns_none(self):
        bb = SharedBlackboard()
        assert bb.read("nonexistent") is None

    def test_overwrite(self):
        bb = SharedBlackboard()
        bb.write(BlackboardEntry(key="k1", value="old", created_by="a1", confidence=0.5))
        bb.write(BlackboardEntry(key="k1", value="new", created_by="a2", confidence=0.9))
        assert bb.read("k1").value == "new"
        assert bb.read("k1").created_by == "a2"

    def test_read_all(self):
        bb = SharedBlackboard()
        bb.write(BlackboardEntry(key="a", value="1", created_by="x", confidence=0.5))
        bb.write(BlackboardEntry(key="b", value="2", created_by="y", confidence=0.5))
        all_entries = bb.read_all()
        assert len(all_entries) == 2

    def test_list_by_agent(self):
        bb = SharedBlackboard()
        bb.write(BlackboardEntry(key="a", value="1", created_by="agent_a", confidence=0.5))
        bb.write(BlackboardEntry(key="b", value="2", created_by="agent_a", confidence=0.5))
        bb.write(BlackboardEntry(key="c", value="3", created_by="agent_b", confidence=0.5))
        a_entries = bb.list_by_agent("agent_a")
        assert len(a_entries) == 2

    def test_clear(self):
        bb = SharedBlackboard()
        bb.write(BlackboardEntry(key="k1", value="v1", created_by="a1", confidence=0.5))
        bb.clear()
        assert len(bb.read_all()) == 0
