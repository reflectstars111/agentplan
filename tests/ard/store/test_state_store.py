"""Tests for StateStore."""

import os
import tempfile
import json

import pytest

from ard.infra.config import Config
from ard.infra.db import Database
from ard.store.event import StoreEvent
from ard.store.event_store import EventStore
from ard.store.projections import Projections
from ard.store.state_store import StateStore
from ard.store.transaction import TransactionManager


@pytest.fixture
def state_store():
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test.db")
    db = Database(db_path)
    db.init_schema()
    es = EventStore(db, Projections())
    ss = StateStore(es)
    # Register projection handler
    es.projections.register("state.created", ss.apply_event)
    es.projections.register("state.updated", ss.apply_event)
    es.projections.register("state.archived", ss.apply_event)
    es.projections.register("state.deleted", ss.apply_event)
    yield ss
    db.close()


def _write_state(state_store, key: str, payload: dict, event_type: str = "created"):
    """Helper: write state through a transaction."""
    txn_mgr = TransactionManager(state_store.event_store)
    txn = txn_mgr.begin()
    evt = state_store.build_event(key, event_type, payload)
    txn.add_event(evt)
    txn_mgr.commit(txn)


class TestStateStore:
    def test_read_nonexistent(self, state_store):
        assert state_store.read("nonexistent:key") is None

    def test_write_and_read(self, state_store):
        _write_state(state_store, "agent:agent_001", {
            "role": "worker",
            "status": "ready",
            "priority": 5,
        })
        value = state_store.read("agent:agent_001")
        assert value is not None
        assert value["role"] == "worker"
        assert value["status"] == "ready"
        assert "_version" in value

    def test_update_existing(self, state_store):
        _write_state(state_store, "task:task_001", {
            "status": "running",
            "progress": 0.5,
        })
        _write_state(state_store, "task:task_001", {
            "status": "completed",
            "progress": 1.0,
        }, event_type="updated")

        value = state_store.read("task:task_001")
        assert value["status"] == "completed"
        assert value["progress"] == 1.0

    def test_history(self, state_store):
        # Write 3 versions
        _write_state(state_store, "project:proj_001", {"decision": "v1"})
        _write_state(state_store, "project:proj_001", {"decision": "v2"}, "updated")
        _write_state(state_store, "project:proj_001", {"decision": "v3"}, "updated")

        history = state_store.history("project:proj_001")
        assert len(history) == 3
        for h in history:
            assert "seq_num" in h
            assert "event_type" in h
            assert "payload" in h
            assert "timestamp" in h

    def test_list_keys(self, state_store):
        _write_state(state_store, "agent:agent_001", {"x": 1})
        _write_state(state_store, "agent:agent_002", {"x": 2})
        _write_state(state_store, "task:task_001", {"x": 3})

        all_keys = state_store.list_keys()
        assert len(all_keys) == 3

        agent_keys = state_store.list_keys("agent:")
        assert len(agent_keys) == 2

    def test_read_with_version(self, state_store):
        _write_state(state_store, "test:vkey", {"v": 1})
        _write_state(state_store, "test:vkey", {"v": 2}, "updated")

        # Read latest (v2)
        latest = state_store.read("test:vkey")
        assert latest is not None

        # Read at specific version via history reconstruction
        history = state_store.history("test:vkey")
        v1_seq = history[0]["seq_num"]

        v1 = state_store.read("test:vkey", version=v1_seq)
        assert v1 is not None

    def test_read_for_transaction_records_read_set(self, state_store):
        _write_state(state_store, "agent:t1", {"status": "ready"})

        txn_mgr = TransactionManager(state_store.event_store)
        txn = txn_mgr.begin()
        value = state_store.read_for_transaction("agent:t1", txn)
        assert value is not None
        assert value["status"] == "ready"
        assert len(txn.read_set) == 1
        assert txn.read_set[0]["stream"] == "state"
        assert txn.read_set[0]["stream_key"] == "agent:t1"

    def test_build_event(self, state_store):
        evt = state_store.build_event("agent:new", "created", {"status": "ready"})
        assert evt.stream == "state"
        assert evt.stream_key == "agent:new"
        assert evt.event_type == "created"
        assert evt.payload["status"] == "ready"
        assert evt.seq_num == -1  # not yet committed

    def test_delete_state(self, state_store):
        _write_state(state_store, "temp:key", {"data": "toBeDeleted"})
        assert state_store.read("temp:key") is not None

        _write_state(state_store, "temp:key", {}, event_type="deleted")
        assert state_store.read("temp:key") is None
