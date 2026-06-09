"""Tests for TransactionManager."""

import os
import tempfile

import pytest

from ard.infra.config import Config
from ard.infra.db import Database
from ard.store.event import StoreEvent
from ard.store.event_store import EventStore
from ard.store.projections import Projections
from ard.store.state_store import StateStore
from ard.store.transaction import Transaction, TransactionManager


@pytest.fixture
def txn_mgr():
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test.db")
    db = Database(db_path)
    db.init_schema()
    es = EventStore(db, Projections())
    tm = TransactionManager(es)
    yield tm
    db.close()


class TestTransaction:
    def test_creation(self):
        txn = Transaction()
        assert txn.status == "pending"
        assert txn.txn_id.startswith("txn_")

    def test_add_event(self):
        txn = Transaction()
        evt = StoreEvent(stream="state", stream_key="k1",
                        event_type="created", payload={"v": 1})
        txn.add_event(evt)
        assert len(txn.write_events) == 1
        assert txn.write_events[0].txn_id == txn.txn_id

    def test_record_read(self):
        txn = Transaction()
        txn.record_read("state", "key1", 5)
        assert len(txn.read_set) == 1
        assert txn.read_set[0]["read_at_seq"] == 5

    def test_record_read_dedup(self):
        txn = Transaction()
        txn.record_read("state", "key1", 5)
        txn.record_read("state", "key1", 7)  # should not duplicate
        assert len(txn.read_set) == 1
        assert txn.read_set[0]["read_at_seq"] == 5  # first read wins


class TestTransactionManager:
    def test_begin_returns_transaction(self, txn_mgr):
        txn = txn_mgr.begin()
        assert isinstance(txn, Transaction)
        assert txn.status == "pending"

    def test_commit_writes_events(self, txn_mgr):
        txn = txn_mgr.begin()
        txn.add_event(StoreEvent(stream="state", stream_key="k1",
                                event_type="created", payload={"v": 1}))
        seqs = txn_mgr.commit(txn)
        assert len(seqs) == 1
        assert txn.status == "committed"

        # Should be readable from event store
        evt = txn_mgr.event_store.get_by_seq(seqs[0])
        assert evt is not None
        assert evt.payload["v"] == 1

    def test_rollback_discards_events(self, txn_mgr):
        txn = txn_mgr.begin()
        txn.add_event(StoreEvent(stream="state", stream_key="k1",
                                event_type="created", payload={"v": 1}))
        txn_mgr.rollback(txn)
        assert txn.status == "rolled_back"
        assert txn_mgr.event_store.count() == 0

    def test_optimistic_lock_conflict(self, txn_mgr):
        # Transaction 1: read key A, write key B
        txn1 = txn_mgr.begin()
        txn1.record_read("state", "key_a", 0)

        # Transaction 2: modifies key A (simulated by appending directly)
        txn_mgr.event_store.append(StoreEvent(
            stream="state", stream_key="key_a",
            event_type="updated", payload={"modified": True},
        ))

        # Transaction 1: verify should fail because key_a changed
        assert not txn_mgr.verify(txn1)

        # Commit should raise
        txn1.add_event(StoreEvent(stream="state", stream_key="key_b",
                                 event_type="created", payload={"v": 1}))
        with pytest.raises(RuntimeError, match="conflict"):
            txn_mgr.commit(txn1)
        assert txn1.status == "rolled_back"

    def test_no_conflict_without_read_set(self, txn_mgr):
        txn = txn_mgr.begin()
        # No read_set recorded — no conflict possible
        txn.add_event(StoreEvent(stream="state", stream_key="k1",
                                event_type="created", payload={"v": 1}))
        seqs = txn_mgr.commit(txn)
        assert len(seqs) == 1

    def test_commit_multiple_events_atomic(self, txn_mgr):
        txn = txn_mgr.begin()
        txn.add_event(StoreEvent(stream="state", stream_key="k1",
                                event_type="created", payload={"v": 1}))
        txn.add_event(StoreEvent(stream="state", stream_key="k2",
                                event_type="created", payload={"v": 2}))
        txn.add_event(StoreEvent(stream="knowledge", stream_key="chunk:c1",
                                event_type="created", payload={"text": "hello"}))

        seqs = txn_mgr.commit(txn)
        assert len(seqs) == 3
        assert txn_mgr.event_store.count() == 3

    def test_projection_failure_rolls_back_events_and_snapshots(self, txn_mgr):
        state_store = StateStore(txn_mgr.event_store)
        txn_mgr.event_store.projections.register("state.created", state_store.apply_event)

        def fail_on_second(payload):
            if payload["_stream_key"] == "k2":
                raise RuntimeError("projection failed")

        txn_mgr.event_store.projections.register("state.created", fail_on_second)
        txn = txn_mgr.begin()
        txn.add_event(StoreEvent(
            stream="state", stream_key="k1",
            event_type="created", payload={"v": 1},
        ))
        txn.add_event(StoreEvent(
            stream="state", stream_key="k2",
            event_type="created", payload={"v": 2},
        ))

        with pytest.raises(RuntimeError, match="projection failed"):
            txn_mgr.commit(txn)

        assert txn_mgr.event_store.count() == 0
        assert state_store.read("k1") is None
        assert state_store.read("k2") is None
        assert txn.status == "rolled_back"
        assert txn_mgr.get_txn(txn.txn_id)["status"] == "rolled_back"
