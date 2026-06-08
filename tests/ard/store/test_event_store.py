"""Tests for EventStore."""

import os
import tempfile

import pytest

from ard.infra.config import Config
from ard.infra.db import Database
from ard.store.event import StoreEvent
from ard.store.event_store import EventStore
from ard.store.projections import Projections


@pytest.fixture
def event_store():
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test.db")
    db = Database(db_path)
    db.init_schema()
    es = EventStore(db, Projections())
    yield es
    db.close()


class TestStoreEvent:
    def test_creation(self):
        evt = StoreEvent(
            stream="state",
            stream_key="agent:agent_001",
            event_type="created",
            payload={"status": "ready"},
        )
        assert evt.seq_num == -1
        assert evt.stream == "state"
        assert evt.stream_key == "agent:agent_001"

    def test_immutable(self):
        evt = StoreEvent(payload={"x": 1})
        # Frozen dataclass prevents attribute reassignment
        with pytest.raises(Exception):
            evt.seq_num = 42

    def test_with_seq(self):
        evt = StoreEvent(payload={"x": 1})
        evt2 = evt.with_seq(42)
        assert evt2.seq_num == 42
        assert evt.seq_num == -1  # original unchanged
        assert evt2.payload == evt.payload


class TestEventStore:
    def test_append_returns_seq_num(self, event_store):
        evt = StoreEvent(stream="state", stream_key="k1",
                        event_type="created", payload={"v": 1})
        seq = event_store.append(evt)
        assert seq == 1

    def test_append_batch(self, event_store):
        events = [
            StoreEvent(stream="state", stream_key=f"k{i}",
                      event_type="created", payload={"v": i})
            for i in range(5)
        ]
        seqs = event_store.append_batch(events)
        assert seqs == [1, 2, 3, 4, 5]

    def test_cannot_reappend(self, event_store):
        evt = StoreEvent(payload={"v": 1})
        event_store.append(evt)
        with pytest.raises(ValueError):
            event_store.append(evt)  # already has seq_num

    def test_replay_all(self, event_store):
        for i in range(10):
            event_store.append(StoreEvent(
                stream="state", stream_key=f"k{i}",
                event_type="created", payload={"i": i},
            ))
        events = event_store.replay(after_seq=0)
        assert len(events) == 10

    def test_replay_after_seq(self, event_store):
        for i in range(10):
            event_store.append(StoreEvent(
                stream="state", stream_key=f"k{i}",
                event_type="created", payload={"i": i},
            ))
        events = event_store.replay(after_seq=5)
        assert len(events) == 5
        assert events[0].seq_num == 6

    def test_replay_filtered_by_stream(self, event_store):
        event_store.append(StoreEvent(stream="state", stream_key="k1",
                                     event_type="created", payload={}))
        event_store.append(StoreEvent(stream="knowledge", stream_key="k2",
                                     event_type="created", payload={}))
        event_store.append(StoreEvent(stream="state", stream_key="k3",
                                     event_type="created", payload={}))

        state_events = event_store.replay(stream="state")
        assert len(state_events) == 2

    def test_replay_filtered_by_stream_key(self, event_store):
        event_store.append(StoreEvent(stream="state", stream_key="agent:a",
                                     event_type="created", payload={}))
        event_store.append(StoreEvent(stream="state", stream_key="agent:b",
                                     event_type="created", payload={}))

        filtered = event_store.replay(stream_key="agent:a")
        assert len(filtered) == 1
        assert filtered[0].stream_key == "agent:a"

    def test_get_by_seq(self, event_store):
        event_store.append(StoreEvent(stream="state", stream_key="k1",
                                     event_type="created", payload={"x": 42}))
        evt = event_store.get_by_seq(1)
        assert evt is not None
        assert evt.payload["x"] == 42

    def test_get_by_seq_missing(self, event_store):
        assert event_store.get_by_seq(999) is None

    def test_latest_seq(self, event_store):
        assert event_store.latest_seq() == 0
        event_store.append(StoreEvent(stream="state", stream_key="k1",
                                     event_type="created", payload={}))
        assert event_store.latest_seq() == 1

    def test_latest_seq_filtered(self, event_store):
        event_store.append(StoreEvent(stream="state", stream_key="a",
                                     event_type="created", payload={}))
        event_store.append(StoreEvent(stream="state", stream_key="b",
                                     event_type="created", payload={}))
        assert event_store.latest_seq(stream_key="a") == 1
        assert event_store.latest_seq(stream_key="b") == 2
        assert event_store.latest_seq(stream_key="c") == 0

    def test_count(self, event_store):
        assert event_store.count() == 0
        event_store.append(StoreEvent(payload={}))
        event_store.append(StoreEvent(payload={}))
        assert event_store.count() == 2
