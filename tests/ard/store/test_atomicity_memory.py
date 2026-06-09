import pytest

from ard.infra.db import Database
from ard.store.event import StoreEvent
from ard.store.event_store import EventStore
from ard.store.projections import Projections
from ard.store.state_store import StateStore
from ard.store.transaction import TransactionManager


@pytest.fixture
def stack():
    db = Database(":memory:")
    db.init_schema()
    projections = Projections()
    event_store = EventStore(db, projections)
    state_store = StateStore(event_store)
    projections.register("state.created", state_store.apply_event)
    manager = TransactionManager(event_store)
    yield manager, state_store
    db.close()


def test_duplicate_event_rolls_back_entire_batch(stack):
    manager, _ = stack
    duplicate_id = "evt_duplicate"
    events = [
        StoreEvent(event_id=duplicate_id, stream_key="one"),
        StoreEvent(event_id=duplicate_id, stream_key="two"),
    ]

    with pytest.raises(ValueError, match="already exists"):
        manager.event_store.append_batch(events)

    assert manager.event_store.count() == 0


def test_projection_failure_leaves_no_event_or_projection(stack):
    manager, state_store = stack

    def fail_on_second(payload):
        if payload["_stream_key"] == "two":
            raise RuntimeError("projection failed")

    manager.event_store.projections.register("state.created", fail_on_second)
    txn = manager.begin()
    txn.add_event(StoreEvent(
        stream="state",
        stream_key="one",
        event_type="created",
        payload={"value": 1},
    ))
    txn.add_event(StoreEvent(
        stream="state",
        stream_key="two",
        event_type="created",
        payload={"value": 2},
    ))

    with pytest.raises(RuntimeError, match="projection failed"):
        manager.commit(txn)

    assert manager.event_store.count() == 0
    assert state_store.read("one") is None
    assert state_store.read("two") is None
    assert manager.get_txn(txn.txn_id)["status"] == "rolled_back"
