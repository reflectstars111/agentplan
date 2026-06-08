"""Tests for TraceStore."""

import os
import tempfile

import pytest

from ard.infra.config import Config
from ard.infra.db import Database
from ard.store.event_store import EventStore
from ard.store.projections import Projections
from ard.store.trace_store import TraceStore, TraceHandle


@pytest.fixture
def trace_store():
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test.db")
    db = Database(db_path)
    db.init_schema()
    es = EventStore(db, Projections())
    ts = TraceStore(es)
    yield ts
    db.close()


class TestTraceHandle:
    def test_creation(self):
        h = TraceHandle(trace_id="trace_001", request_id="req_001")
        assert h.trace_id == "trace_001"
        assert h.request_id == "req_001"
        assert h.created_at

    def test_to_dict(self):
        h = TraceHandle("t1", "r1")
        d = h.to_dict()
        assert d["trace_id"] == "t1"


class TestTraceStore:
    def test_start_trace(self, trace_store):
        handle = trace_store.start_trace("req_test")
        assert isinstance(handle, TraceHandle)
        assert handle.trace_id.startswith("trace_")

    def test_add_step(self, trace_store):
        handle = trace_store.start_trace("req_001")
        step_id = trace_store.add_step(
            trace_id=handle.trace_id,
            step_type="retrieve",
            input_data={"query": "test"},
            output_data={"candidates": 10},
        )
        assert step_id.startswith("step_")

    def test_query_trace(self, trace_store):
        handle = trace_store.start_trace("req_002")
        trace_store.add_step(handle.trace_id, "plan", {"q": "hello"}, {"intent": "general"})
        trace_store.add_step(handle.trace_id, "execute", {}, {"answer": "world"})
        trace_store.add_step(handle.trace_id, "verify", {}, {"verified": True})

        steps = trace_store.query(handle.trace_id)
        assert len(steps) == 3
        step_types = [s["step_type"] for s in steps]
        assert "plan" in step_types
        assert "execute" in step_types
        assert "verify" in step_types

    def test_query_nonexistent(self, trace_store):
        assert trace_store.query("nonexistent") == []

    def test_add_step_with_error(self, trace_store):
        handle = trace_store.start_trace("req_err")
        trace_store.add_step(
            handle.trace_id, "execute",
            input_data={"q": "test"},
            status="failed",
            error="LLM timeout",
        )
        steps = trace_store.query(handle.trace_id)
        assert len(steps) == 1
        assert steps[0]["status"] == "failed"
        assert steps[0]["error"] == "LLM timeout"

    def test_multiple_traces_independent(self, trace_store):
        t1 = trace_store.start_trace("r1")
        t2 = trace_store.start_trace("r2")

        trace_store.add_step(t1.trace_id, "plan", {}, {})
        trace_store.add_step(t2.trace_id, "plan", {}, {})
        trace_store.add_step(t2.trace_id, "execute", {}, {})

        assert len(trace_store.query(t1.trace_id)) == 1
        assert len(trace_store.query(t2.trace_id)) == 2
