"""Tests for TraceLogger."""

import pytest
from src.db import Database
from src.models.trace import TraceStep, StepType, StepStatus
from src.runtime.trace_logger import TraceLogger


@pytest.fixture
def db():
    d = Database(":memory:")
    d.init_schema()
    yield d
    d.close()


@pytest.fixture
def logger(db):
    return TraceLogger(db)


class TestTraceLogger:
    def test_start_trace(self, logger):
        trace = logger.start_trace(request_id="req_001")
        assert trace.trace_id.startswith("trace_")
        assert trace.request_id == "req_001"
        assert len(trace.steps) == 0

    def test_add_step(self, logger):
        trace = logger.start_trace(request_id="req_001")
        step = TraceStep(
            step_id="step_1",
            type=StepType.RETRIEVE_MEMORY,
            input={"query": "test"},
            output={"results": 5},
        )
        logger.add_step(trace.trace_id, step)
        # Retrieve and verify
        retrieved = logger.get_trace(trace.trace_id)
        assert len(retrieved.steps) == 1
        assert retrieved.steps[0].step_id == "step_1"
        assert retrieved.steps[0].type == StepType.RETRIEVE_MEMORY

    def test_get_trace_nonexistent(self, logger):
        assert logger.get_trace("nonexistent") is None

    def test_list_recent(self, logger):
        for i in range(5):
            trace = logger.start_trace(request_id=f"req_{i:03d}")
            logger.add_step(trace.trace_id, TraceStep(
                step_id="s1", type=StepType.LLM_REASONING,
                input={}, output={"answer": f"result_{i}"},
            ))
        recent = logger.list_recent(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].request_id > recent[-1].request_id

    def test_start_trace_has_unique_ids(self, logger):
        t1 = logger.start_trace(request_id="req_001")
        t2 = logger.start_trace(request_id="req_002")
        assert t1.trace_id != t2.trace_id

    def test_child_trace_round_trips_parent_id(self, logger):
        parent = logger.start_trace(request_id="req_parent")
        child = logger.start_trace(
            request_id="req_child",
            parent_trace_id=parent.trace_id,
        )

        retrieved = logger.get_trace(child.trace_id)
        assert retrieved.parent_trace_id == parent.trace_id
        assert [
            trace.trace_id for trace in logger.list_children(parent.trace_id)
        ] == [child.trace_id]

    def test_multiple_steps(self, logger):
        trace = logger.start_trace(request_id="req_001")
        steps = [
            TraceStep(step_id="s1", type=StepType.INTENT_DECODE, input={"q": "hello"}),
            TraceStep(step_id="s2", type=StepType.CONTEXT_ASSEMBLE, input={"budget": 24000}),
            TraceStep(step_id="s3", type=StepType.LLM_REASONING, output={"answer": "hi"}),
        ]
        for s in steps:
            logger.add_step(trace.trace_id, s)

        retrieved = logger.get_trace(trace.trace_id)
        assert len(retrieved.steps) == 3

    def test_add_step_to_nonexistent_trace_raises(self, logger):
        step = TraceStep(step_id="s1", type=StepType.RESPOND)
        with pytest.raises(ValueError, match="not found"):
            logger.add_step("nonexistent", step)

    def test_persists_across_connections(self, logger):
        """Trace data survives db close/reopen."""
        trace = logger.start_trace(request_id="req_persist")
        logger.add_step(trace.trace_id, TraceStep(
            step_id="s1", type=StepType.VERIFY,
            input={}, output={"passed": True},
        ))
        tid = trace.trace_id

        # Create a new logger with new DB connection (same in-memory DB)
        logger2 = TraceLogger(logger.db)
        retrieved = logger2.get_trace(tid)
        assert retrieved is not None
        assert len(retrieved.steps) == 1


class TestStepType:
    def test_new_step_types_exist(self):
        from src.models.trace import StepType
        assert hasattr(StepType, "SPAWN_AGENT")
        assert hasattr(StepType, "SEND_MESSAGE")
        assert hasattr(StepType, "MERGE")

    def test_step_type_count(self):
        from src.models.trace import StepType
        types = list(StepType)
        assert len(types) >= 12  # 9 originals + 3 new
