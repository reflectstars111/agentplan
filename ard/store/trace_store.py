"""TraceStore — records and queries full execution traces.

Every step of the ARD control loop is logged here:
plan, retrieve, execute, verify, writeback, respond.

Maps to ARD design §10: Trace Store is the third pillar of storage.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from ard.store.event_store import EventStore


@runtime_checkable
class TraceStoreProtocol(Protocol):
    """Protocol for execution trace recording and querying."""

    def start_trace(self, request_id: str = "") -> "TraceHandle":
        ...

    def add_step(self, trace_id: str, step_type: str, input_data: dict | None = None,
                 output_data: dict | None = None, status: str = "success",
                 error: str | None = None) -> str:
        ...

    def query(self, trace_id: str) -> list[dict]:
        ...

    def replay(self, trace_id: str) -> list[dict]:
        ...


class TraceHandle:
    """A handle to an active trace being recorded."""

    def __init__(self, trace_id: str, request_id: str):
        self.trace_id = trace_id
        self.request_id = request_id
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "created_at": self.created_at,
        }


class TraceStore(TraceStoreProtocol):
    """Records and queries execution traces via EventStore."""

    STREAM = "trace"

    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.db = event_store.db

    def start_trace(self, request_id: str = "") -> TraceHandle:
        """Begin a new execution trace."""
        if not request_id:
            request_id = f"req_{uuid.uuid4().hex[:12]}"
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        return TraceHandle(trace_id=trace_id, request_id=request_id)

    def add_step(self, trace_id: str, step_type: str,
                 input_data: dict | None = None,
                 output_data: dict | None = None,
                 status: str = "success",
                 error: str | None = None) -> str:
        """Record a step in the trace.

        Args:
            trace_id: The trace this step belongs to.
            step_type: e.g. "plan", "retrieve", "execute", "verify", "writeback".
            input_data: Input to this step.
            output_data: Output from this step.
            status: "success" or "failed".
            error: Error message if status=failed.

        Returns:
            step_id.
        """
        step_id = f"step_{uuid.uuid4().hex[:8]}"
        self.db.execute(
            """INSERT INTO traces (trace_id, step_id, step_type, input, output, status, error, seq_num)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (trace_id, step_id, step_type,
             json.dumps(input_data or {}, default=str),
             json.dumps(output_data or {}, default=str),
             status, error,
             self.event_store.latest_seq(self.STREAM) + 1),
        )
        self.db.commit()
        return step_id

    def query(self, trace_id: str) -> list[dict]:
        """Get all steps for a trace."""
        rows = self.db.execute(
            "SELECT * FROM traces WHERE trace_id = ? ORDER BY created_at",
            (trace_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def replay(self, trace_id: str) -> list[dict]:
        """Replay trace steps (alias for query, may add projection logic later)."""
        return self.query(trace_id)

    @staticmethod
    def _row_to_dict(row) -> dict:
        d = dict(row)
        for field in ("input", "output"):
            if isinstance(d.get(field), str):
                try:
                    d[field] = json.loads(d[field])
                except json.JSONDecodeError:
                    pass
        return d
