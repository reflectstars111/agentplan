"""TraceLogger — execution trace recording and retrieval.

Persists Trace and TraceStep objects to SQLite for audit and debugging.
Maps to agent_os_initial_plan.md §15.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional
from src.db.connection import Database
from src.models.trace import Trace, TraceStep, StepType, StepStatus


class TraceLogger:
    """Records execution traces for multi-agent observability.

    Each trace is a sequence of TraceStep objects representing the
    complete execution path of a single user request.
    """

    def __init__(self, db: Database):
        self.db = db

    def start_trace(
        self,
        request_id: str,
        parent_trace_id: str | None = None,
    ) -> Trace:
        """Create a new execution trace. Returns the Trace object."""
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        self.db.execute(
            """
            INSERT INTO traces
                (trace_id, request_id, parent_trace_id, steps, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (trace_id, request_id, parent_trace_id, json.dumps([]), now),
        )
        self.db.commit()

        return Trace(
            trace_id=trace_id,
            request_id=request_id,
            parent_trace_id=parent_trace_id,
            steps=[],
        )

    def add_step(self, trace_id: str, step: TraceStep) -> None:
        """Append a step to an existing trace."""
        row = self.db.execute(
            "SELECT steps FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()

        if row is None:
            raise ValueError(f"Trace '{trace_id}' not found")

        steps_data = json.loads(row["steps"])
        steps_data.append({
            "step_id": step.step_id,
            "type": step.type.value,
            "input": step.input,
            "output": step.output,
            "status": step.status.value,
            "error": step.error,
            "timestamp": step.timestamp.isoformat() if hasattr(step.timestamp, 'isoformat') else str(step.timestamp),
        })

        self.db.execute(
            "UPDATE traces SET steps = ? WHERE trace_id = ?",
            (json.dumps(steps_data), trace_id),
        )
        self.db.commit()

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Retrieve a complete trace by ID."""
        row = self.db.execute(
            "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()

        if row is None:
            return None

        return self._row_to_trace(dict(row))

    def list_recent(self, limit: int = 20) -> list[Trace]:
        """List most recent traces, newest first."""
        rows = self.db.execute(
            "SELECT * FROM traces ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_trace(dict(r)) for r in rows]

    def list_children(self, parent_trace_id: str) -> list[Trace]:
        """List direct child traces in creation order."""
        rows = self.db.execute(
            """
            SELECT * FROM traces
            WHERE parent_trace_id = ?
            ORDER BY created_at
            """,
            (parent_trace_id,),
        ).fetchall()
        return [self._row_to_trace(dict(row)) for row in rows]

    def _row_to_trace(self, row: dict) -> Trace:
        steps_data = json.loads(row.get("steps", "[]"))
        steps = []
        for s in steps_data:
            step = TraceStep(
                step_id=s.get("step_id", ""),
                type=StepType(s.get("type", "respond")),
                input=s.get("input", {}),
                output=s.get("output", {}),
                status=StepStatus(s.get("status", "success")),
                error=s.get("error"),
            )
            if s.get("timestamp"):
                try:
                    step.timestamp = datetime.fromisoformat(s["timestamp"])
                except (ValueError, TypeError):
                    pass
            steps.append(step)

        return Trace(
            trace_id=row["trace_id"],
            request_id=row["request_id"],
            parent_trace_id=row.get("parent_trace_id"),
            steps=steps,
        )
