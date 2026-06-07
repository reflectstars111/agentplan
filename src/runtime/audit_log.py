"""AuditLog — structured query interface over execution traces.

Maps to agent_os_initial_plan.md §15 (Observability & Audit).
"""

import json
from datetime import datetime
from src.db.connection import Database
from src.runtime.trace_logger import TraceLogger


class AuditLog:
    """Structured audit queries on top of TraceLogger.

    Provides filtering by agent, request, time range, step type, and status.
    """

    def __init__(self, db: Database, trace_logger: TraceLogger):
        self.db = db
        self.trace_logger = trace_logger

    def list_by_agent(self, agent_id: str, limit: int = 50) -> list[dict]:
        """List traces involving a specific agent."""
        rows = self.db.execute(
            """SELECT trace_id, request_id, steps, created_at
               FROM traces WHERE steps LIKE ?
               ORDER BY created_at DESC LIMIT ?""",
            (f"%{agent_id}%", limit),
        ).fetchall()
        return [self._row_to_summary(dict(r)) for r in rows]

    def list_by_time_range(self, start: str, end: str,
                           limit: int = 50) -> list[dict]:
        """List traces in a time range (ISO datetime strings)."""
        rows = self.db.execute(
            "SELECT * FROM traces WHERE created_at >= ? AND created_at <= ? "
            "ORDER BY created_at DESC LIMIT ?",
            (start, end, limit),
        ).fetchall()
        return [self._row_to_summary(dict(r)) for r in rows]

    def list_errors(self, limit: int = 50) -> list[dict]:
        """List traces containing failed steps."""
        rows = self.db.execute(
            "SELECT * FROM traces WHERE steps LIKE ? ORDER BY created_at DESC LIMIT ?",
            ('%"status": "failed"%', limit),
        ).fetchall()
        return [self._row_to_summary(dict(r)) for r in rows]

    def get_step_summary(self, trace_id: str) -> dict:
        """Get a structured summary of a trace's steps."""
        trace = self.trace_logger.get_trace(trace_id)
        if not trace:
            return {}
        return {
            "trace_id": trace.trace_id,
            "request_id": trace.request_id,
            "step_count": len(trace.steps),
            "steps": [
                {
                    "type": s.type.value,
                    "status": s.status.value,
                    "timestamp": s.timestamp.isoformat() if hasattr(s.timestamp, 'isoformat') else str(s.timestamp),
                    "error": s.error,
                }
                for s in trace.steps
            ],
        }

    def _row_to_summary(self, row: dict) -> dict:
        steps = json.loads(row.get("steps", "[]"))
        error_count = sum(1 for s in steps if s.get("status") == "failed")
        return {
            "trace_id": row["trace_id"],
            "request_id": row["request_id"],
            "step_count": len(steps),
            "error_count": error_count,
            "created_at": row.get("created_at", ""),
        }
