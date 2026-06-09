"""EventStore — immutable write-ahead log.

All state changes go through EventStore.append(). Once written, an event
can never be modified or deleted. seq_num is auto-assigned and serves as
the MVCC version identifier for all projections.

Maps to ARD design: Event Store is the sole source of truth.
"""

import json
from datetime import datetime, timezone

from ard.infra.db import Database
from ard.infra.logging import log
from ard.store.event import StoreEvent
from ard.store.projections import Projections


class EventStore:
    """Immutable event log with auto-incrementing seq_num."""

    def __init__(self, db: Database, projections: Projections | None = None):
        self.db = db
        self.projections = projections or Projections()

    # ── Write ──────────────────────────────────────────────

    def append(self, event: StoreEvent) -> int:
        """Append a single event. Returns the assigned seq_num.

        Raises ValueError if the event_id already exists in the store.
        """
        if event.seq_num >= 0:
            raise ValueError(f"Event {event.event_id} already has seq_num={event.seq_num}")
        # Check for duplicate event_id
        existing = self.db.execute(
            "SELECT 1 FROM events WHERE event_id = ?", (event.event_id,)
        ).fetchone()
        if existing:
            raise ValueError(f"Event {event.event_id} already exists — events are immutable")

        payload_json = json.dumps(event.payload, default=str)
        c = self.db.execute(
            """INSERT INTO events (event_id, stream, stream_key, event_type,
               payload, txn_id, causation_seq, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (event.event_id, event.stream, event.stream_key, event.event_type,
             payload_json, event.txn_id,
             event.causation_seq if event.causation_seq >= 0 else None,
             event.timestamp or datetime.now(timezone.utc).isoformat()),
        )
        self.db.commit()
        seq_num = c.lastrowid
        log.debug("event_appended", event_id=event.event_id, seq_num=seq_num,
                  stream=event.stream, stream_key=event.stream_key)

        return seq_num

    def append_batch(self, events: list[StoreEvent]) -> list[int]:
        """Append multiple events atomically. Returns list of assigned seq_nums."""
        with self.db.transaction(immediate=True):
            return [self.append(event) for event in events]

    # ── Read ───────────────────────────────────────────────

    def replay(self, after_seq: int = 0, stream: str | None = None,
               stream_key: str | None = None, limit: int = 1000) -> list[StoreEvent]:
        """Replay events after a given seq_num, optionally filtered.

        Args:
            after_seq: Only return events with seq_num > after_seq.
            stream: Filter by stream name.
            stream_key: Filter by stream_key.
            limit: Max events to return.

        Returns:
            List of StoreEvent in seq_num order.
        """
        sql = "SELECT * FROM events WHERE seq_num > ?"
        params: list = [after_seq]

        if stream:
            sql += " AND stream = ?"
            params.append(stream)
        if stream_key:
            sql += " AND stream_key = ?"
            params.append(stream_key)

        sql += " ORDER BY seq_num ASC LIMIT ?"
        params.append(limit)

        rows = self.db.execute(sql, tuple(params)).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_by_seq(self, seq_num: int) -> StoreEvent | None:
        row = self.db.execute(
            "SELECT * FROM events WHERE seq_num = ?", (seq_num,)
        ).fetchone()
        return self._row_to_event(row) if row else None

    def latest_seq(self, stream: str | None = None,
                   stream_key: str | None = None) -> int:
        """Get the latest seq_num, optionally filtered by stream/stream_key."""
        conditions = []
        params: list = []
        if stream:
            conditions.append("stream = ?")
            params.append(stream)
        if stream_key:
            conditions.append("stream_key = ?")
            params.append(stream_key)

        if conditions:
            sql = f"SELECT MAX(seq_num) as mx FROM events WHERE {' AND '.join(conditions)}"
            row = self.db.execute(sql, tuple(params)).fetchone()
        else:
            row = self.db.execute("SELECT MAX(seq_num) as mx FROM events").fetchone()
        return (row["mx"] or 0) if row else 0

    def count(self) -> int:
        row = self.db.execute("SELECT COUNT(*) as cnt FROM events").fetchone()
        return row["cnt"] if row else 0

    def get_events_for_txn(self, txn_id: str) -> list[StoreEvent]:
        rows = self.db.execute(
            "SELECT * FROM events WHERE txn_id = ? ORDER BY seq_num",
            (txn_id,),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    # ── Helpers ────────────────────────────────────────────

    def _row_to_event(self, row) -> StoreEvent:
        payload = row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return StoreEvent(
            event_id=row["event_id"],
            seq_num=row["seq_num"],
            stream=row["stream"],
            stream_key=row["stream_key"],
            event_type=row["event_type"],
            payload=payload,
            txn_id=row["txn_id"],
            causation_seq=row["causation_seq"] or -1,
            timestamp=row["created_at"],
        )
