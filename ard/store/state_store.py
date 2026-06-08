"""StateStore — manages hierarchical state (L0-L5) with MVCC reads.

Read path: reads from state_snapshots projection table (always current).
History path: replays events from EventStore to reconstruct any version.
Write path: goes through TransactionManager (NOT direct writes).

Maps to ARD design §5: State replaces Memory as the unified abstraction.
"""

import json
from typing import Protocol, runtime_checkable

from ard.infra.logging import log
from ard.store.event_store import EventStore
from ard.store.transaction import Transaction


@runtime_checkable
class StateStoreProtocol(Protocol):
    """Protocol for state read/write operations."""

    def read(self, key: str, version: int | None = None) -> dict | None:
        ...

    def history(self, key: str) -> list[dict]:
        ...

    def list_keys(self, prefix: str = "") -> list[str]:
        ...

    def build_event(self, stream_key: str, event_type: str, payload: dict) -> "StoreEvent":
        ...


class StateStore(StateStoreProtocol):
    """Hierarchical state management backed by EventStore.

    State is organized by stream_key categories:
      state:agent:{agent_id}      — Agent state (L2)
      state:task:{task_id}        — Task state (L2)
      state:project:*             — Project knowledge (L3)
      state:memory:{memory_id}    — Long-term memory (L3)
      state:session:{id}          — Conversation session (L1)
      state:user:preferences      — User preferences (L3)
    """

    STREAM = "state"

    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.db = event_store.db

    # ── Read ───────────────────────────────────────────────

    def read(self, key: str, version: int | None = None) -> dict | None:
        """Read current state for a key, or a historical version.

        Args:
            key: stream_key (e.g. "agent:agent_001").
            version: If set, return state as of this seq_num.
                     If None, return latest projection.

        Returns:
            State dict or None if key doesn't exist.
        """
        if version is not None:
            return self._read_at_version(key, version)

        # Read from projection (current snapshot)
        row = self.db.execute(
            "SELECT value, version FROM state_snapshots WHERE stream_key = ?",
            (key,),
        ).fetchone()

        if not row:
            return None

        value = row["value"]
        if isinstance(value, str):
            value = json.loads(value)
        value["_version"] = row["version"]
        return value

    def _read_at_version(self, key: str, version: int) -> dict | None:
        """Reconstruct state at a specific version by replaying events."""
        events = self.event_store.replay(
            after_seq=0, stream=self.STREAM, stream_key=key, limit=500
        )

        if not events:
            return None

        state: dict = {}
        for evt in events:
            if evt.seq_num > version:
                break
            if evt.event_type == "deleted":
                state = {}
            else:
                state.update(evt.payload)

        state["_version"] = version
        return state if state else None

    def history(self, key: str) -> list[dict]:
        """Return the full event history for a key (all versions)."""
        events = self.event_store.replay(
            after_seq=0, stream=self.STREAM, stream_key=key, limit=500
        )
        return [
            {
                "seq_num": evt.seq_num,
                "event_id": evt.event_id,
                "event_type": evt.event_type,
                "payload": evt.payload,
                "txn_id": evt.txn_id,
                "timestamp": evt.timestamp,
            }
            for evt in events
        ]

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all state keys, optionally filtered by prefix."""
        if prefix:
            rows = self.db.execute(
                "SELECT stream_key FROM state_snapshots WHERE stream_key LIKE ?",
                (f"{prefix}%",),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT stream_key FROM state_snapshots"
            ).fetchall()
        return [r["stream_key"] for r in rows]

    # ── Write (via Transaction) ─────────────────────────────

    def build_event(self, stream_key: str, event_type: str,
                    payload: dict) -> "StoreEvent":
        """Build a StoreEvent for this state key. Not written until committed.

        Usage:
            txn = txn_mgr.begin()
            txn.add_event(state_store.build_event("agent:agent_001", "updated", {...}))
            txn_mgr.commit(txn)
        """
        from ard.store.event import StoreEvent
        return StoreEvent(
            stream=self.STREAM,
            stream_key=stream_key,
            event_type=event_type,
            payload=payload,
        )

    def read_for_transaction(self, key: str, txn: Transaction) -> dict | None:
        """Read state and record the read in the transaction's read_set.

        Call this instead of plain read() when inside a transaction,
        so the TransactionManager can detect conflicts.
        """
        value = self.read(key)
        current_seq = value.get("_version", 0) if value else self.event_store.latest_seq(
            stream=self.STREAM, stream_key=key
        )
        txn.record_read(self.STREAM, key, current_seq)
        return value

    # ── Projection handler ─────────────────────────────────

    def apply_event(self, event_data: dict) -> None:
        """Projection handler: update state_snapshots from event payload."""
        key = event_data.get("_stream_key", "")
        if not key:
            return

        seq_num = event_data.get("_seq_num", 0)
        event_type = event_data.get("event_type", "updated")

        if event_type == "deleted":
            self.db.execute("DELETE FROM state_snapshots WHERE stream_key = ?", (key,))
            self.db.commit()
            return

        payload = {k: v for k, v in event_data.items()
                   if not k.startswith("_")}

        # Merge with existing value
        existing = self.db.execute(
            "SELECT value FROM state_snapshots WHERE stream_key = ?", (key,)
        ).fetchone()

        if existing:
            old_val = existing["value"]
            if isinstance(old_val, str):
                old_val = json.loads(old_val)
            merged = {**old_val, **payload}
        else:
            merged = payload

        self.db.execute(
            """INSERT INTO state_snapshots (stream_key, value, version, stream)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(stream_key) DO UPDATE SET
               value=excluded.value, version=excluded.version, updated_at=datetime('now')""",
            (key, json.dumps(merged), seq_num, self.STREAM),
        )
        self.db.commit()
