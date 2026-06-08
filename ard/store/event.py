"""Immutable StoreEvent — the write-ahead log entry.

Each event is a fact: once written, it can never be modified or deleted.
seq_num (monotonically increasing) serves as the MVCC version identifier.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


@dataclass(frozen=True)
class StoreEvent:
    """A single immutable event in the write-ahead log.

    Fields:
        event_id: Globally unique identifier (UUID7-ish).
        seq_num: Monotonic sequence number assigned by EventStore on append.
                 -1 means not yet appended. This is the MVCC version.
        stream: Logical partition — "state", "knowledge", or "trace".
        stream_key: Entity key within the stream, e.g. "agent:agent_001".
        event_type: "created" | "updated" | "archived" | "deleted".
        payload: The changed content as a JSON-serializable dict.
        txn_id: ID of the owning transaction.
        causation_seq: seq_num of the prior event in the same causal chain.
        timestamp: ISO8601 timestamp, set on creation.
    """
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex}")
    seq_num: int = -1
    stream: str = "state"
    stream_key: str = ""
    event_type: str = "created"
    payload: dict = field(default_factory=dict)
    txn_id: str = ""
    causation_seq: int = -1
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def with_seq(self, seq_num: int) -> "StoreEvent":
        """Return a new event with seq_num set (since the dataclass is frozen)."""
        return StoreEvent(
            event_id=self.event_id,
            seq_num=seq_num,
            stream=self.stream,
            stream_key=self.stream_key,
            event_type=self.event_type,
            payload=self.payload,
            txn_id=self.txn_id,
            causation_seq=self.causation_seq,
            timestamp=self.timestamp,
        )
