"""Transaction Manager — ACID properties for state writes.

Implements optimistic locking:
1. begin() — start a transaction
2. Within the transaction: add events, record read_set
3. verify() — check read_set keys haven't been modified since read
4. commit() — atomically append all events + apply projections
5. rollback() — discard

Maps to ARD design §8 (Transaction Manager).
"""

import json
from datetime import datetime, timezone
import uuid

from ard.infra.logging import log
from ard.store.event import StoreEvent
from ard.store.event_store import EventStore


class Transaction:
    """A pending transaction collecting read and write sets."""

    def __init__(self, txn_id: str | None = None):
        self.txn_id = txn_id or f"txn_{uuid.uuid4().hex[:12]}"
        self.status = "pending"
        self.read_set: list[dict] = []       # [{stream, stream_key, read_at_seq}, ...]
        self.write_events: list[StoreEvent] = []
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.completed_at: str | None = None

    def add_event(self, event: StoreEvent) -> None:
        """Add a write event to this transaction."""
        event = StoreEvent(
            event_id=event.event_id,
            seq_num=-1,  # will be assigned on commit
            stream=event.stream,
            stream_key=event.stream_key,
            event_type=event.event_type,
            payload=event.payload,
            txn_id=self.txn_id,
            causation_seq=event.causation_seq,
            timestamp=event.timestamp,
        )
        self.write_events.append(event)

    def record_read(self, stream: str, stream_key: str, read_at_seq: int) -> None:
        """Record that we read a key at a specific version."""
        # Only record if not already in read_set
        for entry in self.read_set:
            if entry["stream"] == stream and entry["stream_key"] == stream_key:
                return
        self.read_set.append({
            "stream": stream,
            "stream_key": stream_key,
            "read_at_seq": read_at_seq,
        })


class TransactionManager:
    """Manages the lifecycle of transactions with optimistic locking."""

    def __init__(self, event_store: EventStore):
        self.event_store = event_store

    def begin(self) -> Transaction:
        """Start a new transaction."""
        txn = Transaction()
        self._save_txn(txn)
        log.debug("txn_begin", txn_id=txn.txn_id)
        return txn

    def verify(self, txn: Transaction) -> bool:
        """Optimistic lock check: have any read_set keys changed?

        Returns True if all read_set keys are at the same version we read.
        Returns False if any key has been modified by another transaction.
        """
        for entry in txn.read_set:
            current_seq = self.event_store.latest_seq(
                stream=entry["stream"],
                stream_key=entry["stream_key"],
            )
            if current_seq > entry["read_at_seq"]:
                log.info("txn_conflict", txn_id=txn.txn_id,
                         stream_key=entry["stream_key"],
                         read_at=entry["read_at_seq"], current=current_seq)
                return False
        return True

    def commit(self, txn: Transaction) -> list[int]:
        """Commit: verify, append all events atomically, save txn status.

        Returns list of assigned seq_nums.

        Raises RuntimeError if verification fails.
        """
        if txn.status != "pending":
            raise RuntimeError(f"Transaction {txn.txn_id} is {txn.status}, not pending")

        try:
            with self.event_store.db.transaction(immediate=True):
                if not self.verify(txn):
                    raise _TransactionConflict(
                        f"Transaction {txn.txn_id} conflict: read_set changed since read."
                        " Please retry."
                    )

                seq_nums = [
                    self.event_store.append(event)
                    for event in txn.write_events
                ]

                for event, seq_num in zip(txn.write_events, seq_nums):
                    self.event_store.projections.apply(
                        f"{event.stream}.{event.event_type}",
                        {
                            **event.payload,
                            "_seq_num": seq_num,
                            "_stream_key": event.stream_key,
                            "event_type": event.event_type,
                        },
                    )

                txn.status = "committed"
                txn.completed_at = datetime.now(timezone.utc).isoformat()
                self._save_txn(txn)
        except _TransactionConflict as exc:
            txn.status = "rolled_back"
            txn.completed_at = datetime.now(timezone.utc).isoformat()
            self._save_txn(txn)
            raise RuntimeError(str(exc)) from exc
        except Exception as exc:
            log.error("txn_atomic_failure", txn_id=txn.txn_id, error=str(exc))
            txn.status = "rolled_back"
            txn.completed_at = datetime.now(timezone.utc).isoformat()
            self._save_txn(txn)
            raise

        log.info("txn_commit", txn_id=txn.txn_id, events=len(seq_nums))
        return seq_nums

    def rollback(self, txn: Transaction) -> None:
        """Roll back — discard all pending events."""
        txn.status = "rolled_back"
        txn.completed_at = datetime.now(timezone.utc).isoformat()
        txn.write_events.clear()
        self._save_txn(txn)
        log.info("txn_rollback", txn_id=txn.txn_id)

    def get_txn(self, txn_id: str) -> dict | None:
        row = self.event_store.db.execute(
            "SELECT * FROM transactions WHERE txn_id = ?", (txn_id,)
        ).fetchone()
        return dict(row) if row else None

    def _save_txn(self, txn: Transaction) -> None:
        self.event_store.db.execute(
            """INSERT INTO transactions (txn_id, status, read_set, event_count, created_at, completed_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(txn_id) DO UPDATE SET
               status=excluded.status, read_set=excluded.read_set,
               event_count=excluded.event_count, completed_at=excluded.completed_at""",
            (txn.txn_id, txn.status, json.dumps(txn.read_set),
             len(txn.write_events), txn.created_at, txn.completed_at),
        )
        self.event_store.db.commit()


class _TransactionConflict(RuntimeError):
    """Internal signal used to roll back before persisting conflict status."""
