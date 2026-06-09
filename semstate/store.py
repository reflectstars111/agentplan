"""SQLite projections for versioned SemState nodes and dependencies."""

from dataclasses import replace
import json
import uuid

from ard.infra.db import Database
from semstate.models import (
    DependencyEdge,
    EdgeKind,
    NodeStatus,
    StateNode,
    StateWrite,
    TransactionEnvelope,
    ValidationDecision,
)


class SemStateStore:
    def __init__(self, db: Database):
        self.db = db

    def get_node(self, key: str) -> StateNode | None:
        row = self.db.execute(
            "SELECT * FROM semstate_nodes WHERE state_key = ?",
            (key,),
        ).fetchone()
        return self._row_to_node(row) if row else None

    def list_nodes(self) -> list[StateNode]:
        rows = self.db.execute(
            "SELECT * FROM semstate_nodes ORDER BY state_key"
        ).fetchall()
        return [self._row_to_node(row) for row in rows]

    def current_version(self, key: str) -> int:
        node = self.get_node(key)
        return node.version if node else 0

    def upsert_node(
        self,
        write: StateWrite,
        *,
        version: int,
        status: NodeStatus,
        producer_task: str,
    ) -> StateNode:
        self.db.execute(
            """INSERT INTO semstate_nodes
               (state_key, value, version, node_type, status, producer_task, source_refs)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(state_key) DO UPDATE SET
               value=excluded.value, version=excluded.version,
               node_type=excluded.node_type, status=excluded.status,
               producer_task=excluded.producer_task,
               source_refs=excluded.source_refs, updated_at=datetime('now')""",
            (
                write.key,
                json.dumps(write.value, sort_keys=True),
                version,
                write.node_type,
                status.value,
                producer_task,
                json.dumps(write.source_refs),
            ),
        )
        self.db.commit()
        return StateNode(
            key=write.key,
            version=version,
            value=write.value,
            node_type=write.node_type,
            status=status,
            producer_task=producer_task,
            source_refs=write.source_refs,
        )

    def seed_node(self, node: StateNode) -> None:
        write = StateWrite(
            key=node.key,
            value=node.value,
            node_type=node.node_type,
            source_refs=node.source_refs,
        )
        self.upsert_node(
            write,
            version=node.version,
            status=node.status,
            producer_task=node.producer_task,
        )

    def set_status(self, key: str, status: NodeStatus) -> None:
        self.db.execute(
            """UPDATE semstate_nodes
               SET status = ?, updated_at = datetime('now')
               WHERE state_key = ?""",
            (status.value, key),
        )
        self.db.commit()

    def replace_dependencies(
        self,
        target: str,
        edges: list[DependencyEdge],
    ) -> None:
        self.db.execute(
            "DELETE FROM semstate_dependencies WHERE target_key = ?",
            (target,),
        )
        for edge in edges:
            self.db.execute(
                """INSERT INTO semstate_dependencies
                   (source_key, target_key, source_version, origin, confidence, edge_kind)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    edge.source,
                    edge.target,
                    edge.source_version,
                    edge.origin,
                    edge.confidence,
                    edge.kind.value,
                ),
            )
        self.db.commit()

    def outgoing(self, source: str) -> list[DependencyEdge]:
        rows = self.db.execute(
            """SELECT * FROM semstate_dependencies
               WHERE source_key = ? ORDER BY target_key""",
            (source,),
        ).fetchall()
        return [self._row_to_edge(row) for row in rows]

    def incoming(self, target: str) -> list[DependencyEdge]:
        rows = self.db.execute(
            """SELECT * FROM semstate_dependencies
               WHERE target_key = ? ORDER BY source_key""",
            (target,),
        ).fetchall()
        return [self._row_to_edge(row) for row in rows]

    def all_dependencies(self) -> list[DependencyEdge]:
        rows = self.db.execute(
            """SELECT * FROM semstate_dependencies
               ORDER BY source_key, target_key"""
        ).fetchall()
        return [self._row_to_edge(row) for row in rows]

    def refresh_dependency_versions(self, target: str) -> None:
        for edge in self.incoming(target):
            self.db.execute(
                """UPDATE semstate_dependencies SET source_version = ?
                   WHERE source_key = ? AND target_key = ?""",
                (self.current_version(edge.source), edge.source, target),
            )
        self.db.commit()

    def save_conflict(
        self,
        decision: ValidationDecision,
        envelope: TransactionEnvelope,
        *,
        conflict_id: str | None = None,
    ) -> str:
        conflict_id = conflict_id or f"conflict_{uuid.uuid4().hex[:12]}"
        stored_decision = replace(decision, conflict_id=conflict_id)
        self.db.execute(
            """INSERT INTO semstate_conflicts
               (conflict_id, txn_id, anomaly_type, decision, envelope)
               VALUES (?, ?, ?, ?, ?)""",
            (
                conflict_id,
                envelope.txn_id,
                decision.anomaly_type.value if decision.anomaly_type else None,
                json.dumps(stored_decision.to_dict(), sort_keys=True),
                json.dumps(envelope.to_dict(), sort_keys=True),
            ),
        )
        self.db.commit()
        return conflict_id

    def get_conflict(self, conflict_id: str) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM semstate_conflicts WHERE conflict_id = ?",
            (conflict_id,),
        ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["decision"] = json.loads(result["decision"])
        result["envelope"] = json.loads(result["envelope"])
        return result

    def resolve_conflict(self, conflict_id: str) -> None:
        self.db.execute(
            """UPDATE semstate_conflicts
               SET status = 'resolved', resolved_at = datetime('now')
               WHERE conflict_id = ?""",
            (conflict_id,),
        )
        self.db.commit()

    @staticmethod
    def _row_to_node(row) -> StateNode:
        return StateNode(
            key=row["state_key"],
            version=row["version"],
            value=json.loads(row["value"]),
            node_type=row["node_type"],
            status=NodeStatus(row["status"]),
            producer_task=row["producer_task"] or "",
            source_refs=json.loads(row["source_refs"] or "[]"),
        )

    @staticmethod
    def _row_to_edge(row) -> DependencyEdge:
        return DependencyEdge(
            source=row["source_key"],
            target=row["target_key"],
            source_version=row["source_version"],
            origin=row["origin"],
            confidence=row["confidence"],
            kind=EdgeKind(row["edge_kind"]),
        )
