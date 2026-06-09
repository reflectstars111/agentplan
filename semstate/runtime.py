"""Commit-time semantic validation, invalidation, and selective repair."""

from dataclasses import replace

from ard.infra.db import Database
from semstate.models import (
    AnomalyType,
    CommitResult,
    DecisionAction,
    EdgeKind,
    NodeStatus,
    RepairPlan,
    StateNode,
    StateWrite,
    TransactionEnvelope,
    ValidationDecision,
)
from semstate.rules import register_builtin_rules
from semstate.store import SemStateStore
from semstate.validation import DeterministicValidator


class SemStateRuntime:
    def __init__(
        self,
        db: Database,
        *,
        validator: DeterministicValidator | None = None,
        register_defaults: bool = True,
    ):
        self.db = db
        self.db.init_schema()
        self.store = SemStateStore(db)
        self.validator = validator or DeterministicValidator()
        if register_defaults:
            register_builtin_rules(self.validator)

    def validate(self, envelope: TransactionEnvelope) -> ValidationDecision:
        return self.validator.validate(envelope, self.store)

    def seed_node(self, node: StateNode) -> None:
        with self.db.transaction(immediate=True):
            self.store.seed_node(node)

    def commit(self, envelope: TransactionEnvelope) -> CommitResult:
        try:
            with self.db.transaction(immediate=True):
                decision = self.validate(envelope)
                if decision.action == DecisionAction.REJECT:
                    raise _RejectedCommit(decision)

                write_status = (
                    NodeStatus.NEEDS_VERIFICATION
                    if decision.action == DecisionAction.MARK_UNCERTAIN
                    else NodeStatus.VALID
                )
                versions = {}
                for key, write in envelope.write_set.items():
                    version = self.store.current_version(key) + 1
                    self.store.upsert_node(
                        write,
                        version=version,
                        status=write_status,
                        producer_task=envelope.task_id,
                    )
                    versions[key] = version

                for target in envelope.write_set:
                    edges = [
                        edge
                        for edge in envelope.dependencies
                        if edge.target == target
                    ]
                    self.store.replace_dependencies(target, edges)

                affected = self._propagate_invalidations(set(envelope.write_set))
                conflict_id = None
                if affected or decision.action == DecisionAction.MARK_UNCERTAIN:
                    conflict_decision = ValidationDecision(
                        action=DecisionAction.MARK_UNCERTAIN,
                        anomaly_type=(
                            decision.anomaly_type
                            or AnomalyType.DERIVED_ARTIFACT_STALE
                        ),
                        evidence=decision.evidence,
                        affected_states=sorted(
                            set(affected) | set(decision.affected_states)
                        ),
                    )
                    conflict_id = self.store.save_conflict(
                        conflict_decision,
                        envelope,
                    )

                return CommitResult(
                    committed=True,
                    decision=replace(decision, conflict_id=conflict_id),
                    committed_versions=versions,
                    affected_states=affected,
                    conflict_id=conflict_id,
                )
        except _RejectedCommit as rejected:
            conflict_id = self.store.save_conflict(rejected.decision, envelope)
            decision = replace(rejected.decision, conflict_id=conflict_id)
            return CommitResult(
                committed=False,
                decision=decision,
                affected_states=decision.affected_states,
                conflict_id=conflict_id,
            )

    def repair(self, conflict_id: str) -> RepairPlan:
        conflict = self.store.get_conflict(conflict_id)
        if not conflict:
            raise KeyError(f"Unknown conflict: {conflict_id}")

        selected = set(conflict["decision"].get("affected_states", []))
        queue = list(selected)
        while queue:
            source = queue.pop(0)
            for edge in self.store.outgoing(source):
                node = self.store.get_node(edge.target)
                if (
                    node
                    and node.status != NodeStatus.VALID
                    and edge.target not in selected
                ):
                    selected.add(edge.target)
                    queue.append(edge.target)

        selected = {
            key
            for key in selected
            if (
                (node := self.store.get_node(key))
                and node.status != NodeStatus.VALID
            )
        }
        order = self._topological_order(selected)
        tasks = []
        for key in order:
            task = self.store.get_node(key).producer_task
            if task and task not in tasks:
                tasks.append(task)
        return RepairPlan(
            conflict_id=conflict_id,
            invalid_nodes=sorted(selected),
            rerun_tasks=tasks,
            topological_order=order,
            estimated_cost=float(len(tasks) or len(selected)),
        )

    def mark_revalidated(self, conflict_id: str, keys: list[str]) -> None:
        with self.db.transaction(immediate=True):
            for key in keys:
                if self.store.get_node(key):
                    self.store.refresh_dependency_versions(key)
                    self.store.set_status(key, NodeStatus.VALID)
            conflict = self.store.get_conflict(conflict_id)
            affected = conflict["decision"].get("affected_states", []) if conflict else []
            if all(
                not self.store.get_node(key)
                or self.store.get_node(key).status == NodeStatus.VALID
                for key in affected
            ):
                self.store.resolve_conflict(conflict_id)

    def _propagate_invalidations(self, changed: set[str]) -> list[str]:
        affected = set()
        queue = list(changed)
        while queue:
            source = queue.pop(0)
            for edge in self.store.outgoing(source):
                if edge.target in changed:
                    continue
                target = self.store.get_node(edge.target)
                if not target:
                    continue
                status = (
                    NodeStatus.STALE
                    if edge.kind == EdgeKind.HARD
                    else NodeStatus.NEEDS_VERIFICATION
                )
                self.store.set_status(edge.target, status)
                if edge.target not in affected:
                    affected.add(edge.target)
                    if edge.kind == EdgeKind.HARD:
                        queue.append(edge.target)
        return sorted(affected)

    def _topological_order(self, selected: set[str]) -> list[str]:
        if not selected:
            return []
        outgoing = {key: [] for key in selected}
        indegree = {key: 0 for key in selected}
        for edge in self.store.all_dependencies():
            if edge.source in selected and edge.target in selected:
                outgoing[edge.source].append(edge.target)
                indegree[edge.target] += 1
        ready = sorted(key for key, degree in indegree.items() if degree == 0)
        order = []
        while ready:
            key = ready.pop(0)
            order.append(key)
            for target in sorted(outgoing[key]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort()
        order.extend(sorted(selected - set(order)))
        return order


class _RejectedCommit(RuntimeError):
    def __init__(self, decision: ValidationDecision):
        super().__init__(decision.anomaly_type.value if decision.anomaly_type else "rejected")
        self.decision = decision
