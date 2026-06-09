"""Canonical 40-scenario, six-schedule SemStateBench history generator."""

from dataclasses import asdict, dataclass
from typing import Any

from semstate.models import (
    AnomalyType,
    DecisionAction,
    DependencyEdge,
    EdgeKind,
    EvidenceRef,
    StateNode,
    StateWrite,
    TransactionEnvelope,
)


SCHEDULES = (
    "serial_valid",
    "independent_valid",
    "same_key_race",
    "semantic_write_skew",
    "stale_artifact",
    "evidence_drift",
)


@dataclass(frozen=True)
class BenchmarkHistory:
    scenario_id: str
    history_id: str
    domain: str
    schedule: str
    initial_nodes: list[StateNode]
    envelope: TransactionEnvelope
    expected_action: DecisionAction
    expected_anomaly: AnomalyType | None
    task_count: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "history_id": self.history_id,
            "domain": self.domain,
            "schedule": self.schedule,
            "initial_nodes": [_jsonable(asdict(node)) for node in self.initial_nodes],
            "envelope": self.envelope.to_dict(),
            "expected_action": self.expected_action.value,
            "expected_anomaly": (
                self.expected_anomaly.value if self.expected_anomaly else None
            ),
            "task_count": self.task_count,
        }


def build_canonical_histories(
    scenario_count: int = 40,
) -> list[BenchmarkHistory]:
    domains = ("deployment", "migration", "pipeline")
    histories = []
    for index in range(scenario_count):
        domain = domains[index % len(domains)]
        scenario_id = f"{domain}_{index + 1:02d}"
        histories.extend(_domain_histories(domain, scenario_id, index))
    return histories


def _domain_histories(
    domain: str,
    scenario_id: str,
    index: int,
) -> list[BenchmarkHistory]:
    version = index + 1
    if domain == "deployment":
        nodes = [
            _node("deploy:database", version, "task:db",
                  max_connections=100 + index, reserve=10),
            _node("deploy:service", version, "task:service",
                  connection_pool=40),
        ]
        source = "deploy:database"
        target = "deploy:service"
        valid_write = StateWrite(target, {"connection_pool": 60})
        independent = StateWrite("deploy:replicas", {"count": 3})
        invalid_write = StateWrite(
            target,
            {"connection_pool": 100 + index},
        )
        artifact = StateWrite("deploy:plan", {"approved": True})
    elif domain == "migration":
        nodes = [
            _node("migration:database", version, "task:db", schema_version=4),
            _node("migration:app", version, "task:app", min_schema_version=3),
        ]
        source = "migration:database"
        target = "migration:app"
        valid_write = StateWrite(target, {"min_schema_version": 4})
        independent = StateWrite("migration:backup", {"ready": True})
        invalid_write = StateWrite(target, {"min_schema_version": 6})
        artifact = StateWrite("migration:plan", {"target_schema": 5})
    else:
        nodes = [
            _node("pipeline:producer", version, "task:producer",
                  output_schema="v2"),
            _node("pipeline:consumer", version, "task:consumer",
                  expected_schema="v2"),
        ]
        source = "pipeline:producer"
        target = "pipeline:consumer"
        valid_write = StateWrite(target, {"expected_schema": "v2"})
        independent = StateWrite("pipeline:monitor", {"enabled": True})
        invalid_write = StateWrite(source, {"output_schema": "v3"})
        artifact = StateWrite("pipeline:artifact", {"schema": "v2"})

    current_reads = {node.key: node.version for node in nodes}
    histories = [
        _history(
            scenario_id, domain, "serial_valid", nodes,
            current_reads, valid_write, DecisionAction.COMMIT, None,
        ),
        _history(
            scenario_id, domain, "independent_valid", nodes,
            current_reads, independent, DecisionAction.COMMIT, None,
        ),
        _history(
            scenario_id, domain, "same_key_race", nodes,
            {target: version - 1}, valid_write,
            DecisionAction.REJECT, AnomalyType.SAME_KEY_CONFLICT,
        ),
        _history(
            scenario_id, domain, "semantic_write_skew", nodes,
            current_reads, invalid_write,
            DecisionAction.REJECT, AnomalyType.CROSS_KEY_CONSTRAINT,
        ),
        _history(
            scenario_id, domain, "stale_artifact", nodes,
            {}, artifact,
            DecisionAction.REJECT, AnomalyType.DERIVED_ARTIFACT_STALE,
            dependencies=[DependencyEdge(
                source=source,
                target=artifact.key,
                source_version=version - 1,
                origin="derived",
                kind=EdgeKind.HARD,
            )],
        ),
        _history(
            scenario_id, domain, "evidence_drift", nodes,
            {}, artifact,
            DecisionAction.REJECT, AnomalyType.EVIDENCE_VERSION_MISMATCH,
            evidence=[EvidenceRef(
                source_key=source,
                version=version - 1,
                claim="precondition evidence",
            )],
        ),
    ]
    return histories


def _history(
    scenario_id,
    domain,
    schedule,
    nodes,
    read_set,
    write,
    action,
    anomaly,
    *,
    dependencies=None,
    evidence=None,
) -> BenchmarkHistory:
    return BenchmarkHistory(
        scenario_id=scenario_id,
        history_id=f"{scenario_id}:{schedule}",
        domain=domain,
        schedule=schedule,
        initial_nodes=nodes,
        envelope=TransactionEnvelope(
            agent_id="semstate-bench",
            task_id=f"task:{scenario_id}:{schedule}",
            domain=domain,
            read_set=read_set,
            write_set={write.key: write},
            dependencies=dependencies or [],
            evidence=evidence or [],
            txn_id=f"sem_txn:{scenario_id}:{schedule}",
        ),
        expected_action=action,
        expected_anomaly=anomaly,
    )


def _node(key, version, task, **value) -> StateNode:
    return StateNode(
        key=key,
        version=version,
        value=value,
        producer_task=task,
    )


def _jsonable(value):
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
