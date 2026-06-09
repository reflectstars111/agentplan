"""Twelve hand-authored G0 cases for the initial SemStateBench."""

from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    domain: str
    description: str
    initial_nodes: list[StateNode]
    envelope: TransactionEnvelope
    expected_action: DecisionAction
    expected_anomaly: AnomalyType | None = None
    metadata: dict = field(default_factory=dict)


def build_g0_cases() -> list[BenchmarkCase]:
    return [
        _case(
            "deploy_valid_resize",
            "deployment",
            [_node("deploy:database", 2, max_connections=100, reserve=10),
             _node("deploy:service", 3, connection_pool=40)],
            _envelope(
                "deployment",
                {"deploy:database": 2, "deploy:service": 3},
                _write("deploy:service", connection_pool=70),
            ),
            DecisionAction.COMMIT,
        ),
        _case(
            "deploy_semantic_write_skew",
            "deployment",
            [_node("deploy:database", 2, max_connections=100, reserve=10),
             _node("deploy:service", 3, connection_pool=40)],
            _envelope(
                "deployment",
                {"deploy:database": 2, "deploy:service": 3},
                _write("deploy:service", connection_pool=95),
            ),
            DecisionAction.REJECT,
            AnomalyType.CROSS_KEY_CONSTRAINT,
            {"occ_would_commit": True},
        ),
        _case(
            "deploy_stale_hard_dependency",
            "deployment",
            [_node("deploy:database", 4, max_connections=120, reserve=10),
             _node("deploy:service", 2, connection_pool=50)],
            _envelope(
                "deployment",
                {"deploy:service": 2},
                _write("deploy:service", connection_pool=60),
                dependencies=[_edge(
                    "deploy:database", "deploy:service", 3, EdgeKind.HARD
                )],
            ),
            DecisionAction.REJECT,
            AnomalyType.STALE_DEPENDENCY,
        ),
        _case(
            "deploy_evidence_version_mismatch",
            "deployment",
            [_node("deploy:database", 4, max_connections=120, reserve=10),
             _node("deploy:service", 2, connection_pool=50)],
            _envelope(
                "deployment",
                {"deploy:service": 2},
                _write("deploy:service", connection_pool=60),
                evidence=[EvidenceRef("deploy:database", 3, "capacity audit")],
            ),
            DecisionAction.REJECT,
            AnomalyType.EVIDENCE_VERSION_MISMATCH,
        ),
        _case(
            "migration_valid_upgrade",
            "migration",
            [_node("migration:database", 5, schema_version=4),
             _node("migration:app", 2, min_schema_version=3)],
            _envelope(
                "migration",
                {"migration:database": 5, "migration:app": 2},
                _write("migration:app", min_schema_version=4),
            ),
            DecisionAction.COMMIT,
        ),
        _case(
            "migration_same_key_conflict",
            "migration",
            [_node("migration:database", 6, schema_version=5)],
            _envelope(
                "migration",
                {"migration:database": 5},
                _write("migration:database", schema_version=6),
            ),
            DecisionAction.REJECT,
            AnomalyType.SAME_KEY_CONFLICT,
        ),
        _case(
            "migration_derived_artifact_stale",
            "migration",
            [_node("migration:database", 6, schema_version=5),
             _node("migration:plan", 1, target_schema=5)],
            _envelope(
                "migration",
                {"migration:plan": 1},
                _write("migration:plan", target_schema=6),
                dependencies=[DependencyEdge(
                    source="migration:database",
                    target="migration:plan",
                    source_version=5,
                    origin="derived",
                    kind=EdgeKind.HARD,
                )],
            ),
            DecisionAction.REJECT,
            AnomalyType.DERIVED_ARTIFACT_STALE,
        ),
        _case(
            "migration_incompatible_app",
            "migration",
            [_node("migration:database", 5, schema_version=4),
             _node("migration:app", 2, min_schema_version=3)],
            _envelope(
                "migration",
                {"migration:database": 5, "migration:app": 2},
                _write("migration:app", min_schema_version=6),
            ),
            DecisionAction.REJECT,
            AnomalyType.CROSS_KEY_CONSTRAINT,
            {"occ_would_commit": True},
        ),
        _case(
            "pipeline_valid_schema",
            "pipeline",
            [_node("pipeline:producer", 2, output_schema="v2"),
             _node("pipeline:consumer", 4, expected_schema="v2")],
            _envelope(
                "pipeline",
                {"pipeline:producer": 2, "pipeline:consumer": 4},
                _write("pipeline:consumer", expected_schema="v2"),
            ),
            DecisionAction.COMMIT,
        ),
        _case(
            "pipeline_soft_dependency_uncertain",
            "pipeline",
            [_node("pipeline:producer", 3, output_schema="v2"),
             _node("pipeline:consumer", 4, expected_schema="v2")],
            _envelope(
                "pipeline",
                {"pipeline:consumer": 4},
                _write("pipeline:consumer", expected_schema="v2"),
                dependencies=[_edge(
                    "pipeline:producer", "pipeline:consumer", 2, EdgeKind.SOFT
                )],
            ),
            DecisionAction.MARK_UNCERTAIN,
            AnomalyType.STALE_DEPENDENCY,
        ),
        _case(
            "pipeline_schema_type_violation",
            "pipeline",
            [_node("pipeline:consumer", 4, expected_schema="v2")],
            _envelope(
                "pipeline",
                {"pipeline:consumer": 4},
                StateWrite(
                    key="pipeline:consumer",
                    value={"expected_schema": 2},
                    schema={"expected_schema": "str"},
                ),
            ),
            DecisionAction.REJECT,
            AnomalyType.SCHEMA_VIOLATION,
        ),
        _case(
            "pipeline_cross_key_mismatch",
            "pipeline",
            [_node("pipeline:producer", 2, output_schema="v2"),
             _node("pipeline:consumer", 4, expected_schema="v2")],
            _envelope(
                "pipeline",
                {"pipeline:producer": 2, "pipeline:consumer": 4},
                _write("pipeline:producer", output_schema="v3"),
            ),
            DecisionAction.REJECT,
            AnomalyType.CROSS_KEY_CONSTRAINT,
            {"occ_would_commit": True},
        ),
    ]


def _case(
    case_id,
    domain,
    nodes,
    envelope,
    action,
    anomaly=None,
    metadata=None,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        domain=domain,
        description=case_id.replace("_", " "),
        initial_nodes=nodes,
        envelope=envelope,
        expected_action=action,
        expected_anomaly=anomaly,
        metadata=metadata or {},
    )


def _node(key, version, **value) -> StateNode:
    return StateNode(
        key=key,
        version=version,
        value=value,
        producer_task=f"build:{key}",
    )


def _write(key, **value) -> StateWrite:
    return StateWrite(key=key, value=value)


def _edge(source, target, version, kind) -> DependencyEdge:
    return DependencyEdge(
        source=source,
        target=target,
        source_version=version,
        kind=kind,
    )


def _envelope(
    domain,
    read_set,
    write,
    *,
    dependencies=None,
    evidence=None,
) -> TransactionEnvelope:
    return TransactionEnvelope(
        agent_id="benchmark-agent",
        task_id=f"task:{domain}",
        domain=domain,
        read_set=read_set,
        write_set={write.key: write},
        dependencies=dependencies or [],
        evidence=evidence or [],
    )
