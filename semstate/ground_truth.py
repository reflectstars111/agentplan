"""Independent rule reconstruction for canonical benchmark labels."""

from dataclasses import dataclass

from semstate.histories import BenchmarkHistory
from semstate.models import AnomalyType, DecisionAction


@dataclass(frozen=True)
class GroundTruth:
    action: DecisionAction
    anomaly: AnomalyType | None
    reason: str


def reconstruct_ground_truth(history: BenchmarkHistory) -> GroundTruth:
    versions = {node.key: node.version for node in history.initial_nodes}
    for key, expected in history.envelope.read_set.items():
        current = versions.get(key, 0)
        if current != expected:
            anomaly = (
                AnomalyType.SAME_KEY_CONFLICT
                if key in history.envelope.write_set
                else AnomalyType.STALE_DEPENDENCY
            )
            return GroundTruth(
                DecisionAction.REJECT,
                anomaly,
                f"{key} version changed",
            )

    for edge in history.envelope.dependencies:
        if versions.get(edge.source, 0) != edge.source_version:
            anomaly = (
                AnomalyType.DERIVED_ARTIFACT_STALE
                if edge.origin == "derived"
                else AnomalyType.STALE_DEPENDENCY
            )
            action = (
                DecisionAction.REJECT
                if edge.kind.value == "hard"
                else DecisionAction.MARK_UNCERTAIN
            )
            return GroundTruth(action, anomaly, f"{edge.source} dependency changed")

    for write in history.envelope.write_set.values():
        for field, type_name in write.schema.items():
            value = write.value.get(field)
            expected_type = {
                "str": str,
                "int": int,
                "float": (int, float),
                "bool": bool,
                "dict": dict,
                "list": list,
            }[type_name]
            if field not in write.value or not isinstance(value, expected_type):
                return GroundTruth(
                    DecisionAction.REJECT,
                    AnomalyType.SCHEMA_VIOLATION,
                    f"{write.key}.{field} schema mismatch",
                )

    proposed = {
        node.key: dict(node.value)
        for node in history.initial_nodes
    }
    proposed.update({
        key: dict(write.value)
        for key, write in history.envelope.write_set.items()
    })
    reason = _domain_violation(history.domain, proposed)
    if reason:
        return GroundTruth(
            DecisionAction.REJECT,
            AnomalyType.CROSS_KEY_CONSTRAINT,
            reason,
        )

    for evidence in history.envelope.evidence:
        if versions.get(evidence.source_key, 0) != evidence.version:
            return GroundTruth(
                DecisionAction.REJECT,
                AnomalyType.EVIDENCE_VERSION_MISMATCH,
                f"{evidence.source_key} evidence changed",
            )
    return GroundTruth(DecisionAction.COMMIT, None, "all deterministic checks pass")


def _domain_violation(domain: str, state: dict[str, dict]) -> str | None:
    if domain == "deployment":
        service = state.get("deploy:service")
        database = state.get("deploy:database")
        if service and database:
            pool = service.get("connection_pool")
            available = database.get("max_connections", 0) - database.get("reserve", 0)
            if isinstance(pool, int) and pool > available:
                return "connection pool exceeds available capacity"
    elif domain == "migration":
        app = state.get("migration:app")
        database = state.get("migration:database")
        if app and database:
            required = app.get("min_schema_version")
            actual = database.get("schema_version")
            if isinstance(required, int) and isinstance(actual, int) and required > actual:
                return "application requires a newer schema"
    elif domain == "pipeline":
        producer = state.get("pipeline:producer")
        consumer = state.get("pipeline:consumer")
        if producer and consumer:
            produced = producer.get("output_schema")
            expected = consumer.get("expected_schema")
            if produced and expected and produced != expected:
                return "producer and consumer schemas differ"
    return None
