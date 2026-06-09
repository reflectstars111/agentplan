"""Deterministic domain constraints used by SemStateBench."""

from semstate.models import AnomalyType, TransactionEnvelope
from semstate.validation import ValidationIssue


def deployment_capacity_rule(
    state: dict[str, dict],
    envelope: TransactionEnvelope,
) -> list[ValidationIssue]:
    service = state.get("deploy:service")
    database = state.get("deploy:database")
    if not service or not database:
        return []
    pool = service.get("connection_pool")
    maximum = database.get("max_connections")
    reserve = database.get("reserve", 0)
    if isinstance(pool, int) and isinstance(maximum, int) and pool > maximum - reserve:
        return [_constraint(
            "Service connection pool exceeds available database connections",
            ["deploy:service", "deploy:database"],
            {"connection_pool": pool, "available_connections": maximum - reserve},
        )]
    return []


def migration_compatibility_rule(
    state: dict[str, dict],
    envelope: TransactionEnvelope,
) -> list[ValidationIssue]:
    app = state.get("migration:app")
    database = state.get("migration:database")
    if not app or not database:
        return []
    required = app.get("min_schema_version")
    actual = database.get("schema_version")
    if isinstance(required, int) and isinstance(actual, int) and required > actual:
        return [_constraint(
            "Application requires a newer database schema",
            ["migration:app", "migration:database"],
            {"required_schema": required, "actual_schema": actual},
        )]
    return []


def pipeline_schema_rule(
    state: dict[str, dict],
    envelope: TransactionEnvelope,
) -> list[ValidationIssue]:
    producer = state.get("pipeline:producer")
    consumer = state.get("pipeline:consumer")
    if not producer or not consumer:
        return []
    output_schema = producer.get("output_schema")
    expected_schema = consumer.get("expected_schema")
    if output_schema and expected_schema and output_schema != expected_schema:
        return [_constraint(
            "Pipeline producer and consumer schemas are incompatible",
            ["pipeline:producer", "pipeline:consumer"],
            {"output_schema": output_schema, "expected_schema": expected_schema},
        )]
    return []


def register_builtin_rules(validator) -> None:
    validator.register_domain_rule(
        "deployment", "deployment_capacity", deployment_capacity_rule
    )
    validator.register_domain_rule(
        "migration", "migration_compatibility", migration_compatibility_rule
    )
    validator.register_domain_rule(
        "pipeline", "pipeline_schema", pipeline_schema_rule
    )


def _constraint(
    message: str,
    affected: list[str],
    details: dict,
) -> ValidationIssue:
    return ValidationIssue(
        anomaly_type=AnomalyType.CROSS_KEY_CONSTRAINT,
        message=message,
        affected_states=affected,
        details=details,
    )
