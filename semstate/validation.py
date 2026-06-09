"""Ordered deterministic validation for semantic commits."""

from dataclasses import dataclass
from typing import Callable

from semstate.models import (
    AnomalyType,
    DecisionAction,
    EdgeKind,
    TransactionEnvelope,
    ValidationDecision,
)
from semstate.store import SemStateStore


@dataclass(frozen=True)
class ValidationIssue:
    anomaly_type: AnomalyType
    message: str
    affected_states: list[str]
    details: dict
    hard: bool = True

    def to_dict(self) -> dict:
        return {
            "anomaly_type": self.anomaly_type.value,
            "message": self.message,
            "affected_states": self.affected_states,
            "details": self.details,
            "hard": self.hard,
        }


Rule = Callable[[dict[str, dict], TransactionEnvelope], list[ValidationIssue]]


class DeterministicValidator:
    """Validate in the order fixed by PLANCCFA.

    Order: read versions, dependency versions, schema, domain rules,
    executable checks, and evidence versions. A soft dependency mismatch
    marks the commit uncertain; any hard issue rejects it.
    """

    _TYPES = {
        "str": str,
        "int": int,
        "float": (int, float),
        "bool": bool,
        "dict": dict,
        "list": list,
    }

    def __init__(self):
        self._domain_rules: dict[str, list[tuple[str, Rule]]] = {}
        self._executable_checks: dict[str, list[tuple[str, Rule]]] = {}

    def register_domain_rule(self, domain: str, name: str, rule: Rule) -> None:
        self._domain_rules.setdefault(domain, []).append((name, rule))

    def register_executable_check(self, domain: str, name: str, check: Rule) -> None:
        self._executable_checks.setdefault(domain, []).append((name, check))

    def validate(
        self,
        envelope: TransactionEnvelope,
        store: SemStateStore,
    ) -> ValidationDecision:
        uncertain: list[ValidationIssue] = []

        issues = self._check_read_versions(envelope, store)
        if issues:
            return self._reject(issues)

        hard, soft = self._check_dependency_versions(envelope, store)
        if hard:
            return self._reject(hard)
        uncertain.extend(soft)

        issues = self._check_schemas(envelope)
        if issues:
            return self._reject(issues)

        proposed = {
            node.key: dict(node.value)
            for node in store.list_nodes()
        }
        proposed.update({
            key: dict(write.value)
            for key, write in envelope.write_set.items()
        })

        issues = self._run_rules(
            self._domain_rules.get(envelope.domain, []),
            proposed,
            envelope,
        )
        if issues:
            return self._reject(issues)

        issues = self._run_rules(
            self._executable_checks.get(envelope.domain, []),
            proposed,
            envelope,
        )
        if issues:
            return self._reject(issues)

        issues = self._check_evidence_versions(envelope, store)
        if issues:
            return self._reject(issues)

        if uncertain:
            return ValidationDecision(
                action=DecisionAction.MARK_UNCERTAIN,
                anomaly_type=uncertain[0].anomaly_type,
                evidence=[issue.to_dict() for issue in uncertain],
                affected_states=self._affected(uncertain),
            )
        return ValidationDecision(action=DecisionAction.COMMIT)

    def _check_read_versions(
        self,
        envelope: TransactionEnvelope,
        store: SemStateStore,
    ) -> list[ValidationIssue]:
        issues = []
        for key, read_version in envelope.read_set.items():
            current = store.current_version(key)
            if current == read_version:
                continue
            anomaly = (
                AnomalyType.SAME_KEY_CONFLICT
                if key in envelope.write_set
                else AnomalyType.STALE_DEPENDENCY
            )
            issues.append(ValidationIssue(
                anomaly_type=anomaly,
                message=f"{key} changed from version {read_version} to {current}",
                affected_states=[key],
                details={"expected_version": read_version, "current_version": current},
            ))
        return issues

    def _check_dependency_versions(
        self,
        envelope: TransactionEnvelope,
        store: SemStateStore,
    ) -> tuple[list[ValidationIssue], list[ValidationIssue]]:
        hard = []
        soft = []
        for edge in envelope.dependencies:
            current = store.current_version(edge.source)
            if current == edge.source_version:
                continue
            anomaly = (
                AnomalyType.DERIVED_ARTIFACT_STALE
                if edge.origin == "derived"
                else AnomalyType.STALE_DEPENDENCY
            )
            issue = ValidationIssue(
                anomaly_type=anomaly,
                message=(
                    f"{edge.target} depends on {edge.source}@{edge.source_version}, "
                    f"current version is {current}"
                ),
                affected_states=[edge.source, edge.target],
                details={
                    "source": edge.source,
                    "target": edge.target,
                    "expected_version": edge.source_version,
                    "current_version": current,
                    "kind": edge.kind.value,
                    "origin": edge.origin,
                },
                hard=edge.kind == EdgeKind.HARD,
            )
            (hard if issue.hard else soft).append(issue)
        return hard, soft

    def _check_schemas(
        self,
        envelope: TransactionEnvelope,
    ) -> list[ValidationIssue]:
        issues = []
        for key, write in envelope.write_set.items():
            for field_name, type_name in write.schema.items():
                expected = self._TYPES.get(type_name)
                value = write.value.get(field_name)
                if expected is None:
                    raise ValueError(f"Unsupported schema type: {type_name}")
                if field_name not in write.value or not isinstance(value, expected):
                    issues.append(ValidationIssue(
                        anomaly_type=AnomalyType.SCHEMA_VIOLATION,
                        message=f"{key}.{field_name} must be {type_name}",
                        affected_states=[key],
                        details={
                            "field": field_name,
                            "expected_type": type_name,
                            "actual_type": type(value).__name__,
                        },
                    ))
        return issues

    @staticmethod
    def _run_rules(
        rules: list[tuple[str, Rule]],
        proposed: dict[str, dict],
        envelope: TransactionEnvelope,
    ) -> list[ValidationIssue]:
        issues = []
        for name, rule in rules:
            for issue in rule(proposed, envelope):
                details = dict(issue.details)
                details.setdefault("rule", name)
                issues.append(ValidationIssue(
                    anomaly_type=issue.anomaly_type,
                    message=issue.message,
                    affected_states=issue.affected_states,
                    details=details,
                    hard=issue.hard,
                ))
        return issues

    @staticmethod
    def _check_evidence_versions(
        envelope: TransactionEnvelope,
        store: SemStateStore,
    ) -> list[ValidationIssue]:
        issues = []
        for evidence in envelope.evidence:
            current = store.current_version(evidence.source_key)
            if current == evidence.version:
                continue
            issues.append(ValidationIssue(
                anomaly_type=AnomalyType.EVIDENCE_VERSION_MISMATCH,
                message=(
                    f"Evidence for {evidence.source_key} uses version "
                    f"{evidence.version}, current version is {current}"
                ),
                affected_states=[evidence.source_key],
                details={
                    "source_key": evidence.source_key,
                    "evidence_version": evidence.version,
                    "current_version": current,
                    "claim": evidence.claim,
                },
            ))
        return issues

    @staticmethod
    def _reject(issues: list[ValidationIssue]) -> ValidationDecision:
        return ValidationDecision(
            action=DecisionAction.REJECT,
            anomaly_type=issues[0].anomaly_type,
            evidence=[issue.to_dict() for issue in issues],
            affected_states=DeterministicValidator._affected(issues),
        )

    @staticmethod
    def _affected(issues: list[ValidationIssue]) -> list[str]:
        return sorted({
            key
            for issue in issues
            for key in issue.affected_states
        })
