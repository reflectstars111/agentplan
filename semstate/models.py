"""Stable data contracts used by the SemState research runtime."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
import uuid


class DecisionAction(str, Enum):
    COMMIT = "commit"
    REJECT = "reject"
    MARK_UNCERTAIN = "mark_uncertain"


class AnomalyType(str, Enum):
    SAME_KEY_CONFLICT = "same_key_conflict"
    STALE_DEPENDENCY = "stale_dependency"
    DERIVED_ARTIFACT_STALE = "derived_artifact_stale"
    CROSS_KEY_CONSTRAINT = "cross_key_constraint"
    EVIDENCE_VERSION_MISMATCH = "evidence_version_mismatch"
    SCHEMA_VIOLATION = "schema_violation"
    EXECUTABLE_CHECK_FAILED = "executable_check_failed"


class NodeStatus(str, Enum):
    VALID = "valid"
    STALE = "stale"
    INVALID = "invalid"
    NEEDS_VERIFICATION = "needs_verification"


class EdgeKind(str, Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True)
class EvidenceRef:
    source_key: str
    version: int
    claim: str = ""


@dataclass(frozen=True)
class DependencyEdge:
    source: str
    target: str
    source_version: int
    origin: str = "read_set"
    confidence: float = 1.0
    kind: EdgeKind = EdgeKind.HARD


@dataclass(frozen=True)
class StateWrite:
    key: str
    value: dict[str, Any]
    node_type: str = "generic"
    schema: dict[str, str] = field(default_factory=dict)
    source_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StateNode:
    key: str
    version: int
    value: dict[str, Any]
    node_type: str = "generic"
    status: NodeStatus = NodeStatus.VALID
    producer_task: str = ""
    source_refs: list[str] = field(default_factory=list)


@dataclass
class TransactionEnvelope:
    agent_id: str
    task_id: str
    read_set: dict[str, int] = field(default_factory=dict)
    write_set: dict[str, StateWrite] = field(default_factory=dict)
    dependencies: list[DependencyEdge] = field(default_factory=list)
    evidence: list[EvidenceRef] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    domain: str = ""
    txn_id: str = field(default_factory=lambda: f"sem_txn_{uuid.uuid4().hex[:12]}")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class ValidationDecision:
    action: DecisionAction
    anomaly_type: AnomalyType | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    affected_states: list[str] = field(default_factory=list)
    conflict_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class RepairPlan:
    conflict_id: str
    invalid_nodes: list[str]
    rerun_tasks: list[str]
    topological_order: list[str]
    estimated_cost: float


@dataclass(frozen=True)
class CommitResult:
    committed: bool
    decision: ValidationDecision
    committed_versions: dict[str, int] = field(default_factory=dict)
    affected_states: list[str] = field(default_factory=list)
    conflict_id: str | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
