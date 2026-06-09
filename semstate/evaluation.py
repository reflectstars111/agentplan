"""Deterministic baseline evaluation and primary SemState metrics."""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable

from ard.infra.db import Database
from semstate.histories import BenchmarkHistory
from semstate.models import DecisionAction
from semstate.runtime import SemStateRuntime


class Baseline(str, Enum):
    NO_VALIDATION = "no_validation"
    LWW = "lww"
    READ_SET_OCC = "read_set_occ"
    ORACLE_DEPENDENCY = "oracle_dependency"
    INDEPENDENT_VERIFIER = "independent_verifier"
    FULL_RERUN = "full_rerun"
    SEMSTATE_TRACE_ONLY = "semstate_trace_only"
    SEMSTATE_FULL = "semstate_full"


@dataclass(frozen=True)
class EvaluationRecord:
    history_id: str
    scenario_id: str
    baseline: Baseline
    expected_conflict: bool
    predicted_conflict: bool
    committed: bool
    invalid_commit: bool
    final_state_correct: bool
    false_rejection: bool
    repair_calls: int


def evaluate_history(
    history: BenchmarkHistory,
    baseline: Baseline,
) -> EvaluationRecord:
    expected_conflict = history.expected_action != DecisionAction.COMMIT
    predicted_conflict, committed, repair_calls = _run_baseline(history, baseline)
    invalid_commit = expected_conflict and committed
    false_rejection = not expected_conflict and not committed
    final_state_correct = not invalid_commit and not false_rejection
    return EvaluationRecord(
        history_id=history.history_id,
        scenario_id=history.scenario_id,
        baseline=baseline,
        expected_conflict=expected_conflict,
        predicted_conflict=predicted_conflict,
        committed=committed,
        invalid_commit=invalid_commit,
        final_state_correct=final_state_correct,
        false_rejection=false_rejection,
        repair_calls=repair_calls,
    )


def evaluate_baselines(
    histories: Iterable[BenchmarkHistory],
    baselines: Iterable[Baseline] | None = None,
) -> dict[str, dict]:
    history_list = list(histories)
    selected = list(baselines or Baseline)
    return {
        baseline.value: aggregate([
            evaluate_history(history, baseline)
            for history in history_list
        ])
        for baseline in selected
    }


def aggregate(records: list[EvaluationRecord]) -> dict:
    count = len(records)
    actual_positive = sum(record.expected_conflict for record in records)
    predicted_positive = sum(record.predicted_conflict for record in records)
    true_positive = sum(
        record.expected_conflict and record.predicted_conflict
        for record in records
    )
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "histories": count,
        "invalid_commit_rate": (
            sum(record.invalid_commit for record in records) / actual_positive
            if actual_positive
            else 0.0
        ),
        "final_state_correctness": (
            sum(record.final_state_correct for record in records) / count
            if count
            else 0.0
        ),
        "false_rejection_rate": (
            sum(record.false_rejection for record in records)
            / max(count - actual_positive, 1)
        ),
        "conflict_precision": precision,
        "conflict_recall": recall,
        "conflict_f1": f1,
        "repair_calls": sum(record.repair_calls for record in records),
    }


def _run_baseline(
    history: BenchmarkHistory,
    baseline: Baseline,
) -> tuple[bool, bool, int]:
    if baseline in {Baseline.NO_VALIDATION, Baseline.LWW}:
        return False, True, 0
    if baseline == Baseline.READ_SET_OCC:
        versions = {node.key: node.version for node in history.initial_nodes}
        conflict = any(
            versions.get(key, 0) != expected
            for key, expected in history.envelope.read_set.items()
        )
        return conflict, not conflict, 0
    if baseline == Baseline.ORACLE_DEPENDENCY:
        conflict = history.expected_action != DecisionAction.COMMIT
        return conflict, not conflict, 0
    if baseline == Baseline.FULL_RERUN:
        conflict = history.expected_action != DecisionAction.COMMIT
        return conflict, not conflict, history.task_count if conflict else 0

    runtime = SemStateRuntime(
        Database(":memory:"),
        register_defaults=baseline != Baseline.SEMSTATE_TRACE_ONLY,
    )
    for node in history.initial_nodes:
        runtime.seed_node(node)

    envelope = history.envelope
    if baseline == Baseline.INDEPENDENT_VERIFIER:
        envelope = replace(envelope, dependencies=[], evidence=[])
    decision = runtime.validate(envelope)
    runtime.db.close()
    conflict = decision.action != DecisionAction.COMMIT
    committed = decision.action != DecisionAction.REJECT
    return conflict, committed, 0
