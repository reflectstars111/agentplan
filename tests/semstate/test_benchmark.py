import pytest

from ard.infra.db import Database
from semstate.benchmark import build_g0_cases
from semstate.runtime import SemStateRuntime


@pytest.mark.parametrize("case", build_g0_cases(), ids=lambda case: case.case_id)
def test_g0_case_ground_truth_reconstructs(case):
    db = Database(":memory:")
    runtime = SemStateRuntime(db)
    for node in case.initial_nodes:
        runtime.seed_node(node)

    decision = runtime.validate(case.envelope)

    assert decision.action == case.expected_action
    assert decision.anomaly_type == case.expected_anomaly
    db.close()


def test_g0_contains_three_domains_and_twelve_cases():
    cases = build_g0_cases()
    assert len(cases) == 12
    assert {case.domain for case in cases} == {
        "deployment",
        "migration",
        "pipeline",
    }
    assert sum(case.metadata.get("occ_would_commit", False) for case in cases) >= 3
