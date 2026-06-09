from semstate.evaluation import Baseline, evaluate_baselines
from semstate.histories import SCHEDULES, build_canonical_histories


def test_canonical_generator_builds_40_by_6_histories():
    histories = build_canonical_histories()
    assert len(histories) == 240
    assert len({history.scenario_id for history in histories}) == 40
    assert {history.schedule for history in histories} == set(SCHEDULES)
    assert {history.domain for history in histories} == {
        "deployment",
        "migration",
        "pipeline",
    }
    assert len({history.envelope.txn_id for history in histories}) == 240


def test_canonical_history_generation_is_deterministic():
    first = [history.to_dict() for history in build_canonical_histories()]
    second = [history.to_dict() for history in build_canonical_histories()]
    assert first == second


def test_read_set_occ_misses_semantic_and_dependency_conflicts():
    report = evaluate_baselines(
        build_canonical_histories(),
        [Baseline.READ_SET_OCC, Baseline.SEMSTATE_FULL],
    )
    assert report[Baseline.READ_SET_OCC.value]["invalid_commit_rate"] > 0.5
    assert report[Baseline.SEMSTATE_FULL.value]["invalid_commit_rate"] == 0.0
    assert report[Baseline.SEMSTATE_FULL.value]["final_state_correctness"] == 1.0


def test_trace_only_misses_cross_key_constraints():
    report = evaluate_baselines(
        build_canonical_histories(),
        [Baseline.SEMSTATE_TRACE_ONLY, Baseline.SEMSTATE_FULL],
    )
    assert report[Baseline.SEMSTATE_TRACE_ONLY.value]["invalid_commit_rate"] > 0
    assert (
        report[Baseline.SEMSTATE_FULL.value]["final_state_correctness"]
        > report[Baseline.SEMSTATE_TRACE_ONLY.value]["final_state_correctness"]
    )
