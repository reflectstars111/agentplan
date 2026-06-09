from semstate.evaluation import Baseline, evaluate_history
from semstate.histories import build_canonical_histories
from semstate.statistics import paired_cluster_bootstrap


def test_cluster_bootstrap_uses_40_paired_scenarios():
    histories = build_canonical_histories()
    occ = [
        evaluate_history(history, Baseline.READ_SET_OCC)
        for history in histories
    ]
    semstate = [
        evaluate_history(history, Baseline.SEMSTATE_FULL)
        for history in histories
    ]

    result = paired_cluster_bootstrap(
        occ,
        semstate,
        field="final_state_correct",
        iterations=200,
    )

    assert result.clusters == 40
    assert result.observed_delta > 0
    assert result.ci_low > 0
