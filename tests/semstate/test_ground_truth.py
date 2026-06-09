from semstate.ground_truth import reconstruct_ground_truth
from semstate.histories import build_canonical_histories


def test_independent_ground_truth_reconstructs_all_canonical_labels():
    for history in build_canonical_histories():
        truth = reconstruct_ground_truth(history)
        assert truth.action == history.expected_action, history.history_id
        assert truth.anomaly == history.expected_anomaly, history.history_id
