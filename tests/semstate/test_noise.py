from semstate.histories import build_canonical_histories
from semstate.noise import dependency_noise_curves


def test_dependency_noise_has_expected_endpoint_degradation():
    curves = dependency_noise_curves(build_canonical_histories())
    missing = curves["missing_dependency"]
    wrong = curves["wrong_dependency"]

    assert missing[0]["invalid_commit_rate"] == 0.0
    assert missing[-1]["invalid_commit_rate"] > missing[0]["invalid_commit_rate"]
    assert wrong[0]["false_rejection_rate"] == 0.0
    assert wrong[-1]["false_rejection_rate"] > wrong[0]["false_rejection_rate"]
