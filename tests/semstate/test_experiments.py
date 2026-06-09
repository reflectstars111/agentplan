import json
from pathlib import Path
import uuid

import pytest

from semstate.experiments import ExperimentRunner


def test_runner_resumes_from_manifest_without_duplicate_cases():
    run_dir = Path("data") / "semstate_test_runs" / uuid.uuid4().hex
    output = run_dir / "results.jsonl"
    manifest = run_dir / "manifest.json"
    runner = ExperimentRunner(str(output), str(manifest))
    cases = [
        {"case_id": "case-1", "value": 1},
        {"case_id": "case-2", "value": 2},
    ]
    calls = []

    def interrupted(case):
        calls.append(case["case_id"])
        if case["case_id"] == "case-2":
            raise RuntimeError("interrupted")
        return {"score": case["value"]}

    with pytest.raises(RuntimeError, match="interrupted"):
        runner.run(cases, interrupted)

    resumed_calls = []
    runner.run(
        cases,
        lambda case: resumed_calls.append(case["case_id"]) or {
            "score": case["value"]
        },
    )

    assert calls == ["case-1", "case-2"]
    assert resumed_calls == ["case-2"]
    manifest_records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["case_id"] for record in manifest_records] == [
        "case-1",
        "case-2",
    ]
    completed = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["status"] == "completed"
    ]
    assert [record["case_id"] for record in completed] == ["case-1", "case-2"]


def test_parallel_failure_is_recorded_and_not_marked_complete():
    run_dir = Path("data") / "semstate_test_runs" / uuid.uuid4().hex
    output = run_dir / "results.jsonl"
    manifest = run_dir / "manifest.json"
    runner = ExperimentRunner(str(output), str(manifest))

    def evaluator(case):
        if case["case_id"] == "bad":
            raise RuntimeError("model call failed")
        return {"ok": True}

    with pytest.raises(RuntimeError, match="model call failed"):
        runner.run(
            [{"case_id": "bad"}, {"case_id": "good"}],
            evaluator,
            max_workers=2,
        )

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        record["case_id"] == "bad" and record["status"] == "failed"
        for record in records
    )
    if manifest.exists():
        assert "bad" not in manifest.read_text(encoding="utf-8")
