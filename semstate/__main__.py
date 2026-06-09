"""Generate SemStateBench histories and deterministic baseline reports."""

import argparse
import json
from pathlib import Path

from semstate.evaluation import Baseline, evaluate_baselines, evaluate_history
from semstate.histories import build_canonical_histories
from semstate.noise import dependency_noise_curves
from semstate.statistics import paired_cluster_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--histories",
        default="eval_data/semstate_benchmark_v1.json",
    )
    parser.add_argument(
        "--report",
        default="eval_data/semstate_baselines_v1.json",
    )
    args = parser.parse_args()

    histories = build_canonical_histories()
    history_path = Path(args.histories)
    report_path = Path(args.report)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(
            [history.to_dict() for history in histories],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report = evaluate_baselines(histories)
    occ_records = [
        evaluate_history(history, Baseline.READ_SET_OCC)
        for history in histories
    ]
    full_records = [
        evaluate_history(history, Baseline.SEMSTATE_FULL)
        for history in histories
    ]
    report["paired_cluster_bootstrap"] = {
        "final_state_correctness_improvement": paired_cluster_bootstrap(
            occ_records,
            full_records,
            field="final_state_correct",
            higher_is_better=True,
        ).to_dict(),
        "invalid_commit_rate_reduction": paired_cluster_bootstrap(
            occ_records,
            full_records,
            field="invalid_commit",
            higher_is_better=False,
        ).to_dict(),
    }
    report["dependency_noise"] = dependency_noise_curves(histories)
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(histories)} histories to {history_path}")
    print(f"wrote baseline report to {report_path}")


if __name__ == "__main__":
    main()
