"""Dependency-noise ablations for missing and incorrect edges."""

from dataclasses import replace
import random

from semstate.evaluation import Baseline, aggregate, evaluate_history
from semstate.histories import BenchmarkHistory
from semstate.models import DependencyEdge, EdgeKind


def dependency_noise_curves(
    histories: list[BenchmarkHistory],
    *,
    rates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    seed: int = 20260609,
) -> dict[str, list[dict]]:
    return {
        "missing_dependency": [
            _evaluate_missing(histories, rate, seed)
            for rate in rates
        ],
        "wrong_dependency": [
            _evaluate_wrong(histories, rate, seed)
            for rate in rates
        ],
    }


def _evaluate_missing(
    histories: list[BenchmarkHistory],
    rate: float,
    seed: int,
) -> dict:
    candidates = [
        index
        for index, history in enumerate(histories)
        if history.envelope.dependencies
    ]
    selected = _sample(candidates, rate, seed)
    noisy = []
    for index, history in enumerate(histories):
        if index in selected:
            noisy.append(replace(
                history,
                envelope=replace(history.envelope, dependencies=[]),
            ))
        else:
            noisy.append(history)
    metrics = aggregate([
        evaluate_history(history, Baseline.SEMSTATE_FULL)
        for history in noisy
    ])
    return {"rate": rate, "modified_histories": len(selected), **metrics}


def _evaluate_wrong(
    histories: list[BenchmarkHistory],
    rate: float,
    seed: int,
) -> dict:
    candidates = [
        index
        for index, history in enumerate(histories)
        if not history.envelope.dependencies
        and history.expected_action.value == "commit"
    ]
    selected = _sample(candidates, rate, seed + 1)
    noisy = []
    for index, history in enumerate(histories):
        if index not in selected:
            noisy.append(history)
            continue
        source = history.initial_nodes[0]
        target = next(iter(history.envelope.write_set))
        wrong_edge = DependencyEdge(
            source=source.key,
            target=target,
            source_version=max(source.version - 1, 0),
            origin="injected_noise",
            confidence=1.0,
            kind=EdgeKind.HARD,
        )
        noisy.append(replace(
            history,
            envelope=replace(
                history.envelope,
                dependencies=[*history.envelope.dependencies, wrong_edge],
            ),
        ))
    metrics = aggregate([
        evaluate_history(history, Baseline.SEMSTATE_FULL)
        for history in noisy
    ])
    return {"rate": rate, "modified_histories": len(selected), **metrics}


def _sample(indices: list[int], rate: float, seed: int) -> set[int]:
    if not 0 <= rate <= 1:
        raise ValueError("Noise rate must be between 0 and 1")
    count = round(len(indices) * rate)
    rng = random.Random(seed)
    return set(rng.sample(indices, count))
