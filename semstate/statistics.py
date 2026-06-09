"""Paired cluster bootstrap over base scenarios."""

from dataclasses import dataclass
import random

from semstate.evaluation import EvaluationRecord


@dataclass(frozen=True)
class BootstrapResult:
    observed_delta: float
    ci_low: float
    ci_high: float
    iterations: int
    clusters: int

    def to_dict(self) -> dict:
        return {
            "observed_delta": self.observed_delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "iterations": self.iterations,
            "clusters": self.clusters,
        }


def paired_cluster_bootstrap(
    control: list[EvaluationRecord],
    treatment: list[EvaluationRecord],
    *,
    field: str,
    higher_is_better: bool = True,
    iterations: int = 2000,
    seed: int = 20260609,
) -> BootstrapResult:
    """Estimate a paired treatment improvement with scenario resampling."""
    control_by_id = {record.history_id: record for record in control}
    treatment_by_id = {record.history_id: record for record in treatment}
    if control_by_id.keys() != treatment_by_id.keys():
        raise ValueError("Control and treatment histories must match")

    clusters: dict[str, list[tuple[float, float]]] = {}
    for history_id, control_record in control_by_id.items():
        treatment_record = treatment_by_id[history_id]
        clusters.setdefault(control_record.scenario_id, []).append((
            float(getattr(control_record, field)),
            float(getattr(treatment_record, field)),
        ))
    cluster_ids = sorted(clusters)
    if not cluster_ids:
        raise ValueError("At least one scenario cluster is required")

    observed = _improvement(
        [pair for cluster in clusters.values() for pair in cluster],
        higher_is_better,
    )
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        selected = [
            rng.choice(cluster_ids)
            for _ in cluster_ids
        ]
        pairs = [
            pair
            for cluster_id in selected
            for pair in clusters[cluster_id]
        ]
        samples.append(_improvement(pairs, higher_is_better))
    samples.sort()
    return BootstrapResult(
        observed_delta=observed,
        ci_low=_quantile(samples, 0.025),
        ci_high=_quantile(samples, 0.975),
        iterations=iterations,
        clusters=len(cluster_ids),
    )


def _improvement(
    pairs: list[tuple[float, float]],
    higher_is_better: bool,
) -> float:
    deltas = [
        treatment - control if higher_is_better else control - treatment
        for control, treatment in pairs
    ]
    return sum(deltas) / len(deltas)


def _quantile(values: list[float], probability: float) -> float:
    if len(values) == 1:
        return values[0]
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight
