"""Deterministic dependency collection from observable execution inputs."""

from dataclasses import dataclass
from typing import Any, Iterable

from semstate.models import DependencyEdge, EdgeKind


@dataclass(frozen=True)
class DependencyObservation:
    state_key: str
    version: int
    origin: str
    confidence: float = 1.0
    kind: EdgeKind = EdgeKind.HARD


class DependencyCollector:
    """Collect dependencies without inferring latent semantic relations."""

    def collect(
        self,
        *,
        target: str,
        read_set: dict[str, int] | None = None,
        task_inputs: Any = None,
        context_pack: Any = None,
        tool_args: Any = None,
        source_refs: Iterable[Any] | None = None,
        current_versions: dict[str, int] | None = None,
    ) -> list[DependencyEdge]:
        observations = []
        for key, version in (read_set or {}).items():
            observations.append(DependencyObservation(
                state_key=key,
                version=version,
                origin="read_set",
            ))
        observations.extend(self._structured(task_inputs, "task_input"))
        observations.extend(self._structured(tool_args, "tool_arg"))
        observations.extend(self._context(context_pack, current_versions or {}))
        observations.extend(self._refs(source_refs or [], current_versions or {}))

        selected: dict[str, DependencyObservation] = {}
        for observation in observations:
            if observation.state_key == target:
                continue
            existing = selected.get(observation.state_key)
            if existing is None or self._rank(observation) > self._rank(existing):
                selected[observation.state_key] = observation

        return [
            DependencyEdge(
                source=observation.state_key,
                target=target,
                source_version=observation.version,
                origin=observation.origin,
                confidence=observation.confidence,
                kind=observation.kind,
            )
            for observation in sorted(
                selected.values(),
                key=lambda item: item.state_key,
            )
        ]

    def _structured(self, value: Any, origin: str) -> list[DependencyObservation]:
        observations = []
        if isinstance(value, dict):
            if "state_key" in value and "version" in value:
                observations.append(DependencyObservation(
                    state_key=str(value["state_key"]),
                    version=int(value["version"]),
                    origin=origin,
                    confidence=float(value.get("confidence", 1.0)),
                    kind=EdgeKind(value.get("kind", EdgeKind.HARD.value)),
                ))
            for child in value.values():
                observations.extend(self._structured(child, origin))
        elif isinstance(value, (list, tuple)):
            for child in value:
                observations.extend(self._structured(child, origin))
        return observations

    @staticmethod
    def _context(
        context_pack: Any,
        current_versions: dict[str, int],
    ) -> list[DependencyObservation]:
        if context_pack is None:
            return []
        refs = getattr(context_pack, "source_refs", [])
        return DependencyCollector._refs(
            refs,
            current_versions,
            origin="context_pack",
            default_kind=EdgeKind.SOFT,
            default_confidence=0.8,
        )

    @staticmethod
    def _refs(
        refs: Iterable[Any],
        current_versions: dict[str, int],
        *,
        origin: str = "source_ref",
        default_kind: EdgeKind = EdgeKind.SOFT,
        default_confidence: float = 0.8,
    ) -> list[DependencyObservation]:
        observations = []
        for ref in refs:
            if isinstance(ref, str):
                observations.append(DependencyObservation(
                    state_key=ref,
                    version=current_versions.get(ref, 0),
                    origin=origin,
                    confidence=default_confidence,
                    kind=default_kind,
                ))
            elif isinstance(ref, dict) and "state_key" in ref:
                key = str(ref["state_key"])
                observations.append(DependencyObservation(
                    state_key=key,
                    version=int(ref.get("version", current_versions.get(key, 0))),
                    origin=origin,
                    confidence=float(ref.get("confidence", default_confidence)),
                    kind=EdgeKind(ref.get("kind", default_kind.value)),
                ))
        return observations

    @staticmethod
    def _rank(observation: DependencyObservation) -> tuple[int, float]:
        return (
            1 if observation.kind == EdgeKind.HARD else 0,
            observation.confidence,
        )
