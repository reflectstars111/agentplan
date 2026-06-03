"""Merger — 6-stage merge pipeline for multi-agent output unification.

Maps to agent_os_initial_plan.md §11.3 (Merge mechanism).
"""

from dataclasses import dataclass
from src.models.blackboard import BlackboardEntry
from src.runtime.verifier import Verifier


@dataclass
class MergeOutput:
    """Result of the merge pipeline."""
    unified_statement: str
    entries_merged: int
    entries_deduped: int = 0
    conflicts_detected: int = 0
    confidence: float = 0.5


class Merger:
    """Merge pipeline: dedup → source_align → confidence_sort
    → conflict_detect → verifier_check → unify.
    """

    def __init__(self, verifier: Verifier | None = None):
        self.verifier = verifier or Verifier()

    def merge(self, entries: list[BlackboardEntry]) -> MergeOutput:
        """Full merge pipeline. Returns unified statement."""
        if not entries:
            return MergeOutput(
                unified_statement="",
                entries_merged=0,
            )

        orig_count = len(entries)

        # Stage 1: Dedup
        unique = self._dedup(entries)
        dedup_count = orig_count - len(unique)

        # Stage 2: Source alignment
        aligned = self._source_align(unique)

        # Stage 3: Confidence sort
        sorted_entries = self._confidence_sort(aligned)

        # Stage 4: Conflict detection
        conflicts = self._detect_conflicts(sorted_entries)

        # Stage 5: Verifier check
        verified = self._verifier_check(sorted_entries)

        # Stage 6: Unify
        unified = self._unify(sorted_entries)

        return MergeOutput(
            unified_statement=unified,
            entries_merged=orig_count,
            entries_deduped=dedup_count,
            conflicts_detected=len(conflicts),
            confidence=self._compute_aggregate_confidence(sorted_entries),
        )

    def _dedup(self, entries: list[BlackboardEntry]) -> list[BlackboardEntry]:
        """Remove duplicates by value. Keeps the higher-confidence copy."""
        seen: dict[str, BlackboardEntry] = {}
        for e in entries:
            key = e.value.strip().lower()
            if key not in seen or e.confidence > seen[key].confidence:
                seen[key] = e
        return list(seen.values())

    def _source_align(self, entries: list[BlackboardEntry]) -> list[BlackboardEntry]:
        """Group entries by shared source_refs. Merges entries with overlapping sources."""
        if len(entries) <= 1:
            return entries
        # For MVP: merge entries that share at least one source_ref
        merged = []
        used = set()
        for i, e1 in enumerate(entries):
            if i in used:
                continue
            combined_value = e1.value
            combined_sources = list(e1.source_refs)
            combined_confidence = e1.confidence
            for j, e2 in enumerate(entries[i + 1:], start=i + 1):
                if j in used:
                    continue
                if set(e1.source_refs) & set(e2.source_refs):
                    combined_value += "; " + e2.value
                    combined_sources.extend(s for s in e2.source_refs if s not in combined_sources)
                    combined_confidence = max(combined_confidence, e2.confidence)
                    used.add(j)
            merged.append(BlackboardEntry(
                key=f"merged_{i}",
                value=combined_value,
                created_by=e1.created_by,
                confidence=combined_confidence,
                source_refs=combined_sources,
            ))
        return merged

    def _confidence_sort(self, entries: list[BlackboardEntry]) -> list[BlackboardEntry]:
        """Sort by confidence descending."""
        return sorted(entries, key=lambda e: e.confidence, reverse=True)

    def _detect_conflicts(self, entries: list[BlackboardEntry]) -> list[tuple[str, str]]:
        """Detect conflicting claims across entries."""
        conflicts = []
        opposing_pairs = [
            ({"fastapi", "starlette"}, {"django", "flask"}),
            ({"postgresql", "postgres"}, {"mongodb", "mysql", "sqlite"}),
            ({"python"}, {"rust", "golang", "java"}),
        ]
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                v1 = entries[i].value.lower()
                v2 = entries[j].value.lower()
                for group_a, group_b in opposing_pairs:
                    in_a1 = any(t in v1 for t in group_a)
                    in_b1 = any(t in v1 for t in group_b)
                    in_a2 = any(t in v2 for t in group_a)
                    in_b2 = any(t in v2 for t in group_b)
                    if (in_a1 and in_b2) or (in_b1 and in_a2):
                        conflicts.append((entries[i].value[:80], entries[j].value[:80]))
                        break
        return conflicts

    def _verifier_check(self, entries: list[BlackboardEntry]) -> bool:
        """Check if the combined content passes verification."""
        combined = self._unify(entries)
        if not combined:
            return False
        result = self.verifier.verify(combined, context_pack=None)
        return result.is_verified

    def _unify(self, entries: list[BlackboardEntry]) -> str:
        """Produce a single unified statement from sorted entries."""
        if not entries:
            return ""
        # Join high-confidence entries, prefixed by confidence
        parts = []
        for e in entries:
            if e.confidence >= 0.5:
                parts.append(e.value)
        return " ".join(parts) if parts else entries[0].value

    def _compute_aggregate_confidence(self, entries: list[BlackboardEntry]) -> float:
        """Weighted average of entry confidences."""
        if not entries:
            return 0.0
        return round(sum(e.confidence for e in entries) / len(entries), 4)
