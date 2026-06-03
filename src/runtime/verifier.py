"""Verifier — source validation and conflict detection.

Checks that LLM responses are grounded in source references and detects
conflicts with stored memories. Maps to agent_os_initial_plan.md §14, §23.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from src.models.context import ContextPack
from src.models.memory import MemoryItem


@dataclass
class VerifyOutput:
    """Result of verification."""
    is_verified: bool
    unverified_claims: list[str] = field(default_factory=list)
    conflicting_pairs: list[tuple[str, str]] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


# Low-confidence language patterns that should trigger suggestions
LOW_CONFIDENCE_PATTERNS = [
    r'\b(might|may|could|possibly|perhaps|probably)\b',
    r'\b(I think|I believe|maybe|not sure|unclear)\b',
    r'\b(seems? like|appears? to be)\b',
]


class Verifier:
    """Validates LLM responses against source references and stored memories.

    Performs heuristic checks in MVP mode:
    1. Source reference validation (claimed sources must exist in context)
    2. Missing-source detection (claims without accompanying sources)
    3. Conflict detection (response vs working memory)
    4. Low-confidence language flagging
    """

    def verify(
        self,
        response: str,
        context_pack: ContextPack | None = None,
        working_memories: list[MemoryItem] | None = None,
    ) -> VerifyOutput:
        """Verify a response against its context and memories.

        Args:
            response: The LLM-generated response text.
            context_pack: The ContextPack used for this inference.
            working_memories: Current working memory state (L2).

        Returns:
            VerifyOutput with verification results.
        """
        if working_memories is None:
            working_memories = []

        unverified: list[str] = []
        conflicts: list[tuple[str, str]] = []
        suggestions: list[str] = []

        if not response.strip():
            return VerifyOutput(
                is_verified=False,
                suggestions=["Response is empty"],
            )

        if context_pack is None:
            return VerifyOutput(
                is_verified=False,
                suggestions=["No context pack provided — cannot verify sources"],
            )

        # 1. Extract claimed source references from response
        claimed_sources = self._extract_source_refs(response)
        known_sources = set(context_pack.source_refs)

        # 2. Check each claimed source exists in context
        for src in claimed_sources:
            if src not in known_sources:
                unverified.append(f"Claimed source '{src}' not found in context pack")

        # 3. Split response into sentences; flag those without sources
        sentences = re.split(r'(?<=[.!?])\s+', response)
        for sent in sentences:
            sent = sent.strip()
            if not sent or len(sent) < 20:  # skip short fragments
                continue
            # Check if this sentence has a source reference
            has_source = bool(re.search(r'\[file:[^\]]+\]', sent))
            if not has_source and not self._is_meta_sentence(sent):
                unverified.append(f"Statement without source: '{sent[:120]}...'" if len(sent) > 120 else f"Statement without source: '{sent}'")

        # 4. Detect conflicts with working memories
        for mem in working_memories:
            conflict = self._detect_conflict(response, mem)
            if conflict:
                conflicts.append((mem.content[:100], conflict))

        # 5. Check for low-confidence language
        for pattern in LOW_CONFIDENCE_PATTERNS:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                for m in matches[:3]:
                    suggestions.append(
                        f"Low-confidence language detected: '{m}'. "
                        "Consider more definitive phrasing or verify against sources."
                    )

        # 6. Deduplicate suggestions
        suggestions = list(dict.fromkeys(suggestions))

        is_verified = len(unverified) == 0 and len(conflicts) == 0

        return VerifyOutput(
            is_verified=is_verified,
            unverified_claims=unverified,
            conflicting_pairs=conflicts,
            suggestions=suggestions,
        )

    def _extract_source_refs(self, text: str) -> list[str]:
        """Extract source references in [file:...] or [source:...] format.
        Returns the inner reference without brackets."""
        refs = re.findall(r'\[(?:file|source|memory):[^\]]+\]', text)
        # Strip the outer brackets
        return [r[1:-1] for r in refs]

    def _is_meta_sentence(self, sentence: str) -> bool:
        """Check if a sentence is meta/transitional (doesn't need a source)."""
        meta_patterns = [
            r'^(here|let me|I will|let\'s|the|this|in summary|to summarize|overall)',
            r'^(based on|according to|as shown|from the)',
        ]
        return any(re.match(p, sentence.strip(), re.IGNORECASE) for p in meta_patterns)

    def _detect_conflict(self, response: str, memory: MemoryItem) -> str | None:
        """Heuristic conflict detection between response and a memory item.

        Returns conflict description string, or None if no conflict.
        """
        # Simple keyword-based approach for MVP: if memory mentions a
        # specific technology/decision and response mentions a different one
        mem_lower = memory.content.lower()
        resp_lower = response.lower()

        # Known opposing pairs for common technology choices
        opposing_pairs = [
            ({"fastapi", "starlette"}, {"django", "flask"}),
            ({"postgresql", "postgres"}, {"mongodb", "mongo", "mysql", "sqlite"}),
            ({"python"}, {"rust", "golang", "go", "java", "javascript", "typescript"}),
            ({"redis"}, {"memcached"}),
            ({"rest", "restful"}, {"graphql", "grpc"}),
        ]

        for group_a, group_b in opposing_pairs:
            mem_has_a = any(t in mem_lower for t in group_a)
            mem_has_b = any(t in mem_lower for t in group_b)
            resp_has_a = any(t in resp_lower for t in group_a)
            resp_has_b = any(t in resp_lower for t in group_b)

            # Conflict: memory prefers A, response prefers B (or vice versa)
            if mem_has_a and not mem_has_b and resp_has_b and not resp_has_a:
                mem_tech = next(t for t in group_a if t in mem_lower)
                resp_tech = next(t for t in group_b if t in resp_lower)
                return f"Memory prefers '{mem_tech}' but response suggests '{resp_tech}'"
            if mem_has_b and not mem_has_a and resp_has_a and not resp_has_b:
                mem_tech = next(t for t in group_b if t in mem_lower)
                resp_tech = next(t for t in group_a if t in resp_lower)
                return f"Memory prefers '{mem_tech}' but response suggests '{resp_tech}'"

        return None
