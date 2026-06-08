"""Verifier — checks response against sources and detects conflicts.

Phase 2: Source verification (n-gram overlap) + conflict detection.
Phase 3+: Enhanced with citation verification and LLM-based checking.
"""

from dataclasses import dataclass, field
import re

from ard.infra.logging import log


@dataclass
class Verdict:
    """Output of the verification step."""
    verified: bool
    confidence: float
    conflicts: list[dict] = field(default_factory=list)
    orphan_claims: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


class Verifier:
    """Validates LLM responses against retrieved context.

    Two checks:
    1. Source check: Do the response's claims have supporting evidence?
    2. Conflict check: Does the response contradict existing state?
    """

    def __init__(self, ngram_size: int = 4, orphan_threshold: float = 0.3):
        self.ngram_size = ngram_size
        self.orphan_threshold = orphan_threshold  # ratio of unsupported sentences

    def verify(self, response: str, context_pack, state_store=None) -> Verdict:
        """Verify a response against its context pack and optional state store.

        Args:
            response: The LLM's answer text.
            context_pack: The ContextPack used to generate the response.
            state_store: Optional StateStore for conflict detection.

        Returns:
            Verdict with verification status and details.
        """
        conflicts = []
        orphan_claims = []

        # ── 1. Source check: does the response cite sources? ──
        evidence_texts = self._extract_evidence(context_pack)
        sentences = self._split_sentences(response)

        supported = 0
        unsupported = []
        for sentence in sentences:
            if self._has_evidence(sentence, evidence_texts):
                supported += 1
            else:
                # Ignore very short / transitional sentences
                if len(sentence.split()) >= 5:
                    unsupported.append(sentence)

        total_meaningful = max(supported + len(unsupported), 1)
        source_ratio = supported / total_meaningful

        if unsupported:
            orphan_claims = unsupported[:3]  # keep top 3

        # ── 2. Conflict check against existing state ──
        if state_store and len(sentences) > 2:
            try:
                conflicts = self._detect_conflicts(sentences, state_store)
            except Exception as e:
                log.warn("conflict_check_failed", error=str(e))

        # ── Compute confidence ──
        confidence = self._compute_confidence(
            source_ratio=source_ratio,
            source_refs=len(getattr(context_pack, 'source_refs', [])),
            conflicts=len(conflicts),
            orphan_count=len(orphan_claims),
        )

        verified = confidence >= 0.5 and len(orphan_claims) <= 2 and len(conflicts) == 0

        suggestions = []
        if orphan_claims:
            suggestions.append(
                f"{len(orphan_claims)} claims may lack supporting evidence"
            )
        if conflicts:
            suggestions.append(
                f"{len(conflicts)} potential conflicts found with existing state"
            )
        if source_ratio < 0.5:
            suggestions.append("Response has low source coverage")

        return Verdict(
            verified=verified,
            confidence=round(confidence, 3),
            conflicts=conflicts,
            orphan_claims=orphan_claims,
            suggestions=suggestions,
        )

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _extract_evidence(context_pack) -> list[str]:
        """Extract all evidence texts from a context pack."""
        texts = []
        for section in getattr(context_pack, 'sections', []):
            for item in getattr(section, 'items', []):
                text = item.get("text", "") if isinstance(item, dict) else ""
                if text:
                    texts.append(text)
        return texts

    def _has_evidence(self, sentence: str, evidence_texts: list[str]) -> bool:
        """Check if a sentence has n-gram overlap with any evidence text."""
        sent_ngrams = self._get_ngrams(sentence.lower(), self.ngram_size)
        if len(sent_ngrams) == 0:
            return False

        for evt in evidence_texts:
            evt_ngrams = self._get_ngrams(evt.lower(), self.ngram_size)
            overlap = sent_ngrams & evt_ngrams
            if len(overlap) >= max(1, len(sent_ngrams) * 0.2):
                return True
        return False

    @staticmethod
    def _get_ngrams(text: str, n: int) -> set:
        words = re.findall(r'\w+', text)
        if len(words) < n:
            return {frozenset(words)}
        return {frozenset(words[i:i + n]) for i in range(len(words) - n + 1)}

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    @staticmethod
    def _detect_conflicts(sentences: list[str], state_store) -> list[dict]:
        """Detect conflicts between response and existing state."""
        conflicts = []
        memory_keys = []
        try:
            memory_keys = state_store.list_keys("memory:")
        except Exception:
            pass

        for key in memory_keys[:10]:  # limit for performance
            try:
                state = state_store.read(key)
                if not state:
                    continue
                state_content = state.get("content", "")
                if not state_content:
                    continue

                for sentence in sentences[:5]:
                    # Simple keyword overlap as conflict proxy
                    sent_words = set(re.findall(r'\w+', sentence.lower()))
                    state_words = set(re.findall(r'\w+', state_content.lower()))

                    # If they share many words but have opposing keywords
                    shared = sent_words & state_words
                    if len(shared) > 5 and len(sent_words) > 3:
                        # Check for negation/opposition markers
                        negation_words = {"not", "no", "never", "unlike", "however",
                                         "but", "contrary", "opposite", "instead",
                                         "different", "disagree"}
                        has_negation = bool(sent_words & negation_words)
                        if has_negation and len(shared) / max(len(sent_words), 1) > 0.3:
                            conflicts.append({
                                "topic": list(shared)[:5],
                                "existing_state_key": key,
                                "existing_content_preview": state_content[:200],
                                "conflicting_sentence": sentence[:200],
                            })
            except Exception:
                continue

        return conflicts

    @staticmethod
    def _compute_confidence(source_ratio: float, source_refs: int,
                            conflicts: int, orphan_count: int) -> float:
        """Compute a confidence score from verification signals."""
        score = 0.3  # neutral start — needs evidence to reach verified threshold

        score += source_ratio * 0.3  # up to +0.3 for good sourcing
        score += min(source_refs, 5) * 0.02  # up to +0.1 for multiple sources
        score -= conflicts * 0.15  # penalty per conflict
        score -= orphan_count * 0.05  # penalty per orphan claim

        return max(0.0, min(1.0, score))
