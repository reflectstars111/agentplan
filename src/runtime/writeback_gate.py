"""WritebackGate — memory write-back decision engine.

Implements the WriteScore formula from agent_os_initial_plan.md §12.2:
  WriteScore = 0.30*future + 0.25*project + 0.20*importance
             + 0.15*explicit + 0.10*confidence
             - 0.20*sensitivity - 0.20*uncertainty - 0.15*temporary
"""

import re
from dataclasses import dataclass


@dataclass
class WritebackDecision:
    """Result of a write-back evaluation."""
    action: str       # "write" | "skip" | "ask_user"
    location: str     # "working_memory" | "long_term_memory" | "none"
    reason: str
    score: float      # 0.0–1.0 WriteScore


# Content patterns that suggest future usefulness (higher score)
FUTURE_USEFUL_PATTERNS = [
    r'\b(decision|decided|chose|selected|picked)\b',
    r'\b(architecture|design|pattern|structure|schema)\b',
    r'\b(preference|prefers?|like|want)\b',
    r'\b(project|goal|requirement|specification)\b',
    r'\b(config|setting|configuration|environment)\b',
]

# Content patterns that suggest temporary/low-value info (lower score)
TEMPORARY_PATTERNS = [
    r'\b(currently|now|right now|at the moment|just|recently)\b',
    r'\b(debugging|troubleshooting|testing|trying)\b',
    r'\b(error|bug|issue|problem|fix)\b',
    r'\b(temp|temporary|scratch|draft)\b',
]

# Content patterns suggesting sensitive info
SENSITIVE_PATTERNS = [
    r'\b(password|secret|token|key|credential|auth)\b',
    r'\b(address|phone|email|personal|private)\b',
    r'\b(API key|access key|secret key)\b',
]

# Sources that are less trustworthy (score penalty)
UNTRUSTED_SOURCES = {"web_page", "external_untrusted", "pdf", "agent_generated"}


class WritebackGate:
    """Decides whether and where to write information to memory.

    Applies the WriteScore formula to determine if content should be
    persisted to working memory (L2) or long-term memory (L3).
    """

    def __init__(self, min_score: float = 0.5, user_confirm_threshold: float = 0.7):
        self.min_score = min_score
        self.user_confirm_threshold = user_confirm_threshold

    def evaluate(
        self,
        content: str,
        source: str = "conversation",
        importance: float = 0.5,
        confidence: float = 0.5,
        user_explicit: bool = False,
    ) -> WritebackDecision:
        """Evaluate whether content should be written to memory.

        Args:
            content: The proposed memory content.
            source: Where the info came from (conversation, file, web, etc.).
            importance: Estimated importance (0.0–1.0).
            confidence: Confidence in the information (0.0–1.0).
            user_explicit: Whether the user explicitly asked to remember this.

        Returns:
            WritebackDecision with action, location, reason, and score.
        """
        # Compute factor scores
        future_usefulness = self._score_future_usefulness(content)
        project_relevance = self._score_project_relevance(content)
        user_explicitness = 1.0 if user_explicit else 0.0
        sensitivity = self._score_sensitivity(content)
        uncertainty = self._score_uncertainty(content)
        short_livedness = self._score_temporary(content)

        # Source trust penalty
        source_penalty = 0.15 if source in UNTRUSTED_SOURCES else 0.0

        # WriteScore formula (§12.2)
        score = (
            0.30 * future_usefulness
            + 0.25 * project_relevance
            + 0.20 * importance
            + 0.15 * user_explicitness
            + 0.10 * confidence
            - 0.20 * sensitivity
            - 0.20 * uncertainty
            - 0.15 * short_livedness
            - 0.10 * source_penalty
        )

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        # Determine action
        if score < self.min_score:
            action = "skip"
            reason = f"WriteScore {score:.2f} below threshold {self.min_score}"
        elif score >= self.user_confirm_threshold and not user_explicit:
            action = "ask_user"
            reason = f"WriteScore {score:.2f} high but user didn't explicitly request — asking for confirmation"
        else:
            action = "write"
            reason = f"WriteScore {score:.2f} above threshold — persisting"

        # Determine location
        if action == "skip":
            location = "none"
        elif (future_usefulness >= 0.5 and project_relevance >= 0.5) or importance >= 0.7:
            # High project relevance or high importance → long-term
            location = "long_term_memory"
        else:
            location = "working_memory"

        # Refine reason for edge cases
        if len(content.strip()) < 20:
            action = "skip"
            reason = "Content too short to be meaningful"
            location = "none"

        return WritebackDecision(
            action=action,
            location=location,
            reason=reason,
            score=round(score, 4),
        )

    def _score_future_usefulness(self, content: str) -> float:
        """Estimate future usefulness based on keyword patterns."""
        content_lower = content.lower()
        matches = 0
        for pattern in FUTURE_USEFUL_PATTERNS:
            if re.search(pattern, content_lower):
                matches += 1
        # Normalize: 0–5 matches → 0.0–1.0
        return min(1.0, matches / 4.0)

    def _score_project_relevance(self, content: str) -> float:
        """Estimate project relevance (context-dependent heuristic)."""
        content_lower = content.lower()
        # Check for substantive technical content
        tech_indicators = [
            r'\b(api|database|schema|endpoint|service|module)\b',
            r'\b(implement|design|architecture|structure|pattern)\b',
            r'\b(python|fastapi|sqlite|faiss|postgresql|redis)\b',
            r'\b(agent|memory|index|retriev|context|token)\b',
        ]
        matches = sum(1 for p in tech_indicators if re.search(p, content_lower))
        return min(1.0, matches / 3.0)

    def _score_sensitivity(self, content: str) -> float:
        """Detect potentially sensitive content."""
        content_lower = content.lower()
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, content_lower):
                return 1.0
        return 0.0

    def _score_uncertainty(self, content: str) -> float:
        """Detect uncertainty markers. High uncertainty → high score (penalty)."""
        uncertainty_patterns = [
            r'\b(maybe|perhaps|possibly|not sure|unclear|might be)\b',
            r'\b(around|about|approximately|roughly|some)\b',
            r'\b(I think|I guess|probably|seems like)\b',
        ]
        content_lower = content.lower()
        matches = sum(1 for p in uncertainty_patterns if re.search(p, content_lower))
        return min(1.0, matches / 2.0)

    def _score_temporary(self, content: str) -> float:
        """Detect temporary/debugging content. High → more penalty."""
        content_lower = content.lower()
        matches = 0
        for pattern in TEMPORARY_PATTERNS:
            if re.search(pattern, content_lower):
                matches += 1
        return min(1.0, matches / 2.0)
