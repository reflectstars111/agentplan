"""IntentDecoder — keyword-based intent parsing.

Maps natural language queries to structured Intent objects using
regex/keyword dispatch. No LLM call required.

Maps to agent_os_initial_plan.md §6.1 (Intent Decoder).
"""

import re
import uuid
from datetime import datetime, timezone
from src.models.intent import Intent, IntentType


# Trigger patterns per intent type (checked in priority order)
TYPE_TRIGGERS: dict[IntentType, list[str]] = {
    IntentType.CODE_ANALYSIS: [
        r"\bwhere\s+(is|are|can\s+i\s+find)\b",
        r"\bfind\s+(the|all|me)\b",
        r"\bwhich\s+file\b",
        r"\blocate\b",
        r"\bshow\s+me\s+where\b",
        r"\bimplement(ed|ation|s)?\s+(in|at)\b",
    ],
    IntentType.MEMORY_QUERY: [
        r"\bwhat\s+did\s+(we|i|you)\s+decide\b",
        r"\brecall\b",
        r"\bwhat\s+was\s+(the|our|my)\b",
        r"\bprevious\b",
        r"\bremember\b",
        r"\bremind\b",
    ],
    IntentType.DOCUMENT_QA: [
        r"\bwhat\s+is\b",
        r"\bexplain\b",
        r"\bhow\s+(does|do|is|can|should)\b",
        r"\bsummarize\b",
        r"\bdescribe\b",
        r"\btell\s+me\s+about\b",
        r"\bdefine\b",
    ],
    IntentType.MULTI_TURN: [
        r"\blet.?s\s+(design|build|create|start|plan)\b",
        r"\bdesign\s+(a|the)\b",
        r"\bproject\b",
    ],
}

# Entity extraction patterns
ENTITY_PATTERNS = [
    r'\b[A-Z][A-Za-z0-9_]{2,}\b',           # CamelCase identifiers (RAPTOR, FastAPI)
    r'\b[\w-]+\.(?:py|js|ts|rs|go|java)\b',  # File names with extensions
    r'\b[\w-]+\.(?:pdf|md|txt|yaml|json)\b', # Document files
]


class IntentDecoder:
    """Parse user queries into structured Intent objects.

    MVP: keyword/regex-based dispatch against trigger patterns.
    Future: swap in LLM-based decoder behind the same decode() interface.
    """

    def decode(self, query: str, request_id: str = "") -> Intent:
        """Parse a user query into a structured Intent.

        Args:
            query: Raw user input text.
            request_id: Optional request ID for trace linking.

        Returns:
            Intent with intent_type, entities, confidence. Always returns a
            valid Intent (defaults to GENERAL if no patterns match).
        """
        intent_id = f"intent_{uuid.uuid4().hex[:12]}"
        intent_type, match_count = self._classify(query)
        entities = self._extract_entities(query)
        confidence = self._compute_confidence(match_count)

        return Intent(
            intent_id=intent_id,
            intent_type=intent_type,
            original_query=query,
            entities=entities,
            priority=5,
            confidence=confidence,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _classify(self, query: str) -> tuple[IntentType, int]:
        """Classify query into an intent type.

        Returns (intent_type, match_count).
        Checks in priority order: CODE_ANALYSIS > MEMORY_QUERY > DOCUMENT_QA > MULTI_TURN > GENERAL.
        """
        query_lower = query.lower().strip()
        if not query_lower:
            return IntentType.GENERAL, 0

        # Check each type in priority order
        type_order = [
            IntentType.CODE_ANALYSIS,
            IntentType.MEMORY_QUERY,
            IntentType.DOCUMENT_QA,
            IntentType.MULTI_TURN,
        ]

        for intent_type in type_order:
            count = 0
            for pattern in TYPE_TRIGGERS.get(intent_type, []):
                if re.search(pattern, query_lower):
                    count += 1
            if count > 0:
                return intent_type, count

        return IntentType.GENERAL, 0

    def _extract_entities(self, query: str) -> list[str]:
        """Extract named entities from the query."""
        entities = []
        seen = set()
        for pattern in ENTITY_PATTERNS:
            for match in re.findall(pattern, query):
                m_lower = match.lower()
                if m_lower not in seen:
                    entities.append(match)
                    seen.add(m_lower)
        return entities

    def _compute_confidence(self, match_count: int) -> float:
        """Compute confidence based on number of trigger pattern matches."""
        if match_count >= 3:
            return 0.9
        elif match_count >= 2:
            return 0.7
        elif match_count >= 1:
            return 0.5
        else:
            return 0.3  # GENERAL fallback
