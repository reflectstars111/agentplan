"""InputSanitizer — prompt injection detection and input protection.

Maps to agent_os_initial_plan.md §13.1 (Prompt injection risks) and §13.3 (Trust boundaries).
"""

import re


# Patterns that suggest prompt injection attempts
INJECTION_PATTERNS = [
    r'(?i)(ignore|forget|disregard)\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|directives?)',
    r'(?i)(you\s+are\s+now|from\s+now\s+on\s+you\s+are|act\s+as\s+(if\s+)?(you\s+are\s+)?)\s+(a\s+)?(different|new)\s+(role|persona|system)',
    r'(?i)(system\s*prompt|override\s*instruction|bypass\s*filter)',
    r'(?i)<\|(system|instruction|prompt)\|>',
    r'(?i)\[SYSTEM\].*\[/SYSTEM\]',
    r'(?i)print\s*\(\s*["\'](system\s*prompt|instructions?)["\']\s*\)',
    r'(?i)reveal\s+(your|the)\s+(system\s*prompt|instructions?|rules?)',
]


class InputSanitizer:
    """Detect and flag potential prompt injection in user input."""

    def scan(self, text: str) -> dict:
        """Scan text for injection patterns.

        Returns:
            dict with:
              - clean: bool — whether text appears safe
              - risk_level: "low" | "medium" | "high"
              - matched_patterns: list[str] — patterns that triggered
              - sanitized_text: str — cleaned version (original if clean)
        """
        if not text:
            return {"clean": True, "risk_level": "low",
                    "matched_patterns": [], "sanitized_text": text}

        matched = []
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, text):
                matched.append(pattern)

        if not matched:
            return {"clean": True, "risk_level": "low",
                    "matched_patterns": [], "sanitized_text": text}

        risk = "high" if len(matched) >= 2 else "medium"

        # Sanitize: truncate dangerous sections
        sanitized = text
        for pattern in INJECTION_PATTERNS:
            sanitized = re.sub(pattern, "[FILTERED]", sanitized, flags=re.IGNORECASE)

        return {
            "clean": False,
            "risk_level": risk,
            "matched_patterns": matched,
            "sanitized_text": sanitized,
        }

    def is_safe(self, text: str) -> bool:
        """Quick check: is this input safe?"""
        return self.scan(text)["clean"]
