"""Context section and pack models for ARD Phase 1."""

from dataclasses import dataclass, field


@dataclass
class ContextSection:
    """A single section of a context pack (e.g., retrieved_evidence)."""
    name: str
    tokens: int = 0
    priority: int = 5
    items: list[dict] = field(default_factory=list)


@dataclass
class ContextPack:
    """Assembled context ready for LLM inference."""
    context_id: str
    task_id: str
    agent_id: str
    budget: int
    sections: list[ContextSection] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    created_at: str = ""

    def total_tokens_used(self) -> int:
        return sum(s.tokens for s in self.sections)

    def to_text(self) -> str:
        """Render the context pack as a single text string for LLM."""
        parts = []
        for section in sorted(self.sections, key=lambda s: s.priority):
            parts.append(f"[{section.name.upper()}]\n")
            for item in section.items:
                text = item.get("text", "")
                compressed = item.get("compressed", False)
                if compressed:
                    parts.append(f"[compressed] {text}\n")
                else:
                    parts.append(f"{text}\n")
            parts.append("")
        return "\n".join(parts)
