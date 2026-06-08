"""ContextPack — the assembled context sent to the LLM. Maps to agent_os_initial_plan.md §9.2."""

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ContextSection:
    """A named section within a context pack."""
    name: str                    # e.g. "current_task", "working_memory", "retrieved_evidence"
    tokens: int                  # token count of items
    priority: int = 5            # 1 = highest, 10 = lowest
    items: list[dict] = field(default_factory=list)  # [{source_ref, trust_level, text}]


@dataclass_json
@dataclass
class ContextPack:
    """The complete context assembled for one LLM inference call."""

    context_id: str
    task_id: str
    agent_id: str
    budget: int                  # total token budget
    sections: list[ContextSection] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)  # all source references used
    memory_ids: list[str] = field(default_factory=list)
    used_tokens: int = 0
    created_at: str = ""         # ISO datetime

    def remaining_budget(self) -> int:
        return max(0, self.budget - self.used_tokens)

    def add_section(self, section: ContextSection) -> bool:
        """Add section if budget remains. Returns True if added."""
        if self.used_tokens + section.tokens > self.budget:
            return False
        self.sections.append(section)
        self.used_tokens += section.tokens
        return True
