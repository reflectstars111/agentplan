"""ConversationCache — L1 dialog cache with automatic turn collection.

Maps to agent_os_initial_plan.md §4.1 (L1 Dialog Cache), §4.2 (ConversationTurn).
"""

import uuid
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ConversationTurn:
    turn_id: str
    role: str           # "user" | "agent"
    content: str
    timestamp: str = ""


class ConversationCache:
    """L1 dialog cache: auto-collects recent N conversation turns."""

    def __init__(self, max_turns: int = 20):
        self._turns: list[ConversationTurn] = []
        self.max_turns = max_turns

    def add_user_message(self, content: str) -> None:
        self._add("user", content)

    def add_agent_response(self, content: str) -> None:
        self._add("agent", content)

    def get_recent_turns(self, n: int = 10) -> list[dict]:
        """Return recent turns as dicts for ContextMMU compatibility."""
        turns = self._turns[-n:]
        return [{"role": t.role, "content": t.content} for t in turns]

    def clear(self) -> None:
        self._turns.clear()

    def _add(self, role: str, content: str) -> None:
        if not content or len(content) < 2:
            return
        self._turns.append(ConversationTurn(
            turn_id=f"turn_{uuid.uuid4().hex[:12]}",
            role=role, content=content,
        ))
        # Evict oldest if over limit
        while len(self._turns) > self.max_turns:
            self._turns.pop(0)
