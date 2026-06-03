"""SharedBlackboard — inter-agent coordination via shared state.

Maps to agent_os_initial_plan.md §11.2 (Shared Blackboard).
"""

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class BlackboardEntry:
    """A single entry on the shared blackboard.

    Maps to agent_os_initial_plan.md §11.2 blackboard item schema.
    """

    key: str
    value: str
    created_by: str              # agent_id
    confidence: float = 0.5      # 0.0–1.0
    source_refs: list[str] = field(default_factory=list)
    timestamp: str = ""


class SharedBlackboard:
    """In-memory dict-backed shared state for agent collaboration.

    Lives within a single Controller.process() request lifecycle.
    Provides read/write/list/clear operations for agents to exchange
    intermediate results.
    """

    def __init__(self):
        self._store: dict[str, BlackboardEntry] = {}

    def write(self, entry: BlackboardEntry) -> None:
        """Write or overwrite an entry on the blackboard."""
        self._store[entry.key] = entry

    def read(self, key: str) -> BlackboardEntry | None:
        """Read a single entry by key. Returns None if not found."""
        return self._store.get(key)

    def read_all(self) -> dict[str, BlackboardEntry]:
        """Return a copy of all entries on the blackboard."""
        return dict(self._store)

    def list_by_agent(self, agent_id: str) -> list[BlackboardEntry]:
        """List all entries created by a specific agent."""
        return [e for e in self._store.values() if e.created_by == agent_id]

    def clear(self) -> None:
        """Remove all entries from the blackboard."""
        self._store.clear()
