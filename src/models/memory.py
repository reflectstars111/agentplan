"""MemoryItem — the core memory record. Maps to agent_os_initial_plan.md §4.3."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from dataclasses_json import dataclass_json, config


class MemoryType(str, Enum):
    PROJECT_STATE = "project_state"
    USER_PREFERENCE = "user_preference"
    DECISION = "decision"
    FILE_SUMMARY = "file_summary"
    CONVERSATION_SUMMARY = "conversation_summary"
    INTERMEDIATE_RESULT = "intermediate_result"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


@dataclass_json
@dataclass
class MemoryItem:
    """A single memory record in L2/L3 storage."""

    memory_id: str
    type: MemoryType
    content: str
    summary: str = ""
    entities: list[str] = field(default_factory=list)
    importance: float = 0.5       # 0.0–1.0
    confidence: float = 0.5       # 0.0–1.0
    source: str = "conversation"  # conversation | file | agent | user
    scope: str = "project"        # project | user | session
    status: MemoryStatus = MemoryStatus.ACTIVE
    version: int = 1
    source_ref: Optional[str] = None  # e.g. "file:repo_001/README.md"
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        metadata=config(encoder=lambda d: d.isoformat(), decoder=lambda s: datetime.fromisoformat(s)),
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        metadata=config(encoder=lambda d: d.isoformat(), decoder=lambda s: datetime.fromisoformat(s)),
    )

    def to_keywords(self) -> list[str]:
        """Extract keywords for keyword index."""
        import re
        words = set()
        for field_text in [self.content, self.summary] + self.entities:
            tokens = re.findall(r'[\w一-鿿]+', field_text.lower())
            words.update(t for t in tokens if len(t) > 1)
        return list(words)
