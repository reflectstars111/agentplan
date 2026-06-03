"""Intent — structured user intent representation.

Maps to agent_os_initial_plan.md §6.1 (Intent Decoder output).
"""

from dataclasses import dataclass, field
from enum import Enum
from dataclasses_json import dataclass_json


class IntentType(str, Enum):
    DOCUMENT_QA = "document_qa"
    CODE_ANALYSIS = "code_analysis"
    MULTI_TURN = "multi_turn_conversation"
    MEMORY_QUERY = "memory_query"
    GENERAL = "general"


@dataclass_json
@dataclass
class Intent:
    """Structured representation of a user's intent.

    Produced by IntentDecoder and consumed by Planner.
    """

    intent_id: str
    intent_type: IntentType
    original_query: str
    entities: list[str] = field(default_factory=list)
    constraints: dict = field(default_factory=dict)
    priority: int = 5                    # 1 (highest) - 10 (lowest)
    confidence: float = 0.5              # decoder confidence 0.0–1.0
    extracted_params: dict = field(default_factory=dict)
    created_at: str = ""                 # ISO datetime
