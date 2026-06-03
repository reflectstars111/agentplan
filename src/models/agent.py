"""AgentProcess — Agent Process Control Block.

Maps to agent_os_initial_plan.md §7.1 (Agent PCB schema) and §7.2 (Agent states).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from dataclasses_json import dataclass_json


class AgentRole(str, Enum):
    PLANNER = "planner"
    WORKER = "worker"
    VERIFIER = "verifier"


class AgentStatus(str, Enum):
    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    BLOCKED = "blocked"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass_json
@dataclass
class AgentProcess:
    """Agent Process Control Block — represents one Agent instance.

    Defines the agent's identity, capabilities, memory scope, and
    execution state. Maps to the `agents` DB table.
    """

    agent_id: str
    role: AgentRole
    status: AgentStatus = AgentStatus.CREATED
    priority: int = 5
    current_goal: str = ""
    system_prompt_id: str = ""
    available_tools: list[str] = field(default_factory=list)
    memory_scope: dict = field(default_factory=dict)   # {"private": "...", "shared": "...", "external": [...]}
    context_budget: int = 24000
    parent_agent: Optional[str] = None
    created_at: str = ""
    last_active_at: Optional[str] = None
