"""TraceStep and Trace — execution audit log. Maps to agent_os_initial_plan.md §15.3."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from dataclasses_json import dataclass_json, config


class StepType(str, Enum):
    INTENT_DECODE = "intent_decode"
    RETRIEVE_MEMORY = "retrieve_memory"
    RETRIEVE_FILE = "retrieve_file"
    CONTEXT_ASSEMBLE = "context_assemble"
    LLM_REASONING = "llm_reasoning"
    TOOL_CALL = "tool_call"
    VERIFY = "verify"
    WRITE_MEMORY = "write_memory"
    RESPOND = "respond"
    SPAWN_AGENT = "spawn_agent"
    SEND_MESSAGE = "send_message"
    MERGE = "merge"


class StepStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass_json
@dataclass
class TraceStep:
    """One step in an execution trace."""

    step_id: str
    type: StepType
    input: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    status: StepStatus = StepStatus.SUCCESS
    error: Optional[str] = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        metadata=config(encoder=lambda d: d.isoformat(), decoder=lambda s: datetime.fromisoformat(s)),
    )


@dataclass_json
@dataclass
class Trace:
    """Complete execution trace for one user request."""

    trace_id: str
    request_id: str
    steps: list[TraceStep] = field(default_factory=list)

    def add_step(self, step: TraceStep) -> None:
        self.steps.append(step)

    def to_summary(self) -> str:
        """Human-readable trace summary."""
        lines = [f"Trace {self.trace_id} (request: {self.request_id})"]
        for s in self.steps:
            status_icon = "✓" if s.status == StepStatus.SUCCESS else "✗" if s.status == StepStatus.FAILED else "○"
            lines.append(f"  {status_icon} [{s.type.value}] {s.timestamp.isoformat()}")
            if s.error:
                lines.append(f"     Error: {s.error}")
        return "\n".join(lines)
