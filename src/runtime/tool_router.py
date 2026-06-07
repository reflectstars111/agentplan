"""ToolRouter — tool registration, permission check, execution.

Maps to agent_os_initial_plan.md §6.1 (Tool Router), §6.3 (CALL_TOOL), §10.3.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional
from src.runtime.permission_checker import PermissionChecker
from src.models.agent import AgentProcess


@dataclass
class ToolResult:
    success: bool
    output: dict = field(default_factory=dict)
    error: str = ""
    tool_name: str = ""
    duration_ms: float = 0


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict          # {"param_name": {"type": "string", "required": true}}
    handler: Callable
    requires_permission: bool = True
    max_retries: int = 1
    timeout_seconds: int = 30


class ToolRegistry:
    """Registry of available tools with permission-aware listing."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered")
        return self._tools[name]

    def list_all(self) -> list[str]:
        return list(self._tools.keys())

    def list_for_agent(self, agent: AgentProcess) -> list[str]:
        allowed = set(agent.available_tools) if agent.available_tools else set(self._tools.keys())
        return [t for t in self._tools if t in allowed]


class ToolRouter:
    """Route tool calls with permission checks, param validation, and error handling."""

    def __init__(
        self,
        registry: ToolRegistry,
        permission_checker: PermissionChecker,
        trace_logger: "Optional[TraceLogger]" = None,
    ):
        self.registry = registry
        self.permission_checker = permission_checker
        self.trace_logger = trace_logger

    def execute(
        self,
        name: str,
        params: dict,
        agent: AgentProcess,
        trace_id: str = "",
    ) -> ToolResult:
        """Execute a tool call on behalf of an agent.

        Args:
            name: Tool name to execute.
            params: Parameters to pass to the tool handler.
            agent: The AgentProcess executing the tool.
            trace_id: Optional trace ID for audit logging.
        """
        try:
            tool = self.registry.get(name)
        except KeyError:
            return self._audit_and_return(
                ToolResult(success=False, error=f"Tool '{name}' not found", tool_name=name),
                name, params, trace_id,
            )

        if tool.requires_permission:
            if not self.permission_checker.check_tool(agent, name):
                return self._audit_and_return(
                    ToolResult(
                        success=False,
                        error=f"Agent '{agent.agent_id}' lacks permission for tool '{name}'",
                        tool_name=name,
                    ),
                    name, params, trace_id,
                )

        if not self.validate_params(name, params):
            return self._audit_and_return(
                ToolResult(success=False, error=f"Invalid parameters for tool '{name}'", tool_name=name),
                name, params, trace_id,
            )

        start = time.time()
        try:
            output = tool.handler(**params)
            duration = (time.time() - start) * 1000
            result = ToolResult(
                success=True,
                output=output if isinstance(output, dict) else {"result": output},
                tool_name=name,
                duration_ms=round(duration, 2),
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            result = ToolResult(
                success=False,
                error=str(e),
                tool_name=name,
                duration_ms=round(duration, 2),
            )

        return self._audit_and_return(result, name, params, trace_id)

    def _audit_and_return(
        self,
        result: ToolResult,
        tool_name: str,
        params: dict,
        trace_id: str,
    ) -> ToolResult:
        """Log tool execution to trace if configured, then return result."""
        if self.trace_logger and trace_id:
            from src.models.trace import TraceStep, StepType, StepStatus
            self.trace_logger.add_step(trace_id, TraceStep(
                step_id=f"tool_{tool_name}_{uuid.uuid4().hex[:8]}",
                type=StepType.TOOL_CALL,
                input={"tool": tool_name, "params": params},
                output={
                    "success": result.success,
                    "duration_ms": result.duration_ms,
                    "error": result.error or "",
                },
                status=StepStatus.SUCCESS if result.success else StepStatus.FAILED,
            ))
        return result

    def validate_params(self, name: str, params: dict) -> bool:
        """Validate parameters against the tool's schema."""
        try:
            tool = self.registry.get(name)
        except KeyError:
            return False
        for param_name, schema in tool.parameters.items():
            if schema.get("required", False) and param_name not in params:
                return False
        return True
