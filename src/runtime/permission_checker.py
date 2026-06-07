"""PermissionChecker — enforces Agent permissions for tools and memory access.

Maps to agent_os_initial_plan.md §13.2 (Permission Model).
"""

from src.models.agent import AgentProcess


class PermissionChecker:
    """Enforces agent-level permissions on tools, memory, and file access."""

    def check_tool(self, agent: AgentProcess, tool_name: str) -> bool:
        """Check if agent is allowed to use a tool."""
        if not agent.available_tools:
            return True  # No restriction
        return tool_name in agent.available_tools

    def check_memory_read(self, agent: AgentProcess, scope: str) -> bool:
        """Check if agent can read from a memory scope."""
        allowed = agent.memory_scope.get("read_memory", [])
        if not allowed:
            return True  # No restriction
        return scope in allowed

    def check_memory_write(self, agent: AgentProcess, scope: str) -> bool:
        """Check if agent can write to a memory scope."""
        allowed = agent.memory_scope.get("write_memory", [])
        if not allowed:
            return True  # No restriction
        return scope in allowed

    def verify_permissions(
        self, agent: AgentProcess, action: str, **context
    ) -> dict:
        """Verify permissions for an action. Returns {allowed, reason}."""
        if action == "tool_call":
            tool = context.get("tool_name", "")
            allowed = self.check_tool(agent, tool)
            return {
                "allowed": allowed,
                "reason": f"Tool '{tool}' {'allowed' if allowed else 'denied'} for agent {agent.agent_id}",
            }
        elif action == "memory_read":
            scope = context.get("scope", "")
            allowed = self.check_memory_read(agent, scope)
            return {
                "allowed": allowed,
                "reason": f"Memory read '{scope}' {'allowed' if allowed else 'denied'}",
            }
        elif action == "memory_write":
            scope = context.get("scope", "")
            allowed = self.check_memory_write(agent, scope)
            return {
                "allowed": allowed,
                "reason": f"Memory write '{scope}' {'allowed' if allowed else 'denied'}",
            }
        return {"allowed": True, "reason": f"Action '{action}' not restricted"}
