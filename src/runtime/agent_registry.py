"""AgentRegistry — maps agent_type strings to AgentProcess + AgentRuntime pairs.

Used by Scheduler to route tasks to the correct agent at execution time.
"""

from src.models.agent import AgentProcess
from src.runtime.agent_runtime import AgentRuntime


class AgentRegistry:
    """Registry of available agent types and their runtimes.

    Maps agent_type (str) to a (AgentProcess, AgentRuntime) tuple.
    Supports lookup, existence check, and listing of registered types.
    """

    def __init__(self):
        self._agents: dict[str, tuple[AgentProcess, AgentRuntime | None]] = {}

    def register(
        self,
        agent_type: str,
        agent_process: AgentProcess,
        runtime: AgentRuntime | None,
    ) -> None:
        """Register an agent for a given agent_type.

        Args:
            agent_type: The type key (e.g. "worker", "verifier").
            agent_process: The AgentProcess metadata.
            runtime: The AgentRuntime instance (optional for testing).
        """
        self._agents[agent_type] = (agent_process, runtime)

    def get_agent(self, agent_type: str) -> tuple[AgentProcess, AgentRuntime | None]:
        """Look up an agent by type. Raises KeyError if not found."""
        if agent_type not in self._agents:
            raise KeyError(f"Agent type '{agent_type}' not registered")
        return self._agents[agent_type]

    def get_runtime(self, agent_type: str) -> AgentRuntime | None:
        """Return just the runtime for an agent type."""
        return self.get_agent(agent_type)[1]

    def has_agent(self, agent_type: str) -> bool:
        """Check if an agent type is registered."""
        return agent_type in self._agents

    def list_types(self) -> list[str]:
        """Return all registered agent types."""
        return list(self._agents.keys())
