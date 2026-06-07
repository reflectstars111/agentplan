"""Tests for AgentRegistry."""

import pytest
from src.models.agent import AgentProcess, AgentRole
from src.runtime.agent_registry import AgentRegistry


@pytest.fixture
def registry():
    return AgentRegistry()


class TestAgentRegistry:
    def test_register_and_get(self, registry):
        proc = AgentProcess(agent_id="w1", role=AgentRole.WORKER)
        registry.register("worker", proc, None)
        found_proc, _ = registry.get_agent("worker")
        assert found_proc.agent_id == "w1"

    def test_get_missing_raises(self, registry):
        with pytest.raises(KeyError, match="planner"):
            registry.get_agent("planner")

    def test_list_types(self, registry):
        registry.register("worker", AgentProcess(agent_id="w", role=AgentRole.WORKER), None)
        registry.register("verifier", AgentProcess(agent_id="v", role=AgentRole.VERIFIER), None)
        types = registry.list_types()
        assert "worker" in types
        assert "verifier" in types

    def test_has_agent(self, registry):
        assert not registry.has_agent("planner")
        registry.register("planner", AgentProcess(agent_id="p", role=AgentRole.PLANNER), None)
        assert registry.has_agent("planner")

    def test_spawn_creates_new_agent(self, registry):
        """spawn() should dynamically create and register a new agent."""
        process = registry.spawn("worker", "agent_dynamic_001")
        assert process.agent_id == "agent_dynamic_001"
        assert process.status.value == "created"
        assert registry.has_agent("agent_dynamic_001")

    def test_default_runtime_returns_worker(self, registry):
        proc = AgentProcess(agent_id="w", role=AgentRole.WORKER)
        registry.register("worker", proc, None)
        # Should not raise
        _proc, _rt = registry.get_agent("worker")
