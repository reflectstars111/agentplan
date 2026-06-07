"""Tests for PermissionChecker."""

import pytest
from src.models.agent import AgentProcess, AgentRole
from src.runtime.permission_checker import PermissionChecker


@pytest.fixture
def checker():
    return PermissionChecker()


@pytest.fixture
def restricted_agent():
    return AgentProcess(
        agent_id="agent_r",
        role=AgentRole.WORKER,
        available_tools=["retriever", "code_search"],
    )


class TestPermissionChecker:
    def test_check_tool_allowed(self, checker, restricted_agent):
        assert checker.check_tool(restricted_agent, "retriever") is True

    def test_check_tool_denied(self, checker, restricted_agent):
        assert checker.check_tool(restricted_agent, "shell_exec") is False

    def test_check_tool_no_tools_list(self, checker):
        agent = AgentProcess(agent_id="a", role=AgentRole.WORKER)
        assert checker.check_tool(agent, "any_tool") is True

    def test_check_memory_read_allowed(self, checker):
        agent = AgentProcess(
            agent_id="a", role=AgentRole.WORKER,
            memory_scope={"read_memory": ["project", "code_index"]},
        )
        assert checker.check_memory_read(agent, "project") is True

    def test_check_memory_read_denied(self, checker):
        agent = AgentProcess(
            agent_id="a", role=AgentRole.WORKER,
            memory_scope={"read_memory": ["project"]},
        )
        assert checker.check_memory_read(agent, "user_private") is False

    def test_check_memory_write_allowed(self, checker):
        agent = AgentProcess(
            agent_id="a", role=AgentRole.WORKER,
            memory_scope={"write_memory": ["working_memory"]},
        )
        assert checker.check_memory_write(agent, "working_memory") is True

    def test_check_memory_write_denied(self, checker):
        agent = AgentProcess(
            agent_id="a", role=AgentRole.WORKER,
            memory_scope={"write_memory": ["working_memory"]},
        )
        assert checker.check_memory_write(agent, "long_term_memory") is False

    def test_check_memory_no_scope_restriction(self, checker):
        agent = AgentProcess(agent_id="a", role=AgentRole.WORKER)
        assert checker.check_memory_read(agent, "anything") is True
        assert checker.check_memory_write(agent, "anything") is True

    def test_verify_permissions_report(self, checker, restricted_agent):
        report = checker.verify_permissions(restricted_agent, "tool_call",
                                            tool_name="shell_exec")
        assert report["allowed"] is False
        assert "reason" in report
