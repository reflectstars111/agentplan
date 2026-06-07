"""Tests for ToolRouter + ToolRegistry."""

import pytest
from dataclasses import dataclass
from src.runtime.tool_router import ToolRouter, ToolRegistry, ToolDefinition
from src.runtime.permission_checker import PermissionChecker
from src.models.agent import AgentProcess, AgentRole


def _mock_handler(**params):
    return {"result": f"executed with {params}"}


@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register(ToolDefinition(
        name="code_search", description="Search code symbols",
        parameters={"query": {"type": "string"}},
        handler=_mock_handler,
    ))
    r.register(ToolDefinition(
        name="pdf_reader", description="Read PDF files",
        parameters={
            "path": {"type": "string", "required": True},
            "page": {"type": "integer"},
        },
        handler=_mock_handler,
    ))
    return r


@pytest.fixture
def router(registry):
    return ToolRouter(registry, PermissionChecker())


@pytest.fixture
def agent():
    return AgentProcess(agent_id="a1", role=AgentRole.WORKER,
                        available_tools=["code_search", "pdf_reader"])


class TestToolRegistry:
    def test_register_and_get(self, registry):
        t = registry.get("code_search")
        assert t.name == "code_search"

    def test_get_missing_raises(self, registry):
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_all(self, registry):
        assert "code_search" in registry.list_all()
        assert "pdf_reader" in registry.list_all()

    def test_list_for_agent(self, registry, agent):
        tools = registry.list_for_agent(agent)
        assert "code_search" in tools
        assert "pdf_reader" in tools

    def test_list_for_agent_restricted(self, registry):
        restricted = AgentProcess(agent_id="a2", role=AgentRole.WORKER,
                                  available_tools=["code_search"])
        tools = registry.list_for_agent(restricted)
        assert "code_search" in tools
        assert "pdf_reader" not in tools


class TestToolRouter:
    def test_execute_tool(self, router, agent):
        result = router.execute("code_search", {"query": "main"}, agent)
        assert result.success is True
        assert "executed" in result.output.get("result", "")

    def test_validate_params_ok(self, router):
        assert router.validate_params("code_search", {"query": "test"}) is True

    def test_validate_params_missing_required(self, router):
        assert router.validate_params("pdf_reader", {"path": "a.pdf"}) is True
        # Missing path should fail
        assert router.validate_params("pdf_reader", {"page": 1}) is False

    def test_execute_permission_denied(self, router):
        restricted = AgentProcess(agent_id="a3", role=AgentRole.WORKER,
                                  available_tools=["pdf_reader"])  # has pdf_reader but NOT code_search
        result = router.execute("code_search", {"query": "test"}, restricted)
        assert result.success is False
        assert "permission" in result.error.lower()

    def test_execute_nonexistent_tool(self, router, agent):
        result = router.execute("nonexistent", {}, agent)
        assert result.success is False
