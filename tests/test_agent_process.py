"""Tests for AgentProcess model."""

import pytest
from src.models.agent import AgentProcess, AgentRole, AgentStatus


class TestAgentRole:
    def test_three_roles(self):
        assert AgentRole.PLANNER.value == "planner"
        assert AgentRole.WORKER.value == "worker"
        assert AgentRole.VERIFIER.value == "verifier"


class TestAgentStatus:
    def test_nine_states(self):
        states = list(AgentStatus)
        assert len(states) == 9
        assert AgentStatus.CREATED in states
        assert AgentStatus.RUNNING in states
        assert AgentStatus.COMPLETED in states
        assert AgentStatus.FAILED in states
        assert AgentStatus.TERMINATED in states


class TestAgentProcess:
    def test_create_minimal(self):
        p = AgentProcess(agent_id="agent_001", role=AgentRole.WORKER)
        assert p.agent_id == "agent_001"
        assert p.role == AgentRole.WORKER
        assert p.status == AgentStatus.CREATED
        assert p.priority == 5

    def test_create_full(self):
        p = AgentProcess(
            agent_id="agent_planner_001",
            role=AgentRole.PLANNER,
            status=AgentStatus.READY,
            priority=8,
            current_goal="Analyze user request and plan tasks",
            system_prompt_id="prompt_planner_v1",
            available_tools=["retriever", "task_graph"],
            memory_scope={"private": "planner_mem", "shared": "blackboard"},
            context_budget=32000,
        )
        assert p.available_tools == ["retriever", "task_graph"]
        assert p.memory_scope["shared"] == "blackboard"
        assert p.context_budget == 32000

    def test_serialization_roundtrip(self):
        p = AgentProcess(
            agent_id="agent_v_001",
            role=AgentRole.VERIFIER,
            status=AgentStatus.RUNNING,
            current_goal="Verify responses against sources",
        )
        json_str = p.to_json()
        p2 = AgentProcess.from_json(json_str)
        assert p2.agent_id == "agent_v_001"
        assert p2.role == AgentRole.VERIFIER

    def test_valid_transition(self):
        p = AgentProcess(agent_id="a1", role=AgentRole.WORKER)
        assert p.status == AgentStatus.CREATED
        assert p.transition(AgentStatus.READY) is True
        assert p.status == AgentStatus.READY
        assert p.transition(AgentStatus.RUNNING) is True
        assert p.status == AgentStatus.RUNNING
        assert p.transition(AgentStatus.COMPLETED) is True
        assert p.status == AgentStatus.COMPLETED

    def test_invalid_transition_rejected(self):
        p = AgentProcess(agent_id="a2", role=AgentRole.WORKER)
        # CREATED → RUNNING is invalid (must go through READY)
        assert p.transition(AgentStatus.RUNNING) is False
        assert p.status == AgentStatus.CREATED

    def test_failed_can_retry(self):
        p = AgentProcess(agent_id="a3", role=AgentRole.WORKER,
                         status=AgentStatus.FAILED)
        assert p.transition(AgentStatus.READY) is True

    def test_terminal_no_transition(self):
        p = AgentProcess(agent_id="a4", role=AgentRole.WORKER,
                         status=AgentStatus.COMPLETED)
        assert p.transition(AgentStatus.RUNNING) is False
        assert p.status == AgentStatus.COMPLETED

    def test_status_transitions(self):
        p = AgentProcess(agent_id="a1", role=AgentRole.WORKER)
        p.status = AgentStatus.READY
        p.status = AgentStatus.RUNNING
        p.status = AgentStatus.WAITING
        p.status = AgentStatus.BLOCKED
        p.status = AgentStatus.VERIFYING
        p.status = AgentStatus.COMPLETED
        assert p.status == AgentStatus.COMPLETED
