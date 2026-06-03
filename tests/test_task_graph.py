"""Tests for Task model and TaskGraph DAG."""

import pytest
from src.models.task import Task, TaskStatus, TaskGraph


class TestTask:
    def test_create_minimal(self):
        t = Task(task_id="t1", task_type="retrieve")
        assert t.task_id == "t1"
        assert t.task_type == "retrieve"
        assert t.status == TaskStatus.CREATED
        assert t.agent_type == "worker"
        assert t.dependencies == []
        assert t.max_retries == 2

    def test_create_with_dependencies(self):
        t = Task(
            task_id="t2", task_type="reason",
            dependencies=["t1"], input={"query": "test"},
        )
        assert t.dependencies == ["t1"]
        assert t.input["query"] == "test"

    def test_serialization_roundtrip(self):
        t = Task(
            task_id="t3", task_type="verify",
            dependencies=["t1", "t2"], priority=7,
            input={"response": "some text"}, output={},
        )
        json_str = t.to_json()
        t2 = Task.from_json(json_str)
        assert t2.task_id == "t3"
        assert t2.dependencies == ["t1", "t2"]
        assert t2.priority == 7

    def test_status_transitions(self):
        t = Task(task_id="t4", task_type="retrieve")
        t.status = TaskStatus.READY
        assert t.status == TaskStatus.READY
        t.status = TaskStatus.RUNNING
        assert t.status == TaskStatus.RUNNING
        t.status = TaskStatus.COMPLETED
        assert t.status == TaskStatus.COMPLETED


class TestTaskGraph:
    def test_add_node(self):
        g = TaskGraph(intent_id="intent_001")
        t = Task(task_id="t1", task_type="retrieve")
        g.add_node(t)
        assert g.node_count() == 1
        assert g.get_node("t1").task_type == "retrieve"

    def test_add_edge(self):
        g = TaskGraph(intent_id="intent_001")
        g.add_node(Task(task_id="t1", task_type="retrieve"))
        g.add_node(Task(task_id="t2", task_type="reason"))
        g.add_edge("t1", "t2")
        assert "t2" in g.adj_in["t1"] or "t2" in g.adj_out.get("t1", set())

    def test_validate_acyclic_passes(self):
        g = TaskGraph(intent_id="intent_001")
        g.add_node(Task(task_id="t1", task_type="retrieve"))
        g.add_node(Task(task_id="t2", task_type="reason"))
        g.add_node(Task(task_id="t3", task_type="verify"))
        g.add_edge("t1", "t2")
        g.add_edge("t2", "t3")
        assert g.validate_acyclic() is True

    def test_validate_acyclic_detects_cycle(self):
        g = TaskGraph(intent_id="intent_001")
        g.add_node(Task(task_id="t1", task_type="retrieve"))
        g.add_node(Task(task_id="t2", task_type="reason"))
        g.add_edge("t1", "t2")
        g.add_edge("t2", "t1")  # cycle
        assert g.validate_acyclic() is False

    def test_topological_sort_linear(self):
        g = TaskGraph(intent_id="intent_001")
        g.add_node(Task(task_id="t1", task_type="retrieve"))
        g.add_node(Task(task_id="t2", task_type="reason"))
        g.add_node(Task(task_id="t3", task_type="verify"))
        g.add_edge("t1", "t2")
        g.add_edge("t2", "t3")
        order = g.topological_sort()
        assert order == ["t1", "t2", "t3"]

    def test_topological_sort_diamond(self):
        g = TaskGraph(intent_id="intent_001")
        for tid in ["t1", "t2", "t3", "t4"]:
            g.add_node(Task(task_id=tid, task_type="retrieve"))
        g.add_edge("t1", "t2")
        g.add_edge("t1", "t3")
        g.add_edge("t2", "t4")
        g.add_edge("t3", "t4")
        order = g.topological_sort()
        assert order[0] == "t1"
        assert order[-1] == "t4"
        assert order.index("t2") < order.index("t4")
        assert order.index("t3") < order.index("t4")

    def test_topological_sort_cycle_raises(self):
        g = TaskGraph(intent_id="intent_001")
        g.add_node(Task(task_id="t1", task_type="retrieve"))
        g.add_node(Task(task_id="t2", task_type="reason"))
        g.add_edge("t1", "t2")
        g.add_edge("t2", "t1")
        with pytest.raises(ValueError, match="[Cc]ycle"):
            g.topological_sort()

    def test_get_ready_nodes(self):
        g = TaskGraph(intent_id="intent_001")
        g.add_node(Task(task_id="t1", task_type="retrieve"))
        g.add_node(Task(task_id="t2", task_type="reason"))
        g.add_edge("t1", "t2")
        ready = g.get_ready_nodes(set())
        assert ready == ["t1"]

        ready2 = g.get_ready_nodes({"t1"})
        assert "t2" in ready2

    def test_all_completed(self):
        g = TaskGraph(intent_id="intent_001")
        g.add_node(Task(task_id="t1", task_type="retrieve"))
        t = g.get_node("t1")
        t.status = TaskStatus.COMPLETED
        assert g.all_completed() is True

    def test_node_count(self):
        g = TaskGraph(intent_id="intent_001")
        assert g.node_count() == 0
        g.add_node(Task(task_id="t1", task_type="retrieve"))
        g.add_node(Task(task_id="t2", task_type="reason"))
        assert g.node_count() == 2

    def test_to_dict(self):
        g = TaskGraph(intent_id="intent_001")
        g.add_node(Task(task_id="t1", task_type="retrieve"))
        d = g.to_dict()
        assert d["intent_id"] == "intent_001"
        assert d["node_count"] == 1
        assert len(d["nodes"]) == 1
