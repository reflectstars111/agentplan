"""Tests for Planner."""

import pytest
from src.models.intent import Intent, IntentType
from src.runtime.planner import Planner


@pytest.fixture
def planner():
    return Planner()


class TestPlanner:
    def test_plan_doc_qa(self, planner):
        intent = Intent(
            intent_id="i1", intent_type=IntentType.DOCUMENT_QA,
            original_query="What is FastAPI?",
        )
        graph = planner.plan(intent)
        assert graph.node_count() >= 2
        assert graph.validate_acyclic() is True

    def test_plan_code_analysis(self, planner):
        intent = Intent(
            intent_id="i2", intent_type=IntentType.CODE_ANALYSIS,
            original_query="Where is main.py?",
        )
        graph = planner.plan(intent)
        assert graph.node_count() >= 2

    def test_plan_general_single_node(self, planner):
        intent = Intent(
            intent_id="i3", intent_type=IntentType.GENERAL,
            original_query="Hello",
        )
        graph = planner.plan(intent)
        assert graph.node_count() >= 1

    def test_plan_memory_query(self, planner):
        intent = Intent(
            intent_id="i4", intent_type=IntentType.MEMORY_QUERY,
            original_query="What did we decide about auth?",
        )
        graph = planner.plan(intent)
        assert graph.node_count() >= 1

    def test_doc_qa_has_verify_node(self, planner):
        intent = Intent(
            intent_id="i5", intent_type=IntentType.DOCUMENT_QA,
            original_query="Explain RAG.",
        )
        graph = planner.plan(intent)
        task_types = {t.task_type for t in graph.nodes.values()}
        assert "verify" in task_types

    def test_all_nodes_have_unique_ids(self, planner):
        intent = Intent(
            intent_id="i6", intent_type=IntentType.DOCUMENT_QA,
            original_query="Test query",
        )
        graph = planner.plan(intent)
        ids = list(graph.nodes.keys())
        assert len(ids) == len(set(ids))

    def test_graph_is_valid_dag(self, planner):
        for intent_type in IntentType:
            intent = Intent(
                intent_id=f"i_{intent_type.value}",
                intent_type=intent_type,
                original_query="test",
            )
            graph = planner.plan(intent)
            assert graph.validate_acyclic() is True, f"Cycle in {intent_type} graph"
            # Topological sort should succeed
            order = graph.topological_sort()
            assert len(order) == graph.node_count()

    def test_multi_turn_includes_writeback(self, planner):
        intent = Intent(
            intent_id="i7", intent_type=IntentType.MULTI_TURN,
            original_query="Let's design an auth system.",
        )
        graph = planner.plan(intent)
        task_types = {t.task_type for t in graph.nodes.values()}
        assert "writeback" in task_types
