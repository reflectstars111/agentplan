"""Tests for StructureIndex."""

import pytest
from src.db import Database
from src.models.structure_node import StructureNode
from src.index.structure_index import StructureIndex


@pytest.fixture
def db():
    d = Database(":memory:")
    d.init_schema()
    yield d
    d.close()


@pytest.fixture
def index(db):
    return StructureIndex(db)


class TestStructureIndex:
    def test_index_and_retrieve_nodes(self, index):
        nodes = [
            StructureNode(node_id="n1", source_id="file:a.py", node_type="file", name="a.py", depth=0),
            StructureNode(node_id="n2", source_id="file:a.py", node_type="function", name="main", parent_id="n1", depth=1),
        ]
        count = index.index_nodes(nodes)
        assert count == 2
        children = index.get_children("n1")
        assert len(children) == 1
        assert children[0].name == "main"

    def test_get_subtree(self, index):
        nodes = [
            StructureNode(node_id="r", source_id="f", node_type="file", name="f", depth=0),
            StructureNode(node_id="c1", source_id="f", node_type="class", name="A", parent_id="r", depth=1),
            StructureNode(node_id="c2", source_id="f", node_type="method", name="m", parent_id="c1", depth=2),
        ]
        index.index_nodes(nodes)
        subtree = index.get_subtree("r")
        assert len(subtree) == 3

    def test_search_by_name(self, index):
        nodes = [
            StructureNode(node_id="n1", source_id="f", node_type="file", name="main.py", depth=0),
            StructureNode(node_id="n2", source_id="f", node_type="function", name="hello", parent_id="n1", depth=1),
        ]
        index.index_nodes(nodes)
        results = index.search_by_name("hello")
        assert len(results) == 1

    def test_search_case_insensitive(self, index):
        index.index_nodes([
            StructureNode(node_id="n1", source_id="f", node_type="function", name="Main", depth=0),
        ])
        assert len(index.search_by_name("main")) == 1

    def test_get_by_source(self, index):
        index.index_nodes([
            StructureNode(node_id="n1", source_id="file:a.py", node_type="file", name="a", depth=0),
            StructureNode(node_id="n2", source_id="file:b.py", node_type="file", name="b", depth=0),
        ])
        a_nodes = index.get_by_source("file:a.py")
        assert len(a_nodes) == 1

    def test_empty_index_graceful(self, index):
        assert index.get_children("nonexistent") == []
        assert index.search_by_name("nothing") == []
