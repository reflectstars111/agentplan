"""Tests for Merger pipeline."""

import pytest
from src.models.blackboard import BlackboardEntry
from src.runtime.verifier import Verifier
from src.runtime.merger import Merger, MergeOutput


@pytest.fixture
def merger():
    return Merger(Verifier())


class TestMergerDedup:
    def test_dedup_removes_duplicate_values(self, merger):
        entries = [
            BlackboardEntry(key="a", value="FastAPI is great.", created_by="w1", confidence=0.8),
            BlackboardEntry(key="b", value="FastAPI is great.", created_by="w2", confidence=0.6),
            BlackboardEntry(key="c", value="Django is older.", created_by="w1", confidence=0.7),
        ]
        unique = merger._dedup(entries)
        # Should keep the higher-confidence duplicate
        assert len(unique) == 2

    def test_dedup_keeps_higher_confidence(self, merger):
        entries = [
            BlackboardEntry(key="a", value="Use FastAPI.", created_by="w1", confidence=0.5),
            BlackboardEntry(key="b", value="Use FastAPI.", created_by="w2", confidence=0.9),
        ]
        unique = merger._dedup(entries)
        assert len(unique) == 1
        assert unique[0].confidence == 0.9

    def test_dedup_preserves_unique_entries(self, merger):
        entries = [
            BlackboardEntry(key="a", value="FastAPI.", created_by="w1", confidence=0.8),
            BlackboardEntry(key="b", value="PostgreSQL.", created_by="w1", confidence=0.8),
        ]
        unique = merger._dedup(entries)
        assert len(unique) == 2


class TestMergerConfidenceSort:
    def test_confidence_sort_descending(self, merger):
        entries = [
            BlackboardEntry(key="a", value="A", created_by="w1", confidence=0.3),
            BlackboardEntry(key="b", value="B", created_by="w2", confidence=0.9),
            BlackboardEntry(key="c", value="C", created_by="w1", confidence=0.6),
        ]
        sorted_entries = merger._confidence_sort(entries)
        assert sorted_entries[0].confidence == 0.9
        assert sorted_entries[-1].confidence == 0.3


class TestMergerUnify:
    def test_unify_produces_statement(self, merger):
        entries = [
            BlackboardEntry(key="a", value="FastAPI is async.", created_by="w1", confidence=0.9),
            BlackboardEntry(key="b", value="It uses Pydantic.", created_by="w2", confidence=0.8),
        ]
        result = merger._unify(entries)
        assert "FastAPI" in result

    def test_unify_handles_empty(self, merger):
        assert merger._unify([]) == ""


class TestMergerFullPipeline:
    def test_merge_single_entry(self, merger):
        entries = [
            BlackboardEntry(key="r", value="FastAPI is a web framework.", created_by="w1",
                          confidence=0.9, source_refs=["file:fastapi.txt"]),
        ]
        result = merger.merge(entries)
        assert isinstance(result, MergeOutput)
        assert result.entries_merged == 1

    def test_merge_produces_valid_output(self, merger):
        entries = [
            BlackboardEntry(key="r1", value="FastAPI is Python-based.", created_by="w1",
                          confidence=0.85, source_refs=["file:fastapi.txt"]),
            BlackboardEntry(key="r2", value="It supports async operations.", created_by="w2",
                          confidence=0.75, source_refs=["file:async.txt"]),
        ]
        result = merger.merge(entries)
        assert result.entries_merged == 2
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.unified_statement) > 0

    def test_merge_empty_list(self, merger):
        result = merger.merge([])
        assert result.entries_merged == 0
