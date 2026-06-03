"""Tests for Verifier."""

import pytest
from src.models.context import ContextPack, ContextSection
from src.models.memory import MemoryItem, MemoryType
from src.runtime.verifier import Verifier


@pytest.fixture
def context_pack():
    """A context pack with known source refs."""
    pack = ContextPack(
        context_id="ctx_001",
        task_id="task_001",
        agent_id="agent_001",
        budget=1000,
        source_refs=["file:paper.pdf", "file:guide.md"],
    )
    pack.add_section(ContextSection(
        name="retrieved_evidence",
        tokens=300,
        priority=3,
        items=[
            {"source_ref": "file:paper.pdf", "text": "FastAPI is an async Python web framework.", "trust_level": "external_untrusted"},
            {"source_ref": "file:guide.md", "text": "FastAPI uses Pydantic for data validation.", "trust_level": "user_provided_data"},
        ],
    ))
    return pack


@pytest.fixture
def verifier():
    return Verifier()


class TestVerifier:
    def test_verify_passes_with_valid_sources(self, verifier, context_pack):
        response = "FastAPI is an async Python web framework [file:paper.pdf]. It uses Pydantic [file:guide.md]."
        result = verifier.verify(response=response, context_pack=context_pack)
        assert result.is_verified

    def test_verify_flags_missing_source(self, verifier, context_pack):
        response = "FastAPI is great [file:nonexistent.pdf]."
        result = verifier.verify(response=response, context_pack=context_pack)
        assert not result.is_verified
        assert len(result.unverified_claims) > 0

    def test_verify_flags_claims_without_any_source(self, verifier, context_pack):
        response = "FastAPI is the best framework ever created. It was made by geniuses."
        result = verifier.verify(response=response, context_pack=context_pack)
        # Claims without sources should be flagged
        assert len(result.unverified_claims) > 0

    def test_verify_detects_conflicts_with_memory(self, verifier, context_pack):
        working_memories = [
            MemoryItem(memory_id="m1", type=MemoryType.DECISION,
                       content="Use Django for all web projects."),
        ]
        response = "We should use FastAPI for this project [file:paper.pdf]."
        result = verifier.verify(
            response=response,
            context_pack=context_pack,
            working_memories=working_memories,
        )
        # Django vs FastAPI conflict should be detected
        assert len(result.conflicting_pairs) > 0

    def test_verify_no_false_conflict(self, verifier, context_pack):
        working_memories = [
            MemoryItem(memory_id="m1", type=MemoryType.DECISION,
                       content="Use FastAPI for all API services."),
        ]
        response = "FastAPI is a good choice for this project [file:paper.pdf]."
        result = verifier.verify(
            response=response,
            context_pack=context_pack,
            working_memories=working_memories,
        )
        # Both agree on FastAPI — no conflict
        assert len(result.conflicting_pairs) == 0

    def test_verify_empty_response(self, verifier, context_pack):
        result = verifier.verify(response="", context_pack=context_pack)
        assert not result.is_verified

    def test_verify_result_has_suggestions(self, verifier, context_pack):
        response = "FastAPI might be useful [file:paper.pdf]."
        result = verifier.verify(response=response, context_pack=context_pack)
        # Should have suggestions about low-confidence language
        assert len(result.suggestions) > 0

    def test_verify_handles_missing_context_pack(self, verifier):
        result = verifier.verify(
            response="Some claim about something.",
            context_pack=None,
        )
        assert not result.is_verified
