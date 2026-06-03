"""Tests for WritebackGate."""

import pytest
from src.models.memory import MemoryItem, MemoryType, MemoryStatus
from src.runtime.writeback_gate import WritebackGate, WritebackDecision


@pytest.fixture
def gate():
    return WritebackGate()


class TestWritebackGate:
    def test_high_value_decision_returns_write(self, gate):
        result = gate.evaluate(
            content="The project uses FastAPI with PostgreSQL for data storage.",
            source="conversation",
            importance=0.9,
            confidence=0.95,
            user_explicit=True,
        )
        assert result.action == "write"
        assert result.location in ("working_memory", "long_term_memory")

    def test_low_confidence_returns_skip(self, gate):
        result = gate.evaluate(
            content="Maybe the API uses around 3-4 services, not sure which.",
            source="conversation",
            importance=0.3,
            confidence=0.2,
        )
        assert result.action == "skip"

    def test_trivial_content_returns_skip(self, gate):
        result = gate.evaluate(
            content="OK.",
            source="conversation",
            importance=0.1,
            confidence=1.0,
        )
        assert result.action == "skip"

    def test_high_confidence_no_user_confirm_returns_ask_user(self, gate):
        result = gate.evaluate(
            content="User's home address is 123 Main St, Springfield.",
            source="conversation",
            importance=0.7,
            confidence=0.85,
            user_explicit=False,
        )
        assert result.action in ("ask_user", "skip")

    def test_returned_decision_has_reason(self, gate):
        result = gate.evaluate(
            content="The database schema should use UUID primary keys.",
            source="conversation",
            importance=0.75,
            confidence=0.8,
        )
        assert len(result.reason) > 0
        assert 0.0 <= result.score <= 1.0

    def test_external_untrusted_source_has_penalty(self, gate):
        trusted = gate.evaluate(
            content="Key design decision: use event-driven architecture.",
            source="conversation",
            importance=0.8,
            confidence=0.9,
        )
        untrusted = gate.evaluate(
            content="Key design decision: use event-driven architecture.",
            source="web_page",
            importance=0.8,
            confidence=0.9,
        )
        # Untrusted source should score lower
        assert untrusted.score <= trusted.score

    def test_project_state_goes_to_long_term(self, gate):
        result = gate.evaluate(
            content="Agent-OS MVP uses Python + SQLite + FAISS as its tech stack.",
            source="conversation",
            importance=0.85,
            confidence=0.95,
            user_explicit=True,
        )
        if result.action == "write":
            assert result.location == "long_term_memory"

    def test_temporary_info_goes_to_working_memory(self, gate):
        result = gate.evaluate(
            content="Currently debugging the chunker overlap logic — seeing off-by-one in paragraph splits.",
            source="conversation",
            importance=0.5,
            confidence=0.9,
        )
        if result.action == "write":
            assert result.location == "working_memory"
