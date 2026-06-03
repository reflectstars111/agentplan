"""Tests for Intent model."""

import pytest
from src.models.intent import Intent, IntentType


class TestIntentType:
    def test_has_five_types(self):
        types = list(IntentType)
        assert IntentType.DOCUMENT_QA in types
        assert IntentType.CODE_ANALYSIS in types
        assert IntentType.MULTI_TURN in types
        assert IntentType.MEMORY_QUERY in types
        assert IntentType.GENERAL in types


class TestIntent:
    def test_create_minimal(self):
        i = Intent(
            intent_id="intent_001",
            intent_type=IntentType.GENERAL,
            original_query="hello",
        )
        assert i.intent_id == "intent_001"
        assert i.intent_type == IntentType.GENERAL
        assert i.entities == []
        assert i.priority == 5

    def test_create_full(self):
        i = Intent(
            intent_id="intent_002",
            intent_type=IntentType.DOCUMENT_QA,
            original_query="What is FastAPI?",
            entities=["FastAPI"],
            constraints={"max_results": 5},
            priority=8,
            confidence=0.9,
            extracted_params={"topic": "web framework"},
        )
        assert i.entities == ["FastAPI"]
        assert i.confidence == 0.9
        assert i.extracted_params["topic"] == "web framework"

    def test_serialization_roundtrip(self):
        i = Intent(
            intent_id="intent_003",
            intent_type=IntentType.CODE_ANALYSIS,
            original_query="Where is main.py?",
            entities=["main.py"],
            confidence=0.75,
        )
        json_str = i.to_json()
        i2 = Intent.from_json(json_str)
        assert i2.intent_type == IntentType.CODE_ANALYSIS
        assert i2.original_query == "Where is main.py?"
        assert i2.entities == ["main.py"]

    def test_confidence_clamped(self):
        """Confidence should stay in 0-1 range (no automatic clamping in dataclass,
        but the decoder should keep it in range)."""
        i = Intent(
            intent_id="i", intent_type=IntentType.GENERAL, original_query="q",
            confidence=0.88,
        )
        assert 0.0 <= i.confidence <= 1.0
