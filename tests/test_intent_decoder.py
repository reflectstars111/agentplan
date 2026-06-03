"""Tests for IntentDecoder."""

import pytest
from src.models.intent import IntentType
from src.runtime.intent_decoder import IntentDecoder


@pytest.fixture
def decoder():
    return IntentDecoder()


class TestIntentDecoder:
    def test_document_qa_triggers(self, decoder):
        queries = [
            "What is FastAPI?",
            "Explain the RAPTOR algorithm.",
            "How does retrieval-augmented generation work?",
            "Summarize this paper.",
            "Tell me about vector databases.",
        ]
        for q in queries:
            result = decoder.decode(q)
            assert result.intent_type == IntentType.DOCUMENT_QA, f"'{q}' should be DOCUMENT_QA, got {result.intent_type}"

    def test_code_analysis_triggers(self, decoder):
        queries = [
            "Where is the main function defined?",
            "Find the database connection code.",
            "Which file handles API routing?",
            "Locate the authentication middleware.",
            "Show me where error handling is implemented.",
        ]
        for q in queries:
            result = decoder.decode(q)
            assert result.intent_type == IntentType.CODE_ANALYSIS, f"'{q}' should be CODE_ANALYSIS, got {result.intent_type}"

    def test_memory_query_triggers(self, decoder):
        queries = [
            "What did we decide about the database?",
            "Recall our previous discussion about authentication.",
            "What was the outcome of the architecture review?",
        ]
        for q in queries:
            result = decoder.decode(q)
            assert result.intent_type == IntentType.MEMORY_QUERY, f"'{q}' should be MEMORY_QUERY, got {result.intent_type}"

    def test_general_fallback(self, decoder):
        queries = [
            "Hello.",
            "Thanks!",
            "OK, got it.",
            "",
        ]
        for q in queries:
            result = decoder.decode(q)
            assert result.intent_type == IntentType.GENERAL, f"'{q}' should be GENERAL, got {result.intent_type}"

    def test_entity_extraction(self, decoder):
        result = decoder.decode("Explain the RAPTOR paper and its FAISS implementation")
        assert "RAPTOR" in result.entities or any("RAPTOR" in e for e in result.entities)
        assert "FAISS" in result.entities or any("FAISS" in e for e in result.entities)

    def test_file_entity_extraction(self, decoder):
        result = decoder.decode("Where is main.py and database.py?")
        entities_flat = " ".join(result.entities).lower()
        assert "main.py" in entities_flat

    def test_confidence_range(self, decoder):
        result = decoder.decode("What is Python?")
        assert 0.0 <= result.confidence <= 1.0

    def test_returns_intent_with_id(self, decoder):
        result = decoder.decode("Hello world")
        assert result.intent_id.startswith("intent_")
        assert result.original_query == "Hello world"

    def test_priority_default(self, decoder):
        result = decoder.decode("Some query")
        assert 1 <= result.priority <= 10
