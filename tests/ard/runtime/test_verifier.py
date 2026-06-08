"""Tests for Verifier."""

import pytest

from ard.runtime.verifier import Verifier, Verdict
from ard.context.pack import ContextPack, ContextSection


class TestVerifier:
    @pytest.fixture
    def verifier(self):
        return Verifier()

    def test_verify_empty(self, verifier):
        pack = ContextPack("ctx_test", "task_test", "agent_test", 1000)
        verdict = verifier.verify("Some answer.", pack)
        assert isinstance(verdict, Verdict)
        assert not verdict.verified  # no evidence, should fail

    def test_verify_with_matching_evidence(self, verifier):
        pack = ContextPack("ctx", "t", "a", 1000)
        pack.sections = [
            ContextSection("retrieved_evidence", tokens=100, priority=6, items=[
                {"text": "ARD uses hybrid retrieval combining vector and keyword search. The system manages token budgets dynamically."},
            ]),
        ]

        response = "ARD uses hybrid retrieval that combines vector and keyword search methods."
        verdict = verifier.verify(response, pack)
        assert verdict.confidence > 0.5  # evidence matches

    def test_orphan_claims_detected(self, verifier):
        pack = ContextPack("ctx", "t", "a", 1000)
        pack.sections = [
            ContextSection("retrieved_evidence", tokens=100, priority=6, items=[
                {"text": "The sky is blue."},
            ]),
        ]

        response = "The sky is blue. The moon is made of cheese. Unicorns exist in the forest."
        verdict = verifier.verify(response, pack)
        assert len(verdict.orphan_claims) > 0  # cheese and unicorns are unsupported

    def test_confidence_computation(self, verifier):
        pack = ContextPack("ctx", "t", "a", 1000)
        pack.sections = [
            ContextSection("retrieved_evidence", tokens=100, priority=6, items=[
                {"text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data."},
            ]),
        ]
        pack.source_refs = ["src:1"]

        response = "Machine learning is a subset of artificial intelligence."
        verdict = verifier.verify(response, pack)
        assert 0.0 <= verdict.confidence <= 1.0

    def test_verify_no_sections(self, verifier):
        pack = ContextPack("ctx", "t", "a", 1000)
        verdict = verifier.verify("Any response without context.", pack)
        assert not verdict.verified
        assert verdict.confidence <= 0.6
