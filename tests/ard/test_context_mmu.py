"""Tests for Context MMU layer."""

import pytest

from ard.infra.config import Config
from ard.context.token_budgeter import TokenBudgeter, DEFAULT_BUDGET_RATIOS
from ard.context.mmu import ContextMMU
from ard.store import RetrievalResult


class TestTokenBudgeter:
    @pytest.fixture
    def budgeter(self):
        return TokenBudgeter(Config())

    def test_estimate_short_text(self, budgeter):
        tokens = budgeter.estimate("hello")
        assert tokens >= 1

    def test_estimate_empty(self, budgeter):
        assert budgeter.estimate("") == 0

    def test_allocate_all_sections(self, budgeter):
        alloc = budgeter.allocate_default(8000)
        assert len(alloc) == len(DEFAULT_BUDGET_RATIOS)
        total = sum(alloc.values())
        assert abs(total - 8000) < 100  # allow small rounding

    def test_retrieved_evidence_gets_largest_share(self, budgeter):
        alloc = budgeter.allocate_default(8000)
        sections_by_budget = sorted(alloc.items(), key=lambda x: x[1], reverse=True)
        assert sections_by_budget[0][0] == "retrieved_evidence"

    def test_truncate_long_text(self, budgeter):
        long_text = "x" * 2000
        truncated = budgeter.truncate_to_budget(long_text, token_budget=10)
        assert len(truncated) <= 10 * 4  # max 40 chars
        assert "[...truncated...]" in truncated or "..." in truncated

    def test_truncate_short_text_no_change(self, budgeter):
        short = "hello"
        result = budgeter.truncate_to_budget(short, token_budget=100)
        assert result == short

    def test_fit_items_respects_budget(self, budgeter):
        items = [{"text": f"item {i}" * 50} for i in range(10)]
        fitted = budgeter.fit_items_to_budget(items, token_budget=20)
        # Each item is ~350 chars = ~87 tokens, so only 0 or 1 should fit
        assert len(fitted) <= 1

    def test_compress_fallback_to_truncation(self, budgeter):
        long_text = "x" * 500
        compressed, was_compressed = budgeter.compress(long_text, token_budget=5)
        assert len(compressed) <= 5 * 4
        assert was_compressed is False  # Phase 1: no real compression


class TestContextMMU:
    @pytest.fixture
    def mmu(self):
        return ContextMMU(TokenBudgeter(Config()), Config())

    def test_assemble_empty_inputs(self, mmu):
        pack = mmu.assemble(query="test", retrieval_results=[])
        assert pack.context_id
        assert pack.budget > 0
        assert len(pack.sections) >= 1  # at least current_query

    def test_assemble_with_results(self, mmu):
        results = [
            RetrievalResult("c1", "src:1", "Important information about AI.", score=0.9,
                          trust_level="user_provided_data", strategy="vector"),
            RetrievalResult("c2", "src:2", "More details about machine learning.", score=0.7,
                          trust_level="user_provided_data", strategy="keyword"),
        ]
        pack = mmu.assemble(query="Tell me about AI", retrieval_results=results,
                           system_instruction="Be helpful.")
        assert "src:1" in pack.source_refs
        assert "src:2" in pack.source_refs
        assert any(s.name == "retrieved_evidence" for s in pack.sections)
        assert any(s.name == "system_instruction" for s in pack.sections)

    def test_deduplicates_results(self, mmu):
        results = [
            RetrievalResult("c1", "src:1", "Same chunk.", score=0.9,
                          trust_level="user_provided_data", strategy="vector"),
            RetrievalResult("c1", "src:1", "Same chunk.", score=0.8,
                          trust_level="user_provided_data", strategy="keyword"),
        ]
        pack = mmu.assemble(query="test", retrieval_results=results, top_k=10)
        evidence = [s for s in pack.sections if s.name == "retrieved_evidence"]
        if evidence:
            # Should have 1 item, not 2 (dedup by chunk_id)
            assert len(evidence[0].items) == 1

    def test_to_text_renders_pack(self, mmu):
        pack = mmu.assemble(query="Hello world", retrieval_results=[],
                           system_instruction="You are helpful.")
        text = pack.to_text()
        assert "CURRENT_QUERY" in text.upper() or "current_query" in text.lower()
        assert "Hello world" in text

    def test_respects_top_k(self, mmu):
        results = [
            RetrievalResult(f"c{i}", f"src:{i}", f"Text chunk number {i}.", score=0.9 - i * 0.1,
                          trust_level="user_provided_data", strategy="vector")
            for i in range(20)
        ]
        pack = mmu.assemble(query="test", retrieval_results=results, top_k=5)
        evidence = [s for s in pack.sections if s.name == "retrieved_evidence"]
        if evidence:
            assert len(evidence[0].items) <= 5
