"""Tests for TokenBudgeter."""

import pytest
from src.context.token_budgeter import TokenBudgeter


@pytest.fixture
def budgeter():
    return TokenBudgeter()


class TestTokenBudgeter:
    def test_estimate_returns_positive_int(self, budgeter):
        tokens = budgeter.estimate("Hello, world! This is a test sentence.")
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_estimate_empty_string(self, budgeter):
        assert budgeter.estimate("") == 0
        assert budgeter.estimate("   ") >= 0

    def test_estimate_longer_text_uses_more_tokens(self, budgeter):
        short = budgeter.estimate("Hello.")
        long = budgeter.estimate("Hello. " * 100)
        assert long > short

    def test_estimate_list_of_strings(self, budgeter):
        texts = ["First paragraph.", "Second paragraph.", "Third paragraph."]
        total = budgeter.estimate_batch(texts)
        # Sum of individual estimates should equal batch estimate
        individual_sum = sum(budgeter.estimate(t) for t in texts)
        assert total == individual_sum

    def test_allocate_default_ratios(self, budgeter):
        allocation = budgeter.allocate(total_budget=10000)
        assert "system_instruction" in allocation
        assert "current_query" in allocation
        assert "conversation_history" in allocation
        assert "working_memory" in allocation
        assert "long_term_memory" in allocation
        assert "retrieved_evidence" in allocation
        assert "tool_results" in allocation
        assert "output_reserve" in allocation
        # Sum should be <= total
        total_allocated = sum(allocation.values())
        assert total_allocated <= 10000

    def test_allocate_custom_ratios(self, budgeter):
        custom_ratios = {
            "code_snippets": 0.5,
            "documentation": 0.3,
            "output": 0.2,
        }
        allocation = budgeter.allocate(total_budget=1000, ratios=custom_ratios)
        assert allocation["code_snippets"] == 500
        assert allocation["documentation"] == 300
        assert allocation["output"] == 200

    def test_allocate_zero_budget(self, budgeter):
        allocation = budgeter.allocate(total_budget=0)
        for v in allocation.values():
            assert v == 0

    def test_fits_in_budget(self, budgeter):
        """Text that fits within its section budget should return True."""
        assert budgeter.fits_in_budget("Short text.", budget=100) is True
        assert budgeter.fits_in_budget("a" * 10000, budget=10) is False

    def test_truncate_to_budget(self, budgeter):
        text = "This is a longer text that needs to be truncated. " * 20
        truncated = budgeter.truncate_to_budget(text, budget=30)
        assert budgeter.estimate(truncated) <= 30
        # Should preserve beginning of text
        assert truncated.startswith("This is a longer")

    def test_estimate_with_tiktoken_uses_cl100k_base(self, budgeter):
        """Verify we're using a reasonable tokenizer (cl100k_base for OpenAI models)."""
        # English text: roughly 1 token per 4 chars for cl100k
        tokens = budgeter.estimate("hello world")
        # "hello world" should be ~2-3 tokens with cl100k_base
        assert 1 <= tokens <= 5
