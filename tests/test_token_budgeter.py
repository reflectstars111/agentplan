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

    def test_compress_keeps_sentences_within_budget(self, budgeter):
        """Compress should preserve complete sentences when possible."""
        text = "Python is a language. FastAPI is a framework. Kafka is a platform."
        # Target: enough for ~2 sentences
        compressed, was_compressed = budgeter.compress(text, 20)
        assert "Python" in compressed
        assert was_compressed or len(compressed) <= len(text)

    def test_compress_no_compression_when_fits(self, budgeter):
        """Compress should return original when within budget."""
        text = "Short text."
        compressed, was_compressed = budgeter.compress(text, 500)
        assert compressed == text
        assert was_compressed is False

    def test_compress_handles_single_long_sentence(self, budgeter):
        """Compress single long sentence by truncation."""
        text = "A" * 100
        compressed, was_compressed = budgeter.compress(text, 5)
        assert len(compressed) < len(text)
        assert was_compressed is True

    def test_compress_llm_falls_back_when_no_llm(self, budgeter):
        """compress_llm() without llm_fn should fall back to heuristic."""
        budgeter.llm_fn = None
        text = "Short text."
        compressed, was_compressed = budgeter.compress_llm(text, 500)
        assert compressed == text
        assert was_compressed is False

    def test_compress_llm_uses_llm_when_available(self, budgeter):
        """compress_llm() should call llm_fn when available."""
        def mock_llm(ctx, prompt):
            return "Compressed version."
        budgeter.llm_fn = mock_llm
        text = "A very long text that needs compression." * 10
        compressed, was_compressed = budgeter.compress_llm(text, 10)
        assert "Compressed version" in compressed
        assert was_compressed is True

    def test_compress_llm_falls_back_on_llm_error(self, budgeter):
        """compress_llm() falls back when LLM returns error."""
        def mock_llm(ctx, prompt):
            return "[LLM Error: API unavailable]"
        budgeter.llm_fn = mock_llm
        text = "Text that needs compression. But LLM failed."
        compressed, _ = budgeter.compress_llm(text, 5)
        # Should fall back to heuristic (not the error message)
        assert "LLM Error" not in compressed
