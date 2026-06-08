"""TokenBudgeter — estimates, allocates, and truncates context token budgets.

Phase 1: Character-based estimation (len/4). Future: use tiktoken or actual tokenizer.
"""

from dataclasses import dataclass, field

from ard.infra.config import Config


# Default budget ratios per section (ARD design §6)
DEFAULT_BUDGET_RATIOS = {
    "system_instruction": 0.10,
    "current_query": 0.05,
    "conversation_history": 0.10,
    "working_memory": 0.10,
    "long_term_memory": 0.10,
    "retrieved_evidence": 0.35,
    "tool_results": 0.10,
    "output_reserve": 0.10,
}


@dataclass
class BudgetAllocation:
    """Token budget allocated per section."""
    allocations: dict[str, int] = field(default_factory=dict)
    total: int = 0
    consumed: int = 0

    def remaining(self) -> int:
        return max(0, self.total - self.consumed)


class TokenBudgeter:
    """Manages token counting, allocation, and compression for context assembly."""

    # Approximate: 1 token ≈ 4 characters (conservative for English/Chinese mixed)
    CHARS_PER_TOKEN = 4

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    # ── Estimation ──────────────────────────────────────────

    def estimate(self, text: str) -> int:
        """Estimate token count for a text string."""
        if not text:
            return 0
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def estimate_batch(self, texts: list[str]) -> int:
        """Estimate total tokens for a batch of texts."""
        return sum(self.estimate(t) for t in texts)

    # ── Allocation ──────────────────────────────────────────

    def allocate(self, total_budget: int, ratios: dict[str, float]) -> dict[str, int]:
        """Allocate total budget across sections by ratio.

        Returns:
            Dict mapping section name → token budget.
        """
        allocated = {}
        remaining = total_budget

        # Allocate in order, last section gets remainder
        sections = list(ratios.keys())
        for i, section in enumerate(sections[:-1]):
            share = max(10, int(total_budget * ratios[section]))
            allocated[section] = share
            remaining -= share

        if sections:
            allocated[sections[-1]] = max(10, remaining)

        return allocated

    def allocate_default(self, total_budget: int | None = None) -> dict[str, int]:
        """Allocate using default ARD ratios."""
        budget = total_budget or self.config.default_token_budget
        return self.allocate(budget, DEFAULT_BUDGET_RATIOS)

    # ── Truncation & Compression ─────────────────────────────

    def truncate_to_budget(self, text: str, token_budget: int) -> str:
        """Truncate text to fit within a token budget.

        Keeps the beginning and end, cuts from the middle.
        """
        if token_budget <= 0:
            return ""

        max_chars = token_budget * self.CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text

        # Keep 60% from start, 30% from end
        head_ratio = 0.6
        head_chars = int(max_chars * head_ratio)
        tail_chars = max_chars - head_chars - 20  # 20 chars for "[...truncated...]"

        if tail_chars < 20:
            # Not enough budget for both — just return the head
            return text[:max_chars - 3] + "..."

        return (
            text[:head_chars]
            + "\n[...truncated...]\n"
            + text[-tail_chars:]
        )

    def compress(self, text: str, token_budget: int) -> tuple[str, bool]:
        """Attempt semantic compression; fall back to truncation.

        Phase 1: No real semantic compression. Just truncate.
        Phase 2+: Could use LLM-based summarization.

        Returns:
            (compressed_text, was_actually_compressed)
        """
        max_chars = token_budget * self.CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text, False

        # Future: call LLM summarizer here
        return self.truncate_to_budget(text, token_budget), False

    def fit_items_to_budget(
        self, items: list[dict], token_budget: int, text_key: str = "text"
    ) -> list[dict]:
        """Fit as many items as possible into a token budget.

        Items exceeding budget are compressed or truncated.
        Returns the subset of items that fit.
        """
        result = []
        tokens_used = 0

        for item in items:
            text = item.get(text_key, "")
            item_tokens = self.estimate(text)

            if tokens_used + item_tokens <= token_budget:
                result.append(item)
                tokens_used += item_tokens
            else:
                remaining = token_budget - tokens_used
                if remaining > self.estimate("...") + 1:
                    compressed, was_compressed = self.compress(text, remaining)
                    new_item = {**item, text_key: compressed}
                    if was_compressed:
                        new_item["compressed"] = True
                    result.append(new_item)
                break  # no more room

        return result
