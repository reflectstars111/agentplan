"""TokenBudgeter — token estimation and budget allocation for context assembly.

Uses tiktoken (cl100k_base) for accurate token counting with a character-based
fallback when tiktoken is unavailable.
"""

from typing import Optional

# Default budget ratios per agent_os_initial_plan.md §9.3
DEFAULT_BUDGET_RATIOS = {
    "system_instruction": 0.10,     # 10% - role / system prompt
    "current_query": 0.05,          #  5% - current user request
    "conversation_history": 0.10,   # 10% - recent dialogue turns
    "working_memory": 0.10,         # 10% - current task state
    "long_term_memory": 0.10,       # 10% - historical memories
    "retrieved_evidence": 0.35,     # 35% - file chunks / search results
    "tool_results": 0.10,           # 10% - tool call outputs
    "output_reserve": 0.10,         # 10% - reserved for LLM response
}


class TokenBudgeter:
    """Token counting and budget allocation for context management."""

    def __init__(self, model_name: str = "cl100k_base"):
        self.model_name = model_name
        self._encoder = None
        self._init_encoder()

    def _init_encoder(self) -> None:
        """Initialize the tiktoken encoder. Falls back to None if unavailable."""
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding(self.model_name)
        except (ImportError, Exception):
            self._encoder = None

    def estimate(self, text: str) -> int:
        """Estimate token count for a single text string."""
        if not text or not text.strip():
            return 0

        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass

        # Fallback: ~4 characters per token for English text
        return max(1, len(text) // 4)

    def estimate_batch(self, texts: list[str]) -> int:
        """Estimate total token count for a list of texts."""
        return sum(self.estimate(t) for t in texts)

    def allocate(
        self,
        total_budget: int,
        ratios: dict[str, float] | None = None,
    ) -> dict[str, int]:
        """Allocate token budget across sections based on ratios.

        Args:
            total_budget: Total available tokens.
            ratios: Dict of section_name -> ratio (e.g., {"system": 0.10, "docs": 0.35}).
                    If None, uses DEFAULT_BUDGET_RATIOS.

        Returns:
            Dict of section_name -> token_budget (int).
        """
        if total_budget <= 0:
            if ratios is None:
                ratios = DEFAULT_BUDGET_RATIOS
            return {k: 0 for k in (ratios or {})}

        if ratios is None:
            ratios = DEFAULT_BUDGET_RATIOS

        allocation = {}
        remaining = total_budget

        # Allocate proportionally, floor to int
        sections = list(ratios.items())
        for name, ratio in sections[:-1]:
            budget = int(total_budget * ratio)
            allocation[name] = budget
            remaining -= budget

        # Last section gets the remainder (avoids rounding loss)
        if sections:
            allocation[sections[-1][0]] = remaining

        return allocation

    def fits_in_budget(self, text: str, budget: int) -> bool:
        """Check if text fits within the given token budget."""
        return self.estimate(text) <= budget

    def truncate_to_budget(self, text: str, budget: int) -> str:
        """Truncate text to fit within a token budget.

        Truncation preserves the beginning of the text and appends '...'
        if truncation occurred.
        """
        if self.estimate(text) <= budget:
            return text

        if budget <= 0:
            return ""

        # Reserve tokens for the "..." truncation indicator
        ellipsis_tokens = self.estimate("...")
        effective_budget = max(0, budget - ellipsis_tokens)

        if self._encoder is not None:
            try:
                tokens = self._encoder.encode(text)
                truncated_tokens = tokens[:effective_budget]
                return self._encoder.decode(truncated_tokens) + "..."
            except Exception:
                pass

        # Fallback: character-based truncation (~4 chars/token)
        max_chars = effective_budget * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."
