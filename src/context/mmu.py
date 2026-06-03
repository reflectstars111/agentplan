"""ContextMMU — Context Memory Management Unit.

Assembles a ContextPack for LLM inference: allocates token budget across
sections, deduplicates content, truncates to fit, and annotates with
source references and trust levels.

Maps to agent_os_initial_plan.md §9.
"""

import uuid
from datetime import datetime, timezone

from src.config import Config
from src.models.memory import MemoryItem
from src.models.context import ContextPack, ContextSection
from src.context.token_budgeter import TokenBudgeter, DEFAULT_BUDGET_RATIOS
from src.index.hybrid_retriever import RetrievalResult


# Section priority ordering (lower = appears first in context)
SECTION_PRIORITIES = {
    "system_instruction": 1,
    "current_query": 2,
    "working_memory": 3,
    "conversation_history": 4,
    "long_term_memory": 5,
    "retrieved_evidence": 6,
    "tool_results": 7,
    "output_reserve": 8,
}


class ContextMMU:
    """Context Memory Management Unit.

    Responsible for deciding what the LLM sees in each inference call.
    Assembles a ContextPack from query, retrieval results, memories, and
    system instructions, all within a token budget.
    """

    def __init__(self, budgeter: TokenBudgeter, config: Config | None = None):
        self.budgeter = budgeter
        self.config = config or Config()

    def assemble(
        self,
        query: str,
        retrieval_results: list[RetrievalResult] | None = None,
        working_memories: list[MemoryItem] | None = None,
        long_term_memories: list[MemoryItem] | None = None,
        conversation_history: list[dict] | None = None,
        system_instruction: str = "",
        task_id: str = "",
        agent_id: str = "",
    ) -> ContextPack:
        """Assemble a ContextPack for LLM inference.

        Args:
            query: The current user query.
            retrieval_results: Results from hybrid retriever.
            working_memories: Active task state memories (L2).
            long_term_memories: Historical memories (L3).
            conversation_history: Recent dialogue turns.
            system_instruction: System prompt / role instruction.
            task_id: Current task identifier.
            agent_id: Current agent identifier.

        Returns:
            A ContextPack ready for LLM consumption.
        """
        if retrieval_results is None:
            retrieval_results = []
        if working_memories is None:
            working_memories = []
        if long_term_memories is None:
            long_term_memories = []
        if conversation_history is None:
            conversation_history = []

        context_id = f"ctx_{uuid.uuid4().hex[:12]}"
        total_budget = self.config.default_token_budget

        # Allocate budget per section
        allocations = self.budgeter.allocate(total_budget, DEFAULT_BUDGET_RATIOS)

        pack = ContextPack(
            context_id=context_id,
            task_id=task_id or "unknown",
            agent_id=agent_id or "unknown",
            budget=total_budget,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Build sections in priority order

        # 1. System instruction (priority 1)
        if system_instruction:
            pack.add_section(self._build_section(
                name="system_instruction",
                items_text=[system_instruction],
                trust_level="trusted_instruction",
                budget=allocations.get("system_instruction", 0),
                priority=SECTION_PRIORITIES["system_instruction"],
            ))

        # 2. Current query (priority 2)
        pack.add_section(self._build_section(
            name="current_query",
            items_text=[query],
            trust_level="user_instruction",
            budget=allocations.get("current_query", 0),
            priority=SECTION_PRIORITIES["current_query"],
        ))

        # 3. Working memory (priority 3)
        if working_memories:
            mem_texts = [m.content for m in working_memories]
            pack.add_section(self._build_section(
                name="working_memory",
                items_text=mem_texts,
                trust_level="internal_memory",
                budget=allocations.get("working_memory", 0),
                priority=SECTION_PRIORITIES["working_memory"],
                source_ref="memory:working",
            ))

        # 4. Conversation history (priority 4)
        if conversation_history:
            conv_texts = [
                f"[{t.get('role', 'unknown')}]: {t.get('content', '')}"
                for t in conversation_history
            ]
            pack.add_section(self._build_section(
                name="conversation_history",
                items_text=conv_texts,
                trust_level="internal_memory",
                budget=allocations.get("conversation_history", 0),
                priority=SECTION_PRIORITIES["conversation_history"],
            ))

        # 5. Long-term memory (priority 5)
        if long_term_memories:
            ltm_texts = [m.content for m in long_term_memories]
            pack.add_section(self._build_section(
                name="long_term_memory",
                items_text=ltm_texts,
                trust_level="internal_memory",
                budget=allocations.get("long_term_memory", 0),
                priority=SECTION_PRIORITIES["long_term_memory"],
                source_ref="memory:long_term",
            ))

        # 6. Retrieved evidence (priority 6) — the main content
        if retrieval_results:
            evidence_items = []
            seen_texts = set()

            for r in retrieval_results:
                if r.text_preview in seen_texts:
                    continue
                seen_texts.add(r.text_preview)

                evidence_items.append({
                    "source_ref": r.source_ref,
                    "trust_level": r.trust_level,
                    "text": r.text_preview,
                    "score": r.score,
                })
                if r.source_ref not in pack.source_refs:
                    pack.source_refs.append(r.source_ref)

            # Fit items into budget
            evidence_section = self._build_section_from_items(
                name="retrieved_evidence",
                items=evidence_items,
                budget=allocations.get("retrieved_evidence", 0),
                priority=SECTION_PRIORITIES["retrieved_evidence"],
            )
            pack.add_section(evidence_section)

        return pack

    def _build_section(
        self,
        name: str,
        items_text: list[str],
        trust_level: str,
        budget: int,
        priority: int,
        source_ref: str = "",
    ) -> ContextSection:
        """Build a ContextSection from a list of text strings, fitting into budget."""
        items = []
        tokens_used = 0

        for text in items_text:
            if not text:
                continue

            # Check if this item fits in remaining budget
            item_tokens = self.budgeter.estimate(text)
            if tokens_used + item_tokens > budget:
                # Try to truncate
                remaining = budget - tokens_used
                if remaining > 0:
                    text = self.budgeter.truncate_to_budget(text, remaining)
                    item_tokens = self.budgeter.estimate(text)
                else:
                    break

            item = {
                "text": text,
                "trust_level": trust_level,
            }
            if source_ref:
                item["source_ref"] = source_ref

            items.append(item)
            tokens_used += item_tokens

        return ContextSection(
            name=name,
            tokens=tokens_used,
            priority=priority,
            items=items,
        )

    def _build_section_from_items(
        self,
        name: str,
        items: list[dict],
        budget: int,
        priority: int,
    ) -> ContextSection:
        """Build a ContextSection from pre-structured items, fitting into budget."""
        result_items = []
        tokens_used = 0

        for item in items:
            text = item.get("text", "")
            if not text:
                continue

            item_tokens = self.budgeter.estimate(text)
            if tokens_used + item_tokens > budget:
                remaining = budget - tokens_used
                if remaining > 0:
                    item["text"] = self.budgeter.truncate_to_budget(text, remaining)
                    item_tokens = self.budgeter.estimate(item["text"])
                else:
                    break

            result_items.append(item)
            tokens_used += item_tokens

        return ContextSection(
            name=name,
            tokens=tokens_used,
            priority=priority,
            items=result_items,
        )
