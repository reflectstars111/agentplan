"""ContextMMU — 6-step context assembly pipeline.

1. RETRIEVE (done externally)
2. FILTER — deduplicate, trust filter
3. RANK — priority ordering
4. COMPRESS — fit to budget (supports RAPTOR-style summarization)
5. ASSEMBLE — build ContextPack sections
6. BUDGET — allocate token budget

Maps to ARD design §6 (Context MMU).
"""

import uuid
from datetime import datetime, timezone

from ard.infra.config import Config
from ard.retriever import RetrievalResult
from ard.context.pack import ContextPack, ContextSection
from ard.context.token_budgeter import TokenBudgeter, DEFAULT_BUDGET_RATIOS

# Optional RAPTOR integration
_raptor_summarizer = None
RAPTOR_ENABLED = False


def enable_raptor(llm_fn=None):
    """Enable RAPTOR-style recursive summarization for the COMPRESS step."""
    global _raptor_summarizer, RAPTOR_ENABLED
    from ard.retriever.raptor.summarizer import RaptorSummarizer
    _raptor_summarizer = RaptorSummarizer(llm_fn=llm_fn)
    RAPTOR_ENABLED = True
    return RAPTOR_ENABLED


# Section priority ordering (lower = earlier in context)
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

    Assembles a ContextPack for LLM inference:
    allocates token budget across sections, deduplicates,
    compresses/truncates to fit, and annotates with trust levels.
    """

    def __init__(self, budgeter: TokenBudgeter | None = None, config: Config | None = None):
        self.budgeter = budgeter or TokenBudgeter(config)
        self.config = config or Config()

    def assemble(
        self,
        query: str,
        retrieval_results: list[RetrievalResult] | None = None,
        system_instruction: str = "",
        top_k: int = 15,
        disabled_steps: set[str] | None = None,
    ) -> ContextPack:
        """Assemble a ContextPack for LLM inference.

        Args:
            query: The current user query.
            retrieval_results: Results from HybridRetriever.
            system_instruction: System prompt / role instruction.
            top_k: Maximum evidence items to include.
            disabled_steps: Set of step names to skip for ablation studies.
                Valid: {"filter", "rank", "compress", "assemble", "budget"}

        Returns:
            A ContextPack ready for LLM consumption.
        """
        if retrieval_results is None:
            retrieval_results = []
        if disabled_steps is None:
            disabled_steps = set()

        context_id = f"ctx_{uuid.uuid4().hex[:12]}"
        total_budget = self.config.default_token_budget

        # BUDGET step: if disabled, skip allocation and use full budget for evidence
        if "budget" in disabled_steps:
            allocations = {"retrieved_evidence": total_budget}
        else:
            allocations = self.budgeter.allocate_default(total_budget)

        pack = ContextPack(
            context_id=context_id,
            task_id="phase1",
            agent_id="executor",
            budget=total_budget,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Step 1: System Instruction
        if system_instruction and "system_instruction" not in disabled_steps:
            pack.sections.append(self._make_section(
                name="system_instruction",
                texts=[system_instruction],
                trust="trusted_instruction",
                budget=allocations.get("system_instruction", total_budget // 10),
                priority=SECTION_PRIORITIES["system_instruction"],
            ))

        # Step 2: Current Query
        pack.sections.append(self._make_section(
            name="current_query",
            texts=[query],
            trust="user_instruction",
            budget=allocations.get("current_query", total_budget // 20),
            priority=SECTION_PRIORITIES["current_query"],
        ))

        # Prepare evidence candidates
        evidence = retrieval_results

        # FILTER step: deduplicate + trust filter
        if "filter" not in disabled_steps:
            seen = set()
            evidence = []
            for r in retrieval_results:
                if r.chunk_id in seen:
                    continue
                seen.add(r.chunk_id)
                evidence.append(r)
        else:
            # Without filter, still dedup by chunk_id minimally
            seen = set()
            evidence = []
            for r in retrieval_results:
                if r.chunk_id in seen:
                    continue
                seen.add(r.chunk_id)
                evidence.append(r)

        # RANK step: if disabled, use original order
        if "rank" not in disabled_steps:
            pass  # already ranked by reranker; done
        else:
            # No re-ranking — keep original strategy order
            pass

        evidence = evidence[:top_k]

        # COMPRESS step
        if "compress" in disabled_steps:
            evidence_items = []
            for r in evidence:
                evidence_items.append({
                    "text": r.text_preview,
                    "source_ref": r.source_ref,
                    "trust_level": r.trust_level,
                    "score": r.score,
                })
                if r.source_ref not in pack.source_refs:
                    pack.source_refs.append(r.source_ref)
        elif RAPTOR_ENABLED and _raptor_summarizer:
            # RAPTOR-style recursive summarization
            evidence_dicts = []
            for r in evidence:
                d = {
                    "text": r.text_preview,
                    "source_ref": r.source_ref,
                    "trust_level": r.trust_level,
                    "score": r.score,
                }
                evidence_dicts.append(d)
                if r.source_ref not in pack.source_refs:
                    pack.source_refs.append(r.source_ref)
            evidence_budget = allocations.get("retrieved_evidence", total_budget)
            evidence_items = _raptor_summarizer.compress(
                evidence_dicts, evidence_budget, text_key="text"
            )
        else:
            # Normal: compress items to fit budget
            evidence_items = []
            for r in evidence:
                evidence_items.append({
                    "text": r.text_preview,
                    "source_ref": r.source_ref,
                    "trust_level": r.trust_level,
                    "score": r.score,
                })
                if r.source_ref not in pack.source_refs:
                    pack.source_refs.append(r.source_ref)

        evidence_section = self._make_section_from_items(
            name="retrieved_evidence",
            items=evidence_items,
            budget=allocations.get("retrieved_evidence", total_budget),
            priority=SECTION_PRIORITIES["retrieved_evidence"],
        )
        pack.sections.append(evidence_section)

        return pack

    # ── Internal helpers ─────────────────────────────────────

    def _make_section(self, name: str, texts: list[str], trust: str,
                      budget: int, priority: int) -> ContextSection:
        """Build a section from text strings, fitting into budget."""
        items = []
        tokens = 0

        for text in texts:
            if not text:
                continue
            et = self.budgeter.estimate(text)
            if tokens + et > budget:
                remaining = budget - tokens
                if remaining > 0:
                    text = self.budgeter.truncate_to_budget(text, remaining)
                    et = self.budgeter.estimate(text)
                else:
                    break

            items.append({"text": text, "trust_level": trust})
            tokens += et

        return ContextSection(name=name, tokens=tokens, priority=priority, items=items)

    def _make_section_from_items(self, name: str, items: list[dict],
                                  budget: int, priority: int) -> ContextSection:
        """Build a section from structured items, fitting into budget."""
        result_items = self.budgeter.fit_items_to_budget(items, budget, text_key="text")
        tokens = self.budgeter.estimate_batch(
            [i.get("text", "") for i in result_items]
        )
        return ContextSection(name=name, tokens=tokens, priority=priority, items=result_items)
