"""ContextPageFault — "page fault" mechanism to load missing context mid-inference.

Maps to agent_os_initial_plan.md §9.4 (Context Page Fault).

When the LLM or Verifier detects that the current context pack lacks needed
information, a ContextPageFault is triggered:
  1. Identify missing information
  2. Generate a retrieval query
  3. Retrieve from L2/L3/L4 storage
  4. Load the relevant chunk
  5. Update the context pack
  6. Continue (re-run) the task
"""

import re
from dataclasses import dataclass, field
from src.models.context import ContextPack, ContextSection


@dataclass
class PageFaultResult:
    triggered: bool
    retrieved_chunks: list = field(default_factory=list)
    updated_pack: ContextPack | None = None
    query_used: str = ""


class ContextPageFault:
    """Handle page faults when context is insufficient for LLM reasoning."""

    def __init__(self, retriever, mmu, max_faults: int = 2):
        self.retriever = retriever
        self.mmu = mmu
        self.max_faults = max_faults
        self._fault_count = 0

    def check_and_handle(
        self,
        response: str,
        context_pack: ContextPack,
        original_query: str,
        embed_fn,
    ) -> PageFaultResult:
        """Check if the response indicates missing context, and if so, re-retrieve.

        Triggers when:
          - Response contains uncertainty markers indicating missing info
          - Verifier flags unverified claims that need source evidence
          - Response explicitly states "I don't have enough information"
        """
        if self._fault_count >= self.max_faults:
            return PageFaultResult(triggered=False)

        # Detect page fault signals in the response
        if not self._needs_more_context(response):
            return PageFaultResult(triggered=False)

        # Extract what's missing
        missing_query = self._extract_missing_query(response, original_query)
        if not missing_query:
            return PageFaultResult(triggered=False)

        self._fault_count += 1

        # Re-retrieve
        results = self.retriever.retrieve(missing_query, embed_fn, k=5)
        if not results:
            return PageFaultResult(triggered=False, query_used=missing_query)

        # Build new evidence section
        new_items = []
        for r in results:
            new_items.append({
                "source_ref": r.source_ref,
                "trust_level": r.trust_level,
                "text": r.text_preview,
            })

        new_section = ContextSection(
            name="page_fault_evidence",
            tokens=min(2000, context_pack.remaining_budget()),
            priority=7,
            items=new_items,
        )

        updated_pack = context_pack
        if updated_pack.add_section(new_section):
            for ref in set(r.source_ref for r in results):
                if ref not in updated_pack.source_refs:
                    updated_pack.source_refs.append(ref)

        return PageFaultResult(
            triggered=True,
            retrieved_chunks=results,
            updated_pack=updated_pack,
            query_used=missing_query,
        )

    def reset(self) -> None:
        self._fault_count = 0

    def _needs_more_context(self, response: str) -> bool:
        """Check if the response signals insufficient context."""
        uncertainty_patterns = [
            r"(?:i don't have|i do not have|no (?:relevant |specific )?information|not enough (?:context|information))",
            r"(?:cannot|can't|unable to) (?:answer|determine|find|locate)",
            r"(?:insufficient|missing|lack(?:ing)?) (?:context|data|information|details)",
        ]
        response_lower = response.lower()
        return any(re.search(p, response_lower) for p in uncertainty_patterns)

    def _extract_missing_query(self, response: str, original_query: str) -> str:
        """Extract what specific information is missing from the response."""
        # Look for "need more information about X" or "would need X to answer"
        need_patterns = [
            r"need (?:more |additional )?(?:information |context |details )?about\s+(.+?)(?:\.|$)",
            r"would (?:need |require )(.+?)(?:to answer|\.|$)",
            r"missing\s+(.+?)(?:\.|$)",
        ]
        for pattern in need_patterns:
            m = re.search(pattern, response, re.IGNORECASE)
            if m:
                return m.group(1).strip()

        # Fallback: use the original query
        return original_query
