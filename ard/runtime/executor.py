"""Executor — receives ContextPack, calls LLM, returns response.

Phase 1: Minimal executor wrapping LLM factory.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Any

from ard.context.pack import ContextPack


@dataclass
class ExecutorResponse:
    """Output from an executor run."""
    answer: str
    context_id: str
    tokens_used: int = 0
    source_refs: list[str] = field(default_factory=list)
    raw_llm_response: str = ""


@runtime_checkable
class ExecutorProtocol(Protocol):
    """Protocol for an LLM executor."""

    def think(self, context_pack: ContextPack, query: str = "") -> ExecutorResponse:
        ...


class Executor(ExecutorProtocol):
    """Takes a ContextPack, sends it to LLM, returns structured response.

    Supports two LLM function interfaces:
    - `fn(context_pack: ContextPack, query: str) → str`  (LLM factory default)
    - `fn(prompt_text: str, system_text: str) → str`     (simple / mock)
    """

    def __init__(self, llm_fn: Callable | None = None):
        self._raw_llm_fn = llm_fn
        self.llm_fn: Callable = self._adapt(llm_fn) if llm_fn else self._mock_llm

    @staticmethod
    def _adapt(fn: Callable) -> Callable:
        """Wrap LLM function to support (context_pack, query) → str interface.

        Detects and normalizes between:
        - LLM factory style: fn(context_pack, query) → str
        - Simple style: fn(prompt_text, system_text) → str
        """
        import inspect
        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())
            if len(params) >= 2 and params[0] in ("context_pack", "ctx"):
                # Already (context_pack, query) style
                def adapted(ctx_pack: ContextPack, query: str = "") -> str:
                    return fn(ctx_pack, query)
                return adapted
        except (ValueError, TypeError):
            pass

        # Fallback: assume (prompt_text, system_text) style
        def adapted(ctx_pack: ContextPack, query: str = "") -> str:
            system = ""
            for s in ctx_pack.sections:
                if s.name == "system_instruction" and s.items:
                    system = s.items[0].get("text", "")
                    break
            prompt = ctx_pack.to_text()
            try:
                return fn(prompt, system)
            except TypeError:
                # Try single-arg fallback
                return fn(prompt)
        return adapted

    def think(self, context_pack: ContextPack, query: str = "") -> ExecutorResponse:
        """Send context to LLM and return response."""

        try:
            raw = self.llm_fn(context_pack, query)
        except Exception as e:
            raw = f"[LLM_ERROR: {e}]"

        return ExecutorResponse(
            answer=raw,
            context_id=context_pack.context_id,
            tokens_used=context_pack.total_tokens_used(),
            source_refs=context_pack.source_refs,
            raw_llm_response=raw,
        )

    @staticmethod
    def _mock_llm(context_pack: ContextPack, query: str = "") -> str:
        """Mock LLM for testing — returns a template response."""
        # Extract evidence snippets
        evidence = []
        for section in context_pack.sections:
            if section.name == "retrieved_evidence":
                for item in section.items:
                    text = item.get("text", "")
                    if isinstance(text, str) and len(text) > 20:
                        evidence.append(text[:200])
        if evidence:
            return (f"Based on the provided context, I found {len(evidence)} "
                    f"relevant sources. Key information: {chr(10).join(evidence[:3])}")
        return f"No relevant information found for: {query}"
