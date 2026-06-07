"""AgentRuntime — end-to-end pipeline orchestration.

Wires together all MVP components into a single execution loop:
  Input → Retrieve → Context MMU → LLM → Verify → Writeback → Trace → Output

Maps to agent_os_initial_plan.md §18.4 (MVP flow).
"""

import uuid
from pathlib import Path
from typing import Callable, Any, Optional
from src.config import Config
from src.db import Database
from src.storage.file_store import FileStore
from src.storage.memory_store import MemoryStore
from src.index.hybrid_retriever import HybridRetriever, RetrievalResult
from src.context.mmu import ContextMMU
from src.context.token_budgeter import TokenBudgeter
from src.runtime.verifier import Verifier, VerifyOutput
from src.runtime.writeback_gate import WritebackGate, WritebackDecision
from src.runtime.trace_logger import TraceLogger
from src.models.trace import TraceStep, StepType, StepStatus
from src.models.memory import MemoryItem, MemoryType


class AgentRuntime:
    """Orchestrates the full Agent-OS MVP pipeline.

    Usage:
        runtime = AgentRuntime(file_store, memory_store, retriever, mmu,
                               verifier, gate, logger, embed_fn, llm_fn)
        result = runtime.process_query("What is FastAPI?")
        print(result["response"])
        print(result["trace_id"])
    """

    def __init__(
        self,
        file_store: FileStore,
        memory_store: MemoryStore,
        retriever: HybridRetriever,
        mmu: ContextMMU,
        verifier: Verifier,
        writeback_gate: WritebackGate,
        trace_logger: TraceLogger,
        config: Config | None = None,
        embed_fn: Callable | None = None,
        llm_fn: Callable | None = None,
        agent_id: str = "agent_worker_001",
        role: str = "worker",
        memory_scope: dict | None = None,
        page_fault=None,
        entity_index=None,
        dependency_graph=None,
        conversation_cache=None,
    ):
        self.file_store = file_store
        self.memory_store = memory_store
        self.retriever = retriever
        self.mmu = mmu
        self.verifier = verifier
        self.writeback_gate = writeback_gate
        self.trace_logger = trace_logger
        self.config = config or Config()
        self.embed_fn = embed_fn or (lambda texts: [])
        self.llm_fn = llm_fn or self._default_llm
        self.agent_id = agent_id
        self.role = role
        self.memory_scope = memory_scope or {}
        self.page_fault = page_fault
        self.entity_index = entity_index
        self.dependency_graph = dependency_graph
        self.conversation_cache = conversation_cache

    def upload_text(self, content: str, source_name: str) -> str:
        """Upload text content and index it. Returns source_id."""
        source_id = self.file_store.ingest_text(content, source_name)

        # Index in vector store
        chunks = self.file_store.get_chunks(source_id)
        for chunk in chunks:
            if self.embed_fn:
                emb = self.embed_fn([chunk.text])
                if emb is not None and len(emb) > 0:
                    self.retriever.vector_index.add(chunk.chunk_id, emb[0])

        # Extract and index entities
        if self.entity_index:
            self.entity_index.extract_and_index(chunks)

        return source_id

    def upload_file(self, file_path: Path) -> str:
        """Upload a file from disk and index it. Returns source_id."""
        source_id = self.file_store.ingest_file(file_path)

        # Index in vector store
        chunks = self.file_store.get_chunks(source_id)
        for chunk in chunks:
            if self.embed_fn:
                emb = self.embed_fn([chunk.text])
                if emb is not None and len(emb) > 0:
                    self.retriever.vector_index.add(chunk.chunk_id, emb[0])

        # Extract and index entities
        if self.entity_index:
            self.entity_index.extract_and_index(chunks)

        return source_id

    def process_query(
        self,
        query: str,
        request_id: str | None = None,
        model: str = "",
    ) -> dict[str, Any]:
        """Execute the full Agent-OS pipeline for a user query.

        Args:
            query: The user's natural language query.
            request_id: Optional request ID (auto-generated if not provided).
            model: Optional model override (e.g. "deepseek-chat").

        Returns:
            Dict with keys: response, trace_id, verified, context_pack_id.
        """
        if request_id is None:
            request_id = f"req_{uuid.uuid4().hex[:12]}"

        # 1. Start trace
        trace = self.trace_logger.start_trace(request_id)

        try:
            # Record user message in conversation cache
            if self.conversation_cache:
                self.conversation_cache.add_user_message(query)

            # 2. Retrieve relevant chunks
            retrieval_results = self._step_retrieve(query, trace)

            # 3. Assemble context pack (with conversation history)
            context_pack = self._step_assemble(query, retrieval_results, trace)

            # 4. LLM reasoning
            response = self._step_reason(context_pack, query, trace, model)

            # Record agent response in conversation cache
            if self.conversation_cache:
                self.conversation_cache.add_agent_response(response)

            # 5. Verify the response
            verify_result = self._step_verify(response, context_pack, trace)

            # 6. Evaluate memory writeback
            self._step_writeback(query, response, verify_result, trace)

            return {
                "response": response,
                "trace_id": trace.trace_id,
                "verified": verify_result.is_verified,
                "context_pack_id": context_pack.context_id,
                "unverified_claims": verify_result.unverified_claims,
            }
        except Exception as e:
            self.trace_logger.add_step(trace.trace_id, TraceStep(
                step_id=f"step_error",
                type=StepType.RESPOND,
                status=StepStatus.FAILED,
                error=str(e),
            ))
            return {
                "response": f"Error processing query: {e}",
                "trace_id": trace.trace_id,
                "verified": False,
                "context_pack_id": "",
                "unverified_claims": [],
            }

    def process_query_with_page_fault(
        self,
        query: str,
        request_id: str | None = None,
        model: str = "",
    ) -> dict[str, Any]:
        """Process query with automatic ContextPageFault retry.

        If the LLM response signals missing context, triggers page fault
        retrieval and re-runs reasoning up to 2 times. Falls back to
        process_query() if no page_fault is configured.
        """
        if self.page_fault is None:
            return self.process_query(query, request_id, model)

        if request_id is None:
            request_id = f"req_{uuid.uuid4().hex[:12]}"

        trace = self.trace_logger.start_trace(request_id)

        try:
            # First attempt: normal pipeline
            retrieval_results = self._step_retrieve(query, trace)
            context_pack = self._step_assemble(query, retrieval_results, trace)
            response = self._step_reason(context_pack, query, trace, model)

            # Page fault loop
            self.page_fault.reset()
            for _ in range(3):
                pf_result = self.page_fault.check_and_handle(
                    response=response,
                    context_pack=context_pack,
                    original_query=query,
                    embed_fn=self.embed_fn,
                )
                if not pf_result.triggered:
                    break
                # Use updated context pack from page fault
                if pf_result.updated_pack:
                    context_pack = pf_result.updated_pack
                response = self._step_reason(context_pack, query, trace, model)

            # Verify
            verify_result = self._step_verify(response, context_pack, trace)

            # Writeback
            self._step_writeback(query, response, verify_result, trace)

            return {
                "response": response,
                "trace_id": trace.trace_id,
                "verified": verify_result.is_verified,
                "context_pack_id": context_pack.context_id,
                "unverified_claims": verify_result.unverified_claims,
            }
        except Exception as e:
            self.trace_logger.add_step(trace.trace_id, TraceStep(
                step_id="step_error",
                type=StepType.RESPOND,
                status=StepStatus.FAILED,
                error=str(e),
            ))
            return {
                "response": f"Error: {e}",
                "trace_id": trace.trace_id,
                "verified": False,
                "context_pack_id": "",
                "unverified_claims": [],
            }

    def get_trace(self, trace_id: str):
        """Retrieve a trace by ID."""
        return self.trace_logger.get_trace(trace_id)

    # ── Pipeline Steps ────────────────────────────────────────────

    def _step_retrieve(self, query: str, trace) -> list[RetrievalResult]:
        results = self.retriever.retrieve_and_rerank(query, self.embed_fn, k=self.config.top_k_after_rerank)
        self.trace_logger.add_step(trace.trace_id, TraceStep(
            step_id="step_retrieve",
            type=StepType.RETRIEVE_FILE,
            input={"query": query},
            output={"num_results": len(results), "top_scores": [r.score for r in results[:3]]},
        ))
        return results

    def _step_assemble(self, query: str, results, trace):
        working_mems = self.memory_store.list_active()
        conv_history = (
            self.conversation_cache.get_recent_turns(10)
            if self.conversation_cache else None
        )
        context_pack = self.mmu.assemble(
            query=query,
            retrieval_results=results,
            working_memories=working_mems,
            conversation_history=conv_history,
            task_id="",
            agent_id=self.agent_id,
        )
        self.trace_logger.add_step(trace.trace_id, TraceStep(
            step_id="step_assemble",
            type=StepType.CONTEXT_ASSEMBLE,
            input={"num_results": len(results)},
            output={
                "context_id": context_pack.context_id,
                "used_tokens": context_pack.used_tokens,
                "num_sections": len(context_pack.sections),
                "source_refs": context_pack.source_refs,
            },
        ))
        return context_pack

    def _step_reason(self, context_pack, query: str, trace, model: str = "") -> str:
        # Pass model override; llm_fn handles it if supported, ignores if not
        try:
            response = self.llm_fn(context_pack, query, model_override=model)
        except TypeError:
            response = self.llm_fn(context_pack, query)
        self.trace_logger.add_step(trace.trace_id, TraceStep(
            step_id="step_reason",
            type=StepType.LLM_REASONING,
            input={"query": query, "context_id": context_pack.context_id, "model": model},
            output={"response_length": len(response), "response_preview": response[:200]},
        ))
        return response

    def _step_verify(self, response: str, context_pack, trace) -> VerifyOutput:
        working_mems = self.memory_store.list_active()
        result = self.verifier.verify(response, context_pack, working_mems)
        self.trace_logger.add_step(trace.trace_id, TraceStep(
            step_id="step_verify",
            type=StepType.VERIFY,
            input={"response_length": len(response)},
            output={
                "is_verified": result.is_verified,
                "num_unverified": len(result.unverified_claims),
                "num_conflicts": len(result.conflicting_pairs),
                "num_suggestions": len(result.suggestions),
            },
            status=StepStatus.SUCCESS if result.is_verified else StepStatus.FAILED,
        ))
        return result

    def _step_writeback(self, query: str, response: str, verify_result, trace) -> None:
        # Only consider writeback if verification passed or has minor issues
        if verify_result.is_verified or len(verify_result.unverified_claims) <= 2:
            # Evaluate: should we write the query+response context to memory?
            content = f"Q: {query}\nA: {response[:300]}"
            decision = self.writeback_gate.evaluate(
                content=content,
                source="conversation",
                importance=0.6,
                confidence=0.8 if verify_result.is_verified else 0.5,
            )

            if decision.action == "write":
                item = MemoryItem(
                    memory_id=f"mem_{uuid.uuid4().hex[:12]}",
                    type=MemoryType.CONVERSATION_SUMMARY,
                    content=content,
                    summary=response[:100],
                    importance=0.6,
                    confidence=0.8,
                )
                self.memory_store.insert(item)

            self.trace_logger.add_step(trace.trace_id, TraceStep(
                step_id="step_writeback",
                type=StepType.WRITE_MEMORY,
                input={"write_score": decision.score},
                output={"action": decision.action, "location": decision.location, "reason": decision.reason},
                status=StepStatus.SUCCESS if decision.action != "skip" else StepStatus.SKIPPED,
            ))

    @staticmethod
    def _default_llm(context_pack, query: str) -> str:
        """Default LLM: extract answer from retrieved evidence without external API."""
        for section in context_pack.sections:
            if section.name == "retrieved_evidence" and section.items:
                parts = []
                for item in section.items[:3]:
                    src = item.get("source_ref", "unknown")
                    text = item.get("text", "")
                    if text:
                        parts.append(f"According to {src}: {text}")
                if parts:
                    return " ".join(parts)
        return f"I don't have enough information to answer: {query}"
