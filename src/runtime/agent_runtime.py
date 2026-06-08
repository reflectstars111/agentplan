"""AgentRuntime — end-to-end pipeline orchestration.

Wires together all MVP components into a single execution loop:
  Input → Retrieve → Context MMU → LLM → Verify → Writeback → Trace → Output

Maps to agent_os_initial_plan.md §18.4 (MVP flow).
"""

import uuid
import hashlib
from pathlib import Path
from typing import Callable, Any, Optional
from src.config import Config
from src.db import Database
from src.storage.file_store import FileStore
from src.storage.memory_store import MemoryStore
from src.index.hybrid_retriever import HybridRetriever, RetrievalResult
from src.index.memory_retriever import MemoryRetriever
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
        permission_checker=None,
        memory_retriever=None,
        input_sanitizer=None,
        audit_log=None,
        tool_router=None,
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
        self.permission_checker = permission_checker
        self.memory_retriever = memory_retriever or MemoryRetriever(
            memory_store,
            retriever.keyword_index,
        )
        self.input_sanitizer = input_sanitizer
        self.audit_log = audit_log
        self.tool_router = tool_router

        # Build a default AgentProcess for permission checks
        if permission_checker:
            from src.models.agent import AgentProcess, AgentRole, AgentStatus
            self._agent_process = AgentProcess(
                agent_id=self.agent_id,
                role=AgentRole(self.role) if self.role in [r.value for r in AgentRole] else AgentRole.WORKER,
                status=AgentStatus.READY,
                available_tools=[],
                memory_scope=self.memory_scope,
            )
        else:
            self._agent_process = None

    def upload_text(self, content: str, source_name: str) -> str:
        """Upload text content and index it. Returns source_id."""
        source_id = self.file_store.ingest_text(content, source_name)
        self.index_source(source_id)

        # Extract dependency graph for code files
        if self.dependency_graph:
            ext = source_name.rsplit('.', 1)[-1] if '.' in source_name else ''
            if ext in ('py', 'js', 'ts', 'jsx', 'tsx'):
                try:
                    lang = {'py': 'python', 'js': 'javascript', 'jsx': 'javascript',
                            'ts': 'typescript', 'tsx': 'typescript'}.get(ext, 'python')
                    from src.parsing.code_parser import CodeParser
                    parser = CodeParser(lang)
                    symbols = parser.extract_symbols(content, source_id)
                    if symbols:
                        self.dependency_graph.extract_from_symbols(symbols, source_id)
                except Exception:
                    pass

        return source_id

    def upload_file(self, file_path: Path) -> str:
        """Upload a file from disk and index it. Returns source_id."""
        source_id = self.file_store.ingest_file(file_path)
        self.index_source(source_id)
        return source_id

    def index_source(self, source_id: str) -> int:
        """Update all secondary indexes for a persisted source."""
        chunks = self.file_store.get_chunks(source_id)
        for chunk in chunks:
            if self.embed_fn:
                emb = self.embed_fn([chunk.text])
                if emb is not None and len(emb) > 0:
                    self.retriever.vector_index.add(chunk.chunk_id, emb[0])
        self.retriever.vector_index.persist()

        # Extract and index entities
        if self.entity_index:
            self.entity_index.extract_and_index(chunks)
        self._index_structure(source_id, chunks)
        return len(chunks)

    def _index_structure(self, source_id: str, chunks: list) -> None:
        structure_index = getattr(self.retriever, "structure_index", None)
        if structure_index is None:
            return

        from src.models.structure_node import StructureNode

        source_hash = hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:12]
        root_id = f"structure_{source_hash}"
        nodes = [
            StructureNode(
                node_id=root_id,
                source_id=source_id,
                node_type="file",
                name=source_id.split(":", 1)[-1],
                chunk_ids=[chunk.chunk_id for chunk in chunks],
            )
        ]
        for index, chunk in enumerate(chunks):
            location = chunk.location
            label = (
                location.section
                or (f"page {location.page}" if location.page is not None else "")
                or f"chunk {index + 1}"
            )
            nodes.append(
                StructureNode(
                    node_id=f"{root_id}_{index}",
                    source_id=source_id,
                    node_type=chunk.chunk_type.value,
                    name=label,
                    parent_id=root_id,
                    depth=1,
                    location_page=location.page,
                    location_section=location.section,
                    location_line_start=location.line_start,
                    location_line_end=location.line_end,
                    chunk_ids=[chunk.chunk_id],
                )
            )
        structure_index.delete_source(source_id)
        structure_index.index_nodes(nodes)

    def process_query(
        self,
        query: str,
        request_id: str | None = None,
        model: str = "",
        parent_trace_id: str | None = None,
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

        # Drive agent state machine
        if self._agent_process:
            from src.models.agent import AgentStatus
            self._agent_process.transition(AgentStatus.RUNNING)

        # 1. Start trace
        trace = self.trace_logger.start_trace(request_id, parent_trace_id)
        security = self.sanitize_input(query)
        query = security["sanitized_text"]
        self._trace_security(trace.trace_id, security)

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
            writeback = self._step_writeback(
                query, response, verify_result, trace
            )

            if self._agent_process:
                self._agent_process.transition(AgentStatus.COMPLETED)

            return {
                "response": response,
                "trace_id": trace.trace_id,
                "verified": verify_result.is_verified,
                "context_pack_id": context_pack.context_id,
                "unverified_claims": verify_result.unverified_claims,
                "conflicting_pairs": verify_result.conflicting_pairs,
                "suggestions": verify_result.suggestions,
                "writeback": writeback,
                "writeback_confirmation_required": (
                    writeback["action"] == "ask_user"
                ),
                "security": security,
            }
        except Exception as e:
            if self._agent_process:
                self._agent_process.transition(AgentStatus.FAILED)
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
                "conflicting_pairs": [],
                "suggestions": [],
                "writeback": self._skipped_writeback(str(e)),
                "writeback_confirmation_required": False,
                "security": security,
            }

    def process_query_with_page_fault(
        self,
        query: str,
        request_id: str | None = None,
        model: str = "",
        parent_trace_id: str | None = None,
    ) -> dict[str, Any]:
        """Process query with automatic ContextPageFault retry.

        If the LLM response signals missing context, triggers page fault
        retrieval and re-runs reasoning up to 2 times. Falls back to
        process_query() if no page_fault is configured.
        """
        if self.page_fault is None:
            return self.process_query(
                query, request_id, model, parent_trace_id
            )

        if request_id is None:
            request_id = f"req_{uuid.uuid4().hex[:12]}"

        trace = self.trace_logger.start_trace(request_id, parent_trace_id)
        security = self.sanitize_input(query)
        query = security["sanitized_text"]
        self._trace_security(trace.trace_id, security)

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
            writeback = self._step_writeback(
                query, response, verify_result, trace
            )

            return {
                "response": response,
                "trace_id": trace.trace_id,
                "verified": verify_result.is_verified,
                "context_pack_id": context_pack.context_id,
                "unverified_claims": verify_result.unverified_claims,
                "conflicting_pairs": verify_result.conflicting_pairs,
                "suggestions": verify_result.suggestions,
                "writeback": writeback,
                "writeback_confirmation_required": (
                    writeback["action"] == "ask_user"
                ),
                "security": security,
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
                "conflicting_pairs": [],
                "suggestions": [],
                "writeback": self._skipped_writeback(str(e)),
                "writeback_confirmation_required": False,
                "security": security,
            }

    def sanitize_input(self, text: str) -> dict:
        """Return a normalized security scan for a user-controlled string."""
        if self.input_sanitizer is None:
            return {
                "clean": True,
                "risk_level": "low",
                "matched_patterns": [],
                "sanitized_text": text,
            }
        return self.input_sanitizer.scan(text)

    def _trace_security(self, trace_id: str, security: dict) -> None:
        self.trace_logger.add_step(trace_id, TraceStep(
            step_id="step_security",
            type=StepType.SECURITY,
            input={
                "risk_level": security["risk_level"],
                "matched_count": len(security["matched_patterns"]),
            },
            output={"clean": security["clean"]},
            status=(
                StepStatus.SUCCESS
                if security["clean"]
                else StepStatus.FAILED
            ),
        ))

    def execute_tool(self, name: str, params: dict, trace_id: str = ""):
        """Execute a registered tool through the permission-aware router."""
        if self.tool_router is None or self._agent_process is None:
            raise RuntimeError("Tool routing is not configured")
        return self.tool_router.execute(
            name,
            params,
            self._agent_process,
            trace_id=trace_id,
        )

    def get_trace(self, trace_id: str):
        """Retrieve a trace by ID."""
        return self.trace_logger.get_trace(trace_id)

    def close(self) -> None:
        """Flush persistent indexes and close the shared database."""
        self.retriever.vector_index.persist()
        self.file_store.db.close()

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
        # Check memory read permission
        working_mems = []
        long_term_mems = []
        if self.permission_checker and self._agent_process:
            can_read = self.permission_checker.verify_permissions(
                self._agent_process, "memory_read", scope="working_memory"
            )
            if can_read.get("allowed", True):
                selection = self.memory_retriever.retrieve(
                    query,
                    scopes=self.memory_scope.get("read_memory") or None,
                )
                working_mems = selection.working
                long_term_mems = selection.long_term
        else:
            selection = self.memory_retriever.retrieve(
                query,
                scopes=self.memory_scope.get("read_memory") or None,
            )
            working_mems = selection.working
            long_term_mems = selection.long_term

        selected_memories = [*working_mems, *long_term_mems]
        self.trace_logger.add_step(trace.trace_id, TraceStep(
            step_id="step_retrieve_memory",
            type=StepType.RETRIEVE_MEMORY,
            input={"query": query},
            output={
                "num_results": len(selected_memories),
                "memory_ids": [item.memory_id for item in selected_memories],
            },
        ))

        conv_history = (
            self.conversation_cache.get_recent_turns(10)
            if self.conversation_cache else None
        )
        context_pack = self.mmu.assemble(
            query=query,
            retrieval_results=results,
            working_memories=working_mems,
            long_term_memories=long_term_mems,
            conversation_history=conv_history,
            task_id="",
            agent_id=self.agent_id,
        )
        context_pack.memory_ids = [
            item.memory_id for item in selected_memories
        ]
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
        working_mems = [
            item
            for memory_id in context_pack.memory_ids
            if (item := self.memory_store.get(memory_id)) is not None
        ]
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

    def _step_writeback(
        self, query: str, response: str, verify_result, trace
    ) -> dict:
        # Only consider writeback if verification passed or has minor issues
        decision = WritebackDecision(
            action="skip",
            location="none",
            reason="Verification blocked memory writeback",
            score=0.0,
        )
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
                # Check memory write permission
                can_write = True
                if self.permission_checker and self._agent_process:
                    result_p = self.permission_checker.verify_permissions(
                        self._agent_process, "memory_write", scope="working_memory"
                    )
                    can_write = result_p.get("allowed", True)

                if can_write:
                    item = MemoryItem(
                        memory_id=f"mem_{uuid.uuid4().hex[:12]}",
                        type=(
                            MemoryType.DECISION
                            if decision.location == "long_term_memory"
                            else MemoryType.CONVERSATION_SUMMARY
                        ),
                        content=content,
                        summary=response[:100],
                        importance=0.6,
                        confidence=0.8,
                        scope=(
                            "project"
                            if decision.location == "long_term_memory"
                            else "session"
                        ),
                    )
                    self.memory_store.insert(item)

        self.trace_logger.add_step(trace.trace_id, TraceStep(
            step_id="step_writeback",
            type=StepType.WRITE_MEMORY,
            input={"write_score": decision.score},
            output={
                "action": decision.action,
                "location": decision.location,
                "reason": decision.reason,
            },
            status=(
                StepStatus.SUCCESS
                if decision.action != "skip"
                else StepStatus.SKIPPED
            ),
        ))
        return {
            "action": decision.action,
            "location": decision.location,
            "reason": decision.reason,
            "score": decision.score,
        }

    @staticmethod
    def _skipped_writeback(reason: str) -> dict:
        return {
            "action": "skip",
            "location": "none",
            "reason": reason,
            "score": 0.0,
        }

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
