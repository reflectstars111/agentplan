"""Controller — ARD core execution loop.

Orchestrates the full cycle:
  FETCH → PLAN → LOAD → REASON → VERIFY → WRITE → OUTPUT

Maps to ARD design §8 (Controller core cycle).
"""

import uuid

from ard.infra.logging import log
from ard.store import RetrievalResult
from ard.store.state_store import StateStore
from ard.store.trace_store import TraceStore, TraceHandle
from ard.store.transaction import TransactionManager
from ard.retriever.hybrid import HybridRetriever
from ard.context.mmu import ContextMMU
from ard.runtime.executor import Executor, ExecutorResponse
from ard.runtime.verifier import Verifier, Verdict


class Controller:
    """Orchestrates the ARD execution cycle.

    Process flow:
        1. FETCH    — receive request
        2. PLAN     — decompose into tasks
        3. LOAD     — assemble context via ContextMMU
        4. REASON   — call LLM via Executor
        5. VERIFY   — check sources and conflicts via Verifier
        6. WRITE    — transactional state writeback via TransactionManager
        7. OUTPUT   — return response to user
    """

    def __init__(
        self,
        state_store: StateStore,
        trace_store: TraceStore,
        transaction_manager: TransactionManager,
        hybrid_retriever: HybridRetriever,
        context_mmu: ContextMMU,
        executor: Executor,
        verifier: Verifier | None = None,
    ):
        self.state_store = state_store
        self.trace_store = trace_store
        self.txn_mgr = transaction_manager
        self.hybrid = hybrid_retriever
        self.mmu = context_mmu
        self.executor = executor
        self.verifier = verifier or Verifier()

    def process(self, query: str, request_id: str = "",
                model: str = "") -> dict:
        """Execute the full ARD control loop.

        Args:
            query: User's natural language request.
            request_id: Optional request ID.
            model: Optional LLM model override.

        Returns:
            Dict with: response, trace_id, verdict, writeback, state_keys
        """
        if not request_id:
            request_id = f"req_{uuid.uuid4().hex[:12]}"

        # Start trace
        trace = self.trace_store.start_trace(request_id)

        # ── 1. FETCH ──
        log.info("controller_fetch", request_id=request_id,
                 query_len=len(query))

        # ── 2. PLAN ── (Phase 2: simple pass-through plan)
        plan = {"intent": "general", "query": query}
        self.trace_store.add_step(trace.trace_id, "plan",
                                  input_data={"query": query},
                                  output_data=plan)

        # ── 3. LOAD ──
        candidates = self.hybrid.retrieve(query)
        self.trace_store.add_step(trace.trace_id, "retrieve",
                                  input_data={"query": query},
                                  output_data={"candidates": len(candidates)})

        # Load relevant state
        task_state = self._load_working_state()
        context_pack = self.mmu.assemble(
            query=query,
            retrieval_results=candidates,
            system_instruction=self._default_system_prompt(),
            top_k=15,
        )
        self.trace_store.add_step(trace.trace_id, "context_assemble",
                                  input_data={"query": query},
                                  output_data={"tokens": context_pack.total_tokens_used(),
                                              "sections": len(context_pack.sections),
                                              "sources": context_pack.source_refs})

        # ── 4. REASON ──
        response: ExecutorResponse = self.executor.think(context_pack)
        self.trace_store.add_step(trace.trace_id, "execute",
                                  input_data={"tokens": context_pack.total_tokens_used()},
                                  output_data={"answer_len": len(response.answer)})

        # ── 5. VERIFY ──
        verdict: Verdict = self.verifier.verify(
            response=response.answer,
            context_pack=context_pack,
            state_store=self.state_store,
        )
        self.trace_store.add_step(trace.trace_id, "verify",
                                  input_data={"answer_len": len(response.answer)},
                                  output_data={"verified": verdict.verified,
                                              "confidence": verdict.confidence,
                                              "conflicts": len(verdict.conflicts),
                                              "orphan_claims": len(verdict.orphan_claims)})

        # ── 6. WRITE ──
        writeback_info = self._writeback(query, response, verdict, candidates,
                                         context_pack, trace)
        # ── 7. OUTPUT ──
        self.trace_store.add_step(trace.trace_id, "respond",
                                  input_data={},
                                  output_data={"status": "success"})

        return {
            "response": response.answer,
            "trace_id": trace.trace_id,
            "verdict": {
                "verified": verdict.verified,
                "confidence": verdict.confidence,
                "conflicts": verdict.conflicts,
                "orphan_claims": verdict.orphan_claims,
            },
            "writeback": writeback_info,
            "sources": context_pack.source_refs,
            "tokens_used": context_pack.total_tokens_used(),
            "state_keys": self.state_store.list_keys(),
        }

    # ── Internal ────────────────────────────────────────────

    def _load_working_state(self) -> dict:
        """Load current working state (L2)."""
        keys = self.state_store.list_keys("task:")
        state = {}
        for key in keys:
            value = self.state_store.read(key)
            if value:
                state[key] = value
        return state

    def _writeback(self, query: str, response: ExecutorResponse,
                   verdict: Verdict, candidates: list[RetrievalResult],
                   context_pack, trace: TraceHandle) -> dict:
        """Write back state changes through TransactionManager."""
        txn = self.txn_mgr.begin()

        try:
            # Write task state
            task_key = f"task:{trace.trace_id}"
            task_event = self.state_store.build_event(
                stream_key=task_key,
                event_type="created" if not self.state_store.read(task_key) else "updated",
                payload={
                    "query": query,
                    "response_summary": response.answer[:500],
                    "verdict_confidence": verdict.confidence,
                    "source_count": len(candidates),
                    "tokens_used": context_pack.total_tokens_used(),
                },
            )
            txn.add_event(task_event)

            # Write long-term memory if verdict is confident
            if verdict.verified and verdict.confidence > 0.7:
                mem_key = f"memory:mem_{uuid.uuid4().hex[:8]}"
                mem_event = self.state_store.build_event(
                    stream_key=mem_key,
                    event_type="created",
                    payload={
                        "content": response.answer[:1000],
                        "type": "project_state",
                        "importance": verdict.confidence,
                        "confidence": verdict.confidence,
                        "source": "conversation",
                        "entities": [],
                    },
                )
                txn.add_event(mem_event)

            # Commit
            seqs = self.txn_mgr.commit(txn)
            self.trace_store.add_step(trace.trace_id, "writeback",
                                      input_data={"verdict_confidence": verdict.confidence},
                                      output_data={"committed": True, "seq_nums": seqs,
                                                  "task_key": task_key,
                                                  "memory_persisted": verdict.confidence > 0.7})
            return {"action": "committed", "seq_nums": seqs, "task_key": task_key}

        except RuntimeError as e:
            self.txn_mgr.rollback(txn)
            self.trace_store.add_step(trace.trace_id, "writeback",
                                      input_data={"error": str(e)},
                                      output_data={"committed": False, "rolled_back": True})
            return {"action": "rolled_back", "error": str(e)}

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are a research assistant with access to a knowledge base. "
            "Answer questions based ONLY on the provided context. "
            "If the context does not contain enough information to answer, "
            "state that clearly rather than guessing. "
            "When possible, reference the source of information."
        )
