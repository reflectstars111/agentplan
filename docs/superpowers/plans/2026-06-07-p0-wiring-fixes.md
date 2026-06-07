# P0 Wiring Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire 4 already-implemented modules into the main execution loop: ContextPageFault in Scheduler, last_used_at DB column, Reranker in HybridRetriever, and full trace recording in Controller.

**Architecture:** Four independent fixes touching `scheduler.py`, `agent_runtime.py`, `migrations.py`, `memory_store.py`, `hybrid_retriever.py`, `trace.py`, and `controller.py`. Each fix adds a backward-compatible optional parameter and a new code path gated on its presence. No API changes.

**Tech Stack:** Python 3.12+, SQLite, FAISS, pytest

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/db/migrations.py:1-10` | Modify | MEMORIES_TABLE DDL + `last_used_at` |
| `src/storage/memory_store.py` | Modify | insert/read/touch for `last_used_at` |
| `src/models/trace.py:9-22` | Modify | StepType enum +PLAN, +SCHEDULE |
| `src/index/hybrid_retriever.py` | Modify | +reranker param, +retrieve_and_rerank() |
| `src/runtime/scheduler.py` | Modify | +page_fault, +trace_logger opts; retry+logging in _execute_one |
| `src/runtime/agent_runtime.py` | Modify | _step_reason supports page_fault retry context |
| `src/runtime/controller.py` | Modify | +plan/schedule trace steps |
| `tests/test_memory_store.py` | Modify | test last_used_at round-trip |
| `tests/test_hybrid_retriever.py` | Modify | test retrieve_and_rerank |
| `tests/test_controller.py` | Modify | verify ≥4 trace steps |
| `tests/test_scheduler_page_fault.py` | Create | test page fault retry loop |

---

### Task 1: DB Schema — Add `last_used_at` to memories table

**Files:**
- Modify: `src/db/migrations.py:1-19`
- Modify: `src/storage/memory_store.py`
- Modify: `tests/test_memory_store.py`

**Background:** `MemoryItem` model has `last_used_at` field but the DB table lacks the column. Insert silently drops it.

- [ ] **Step 1: Add column to DDL**

In `src/db/migrations.py`, change `MEMORIES_TABLE` to include `last_used_at TEXT` after the `updated_at TEXT NOT NULL` line, and before the closing `);`:

```python
MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT DEFAULT '',
    entities TEXT DEFAULT '[]',
    importance REAL DEFAULT 0.5,
    confidence REAL DEFAULT 0.5,
    source TEXT DEFAULT 'conversation',
    scope TEXT DEFAULT 'project',
    status TEXT DEFAULT 'active',
    version INTEGER DEFAULT 1,
    source_ref TEXT,
    last_used_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""
```

- [ ] **Step 2: Update MemoryStore.insert() to write last_used_at**

In `src/storage/memory_store.py`, change the `insert()` SQL to include `last_used_at`:

```python
def insert(self, item: MemoryItem) -> None:
    """Insert or replace a memory item."""
    now = datetime.now(timezone.utc).isoformat()
    last_used = (
        item.last_used_at.isoformat()
        if item.last_used_at and hasattr(item.last_used_at, 'isoformat')
        else None
    )
    sql = """
    INSERT OR REPLACE INTO memories
        (memory_id, type, content, summary, entities, importance, confidence,
         source, scope, status, version, source_ref, last_used_at, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    self.db.execute(sql, (
        item.memory_id,
        item.type.value,
        item.content,
        item.summary,
        json.dumps(item.entities),
        item.importance,
        item.confidence,
        item.source,
        item.scope,
        item.status.value,
        item.version,
        item.source_ref,
        last_used,
        item.created_at.isoformat() if hasattr(item.created_at, 'isoformat') else str(item.created_at),
        now,
    ))
    self.db.commit()
```

- [ ] **Step 3: Update _row_to_item() to read last_used_at**

```python
def _row_to_item(self, row: dict) -> MemoryItem:
    """Convert a database row dict to a MemoryItem."""
    last_used = None
    if row.get("last_used_at"):
        try:
            last_used = datetime.fromisoformat(row["last_used_at"])
        except (ValueError, TypeError):
            pass

    return MemoryItem(
        memory_id=row["memory_id"],
        type=MemoryType(row["type"]),
        content=row["content"],
        summary=row.get("summary") or "",
        entities=json.loads(row.get("entities", "[]")) if row.get("entities") else [],
        importance=row.get("importance") or 0.5,
        confidence=row.get("confidence") or 0.5,
        source=row.get("source", "conversation"),
        scope=row.get("scope", "project"),
        status=MemoryStatus(row["status"]) if row.get("status") else MemoryStatus.ACTIVE,
        version=row.get("version", 1),
        source_ref=row.get("source_ref"),
        last_used_at=last_used,
        created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else datetime.now(timezone.utc),
        updated_at=datetime.fromisoformat(row["updated_at"]) if row.get("updated_at") else datetime.now(timezone.utc),
    )
```

- [ ] **Step 4: Add touch() method to MemoryStore**

```python
def touch(self, memory_id: str) -> None:
    """Update last_used_at to now for a memory record."""
    now = datetime.now(timezone.utc).isoformat()
    self.db.execute(
        "UPDATE memories SET last_used_at = ? WHERE memory_id = ?",
        (now, memory_id),
    )
    self.db.commit()
```

- [ ] **Step 5: Add test for last_used_at round-trip**

In `tests/test_memory_store.py`, add:

```python
def test_last_used_at_round_trip(memory_store):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    item = MemoryItem(
        memory_id="mem_last_used_test",
        type=MemoryType.PROJECT_STATE,
        content="test last_used_at",
        last_used_at=now,
    )
    memory_store.insert(item)

    retrieved = memory_store.get("mem_last_used_test")
    assert retrieved is not None
    assert retrieved.last_used_at is not None
    # Timestamps should be within 5 seconds
    diff = abs((retrieved.last_used_at - now).total_seconds())
    assert diff < 5


def test_touch_updates_last_used_at(memory_store):
    from datetime import datetime, timezone
    item = MemoryItem(
        memory_id="mem_touch_test",
        type=MemoryType.PROJECT_STATE,
        content="test touch",
    )
    memory_store.insert(item)

    memory_store.touch("mem_touch_test")
    retrieved = memory_store.get("mem_touch_test")
    assert retrieved is not None
    assert retrieved.last_used_at is not None
```

- [ ] **Step 6: Run tests**

```bash
cd f:/agentplan && python -m pytest tests/test_memory_store.py::test_last_used_at_round_trip tests/test_memory_store.py::test_touch_updates_last_used_at -v
```
Expected: 2 PASS

- [ ] **Step 7: Commit**

```bash
cd f:/agentplan && git add src/db/migrations.py src/storage/memory_store.py tests/test_memory_store.py && git commit -m "fix: add last_used_at column to memories table, add touch()"
```

---

### Task 2: StepType Enum — Add PLAN and SCHEDULE

**Files:**
- Modify: `src/models/trace.py:9-22`

- [ ] **Step 1: Add PLAN and SCHEDULE to StepType**

In `src/models/trace.py`, add two new members to the `StepType` enum:

```python
class StepType(str, Enum):
    INTENT_DECODE = "intent_decode"
    PLAN = "plan"                         # ← new
    SCHEDULE = "schedule"                 # ← new
    RETRIEVE_MEMORY = "retrieve_memory"
    RETRIEVE_FILE = "retrieve_file"
    CONTEXT_ASSEMBLE = "context_assemble"
    LLM_REASONING = "llm_reasoning"
    TOOL_CALL = "tool_call"
    VERIFY = "verify"
    WRITE_MEMORY = "write_memory"
    RESPOND = "respond"
    SPAWN_AGENT = "spawn_agent"
    SEND_MESSAGE = "send_message"
    MERGE = "merge"
```

- [ ] **Step 2: Verify no import errors**

```bash
cd f:/agentplan && python -c "from src.models.trace import StepType; assert StepType.PLAN.value == 'plan'; assert StepType.SCHEDULE.value == 'schedule'; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd f:/agentplan && git add src/models/trace.py && git commit -m "feat: add PLAN and SCHEDULE to StepType enum"
```

---

### Task 3: HybridRetriever — Integrate Reranker

**Files:**
- Modify: `src/index/hybrid_retriever.py`
- Modify: `src/runtime/agent_runtime.py:161-169`
- Modify: `tests/test_hybrid_retriever.py`

- [ ] **Step 1: Add reranker parameter and retrieve_and_rerank() to HybridRetriever**

In `src/index/hybrid_retriever.py`, modify `__init__` to accept optional `reranker`:

```python
class HybridRetriever:
    """Combined retriever that fuses vector and keyword search results."""

    def __init__(
        self,
        vector_index: VectorIndex,
        keyword_index: KeywordIndex,
        db: Database,
        config: Config | None = None,
        structure_index=None,
        entity_index=None,
        reranker=None,              # ← new: optional Reranker
    ):
        self.vector_index = vector_index
        self.keyword_index = keyword_index
        self.db = db
        self.config = config or Config()
        self.structure_index = structure_index
        self.entity_index = entity_index
        self.reranker = reranker    # ← new
```

Then add the new method `retrieve_and_rerank()` after the existing `retrieve()` method (after line 143):

```python
    def retrieve_and_rerank(
        self,
        query: str,
        embed_fn,
        k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve candidates then rerank for precision.

        Fetches 2*k candidates from retrieve(), then applies the reranker
        to boost genuinely relevant chunks and suppress noise. Falls back
        to plain retrieve() if no reranker is configured.
        """
        if self.reranker is None:
            return self.retrieve(query, embed_fn, k=k, filters=filters)

        # Fetch 2x candidates for the reranker to filter down
        candidates = self.retrieve(query, embed_fn, k=k * 2, filters=filters)
        if not candidates:
            return []

        return self.reranker.rerank(candidates, query, top_k=k)
```

- [ ] **Step 2: Wire AgentRuntime._step_retrieve() to use retrieve_and_rerank**

In `src/runtime/agent_runtime.py`, change `_step_retrieve()` (line 162) from:

```python
results = self.retriever.retrieve(query, self.embed_fn, k=self.config.top_k_after_rerank)
```

To:

```python
results = self.retriever.retrieve_and_rerank(query, self.embed_fn, k=self.config.top_k_after_rerank)
```

- [ ] **Step 3: Add test for retrieve_and_rerank**

In `tests/test_hybrid_retriever.py`, add:

```python
def test_retrieve_and_rerank_falls_back_without_reranker(hybrid_retriever):
    """retrieve_and_rerank() without a reranker should match retrieve()."""
    # Ensure no reranker
    hybrid_retriever.reranker = None
    results_a = hybrid_retriever.retrieve("test query", mock_embed_fn, k=5)
    results_b = hybrid_retriever.retrieve_and_rerank("test query", mock_embed_fn, k=5)
    assert len(results_a) == len(results_b)
    assert [r.chunk_id for r in results_a] == [r.chunk_id for r in results_b]


def test_retrieve_and_rerank_with_reranker(hybrid_retriever):
    """retrieve_and_rerank() with a reranker should return ≤k results."""
    from src.index.reranker import Reranker
    hybrid_retriever.reranker = Reranker()
    results = hybrid_retriever.retrieve_and_rerank("test query", mock_embed_fn, k=3)
    assert len(results) <= 3
```

- [ ] **Step 4: Run tests**

```bash
cd f:/agentplan && python -m pytest tests/test_hybrid_retriever.py::test_retrieve_and_rerank_falls_back_without_reranker tests/test_hybrid_retriever.py::test_retrieve_and_rerank_with_reranker -v
```
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
cd f:/agentplan && git add src/index/hybrid_retriever.py src/runtime/agent_runtime.py tests/test_hybrid_retriever.py && git commit -m "feat: integrate Reranker into HybridRetriever.retrieve_and_rerank()"
```

---

### Task 4: Controller — Full trace recording (Plan + Schedule steps)

**Files:**
- Modify: `src/runtime/controller.py`
- Modify: `tests/test_controller.py`

- [ ] **Step 1: Add Plan and Schedule trace steps in Controller.process()**

In `src/runtime/controller.py`, after the planner.plan() call (after line 82-83), add a Plan trace step:

```python
        # Phase 2: Plan
        task_graph = self.planner.plan(intent)

        self.trace_logger.add_step(trace.trace_id, TraceStep(
            step_id="step_plan",
            type=StepType.PLAN,
            input={"intent_type": intent.intent_type.value, "entities": intent.entities},
            output={
                "node_count": task_graph.node_count(),
                "nodes": list(task_graph.nodes.keys()),
            },
        ))
```

After the scheduler.execute() call (after line 85), add a Schedule trace step:

```python
        # Phase 3: Schedule + Execute
        exec_result = self.scheduler.execute(task_graph, request_id)

        self.trace_logger.add_step(trace.trace_id, TraceStep(
            step_id="step_schedule",
            type=StepType.SCHEDULE,
            input={"node_count": task_graph.node_count()},
            output={
                "completed": len(exec_result["results"]),
                "failed": len(exec_result.get("failed_tasks", [])),
                "status": exec_result["status"],
            },
        ))
```

- [ ] **Step 2: Add test verifying ≥4 trace steps**

In `tests/test_controller.py`, add:

```python
def test_controller_trace_has_plan_and_schedule_steps(controller):
    """Controller.process() should record intent_decode + plan + schedule + respond."""
    result = controller.process("explain what FastAPI is")
    trace = controller.trace_logger.get_trace(result["trace_id"])
    assert trace is not None

    step_types = [s.type.value for s in trace.steps]
    assert "intent_decode" in step_types
    assert "plan" in step_types
    assert "schedule" in step_types
    assert "respond" in step_types
    assert len(trace.steps) >= 4
```

- [ ] **Step 3: Run test**

```bash
cd f:/agentplan && python -m pytest tests/test_controller.py::test_controller_trace_has_plan_and_schedule_steps -v
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
cd f:/agentplan && git add src/runtime/controller.py tests/test_controller.py && git commit -m "feat: add Plan and Schedule trace steps to Controller"
```

---

### Task 5: Scheduler — Add trace_logger support

**Files:**
- Modify: `src/runtime/scheduler.py`
- Modify: `tests/test_scheduler.py` (verify trace steps recorded)

- [ ] **Step 1: Add optional trace_logger to Scheduler.__init__**

In `src/runtime/scheduler.py`, change the `__init__` signature (line 29-37) to:

```python
    def __init__(
        self,
        agent_runtime: AgentRuntime,
        agent_registry=None,
        blackboard=None,
        trace_logger=None,         # ← new: optional TraceLogger
    ):
        self.agent_runtime = agent_runtime
        self.agent_registry = agent_registry
        self.blackboard = blackboard
        self.trace_logger = trace_logger  # ← new
```

- [ ] **Step 2: Record trace step per task in _execute_one()**

In `src/runtime/scheduler.py`, inside `_execute_one()`, after `results[task.task_id] = result` (after line 129), add trace logging:

```python
                # Record trace step if trace_logger is configured
                if self.trace_logger:
                    from src.models.trace import TraceStep, StepType, StepStatus
                    self.trace_logger.add_step(task.trace_id or f"trace_{task.task_id}", TraceStep(
                        step_id=f"step_{task.task_id}",
                        type=StepType.LLM_REASONING,
                        input={"task_type": task.task_type, "query": task.input.get("query", task.input.get("task", ""))},
                        output={
                            "response_length": len(result.get("response", "")),
                            "verified": result.get("verified", False),
                        },
                        status=StepStatus.SUCCESS,
                    ))
```

- [ ] **Step 3: Run existing scheduler tests to verify no regression**

```bash
cd f:/agentplan && python -m pytest tests/test_scheduler.py -v
```
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
cd f:/agentplan && git add src/runtime/scheduler.py && git commit -m "feat: add optional trace_logger to Scheduler for per-task trace recording"
```

---

### Task 6: ContextPageFault — Wire into Scheduler with retry loop

**Files:**
- Modify: `src/runtime/scheduler.py`
- Modify: `src/runtime/agent_runtime.py`
- Create: `tests/test_scheduler_page_fault.py`

- [ ] **Step 1: Add optional page_fault to Scheduler.__init__**

In `src/runtime/scheduler.py`, extend `__init__`:

```python
    def __init__(
        self,
        agent_runtime: AgentRuntime,
        agent_registry=None,
        blackboard=None,
        trace_logger=None,
        page_fault=None,           # ← new: optional ContextPageFault
    ):
        self.agent_runtime = agent_runtime
        self.agent_registry = agent_registry
        self.blackboard = blackboard
        self.trace_logger = trace_logger
        self.page_fault = page_fault  # ← new
```

- [ ] **Step 2: Add page fault detection + retry in _execute_one()**

In `src/runtime/scheduler.py`, inside `_execute_one()`, after `results[task.task_id] = result` and the trace logging block, add:

```python
                # Page fault: if response indicates missing context, re-retrieve and retry
                if self.page_fault and result.get("context_pack_id"):
                    self.page_fault.reset()  # reset per-task fault counter
                    max_pf_retries = 2
                    pf_attempt = 0
                    response_text = result.get("response", "")
                    context_pack = None
                    # Try to get context_pack from runtime — we reconstruct it
                    while pf_attempt < max_pf_retries:
                        pf_result = self.page_fault.check_and_handle(
                            response=response_text,
                            context_pack=context_pack,  # will be None first time
                            original_query=query,
                            embed_fn=getattr(runtime, 'embed_fn', None),
                        )
                        if not pf_result.triggered:
                            break
                        # Re-run reasoning with updated context
                        if pf_result.updated_pack and hasattr(runtime, 'llm_fn'):
                            context_pack = pf_result.updated_pack
                            response_text = runtime.llm_fn(context_pack, query)
                            result["response"] = response_text
                            results[task.task_id] = result
                        pf_attempt += 1

                # Update trace step with final result
                if self.trace_logger and task.trace_id:
                    from src.models.trace import TraceStep, StepType, StepStatus
                    self.trace_logger.add_step(task.trace_id, TraceStep(
                        step_id=f"step_{task.task_id}",
                        type=StepType.LLM_REASONING,
                        input={"task_type": task.task_type, "query": task.input.get("query", task.input.get("task", ""))},
                        output={
                            "response_length": len(result.get("response", "")),
                            "verified": result.get("verified", False),
                        },
                        status=StepStatus.SUCCESS,
                    ))
```

Wait — that approach is complex because `context_pack` isn't available in Scheduler. Let me rethink. The Scheduler calls `runtime.process_query()` which returns `{"response", "context_pack_id", ...}`. But it doesn't return the full `ContextPack` object.

The cleaner approach: add a `process_query_with_page_fault()` method to `AgentRuntime` that wraps the existing pipeline with page fault retry. Then Scheduler calls that instead.

Let me revise the design:

**AgentRuntime** gets a new optional `page_fault` parameter + method:

```python
# In AgentRuntime.__init__, add:
self.page_fault = page_fault  # ContextPageFault | None

# New method:
def process_query_with_page_fault(self, query, request_id=None, model="") -> dict:
    """Process query with automatic page fault retry."""
    # First attempt
    result = self.process_query(query, request_id, model)
    
    if not self.page_fault:
        return result
    
    # Get context pack for page fault handling
    context_pack_id = result.get("context_pack_id", "")
    response = result.get("response", "")
    
    self.page_fault.reset()
    max_retries = 2
    
    for _ in range(max_retries):
        # Reconstruct context pack (simplified: pass None, let page_fault work
        # with what it can extract from the response text)
        pf_result = self.page_fault.check_and_handle(
            response=response,
            context_pack=None,  # page_fault._needs_more_context works on text
            original_query=query,
            embed_fn=self.embed_fn,
        )
        if not pf_result.triggered:
            break
        
        # Re-run with updated retrieval
        retrieval_results = self.retriever.retrieve(
            pf_result.query_used, self.embed_fn, k=self.config.top_k_after_rerank
        )
        context_pack = self.mmu.assemble(
            query=query,
            retrieval_results=retrieval_results,
            working_memories=self.memory_store.list_active(),
            task_id="",
            agent_id=self.agent_id,
        )
        response = self.llm_fn(context_pack, query)
        result["response"] = response
        result["context_pack_id"] = context_pack.context_id
    
    return result
```

Then **Scheduler._execute_one()** changes to use `process_query_with_page_fault` if page_fault is configured, otherwise `process_query`:

```python
# In _execute_one, change:
if self.page_fault and hasattr(runtime, 'process_query_with_page_fault'):
    result = runtime.process_query_with_page_fault(query, request_id=task.task_id, model=model if 'model' in dir() else "")
else:
    result = runtime.process_query(query, request_id=task.task_id)
```

This is cleaner — page fault logic lives in AgentRuntime where it has access to embed_fn, retriever, mmu, and llm_fn. Scheduler just chooses which method to call.

Let me rewrite Task 6 with this approach.

- [ ] **Step 1: Add page_fault to AgentRuntime.__init__**

In `src/runtime/agent_runtime.py`, add to the constructor parameters (after `memory_scope`):

```python
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
        page_fault=None,            # ← new: optional ContextPageFault
    ):
        # ... existing assignments ...
        self.page_fault = page_fault  # ← new
```

- [ ] **Step 2: Add process_query_with_page_fault() to AgentRuntime**

Add this method after `process_query()` (after line 153):

```python
    def process_query_with_page_fault(
        self,
        query: str,
        request_id: str | None = None,
        model: str = "",
    ) -> dict[str, Any]:
        """Process query with automatic ContextPageFault retry.

        If the LLM response signals missing context, triggers page fault
        retrieval and re-runs reasoning up to 2 times.
        Falls back to process_query() if no page_fault is configured.
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
                # Re-assemble context with new evidence
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
```

- [ ] **Step 3: Wire Scheduler to use process_query_with_page_fault**

In `src/runtime/scheduler.py`, add `page_fault` to `__init__`:

```python
    def __init__(
        self,
        agent_runtime: AgentRuntime,
        agent_registry=None,
        blackboard=None,
        trace_logger=None,
        page_fault=None,           # ← new
    ):
        self.agent_runtime = agent_runtime
        self.agent_registry = agent_registry
        self.blackboard = blackboard
        self.trace_logger = trace_logger
        self.page_fault = page_fault  # ← new
```

Then in `_execute_one()`, change the `runtime.process_query(...)` call (line 122-123) from:

```python
                result = runtime.process_query(
                    query, request_id=task.task_id
                )
```

To:

```python
                # Use page-fault-aware execution if configured
                if self.page_fault and hasattr(runtime, 'process_query_with_page_fault'):
                    result = runtime.process_query_with_page_fault(
                        query, request_id=task.task_id,
                    )
                else:
                    result = runtime.process_query(
                        query, request_id=task.task_id,
                    )
```

- [ ] **Step 4: Write the page fault test**

Create `tests/test_scheduler_page_fault.py`:

```python
"""Tests for ContextPageFault integration with Scheduler."""
import pytest
from src.context.page_fault import ContextPageFault
from src.runtime.agent_runtime import AgentRuntime


class TestPageFaultInAgentRuntime:
    """Verify process_query_with_page_fault retry loop."""

    def test_no_page_fault_falls_back_to_normal(self, agent_runtime):
        """Without page_fault configured, process_query_with_page_fault == process_query."""
        agent_runtime.page_fault = None
        result = agent_runtime.process_query_with_page_fault("what is Python?")
        assert "response" in result
        assert "trace_id" in result

    def test_page_fault_no_trigger_returns_first_response(self, agent_runtime, monkeypatch):
        """When response is confident, page fault should not trigger retry."""
        page_fault = ContextPageFault(
            retriever=agent_runtime.retriever,
            mmu=agent_runtime.mmu,
            max_faults=2,
        )
        agent_runtime.page_fault = page_fault

        # Mock LLM to return a confident response (no uncertainty markers)
        def mock_llm(ctx, query, model_override=""):
            return "Python is a high-level programming language created by Guido van Rossum."
        agent_runtime.llm_fn = mock_llm

        result = agent_runtime.process_query_with_page_fault("what is Python?")
        assert "Python" in result["response"]
        # Should not contain uncertainty markers
        assert "don't have" not in result["response"].lower()

    def test_page_fault_triggers_on_uncertain_response(self, agent_runtime, monkeypatch):
        """When response is uncertain, page fault should trigger re-retrieval."""
        page_fault = ContextPageFault(
            retriever=agent_runtime.retriever,
            mmu=agent_runtime.mmu,
            max_faults=2,
        )
        agent_runtime.page_fault = page_fault

        call_count = [0]

        def mock_llm(ctx, query, model_override=""):
            call_count[0] += 1
            if call_count[0] == 1:
                return "I don't have enough information about this topic to answer properly."
            return "Based on the retrieved documents, here is the answer."

        agent_runtime.llm_fn = mock_llm

        result = agent_runtime.process_query_with_page_fault("what is FastAPI?")
        # Should have retried (call_count >= 2 if page fault triggered)
        assert call_count[0] >= 2
        assert "response" in result
```

- [ ] **Step 5: Run tests**

```bash
cd f:/agentplan && python -m pytest tests/test_scheduler_page_fault.py -v
```
Expected: 3 PASS

- [ ] **Step 6: Run full test suite to check no regressions**

```bash
cd f:/agentplan && python -m pytest -x --tb=short
```
Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
cd f:/agentplan && git add src/runtime/scheduler.py src/runtime/agent_runtime.py tests/test_scheduler_page_fault.py && git commit -m "feat: wire ContextPageFault into AgentRuntime with retry loop in Scheduler"
```

---

### Task 7: Final integration — run full test suite

- [ ] **Step 1: Run all tests**

```bash
cd f:/agentplan && python -m pytest -v
```
Expected: all tests PASS (current count: 273 + new tests)

- [ ] **Step 2: Start server smoke test**

```bash
cd f:/agentplan && timeout 5 python -m src --llm mock --embed mock --port 8765 || true
```
Expected: server starts without import errors

- [ ] **Step 3: Final commit (if needed)**

Only if any fixes were needed from the full test run.

---

## Dependency Order

```
Task 1 (DB last_used_at)  ──┐
Task 2 (StepType enum)    ──┼──→ Task 4 (Controller trace) ──→ Task 7 (final)
Task 3 (Reranker)         ──┤
                            └──→ Task 5 (Scheduler trace) ──→ Task 6 (PageFault) ──→ Task 7
```

Tasks 1, 2, 3 are independent and can run in parallel. Tasks 4-6 have sequential dependencies. Task 7 is the final validation gate.
