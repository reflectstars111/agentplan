# ARD Core Rebuild — Architecture & Design

Date: 2026-06-08
Status: Approved
Design target: Rebuild ARD runtime core on top of existing parsing/embedding/LLM infrastructure

---

## 1. Problem

The existing Agent-OS codebase implements a Von Neumann-inspired multi-agent runtime. The [Agent_Runtime_Database_ARD_Reconstructed.md](../../../Agent_Runtime_Database_ARD_Reconstructed.md) design doc repositions the system as a **database-centric Agent Runtime Database (ARD)**, introducing:

- **State** as the unified abstraction (replacing Memory)
- **Transaction-based write-back** with ACID properties
- **MVCC** for versioned, auditable state evolution
- **Context MMU** as the central 6-step context assembly pipeline

The old codebase does not reflect these new primitives. This spec defines a **core rebuild** — preserving the proven parsing, embedding, and LLM infrastructure layers while rewriting the runtime core to strict ARD design.

---

## 2. Architecture Overview

### 2.1 System Layers

```
┌─────────────────────────────────────────────────┐
│              API / CLI Layer                     │
├─────────────────────────────────────────────────┤
│  Controller → Planner → Scheduler → Verifier     │
├─────────────────────────────────────────────────┤
│          Context MMU  (6-step pipeline)          │
│  Retrieve │ Filter │ Rank │ Compress │ Assemble  │
├─────────────────────────────────────────────────┤
│              Retriever                           │
│  Vector │ Keyword │ Entity │ Structure │ Time    │
├───────────────┬───────────────┬─────────────────┤
│  StateStore   │ KnowledgeStore│   TraceStore     │
│  (L0-L5)      │ (chunks+sources)│ (audit log)    │
├───────────────┴───────────────┴─────────────────┤
│         TransactionManager  │  MVCC Engine       │
├─────────────────────────────────────────────────┤
│    Event Store  (Write-Ahead Log — sole truth)   │
├─────────────────────────────────────────────────┤
│  SQLite  │  FAISS  │  Filesystem  │  BGE-M3     │
└─────────────────────────────────────────────────┘
```

### 2.2 Read vs Write Path Separation (CQRS within the Store layer)

```
                     READ PATH                    WRITE PATH
              ┌────────────────────┐     ┌────────────────────┐
              │   Context MMU      │     │  TransactionManager │
              │   QueryPlanner     │     │  Verifier           │
              │   HybridRetriever  │     │  WritebackGate      │
              └────────┬───────────┘     └────────┬───────────┘
                       │                          │
              ┌────────▼──────────────────────────▼───────────┐
              │            STORE INTERFACE (Protocol)          │
              │  StateStore │ KnowledgeStore │ TraceStore      │
              │  .read()    │ .search()      │ .record()       │
              │  .history() │ .index()       │ .query()        │
              └────────┬──────────────────────┬───────────────┘
                       │                      │
              ┌────────▼──────────────────────▼───────────────┐
              │          EVENT STORE (Write-Ahead Log)         │
              │  append(event) → seq_num (immutable)           │
              │  replay(after_seq) → [events]                  │
              └────────┬──────────────────────┬───────────────┘
                       │                      │
        ┌──────────────▼──────┐  ┌─────────────▼──────────────┐
        │  STATE PROJECTION   │  │  KNOWLEDGE PROJECTION      │
        │  (SQLite views)     │  │  (SQLite + FAISS views)    │
        └─────────────────────┘  └────────────────────────────┘
```

Key design decisions:
- **Event Store is the sole source of truth** — all writes go through it
- **Synchronous projections** — commit → projections applied → immediately readable (database semantics)
- **seq_num IS the MVCC version** — no separate version table needed
- **Upper layers depend on Store Protocols, not implementations** — testable in isolation
- **Optimistic locking** — read_set records read-at seq_num; verify checks no concurrent writes

---

## 3. Module Layout

```
ard/                          # New system root (lives alongside old src/)
├── __init__.py
├── __main__.py
│
├── infra/                    # Infrastructure — zero business logic
│   ├── __init__.py
│   ├── db.py                 # SQLite connection pool + schema migration
│   ├── config.py             # Centralized config
│   └── logging.py            # Structured JSON-line logger
│
├── store/                    # Storage layer — system foundation
│   ├── __init__.py
│   ├── event.py              # StoreEvent dataclass
│   ├── event_store.py        # EventStore: append() / replay()
│   ├── projections.py        # Projection framework: register handler, apply()
│   ├── transaction.py        # TransactionManager + Transaction
│   ├── state_store.py        # StateStore: read / history / list_keys (Protocol)
│   ├── knowledge_store.py    # KnowledgeStore: search / index / get_chunks (Protocol)
│   └── trace_store.py        # TraceStore: record / query / replay (Protocol)
│
├── context/                  # Context assembly pipeline
│   ├── __init__.py
│   ├── mmu.py                # ContextMMU (6-step pipeline)
│   ├── page_fault.py         # ContextPageFault
│   └── token_budgeter.py     # TokenBudgeter
│
├── retriever/                # Multi-strategy retrieval
│   ├── __init__.py
│   ├── hybrid.py             # HybridRetriever
│   ├── query_planner.py      # QueryPlanner
│   ├── reranker.py           # Reranker (8-factor scoring)
│   └── strategies/
│       ├── vector.py         # FAISS search
│       ├── keyword.py        # SQLite FTS
│       ├── entity.py         # Entity index search
│       ├── structure.py      # Section/file/symbol structure search
│       └── temporal.py       # Time-range filter
│
├── runtime/                  # Execution engine — minimal
│   ├── __init__.py
│   ├── controller.py         # Controller (core loop: plan→load→reason→verify→write)
│   ├── planner.py            # Planner (query → TaskGraph DAG)
│   ├── executor.py           # Executor (ContextPack → LLM → Response)
│   ├── verifier.py           # Verifier (source check + conflict detection)
│   └── scheduler.py          # Scheduler (DAG topological execution)
│
├── io/                       # Input/Output adapters
│   ├── __init__.py
│   ├── sources/              # Input sources (migrated from old src/sources/)
│   │   ├── file.py
│   │   ├── github.py
│   │   └── web.py
│   └── sinks/                # Output formatters (NEW)
│       ├── text.py
│       ├── diff.py
│       └── report.py
│
├── parsing/                  # Document parsing (reuse from old src/parsing/)
│   ├── pdf.py
│   ├── word.py
│   ├── code.py
│   └── image.py
│
├── embedding/                # Embedding (reuse from old src/embedding.py)
│   └── bge.py
│
├── llm/                      # LLM adapter (reuse from old src/llm/)
│   └── factory.py
│
└── api/                      # HTTP API (new, minimal)
    ├── server.py
    └── routes.py
```

### 3.1 Old → New Migration Map

| Old module | Disposition | Why |
|---|---|---|
| `src/models/agent.py` | Deleted | Agent state is StateStore entries |
| `src/models/memory.py` | Deleted | Replaced by State (StateStore) |
| `src/models/task.py` | Merged into `runtime/planner.py` | Task as plan output, not persisted model |
| `src/models/trace.py` | Merged into `store/trace_store.py` | Trace model lives with its store |
| `src/models/blackboard.py` | Deleted | StateStore IS the blackboard |
| `src/runtime/agent_runtime.py` | Deleted | Controller replaces it |
| `src/runtime/agent_registry.py` | Deleted | No multi-agent in MVP |
| `src/runtime/audit_log.py` | Merged into `store/trace_store.py` | TraceStore = audit log |
| `src/runtime/writeback_gate.py` | Merged into `store/transaction.py` | Write gating = TransactionManager.verify() |
| `src/runtime/permission_checker.py` | Deferred | YAGNI for MVP |
| `src/runtime/input_sanitizer.py` | Deferred | Security phase later |
| `src/runtime/message_bus.py` | Deleted | ARD de-emphasizes agents |
| `src/runtime/merger.py` | Deleted | Single executor, no merges needed |
| `src/runtime/interrupt_handler.py` | Deferred | Not in MVP scope |
| `src/storage/memory_store.py` | Integrated | Becomes `store/state_store.py` |
| `src/storage/conversation_cache.py` | Integrated | L1 Session in StateStore |
| `src/storage/dependency_graph.py` | Integrated | Becomes `retriever/strategies/structure.py` |
| `src/index/*` | Migrated | To `retriever/`, adapted to KnowledgeStoreProtocol |
| `src/sources/*` | Migrated | To `io/sources/` |
| `src/parsing/*` | Reused verbatim | Proven, no changes needed |
| `src/embedding.py` | Reused | BGE-M3 1024-dim works |
| `src/llm/*` | Reused | Factory pattern works |

### 3.2 Dependencies (strictly acyclic)

```
infra/         ← depends on nothing
store/         ← depends on infra/
retriever/     ← depends on store/ + infra/
context/       ← depends on retriever/ + store/ + infra/
runtime/       ← depends on context/ + store/ + infra/
api/           ← depends on runtime/ + infra/
io/            ← depends on parsing/ + store/ + infra/
parsing/       ← depends on infra/ (or nothing)
embedding/     ← depends on infra/
llm/           ← depends on infra/
```

---

## 4. Data Models

### 4.1 StoreEvent (immutable, write-once)

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class StoreEvent:
    event_id: str           # globally unique UUID7
    seq_num: int = -1       # assigned by EventStore on append; -1 before commit
    stream: str             # "state" | "knowledge" | "trace"
    stream_key: str         # "agent:agent_001" | "chunk:chunk_042" | "trace:trace_001"
    event_type: str         # "created" | "updated" | "archived" | "deleted"
    payload: dict[str, Any] # the changed content
    txn_id: str             # owning transaction
    causation_seq: int = -1 # prior event seq_num in same causal chain
    timestamp: str = ""     # ISO8601 UTC; filled on append
```

### 4.2 Transaction

```python
class Transaction:
    txn_id: str
    status: str                    # "pending" | "committed" | "rolled_back"
    read_set: list[dict]           # [{"stream": ..., "stream_key": ..., "read_at_seq": ...}]
    write_events: list[StoreEvent]
    created_at: str
```

### 4.3 ContextPack (existing model, reused)

```python
class ContextPack:
    context_id: str
    task_id: str
    agent_id: str
    budget: int
    sections: list[ContextSection]
    source_refs: list[str]
    created_at: str
```

### 4.4 RetrievalResult

```python
@dataclass
class RetrievalResult:
    chunk_id: str
    source_ref: str
    text_preview: str
    score: float
    trust_level: str       # "internal_memory" | "user_provided_data" | "external_untrusted"
    strategy: str          # which strategy found it
    location: dict | None  # {page, section, line}
```

### 4.5 Verdict

```python
@dataclass
class Verdict:
    verified: bool
    confidence: float
    conflicts: list[dict]      # [{"topic": ..., "old": ..., "new": ..., "old_version": ...}]
    orphan_claims: list[str]   # claims without source evidence
    suggestions: list[str]
```

---

## 5. DB Schema

### 5.1 Event Store tables

```sql
CREATE TABLE events (
    seq_num INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    stream TEXT NOT NULL,
    stream_key TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload JSON NOT NULL,
    txn_id TEXT NOT NULL,
    causation_seq INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_events_stream ON events(stream, stream_key);
CREATE INDEX idx_events_txn ON events(txn_id);

CREATE TABLE transactions (
    txn_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    read_set JSON,
    event_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT
);
```

### 5.2 State projection tables

```sql
CREATE TABLE state_snapshots (
    stream_key TEXT PRIMARY KEY,
    value JSON NOT NULL,
    version INTEGER NOT NULL,
    stream TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

### 5.3 Knowledge projection tables

```sql
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    text TEXT NOT NULL,
    summary TEXT,
    location JSON,
    keywords TEXT,
    trust_level TEXT DEFAULT 'external_untrusted',
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id, text, summary, keywords);

CREATE TABLE sources (
    source_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    file_name TEXT,
    file_path TEXT,
    chunk_count INTEGER DEFAULT 0,
    total_size INTEGER,
    status TEXT DEFAULT 'active',
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

### 5.4 Trace projection table

```sql
CREATE TABLE traces (
    trace_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    step_type TEXT NOT NULL,
    input JSON,
    output JSON,
    status TEXT DEFAULT 'success',
    error TEXT,
    seq_num INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (trace_id, step_id)
);
```

---

## 6. Key Algorithms

### 6.1 TransactionManager — Optimistic Locking

```
begin() → new Transaction(txn_id, status="pending")

within txn:
  any StateStore.read(key) → records {stream, stream_key, read_at_seq} in txn.read_set
  txn.add_event(event) → appends to txn.write_events

verify(txn):
  for each entry in txn.read_set:
    current_seq = EventStore.latest_seq(entry.stream, entry.stream_key)
    if current_seq > entry.read_at_seq:
      return False  # concurrent write detected
  return True

commit(txn):
  assert verify(txn)
  for each event in txn.write_events:
    seq = EventStore.append(event)
    event = event with seq_num=seq
  Projections.apply(txn.write_events)
  txn.status = "committed"
  return [seq_nums]

rollback(txn):
  txn.status = "rolled_back"
  txn.write_events = []
```

### 6.2 StateStore — MVCC Read

```
read(key, version=None):
  if version is None:
    return latest state_snapshot for key
  else:
    replay events for key up to seq_num=version
    apply sequentially to build snapshot at that version
    return snapshot

history(key):
  SELECT * FROM events WHERE stream_key = key ORDER BY seq_num
  return [events]
```

### 6.3 HybridRetriever — 8-Factor Scoring

```
Score = 0.35 * semantic_similarity
      + 0.20 * keyword_match
      + 0.15 * entity_relevance
      + 0.10 * recency_score
      + 0.10 * importance_score
      + 0.10 * structural_relevance
      - 0.10 * token_cost
      - 0.20 * trust_penalty
```

### 6.4 ContextMMU — 6-Step Assembly

```
assemble(query, retrieval_results, state, history):
  1. RETRIEVE  ← already done, results passed in
  2. FILTER    → deduplicate by text hash, filter by trust_level
  3. RANK      → sort by priority * score
  4. COMPRESS  → for each item over budget: compress then truncate
  5. ASSEMBLE  → build sections in priority order:
       system_instruction(1) → current_query(2) → working_memory(3)
       → conversation_history(4) → long_term_memory(5)
       → retrieved_evidence(6) → tool_results(7) → output_reserve(8)
  6. BUDGET    → allocate by ratio:
       system:10% | query:5% | conversation:10% | working:10%
       | long_term:10% | evidence:35% | tools:10% | reserve:10%
  → return ContextPack
```

### 6.5 Controller — Core Cycle

```
process(user_request):
  request_id = gen()
  trace = TraceStore.start(request_id)

  intent = IntentDecoder.decode(user_request)
  TraceStore.record(trace, "intent_decode", intent)

  task_graph = Planner.plan(intent)
  TraceStore.record(trace, "plan", task_graph)

  context_pack = ContextMMU.assemble(
    query=user_request,
    retrieval_results=HybridRetriever.retrieve(user_request, plan),
    state=StateStore.read("working:*"),
    history=StateStore.read("conversation:*")
  )
  TraceStore.record(trace, "context_assemble", context_pack)

  response = Executor.think(context_pack)
  TraceStore.record(trace, "execute", response)

  verdict = Verifier.verify(response, context_pack)
  TraceStore.record(trace, "verify", verdict)

  txn = TransactionManager.begin()
  if verdict.verified:
    txn.add_event(state_updated_event)
  txn.add_event(trace_complete_event)
  TransactionManager.commit(txn)
  TraceStore.record(trace, "writeback", txn.status)

  return response
```

---

## 7. Storage Hierarchy → Event Stream Mapping

| ARD Level | Event Stream | Read Projection |
|---|---|---|
| L0 Context Buffer | Not persisted (in-memory) | — |
| L1 Session Store | `state:session:*` | state_snapshots (recent conversations) |
| L2 Working State | `state:task:*` `state:agent:*` | state_snapshots (current task/agent state) |
| L3 Long-Term Knowledge | `state:memory:*` `knowledge:memory:*` | memories in state_snapshots + chunks |
| L4 External Knowledge | `knowledge:chunk:*` `knowledge:source:*` | chunks + FAISS + sources tables |
| L5 Archive | all streams, `event_type="archived"` | filtered by event_type |

---

## 8. Implementation Phases

### Phase 0: Infra + Event Store
**Goal**: Physical storage foundation. A working EventStore with append/replay and a projection framework.
- `infra/db.py`, `infra/config.py`, `infra/logging.py`
- `store/event.py` (StoreEvent)
- `store/event_store.py` (EventStore)
- `store/projections.py` (Projections framework)
- DB schema: events table, transactions table
- **Verify**: append 1000 events, replay from arbitrary seq_num, projections fire on apply
- **NOT building**: StateStore, KnowledgeStore, business logic

### Phase 1: State Store + Transaction Manager
**Goal**: Transactional state read/write with MVCC versioning and optimistic locking.
- `store/state_store.py` (StateStore with Protocol)
- `store/transaction.py` (Transaction + TransactionManager)
- Projection table: state_snapshots
- **Verify**: read(key), read(key, version), history(key); concurrent write conflict → rollback; commit → immediately readable
- **NOT building**: KnowledgeStore, file ingestion, retrieval

### Phase 2: Knowledge Store + File Ingestion
**Goal**: Receive external files, chunk, index. KnowledgeStore supports search/get_chunks/list_sources.
- `store/knowledge_store.py` (KnowledgeStore with Protocol)
- `io/sources/file.py` (FileSource)
- Reuse `src/parsing/pdf.py`, `src/parsing/code.py`, `src/parsing/word.py`
- Reuse `src/embedding.py` (BGE-M3)
- Projection tables: chunks, chunks_fts, sources
- FAISS index persistence
- **Verify**: upload PDF → chunk → FAISS indexed; KnowledgeStore.get_chunks() returns results; restart → index recoverable
- **NOT building**: hybrid retrieval, ContextMMU, query reasoning

### Phase 3: Retriever Layer
**Goal**: Multi-strategy hybrid retrieval with quality evaluation.
- `retriever/strategies/vector.py`, `keyword.py`, `entity.py`, `structure.py`
- `retriever/hybrid.py` (HybridRetriever with Protocol)
- `retriever/query_planner.py` (QueryPlanner with Protocol)
- `retriever/reranker.py` (Reranker with 8-factor scoring)
- Evaluation: fixed dataset, MRR/nDCG metrics, baseline vs hybrid
- **Verify**: 5 strategies individually testable; hybrid returns ranked results; MRR > baseline; structure retrieval locates by section/symbol
- **NOT building**: ContextMMU (uses retrieval but phase 4), Controller

### Phase 4: Context MMU + Executor
**Goal**: Assemble context packs from retrieval results and state, feed LLM, get responses.
- `context/token_budgeter.py` (TokenBudgeter)
- `context/mmu.py` (ContextMMU with Protocol)
- `context/page_fault.py` (ContextPageFault)
- `runtime/executor.py` (Executor with Protocol)
- Reuse `src/llm/factory.py` (LLM adapter)
- **Verify**: ContextMMU produces valid ContextPack with source_refs/trust_levels; token budget respected; page fault triggers secondary retrieval; Executor with mock LLM returns formatted response; end-to-end with real LLM answers doc questions
- **NOT building**: Controller, Planner, Verifier

### Phase 5: Controller + Planner + Verifier
**Goal**: Complete ARD control loop: plan → load → reason → verify → writeback.
- `runtime/planner.py` (Planner with Protocol)
- `runtime/scheduler.py` (Scheduler)
- `runtime/controller.py` (Controller)
- `runtime/verifier.py` (Verifier)
- `store/trace_store.py` (TraceStore with Protocol)
- **Verify**: Controller.process() completes full cycle; TraceStore records every step; Verifier detects orphan claims and lowers confidence; writeback persists to StateStore; version history traceable
- **NOT building**: API, IO sinks, evaluation harness

### Phase 6: API + I/O + Integration
**Goal**: Serveable system with evaluation benchmark.
- `api/server.py`, `api/routes.py`
- `io/sinks/text.py`, `diff.py`, `report.py`
- `ard/__main__.py` (entry point)
- `eval/benchmark.py` (scenario-based eval)
- **API routes**: POST /query, POST /upload, GET /trace/{id}, GET /state/{key}/history, GET /sources, GET /health
- **Eval scenarios**: doc QA, cross-file reasoning, state evolution history, page fault detection
- **Verify**: full end-to-end tests pass; 4 eval scenarios pass; API starts and serves; restart preserves state and indices

---

## 9. Dependencies Between Phases

```
Phase 0 ──→ Phase 1 ──→ Phase 2
               │           │
               │           ▼
               │      Phase 3 ──→ Phase 4
               │                     │
               ▼                     ▼
            (StateStore)        Phase 5 ──→ Phase 6
            used by Phase 5       │
                                  ▼
                             (Complete System)
```

---

## 10. Design Principles

1. **Protocol-driven interfaces**: Every store exposes a Protocol. Upper layers depend on Protocols, not implementations. Tests inject mocks.
2. **One file, one class, one test file**: `store/event_store.py` → `tests/store/test_event_store.py`
3. **No circular dependencies**: Dependency graph is strictly acyclic
4. **Synchronous projections**: Commit → apply → immediately readable. Database semantics, not message-bus semantics.
5. **seq_num IS version**: No separate version column. Every event has a seq_num that serves as the MVCC version identifier.
6. **Optimistic locking, not pessimistic**: Read records seq_num; verify checks no intervening writes; commit or retry.
7. **YAGNI**: No permission checker, no message bus, no multi-agent registry, no input sanitizer — not in MVP.
8. **Each phase produces a running, testable system**: No phase ends with just library code. Every phase has verification criteria.

---

## 11. Risk Mitigation

| Risk | Mitigation |
|---|---|
| Event Store performance degrades with event volume | Phase 0 test with 100k events; add snapshot compaction later if needed |
| Optimistic lock contention under concurrent writes | MVP is single-user; can add retry loop or pessimistic lock later |
| FAISS index out of sync with chunks table | Rebuild from chunks table on mismatch; Phase 2 tests cover recovery |
| ContextMMU token budget logic is inaccurate | TokenBudgeter uses conservative estimates; Phase 4 tests verify with known inputs |
| Verifier false positives on orphan claims | Use n-gram overlap as weak signal; manual review of flagged claims; tune threshold |
| Old parsing code incompatible with new store interfaces | Phase 2 wraps parsers in adapters; parsers themselves unchanged |

---

## 12. Verification Strategy

Each phase has concrete verification criteria (see §8). Additionally:

- **Unit tests**: One test file per source file, testing each class in isolation with mocked dependencies
- **Integration tests**: Cross-layer tests (e.g., TransactionManager → EventStore → StateStore)
- **Eval benchmarks**: Fixed dataset of 20 documents + 50 queries, measured with MRR/nDCG/Precision@k
- **Scenario tests**: 4 end-to-end scenarios (doc QA, cross-file, state evolution, page fault)
- **Property-based tests**: For EventStore (append N then replay N = original events) and TransactionManager (no dirty reads after rollback)
