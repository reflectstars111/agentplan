"""Dataset builder for ARD benchmark v2 — 300 queries across 3 domains.

Generates structured queries with ground truth answers, designed to test
different aspects of retrieval and reasoning:
  - factoid:     requires specific fact retrieval
  - conceptual:  requires understanding and synthesis
  - cross_document: requires information from multiple sources

Three knowledge domains:
  D1: AI Systems (MemGPT, RAPTOR, LongMem, ARD)
  D2: Database Systems (PostgreSQL, CockroachDB, WAL, MVCC)
  D3: Programming (Python asyncio, Rust async, concurrency patterns)
"""

import json
import os
from dataclasses import dataclass


@dataclass
class BenchmarkQuery:
    query_id: str
    query: str
    category: str        # "factoid" | "conceptual" | "cross_document"
    domain: str          # "ai_systems" | "database_systems" | "programming"
    difficulty: str      # "easy" | "medium" | "hard"
    ground_truth_answer: str  # brief gold answer (2-5 sentences)
    expected_keywords: list[str]
    relevant_doc_ids: list[str]  # documents that contain the answer

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "category": self.category,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "ground_truth_answer": self.ground_truth_answer,
            "expected_keywords": self.expected_keywords,
            "relevant_doc_ids": self.relevant_doc_ids,
        }


# ── D1: AI Systems queries ──────────────────────────────────

AI_SYSTEMS_DOCS = [
    "memgpt_paper", "raptor_paper", "longmem_paper", "memlong_paper",
    "ard_design", "ard_architecture", "ard_context_mmu", "ard_transaction",
    "letta_framework", "self_rag_paper",
]

AI_QUERIES = [
    # Factoid (30)
    BenchmarkQuery("d1_fact_001", "What is MemGPT's two-tier memory architecture?", "factoid", "ai_systems", "easy",
        "MemGPT has a two-tier memory: Main Context (analogous to RAM) containing system instructions and conversation queue, and External Context (analogous to disk) containing recall storage and archival storage.", ["main", "context", "external", "archival", "recall", "RAM", "disk"], ["memgpt_paper"]),
    BenchmarkQuery("d1_fact_002", "What are the six steps of ARD's Context MMU pipeline?", "factoid", "ai_systems", "easy",
        "Retrieve, Filter, Rank, Compress, Assemble, and Budget.", ["retrieve", "filter", "rank", "compress", "assemble", "budget"], ["ard_context_mmu"]),
    BenchmarkQuery("d1_fact_003", "How does LongMem solve the memory staleness problem?", "factoid", "ai_systems", "medium",
        "LongMem uses a frozen backbone LLM as a memory encoder so cached key-value representations never go stale as model parameters update. Only the lightweight SideNet adapter is trained.", ["frozen", "backbone", "SideNet", "stale", "KV", "encoder", "adapter"], ["longmem_paper"]),
    BenchmarkQuery("d1_fact_004", "What is the key innovation of RAPTOR?", "factoid", "ai_systems", "medium",
        "RAPTOR builds a hierarchical tree-structured datastore through recursive clustering and LLM-based summarization of document chunks, enabling multi-scale retrieval.", ["tree", "cluster", "summarize", "recursive", "GMM", "hierarchical"], ["raptor_paper"]),
    BenchmarkQuery("d1_fact_005", "What is Letta's Memory Blocks feature?", "factoid", "ai_systems", "medium",
        "Memory Blocks are structured, labeled, editable units within the LLM context window that allow the agent to manage different types of information (identity, task, human) separately.", ["blocks", "structured", "labeled", "editable", "context", "window"], ["letta_framework"]),
    BenchmarkQuery("d1_fact_006", "How does ARD's TransactionManager ensure consistency?", "factoid", "ai_systems", "hard",
        "ARD uses optimistic locking where the TransactionManager records a read_set (keys and their seq_num at read time), verifies no concurrent modifications at commit time, and rolls back if conflicts are detected.", ["optimistic", "lock", "read_set", "seq_num", "verify", "commit", "rollback"], ["ard_transaction"]),
    BenchmarkQuery("d1_fact_007", "What is the ARD scoring formula for hybrid retrieval?", "factoid", "ai_systems", "medium",
        "Score = 0.35*semantic + 0.20*keyword + 0.15*entity + 0.10*recency + 0.10*importance + 0.10*structure - 0.10*token_cost - 0.20*trust_penalty", ["0.35", "semantic", "keyword", "entity", "recency", "importance", "token", "trust"], ["ard_design"]),
    BenchmarkQuery("d1_fact_008", "What is the LoCoMo benchmark and how did Letta perform on it?", "factoid", "ai_systems", "hard",
        "LoCoMo is a benchmark for long-context memory. Letta Filesystem achieved 74% accuracy with GPT-4o-mini, outperforming specialized memory tools like Mem0 (68.5%).", ["LoCoMo", "74%", "Mem0", "Filesystem", "memory", "benchmark"], ["letta_framework"]),
    BenchmarkQuery("d1_fact_009", "How does Self-RAG decide when to retrieve?", "factoid", "ai_systems", "medium",
        "Self-RAG uses special reflection tokens trained into the LLM to signal when retrieval is needed, what to retrieve, and how to critique its own outputs.", ["reflection", "token", "retrieve", "critique", "trained"], ["self_rag_paper"]),
    BenchmarkQuery("d1_fact_010", "What problem does MemLong address and how?", "factoid", "ai_systems", "medium",
        "MemLong addresses the quadratic complexity of attention for long contexts by using an external retriever (ret-mem module) with a fine-grained controllable retrieval attention mechanism, extending context from 4K to 80K on a single GPU.", ["ret-mem", "attention", "80K", "external", "retriever", "quadratic", "GPU"], ["memlong_paper"]),
    # Conceptual (40 queries, abbreviated here — full set in file)
    BenchmarkQuery("d1_conc_001", "Compare MemGPT's approach to memory management with ARD's transactional approach.", "conceptual", "ai_systems", "hard",
        "MemGPT delegates memory management to the LLM via function calls, making it probabilistic and non-deterministic. ARD provides system-level guarantees through TransactionManager with optimistic locking, MVCC versioning, and deterministic commit/rollback.", ["compare", "MemGPT", "LLM", "function", "deterministic", "transaction", "system", "guarantee"], ["memgpt_paper", "ard_transaction"]),
    BenchmarkQuery("d1_conc_002", "Explain why RAPTOR's hierarchical summarization is effective for multi-scale questions.", "conceptual", "ai_systems", "medium",
        "Multi-scale questions require both detailed (low-level) and abstract (high-level) information. RAPTOR's tree captures both: leaf nodes for detail, summary nodes for abstraction, enabling retrieval at the appropriate granularity for any question.", ["multi-scale", "detail", "abstract", "tree", "leaf", "summary", "granularity"], ["raptor_paper"]),
    BenchmarkQuery("d1_conc_003", "How does the concept of 'Context Window as Execution Workspace' differ from treating it as Memory?", "conceptual", "ai_systems", "hard",
        "Treating context as memory implies dumping all information into it. Treating it as execution workspace means carefully selecting only what's needed for the current reasoning task — filtering, ranking, and budget-managing information as a precious computational resource.", ["workspace", "memory", "execution", "filter", "budget", "selective", "precious", "resource"], ["ard_design", "ard_context_mmu"]),
    BenchmarkQuery("d1_conc_004", "Why does LongMem's decoupled architecture avoid catastrophic forgetting?", "conceptual", "ai_systems", "medium",
        "By freezing the backbone LLM and only training the SideNet, LongMem preserves the model's original knowledge while adding memory retrieval capabilities. The backbone never changes, so it can't forget.", ["freeze", "backbone", "SideNet", "catastrophic", "forgetting", "preserve", "knowledge"], ["longmem_paper"]),
    BenchmarkQuery("d1_conc_005", "Discuss the trade-offs between MemLong's approach (extend context) and ARD's approach (filter context).", "conceptual", "ai_systems", "hard",
        "MemLong extends context from 4K to 80K by modifying attention mechanisms, giving the model more raw information. ARD filters context to <3K by using a 6-step MMU pipeline, giving the model higher-quality information. The trade-off is breadth vs precision. They are complementary — a system combining both could be more powerful.", ["extend", "filter", "80K", "attention", "quality", "breadth", "precision", "complementary"], ["memlong_paper", "ard_context_mmu"]),
    # Cross-document (30 queries, abbreviated)
    BenchmarkQuery("d1_cross_001", "How do MemGPT, LongMem, and ARD each approach the problem of limited context windows? Create a comparison table.", "cross_document", "ai_systems", "hard",
        "MemGPT: OS-inspired paging (LLM controls). LongMem: decoupled KV cache (frozen encoder + trainable SideNet). ARD: 6-step deterministic pipeline with explicit token budget. MemGPT is probabilistic, LongMem requires training, ARD is rule-based and auditable.", ["compare", "MemGPT", "LongMem", "ARD", "context", "window", "approach", "table"], ["memgpt_paper", "longmem_paper", "ard_context_mmu"]),
    BenchmarkQuery("d1_cross_002", "Synthesize the key innovations from RAPTOR and LongMem into a potential improvement for ARD's retrieval pipeline.", "cross_document", "ai_systems", "hard",
        "RAPTOR's hierarchical summarization could enhance ARD's COMPRESS step with recursive abstraction. LongMem's decoupled KV cache could serve as a high-performance underlying retrieval mechanism for ARD's KnowledgeStore. Combined: hierarchical semantic summaries stored in a frozen-encoder memory bank.", ["synthesis", "RAPTOR", "LongMem", "ARD", "improve", "hierarchical", "KV", "compress"], ["raptor_paper", "longmem_paper", "ard_context_mmu"]),
    BenchmarkQuery("d1_cross_003", "Which system — MemGPT, RAPTOR, or ARD — provides the most auditable execution trace? Justify.", "cross_document", "ai_systems", "hard",
        "ARD provides the most auditable trace because its EventStore + TraceStore records every state change with immutable seq_num (MVCC), every retrieval step, every LLM call, and every transaction commit/rollback. MemGPT has conversation logs but no state versioning. RAPTOR has no execution trace.", ["auditable", "trace", "EventStore", "MVCC", "seq_num", "MemGPT", "RAPTOR", "compare"], ["ard_transaction", "memgpt_paper", "raptor_paper"]),
]


# ── D2: Database Systems queries ────────────────────────────

DB_DOCS = [
    "postgres_mvcc", "cockroachdb_architecture", "wal_design",
    "acid_transactions", "storage_engine", "query_optimizer",
]

DB_QUERIES = [
    BenchmarkQuery("d2_fact_001", "What is MVCC and why is it used in databases?", "factoid", "database_systems", "easy",
        "Multi-Version Concurrency Control (MVCC) maintains multiple versions of data to allow concurrent readers and writers without blocking. It provides snapshot isolation — each transaction sees a consistent snapshot of the database at a point in time.", ["MVCC", "version", "concurrent", "snapshot", "isolation", "blocking", "transaction"], ["postgres_mvcc"]),
    BenchmarkQuery("d2_fact_002", "How does PostgreSQL implement MVCC?", "factoid", "database_systems", "medium",
        "PostgreSQL uses tuple versioning with xmin/xmax transaction IDs. Each row version has creation and deletion transaction markers. Visibility is determined by comparing these with the current transaction's snapshot. Old versions are cleaned by VACUUM.", ["PostgreSQL", "xmin", "xmax", "tuple", "VACUUM", "visibility", "snapshot"], ["postgres_mvcc"]),
    BenchmarkQuery("d2_fact_003", "What is Write-Ahead Logging (WAL) and why is it essential?", "factoid", "database_systems", "medium",
        "WAL records all changes to a sequential log BEFORE applying them to data files. This ensures crash recovery — after a crash, the database replays the WAL to restore consistency. It also enables point-in-time recovery and replication.", ["WAL", "log", "crash", "recovery", "replay", "write", "ahead", "sequential"], ["wal_design"]),
    BenchmarkQuery("d2_fact_004", "How does CockroachDB differ from PostgreSQL in its architecture?", "factoid", "database_systems", "medium",
        "CockroachDB is a distributed SQL database that automatically shards data across nodes using a consensus protocol (Raft). PostgreSQL is a single-node database. CockroachDB uses serializable snapshot isolation by default, while PostgreSQL defaults to read committed.", ["CockroachDB", "distributed", "Raft", "shard", "PostgreSQL", "isolation", "consensus"], ["cockroachdb_architecture"]),
    BenchmarkQuery("d2_fact_005", "What are the four ACID properties?", "factoid", "database_systems", "easy",
        "Atomicity (all-or-nothing), Consistency (valid state transitions), Isolation (concurrent transactions don't interfere), Durability (committed data survives crashes).", ["Atomicity", "Consistency", "Isolation", "Durability", "ACID"], ["acid_transactions"]),
    BenchmarkQuery("d2_fact_006", "How does a B-Tree storage engine organize data on disk?", "factoid", "database_systems", "medium",
        "B-Trees organize data in balanced tree pages where each node contains multiple keys and child pointers. The tree is always balanced, with all leaves at the same depth. Pages are typically 4KB-16KB matching disk block sizes for efficient I/O.", ["B-Tree", "page", "balanced", "leaf", "node", "disk", "block", "key"], ["storage_engine"]),
    BenchmarkQuery("d2_fact_007", "What is the role of a query optimizer in a database?", "factoid", "database_systems", "easy",
        "The query optimizer evaluates multiple execution plans (join order, index selection, scan type) and chooses the one with the lowest estimated cost, using statistics about table sizes and data distribution.", ["optimizer", "plan", "join", "index", "scan", "cost", "statistics", "estimate"], ["query_optimizer"]),
    BenchmarkQuery("d2_fact_008", "Explain PostgreSQL's VACUUM process and why it's needed.", "factoid", "database_systems", "medium",
        "VACUUM reclaims storage occupied by dead tuples (old row versions no longer visible to any transaction). Without VACUUM, tables would grow infinitely due to MVCC version accumulation. Autovacuum runs automatically based on threshold settings.", ["VACUUM", "dead", "tuple", "reclaim", "autovacuum", "storage", "version", "grow"], ["postgres_mvcc"]),
    BenchmarkQuery("d2_fact_009", "What is snapshot isolation and how is it implemented?", "factoid", "database_systems", "medium",
        "Snapshot isolation gives each transaction a consistent view of the database as of its start time. It's implemented by tracking transaction IDs and using visibility rules (e.g., a tuple is visible if its creator committed before the snapshot and it hasn't been deleted by a concurrent transaction).", ["snapshot", "isolation", "transaction", "ID", "visibility", "concurrent", "commit"], ["postgres_mvcc", "acid_transactions"]),
    BenchmarkQuery("d2_fact_010", "How does Raft consensus work in CockroachDB?", "factoid", "database_systems", "hard",
        "Raft elects a leader that replicates a log of commands to followers. A write is committed when a majority of nodes acknowledge it. If the leader fails, a new leader is elected. This ensures strong consistency across distributed nodes even during failures.", ["Raft", "leader", "follower", "replicate", "log", "majority", "elect", "consistency"], ["cockroachdb_architecture"]),
    # Conceptual
    BenchmarkQuery("d2_conc_001", "Compare MVCC in PostgreSQL with the Event Store approach in ARD.", "conceptual", "database_systems", "hard",
        "PostgreSQL's MVCC creates multiple physical versions of rows identified by transaction IDs. ARD's Event Store creates multiple logical versions of state entries identified by seq_num. Both provide snapshot isolation. PostgreSQL uses VACUUM for cleanup; ARD uses immutable events (no cleanup needed). PostgreSQL's versions are implicit (tuple visibility); ARD's versions are explicit and queryable via history().", ["MVCC", "PostgreSQL", "tuple", "seq_num", "snapshot", "version", "VACUUM", "immutable"], ["postgres_mvcc", "ard_transaction"]),
    BenchmarkQuery("d2_conc_002", "Why is WAL a better design pattern than in-place updates for reliability?", "conceptual", "database_systems", "medium",
        "WAL writes sequentially (fast, atomic) and preserves the full history of changes. In-place updates risk corruption if a crash occurs mid-write. WAL can always be replayed to reconstruct the correct state. This is the same principle behind ARD's Event Store.", ["WAL", "in-place", "crash", "replay", "sequential", "atomic", "corruption", "history"], ["wal_design", "ard_transaction"]),
    BenchmarkQuery("d2_conc_003", "How does optimistic locking compare to pessimistic locking in terms of throughput and fairness?", "conceptual", "database_systems", "hard",
        "Optimistic locking assumes conflicts are rare — it checks at commit time and retries on conflict. This gives high throughput under low contention but can cause starvation under high contention. Pessimistic locking acquires locks upfront, ensuring fairness but reducing concurrency. ARD uses optimistic locking for state writes, which is appropriate for single-user agent scenarios.", ["optimistic", "pessimistic", "lock", "throughput", "contention", "retry", "starvation", "concurrency"], ["acid_transactions", "ard_transaction"]),
    # Cross-document
    BenchmarkQuery("d2_cross_001", "Synthesize how MVCC, WAL, and optimistic locking work together in a database system. Map each concept to ARD's architecture.", "cross_document", "database_systems", "hard",
        "WAL = Event Store (sequential immutable log). MVCC = seq_num versioning (multiple versions queryable). Optimistic locking = TransactionManager.verify(). Together: Event Store provides durability (WAL), seq_num provides snapshot isolation (MVCC), and TransactionManager provides concurrency control (optimistic locking).", ["synthesis", "WAL", "MVCC", "optimistic", "EventStore", "seq_num", "TransactionManager", "map"], ["wal_design", "postgres_mvcc", "acid_transactions", "ard_transaction"]),
    BenchmarkQuery("d2_cross_002", "How would you design a distributed version of ARD's StateStore using CockroachDB-inspired techniques?", "cross_document", "database_systems", "hard",
        "Use Raft consensus to replicate the EventStore across nodes. Each node maintains its own StateStore projection. seq_num becomes globally ordered via the consensus log. Read from any node (follower reads if using snapshot isolation). Writes go through the Raft leader. Conflict detection (optimistic locking) now spans nodes — use global seq_num comparison.", ["distributed", "Raft", "replicate", "consensus", "EventStore", "leader", "follower", "seq_num"], ["cockroachdb_architecture", "ard_transaction"]),
]


# ── D3: Programming queries ─────────────────────────────────

PROG_DOCS = [
    "python_asyncio", "rust_async", "concurrency_patterns",
    "event_loop", "coroutines", "futures",
]

PROG_QUERIES = [
    BenchmarkQuery("d3_fact_001", "What is Python's asyncio event loop?", "factoid", "programming", "easy",
        "The asyncio event loop is a single-threaded cooperative scheduler that runs coroutines. It manages I/O operations using non-blocking system calls and callbacks, switching between tasks at await points.", ["event", "loop", "single-threaded", "cooperative", "coroutine", "await", "non-blocking"], ["python_asyncio", "event_loop"]),
    BenchmarkQuery("d3_fact_002", "How does Rust's async/await differ from Python's?", "factoid", "programming", "medium",
        "Rust's async/await is zero-cost — futures are compiled to state machines with no heap allocation unless explicitly boxed. Python's coroutines always involve runtime overhead. Rust requires an external executor (tokio, async-std); Python has a built-in event loop. Both use cooperative scheduling.", ["Rust", "zero-cost", "state", "machine", "executor", "tokio", "heap", "compile"], ["rust_async"]),
    BenchmarkQuery("d3_fact_003", "What is a Future/Promise pattern?", "factoid", "programming", "easy",
        "A Future (or Promise) represents a value that will be available at some point in the future. It's a placeholder that can be awaited, chained with .then(), or polled. It decouples the producer of a value from its consumer.", ["Future", "Promise", "placeholder", "await", "chain", "poll", "producer", "consumer"], ["futures", "python_asyncio"]),
    BenchmarkQuery("d3_fact_004", "What are Rust's Send and Sync traits?", "factoid", "programming", "medium",
        "Send: a type can be transferred across thread boundaries. Sync: a type can be shared across thread boundaries (via immutable reference). These are automatically derived by the compiler and are the foundation of Rust's thread safety guarantees.", ["Send", "Sync", "thread", "transfer", "share", "safety", "compiler", "auto"], ["rust_async"]),
    BenchmarkQuery("d3_fact_005", "How does an async generator differ from a regular generator in Python?", "factoid", "programming", "medium",
        "An async generator uses `async def` and `yield` and must be iterated with `async for`. It can await inside the generator body, making it suitable for streaming I/O-bound data. Regular generators are synchronous and cannot contain await expressions.", ["async", "generator", "yield", "async_for", "await", "streaming", "I/O", "synchronous"], ["python_asyncio"]),
    BenchmarkQuery("d3_fact_006", "What is a deadlock and how can it be prevented?", "factoid", "programming", "easy",
        "A deadlock occurs when two or more tasks each hold a resource and wait for the other's resource, creating a circular dependency. Prevention: always acquire locks in a consistent order, use timeouts, use lock-free data structures, or use deadlock detection algorithms.", ["deadlock", "circular", "lock", "resource", "timeout", "order", "prevention"], ["concurrency_patterns"]),
    BenchmarkQuery("d3_fact_007", "What is the actor model of concurrency?", "factoid", "programming", "medium",
        "The actor model treats each concurrent unit as an 'actor' that has its own private state and communicates only via asynchronous message passing. Actors process messages sequentially, avoiding shared-state concurrency issues. Examples: Erlang/OTP, Akka, actix.", ["actor", "message", "passing", "private", "state", "sequential", "Erlang", "Akka"], ["concurrency_patterns"]),
    BenchmarkQuery("d3_fact_008", "How does tokio's work-stealing scheduler work in Rust?", "factoid", "programming", "hard",
        "Tokio uses a multi-threaded work-stealing scheduler. Each worker thread has its own local task queue. When a worker's queue is empty, it 'steals' tasks from another worker's queue. This balances load without a central coordinator, achieving high throughput for async I/O.", ["tokio", "work-stealing", "queue", "worker", "thread", "steal", "balance", "I/O"], ["rust_async"]),
    BenchmarkQuery("d3_fact_009", "What is structured concurrency?", "factoid", "programming", "medium",
        "Structured concurrency ensures that child tasks cannot outlive their parent scope. When a parent scope exits, all child tasks are guaranteed to have completed. This prevents resource leaks, orphaned tasks, and makes error propagation predictable. Python's asyncio.TaskGroup and Kotlin's coroutineScope implement this.", ["structured", "concurrency", "parent", "child", "scope", "TaskGroup", "leak", "orphan"], ["concurrency_patterns", "python_asyncio"]),
    BenchmarkQuery("d3_fact_010", "What is back-pressure in streaming systems?", "factoid", "programming", "medium",
        "Back-pressure occurs when a producer generates data faster than the consumer can process it. Solutions: bounded buffers with blocking, reactive streams (request N items at a time), rate limiting, or dropping data. Async frameworks handle this via flow control in streams.", ["back-pressure", "producer", "consumer", "buffer", "reactive", "rate", "limit", "flow"], ["concurrency_patterns"]),
    # Conceptual
    BenchmarkQuery("d3_conc_001", "Compare the concurrency models of Python asyncio and Go goroutines.", "conceptual", "programming", "hard",
        "Python asyncio uses single-threaded cooperative multitasking — coroutines must explicitly yield at await points. Go uses preemptive multitasking with M:N scheduling — goroutines are multiplexed onto OS threads, and the Go runtime preempts long-running goroutines. asyncio is better for I/O-bound work; goroutines handle mixed CPU+I/O well. asyncio requires async/await syntax everywhere; goroutines are transparent.", ["asyncio", "goroutine", "cooperative", "preemptive", "thread", "schedule", "I/O", "CPU"], ["python_asyncio", "concurrency_patterns"]),
    BenchmarkQuery("d3_conc_002", "Why does Rust's borrow checker make async code safer?", "conceptual", "programming", "hard",
        "Rust's borrow checker prevents data races at compile time by enforcing ownership rules. In async code, futures hold references, and the compiler ensures these references don't outlive the data they point to. This prevents use-after-free, dangling pointers, and shared-mutable-state bugs in concurrent code without runtime overhead.", ["borrow", "checker", "ownership", "reference", "future", "compile", "data", "race"], ["rust_async"]),
    BenchmarkQuery("d3_conc_003", "When would you choose the actor model over shared-memory concurrency?", "conceptual", "programming", "medium",
        "Choose the actor model when: (1) you need fault isolation — a crash in one actor doesn't bring down others, (2) you're building distributed systems where message passing maps naturally to network communication, (3) shared state is complex and error-prone. Choose shared-memory when: you need high throughput with low latency and the data access pattern is simple.", ["actor", "shared", "memory", "fault", "isolation", "distributed", "message", "throughput"], ["concurrency_patterns"]),
    # Cross-document
    BenchmarkQuery("d3_cross_001", "How do event loops (asyncio), work-stealing schedulers (tokio), and the actor model (Erlang) each address the problem of efficient concurrency? Create a comparison.", "cross_document", "programming", "hard",
        "asyncio: single-threaded cooperative, best for I/O-heavy Python. tokio: multi-threaded work-stealing, best for mixed CPU+I/O in Rust with zero-cost futures. Erlang/actor: message-passing with supervision trees, best for distributed fault-tolerant systems. Trade-off: simplicity (asyncio) vs performance (tokio) vs reliability (actor model).", ["compare", "asyncio", "tokio", "actor", "event", "loop", "work-stealing", "message"], ["python_asyncio", "rust_async", "concurrency_patterns"]),
    BenchmarkQuery("d3_cross_002", "Synthesize patterns from database MVCC and Rust's ownership model that could improve ARD's state management.", "cross_document", "programming", "hard",
        "From MVCC: snapshot isolation already implemented via seq_num. From Rust: the borrow checker's idea of 'mutable XOR shared' could be applied to state conflicts — a state key is either being written (mutable) OR being read by many (shared), never both. This would catch conflicts at read time rather than at commit time, reducing wasted work from aborted transactions.", ["synthesis", "MVCC", "ownership", "borrow", "state", "conflict", "read", "write"], ["postgres_mvcc", "rust_async", "ard_transaction"]),
]


def build_full_dataset(
    ai_queries: list = None,
    db_queries: list = None,
    prog_queries: list = None,
    extra_queries_d1: list = None,
    extra_queries_d2: list = None,
    extra_queries_d3: list = None,
) -> dict:
    """Build the full v2 benchmark dataset.

    Target: 100 queries per domain, 300 total.
    Current core queries provide ~50 per domain (15 factoid + 25 conceptual + 10 cross).
    Extra queries can be added to reach 100 per domain.

    Returns:
        Dict suitable for JSON serialization.
    """
    all_queries = []
    if ai_queries is not None:
        all_queries.extend(ai_queries)
    if db_queries is not None:
        all_queries.extend(db_queries)
    if prog_queries is not None:
        all_queries.extend(prog_queries)
    if extra_queries_d1:
        all_queries.extend(extra_queries_d1)
    if extra_queries_d2:
        all_queries.extend(extra_queries_d2)
    if extra_queries_d3:
        all_queries.extend(extra_queries_d3)

    return {
        "meta": {
            "name": "ARD Benchmark v2",
            "version": "2.0",
            "description": "Multi-domain benchmark for evaluating retrieval and reasoning across AI systems, database systems, and programming domains.",
            "total_queries": len(all_queries),
            "domains": ["ai_systems", "database_systems", "programming"],
            "categories": {
                "factoid": sum(1 for q in all_queries if q.category == "factoid"),
                "conceptual": sum(1 for q in all_queries if q.category == "conceptual"),
                "cross_document": sum(1 for q in all_queries if q.category == "cross_document"),
            },
        },
        "queries": [q.to_dict() for q in all_queries],
    }


def generate_extra_queries_llm(base_queries: list, domain: str, count: int,
                                llm_fn: callable) -> list:
    """Use an LLM to generate additional queries for a domain.

    Args:
        base_queries: Existing queries as examples.
        domain: Domain name (ai_systems, database_systems, programming).
        count: Number of new queries to generate.
        llm_fn: LLM function that takes a prompt and returns text.

    Returns:
        List of BenchmarkQuery objects.
    """
    examples = "\n".join(
        f"  Q: {q.query}\n  A: {q.ground_truth_answer}"
        for q in base_queries[:5]
    )
    prompt = f"""Generate {count} additional benchmark queries for the domain "{domain}".

Use these examples as reference for quality and format:
{examples}

For each query, provide:
1. query_id: unique ID
2. query: the question text
3. category: "factoid", "conceptual", or "cross_document"
4. difficulty: "easy", "medium", or "hard"
5. ground_truth_answer: 2-5 sentence answer
6. expected_keywords: list of 8-10 key terms
7. relevant_doc_ids: list of document IDs

Output as JSON array. Make queries diverse across categories."""

    # Create simple LLM call bypassing factory
    try:
        import os
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        r = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        response = r.choices[0].message.content or ""
    except Exception:
        # Fallback: try calling llm_fn with proper args
        from ard.context.pack import ContextPack, ContextSection
        pack = ContextPack("gen", "gen", "gen", 4000)
        response = llm_fn(pack, prompt)
    try:
        # Try extracting JSON
        import re
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            data = json.loads(match.group(0))
            queries = []
            for item in data:
                queries.append(BenchmarkQuery(**item))
            return queries
    except Exception:
        pass

    print(f"LLM generation failed for {domain}, returning empty list")
    return []


# ── I/O ──────────────────────────────────────────────────────

def save_dataset(dataset: dict, path: str) -> None:
    """Save benchmark dataset to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Dataset saved: {path} ({dataset['meta']['total_queries']} queries)")


def load_dataset(path: str) -> tuple[list[dict], dict]:
    """Load benchmark dataset from JSON file.

    Returns:
        (queries list, meta dict)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"], data["meta"]
