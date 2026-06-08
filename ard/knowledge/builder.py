"""Cross-domain knowledge base builder.

Generates structured knowledge documents across AI Systems, Database Systems,
and Programming domains using LLM for domain diversity.
"""


def build_knowledge_base(store, llm_fn=None, domains: list[str] | None = None):
    """Build or expand a multi-domain knowledge base.

    Args:
        store: KnowledgeStore to index into.
        llm_fn: Optional LLM function for generating domain documents.
        domains: Domains to build. Default: all 3.

    Returns:
        Number of new chunks indexed.
    """
    if domains is None:
        domains = ["ai_systems", "database_systems", "programming"]

    import uuid

    # Core documents (manually curated for each domain)
    docs = _get_core_docs()

    # LLM-generated expansion documents
    if llm_fn:
        for domain in domains:
            if domain in docs:
                extra = _generate_domain_docs(domain, count=5, llm_fn=llm_fn)
                docs[domain].extend(extra)

    total = 0
    for domain, domain_docs in docs.items():
        for doc in domain_docs:
            try:
                store.index_chunks(
                    [{"text": doc, "source_type": "text",
                      "file_name": f"{domain}_kb.txt",
                      "trust_level": "user_provided_data"}],
                    f"src_{domain}_{uuid.uuid4().hex[:6]}"
                )
                total += 1
            except Exception as e:
                print(f"  SKIP: {e}")
    return total


def _get_core_docs() -> dict[str, list[str]]:
    """Return manually curated domain documents."""
    return {
        "ai_systems": [
            "MemGPT (2023) treats LLM as OS kernel. Two-tier memory: Main Context (RAM) with system instructions and FIFO conversation queue, External Context (disk) with recall storage and archival storage via vector DB. LLM controls paging through function calls like core_memory_replace() and archival_memory_insert(). Letta framework (Sep 2024) evolved from MemGPT with Memory Blocks and sleeptime compute. Achieved 74% on LoCoMo benchmark.",
            "RAPTOR (Stanford 2024) builds hierarchical tree-structured datastore through recursive GMM clustering with UMAP dimensionality reduction, SBERT embedding (multi-qa-mpnet-base-cos-v1), and GPT-3.5-turbo summarization. 20% absolute improvement on QuALITY over prior SOTA. Collapsed tree retrieval outperforms tree traversal.",
            "LongMem (NeurIPS 2023) decoupled network: frozen backbone LLM as memory encoder prevents KV cache staleness, trainable lightweight SideNet adapter serves as memory retriever/reader via joint-attention mechanism. Cached memory bank stores attention KV pairs from up to 65K tokens. Achieved 40.5% on ChapterBreak benchmark.",
            "MemLong (2024) extends context from 4K to 80K tokens on single NVIDIA 3090 GPU using external retriever (non-differentiable ret-mem module) with fine-grained controllable retrieval attention mechanism. Only a small portion of model needs training. Consistently outperforms other LLMs on long-context benchmarks.",
            "ARD Context MMU applies deterministic 6-step pipeline: RETRIEVE (multi-strategy hybrid: vector+keyword+entity+structure), FILTER (dedup+trust-level filtering+source tracking), RANK (priority-based ordering with 8-factor scoring), COMPRESS (token budget compression), ASSEMBLE (build ContextPack with priority-ordered sections), BUDGET (explicit allocation: system 10%, query 5%, evidence 35%, conversation 10%, working 10%, long-term 10%, tools 10%, output 10%).",
            "ARD TransactionManager implements optimistic locking: begin() starts transaction, add_event() buffers writes, record_read() logs read keys with seq_num, verify() checks no concurrent modifications, commit() atomically appends all events to EventStore and applies synchronous projections, rollback() discards on conflict. MVCC via seq_num enables point-in-time queries and full audit trails.",
            "Self-RAG (ICLR 2024) trains LLM with special reflection tokens: Retrieve token signals retrieval necessity, Critique token evaluates answer quality. Requires custom training data with reflection annotations. CRAG (Corrective RAG, 2024) evaluates retrieval quality first, triggers web search fallback for low-quality results. HyDE generates hypothetical answer document from query via LLM, then uses it for retrieval.",
            "HyDE (Hypothetical Document Embeddings, 2023) generates a hypothetical answer document from the query via LLM, embeds the hypothetical, and uses it to retrieve real documents. Improves retrieval relevance for abstract queries but adds latency from extra LLM call. Particularly effective for zero-shot dense retrieval without relevance labels.",
        ],
        "database_systems": [
            "PostgreSQL MVCC implements tuple versioning: each row has xmin (creation transaction ID) and xmax (deletion transaction ID). Visibility determined by snapshot comparison — a tuple is visible if its creator committed before the snapshot and it hasn't been deleted by a concurrent transaction. VACUUM reclaims storage from dead tuples. Default isolation: read committed.",
            "CockroachDB is a distributed SQL database using Raft consensus protocol for replication. Data automatically sharded across nodes via range partitioning (each range ~64MB). Uses serializable snapshot isolation by default, stronger than PostgreSQL's read committed. Each range replicates via Raft log — leader handles writes, followers can serve reads (follower reads).",
            "Write-Ahead Logging (WAL) records all changes to sequential log file BEFORE applying modifications to data files. Essential for crash recovery — database replays WAL from last checkpoint to restore consistency. Enables point-in-time recovery (PITR) and streaming replication. WAL segments are typically 16MB in PostgreSQL.",
            "ACID: Atomicity ensures all-or-nothing execution via rollback (undo log). Consistency guarantees valid state transitions through constraints (PK, FK, CHECK). Isolation prevents interference between concurrent transactions via MVCC or 2PL. Durability ensures committed data survives crashes via WAL fsync.",
            "B-Tree storage engine: balanced multi-level tree where each node (page, 4-16KB) contains multiple keys and child pointers. All leaves at same depth. B+Tree variant places all data in leaf nodes with linked list for efficient range scans (ORDER BY, BETWEEN). Fill factor typically 70-90%. Query optimizer uses table statistics (pg_statistic) and cost model to select optimal execution plan from many possible join orders and index combinations.",
            "Optimistic Concurrency Control (OCC) assumes conflicts are rare: validates at commit time by checking whether any data read has been modified since read. If conflict detected, transaction aborts and retries. Pessimistic locking (2PL) acquires locks upfront, blocking other transactions. OCC yields higher throughput under low contention but can cause starvation under high. PostgreSQL uses a hybrid: MVCC for reads + row-level locks for writes.",
            "Snapshot Isolation (SI): each transaction sees a consistent snapshot of committed data as of its start time. Prevents dirty reads, non-repeatable reads, and phantom reads, but allows write skew anomalies. Implemented via transaction ID comparison: T_i sees data from transactions with ID < T_i that committed before T_i started. Serializable Snapshot Isolation (SSI) adds write conflict detection to prevent write skew.",
            "LSM-Tree (Log-Structured Merge-Tree): writes go to in-memory memtable, flushed to sorted SSTable files on disk when memtable is full. Reads check memtable first, then SSTables (with Bloom filters). Background compaction merges SSTables. Optimized for write-heavy workloads (RocksDB, LevelDB, Cassandra). B-Tree optimized for read-heavy workloads (PostgreSQL, MySQL InnoDB).",
        ],
        "programming": [
            "Python asyncio event loop: single-threaded cooperative scheduler running coroutines. Coroutines explicitly yield control at await points. Non-blocking I/O via epoll (Linux), kqueue (macOS), IOCP (Windows). TaskGroup (Python 3.11+) ensures structured concurrency — child tasks cannot outlive parent scope. asyncio.gather() for concurrent execution, asyncio.create_task() for fire-and-forget.",
            "Rust async/await: zero-cost futures compiled to state machines via compiler transformation. No heap allocation unless Box::pin used (for self-referential futures). External executors required: tokio (multi-threaded work-stealing, most popular), async-std (simpler API). Send+Sync traits enforce compile-time thread safety. Futures are lazy — nothing happens until polled by executor.",
            "Actor model: each actor has private mutable state, communicates exclusively via asynchronous message passing. Messages processed sequentially within each actor (no locks needed). Supervision trees: parent actors monitor children, restart on failure (Erlang/OTP let-it-crash philosophy). Examples: Erlang processes (per-actor GC, millions of actors), Akka (JVM), actix (Rust), Orleans (C#/.NET virtual actors).",
            "Tokio work-stealing scheduler: multi-threaded runtime where each worker thread maintains a local LIFO task queue. Idle workers steal tasks from other workers' queues (FIFO stealing). Provides automatic load balancing without central coordinator. Based on Go runtime's work-stealing design. Configurable: single-threaded (basic_scheduler) for testing, multi-threaded (threaded_scheduler) for production.",
            "Goroutines (Go): lightweight userspace threads multiplexed onto OS threads via M:N scheduling (GOMAXPROCS). Go runtime preempts goroutines at function calls and loop back-edges (since Go 1.14, async preemption). Channels for CSP-style communication: 'Don't communicate by sharing memory; share memory by communicating.' Select statement for multi-channel operations. sync.WaitGroup, errgroup for coordination.",
            "Structured concurrency: child tasks cannot outlive parent scope. When parent scope exits, all children are guaranteed completed (via cancellation or join). Prevents resource leaks, orphaned tasks, and makes error propagation predictable. Key insight: the syntactic structure of code (indentation, scopes) should match the runtime structure (task lifetimes). Python TaskGroup, Kotlin coroutineScope, Swift async let, Java StructuredTaskScope (Java 21, preview).",
            "Back-pressure: occurs when producer generates data faster than consumer processes. Solutions: bounded buffers with blocking (block when full), reactive streams (consumer requests N items: request(N)), rate limiting (token bucket, leaky bucket), dropping data (sampling), or dynamic adaptation (slow down producer). Async frameworks handle via flow control in Stream/AsyncIterator interfaces.",
            "Futures/Promises: represent a value that may not yet be available. Composition: then() chaining, async/await syntax sugar, Promise.all() for concurrent, Promise.race() for first-to-complete. Error handling via .catch() or try/catch with await. Rust futures are poll-based (push); JS promises are callback-based (pull). Futures can be eager (JS: start immediately) or lazy (Rust: start when polled).",
        ],
    }


def _generate_domain_docs(domain: str, count: int, llm_fn) -> list[str]:
    """Generate additional domain documents via LLM."""
    topics = {
        "ai_systems": [
            "memory-augmented neural networks and their limitations",
            "retrieval-augmented generation evaluation benchmarks beyond accuracy",
            "comparison of vector databases for AI memory (Chroma, Pinecone, Weaviate, Qdrant)",
            "in-context learning vs fine-tuning for long-horizon task adaptation",
            "hybrid search strategies combining sparse and dense retrieval",
        ],
        "database_systems": [
            "columnar storage engines vs row-based for analytical workloads",
            "materialized views and incremental view maintenance strategies",
            "distributed consensus protocols beyond Raft: Paxos, EPaxos, Viewstamped Replication",
            "database migration strategies: blue-green, canary, expand-contract patterns",
            "query optimization for distributed SQL: cost models, statistics, join ordering across nodes",
        ],
        "programming": [
            "coroutine implementation internals: stackful vs stackless, symmetric vs asymmetric",
            "memory models across languages: Java happens-before, C++11 atomics, Rust ownership",
            "reactive programming paradigms: RxJS, Project Reactor, Kotlin Flow vs async/await",
            "concurrency testing: deterministic simulation, property-based testing, Jepsen for distributed systems",
            "green threads vs async/await tradeoffs: Go goroutines, Java virtual threads (Project Loom), Erlang processes",
        ],
    }

    domain_topics = topics.get(domain, topics["ai_systems"])
    if count > len(domain_topics):
        count = len(domain_topics)

    docs = []
    for i in range(count):
        prompt = f"""Write a 3-5 sentence technical summary about this {domain} topic: {domain_topics[i]}.

Focus on: key concepts, important facts, and practical implications. Be concise and information-dense. Write as if for a technical encyclopedia."""
        try:
            from ard.context.pack import ContextPack
            p = ContextPack("gen", "gen", "gen", 2000)
            response = llm_fn(p, prompt)
            if isinstance(response, str) and len(response) > 50:
                docs.append(response.strip())
        except Exception:
            pass

    return docs
