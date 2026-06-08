"""Run E1 experiment with real BGE-M3 embeddings + DeepSeek LLM.

REQUIRE: set DEEPSEEK_API_KEY environment variable before running.
    set DEEPSEEK_API_KEY=sk-your-key-here
    python run_experiment_bge.py
"""
import os, json, time, uuid

if not os.environ.get("DEEPSEEK_API_KEY"):
    print("ERROR: DEEPSEEK_API_KEY not set. Run: set DEEPSEEK_API_KEY=sk-...")
    exit(1)

from ard.infra.config import Config
from ard.infra.db import Database
from ard.store.knowledge_store import KnowledgeStore
from ard.retriever.vector_index import VectorIndex
from src.embedding import create_bge_embed_fn
from src.llm.llm_factory import create_llm_fn
from ard.retriever.reranker import Reranker
from ard.retriever.query_planner import QueryPlanner
from ard.retriever.hybrid import HybridRetriever
from ard.context.token_budgeter import TokenBudgeter
from ard.context.mmu import ContextMMU
from ard.runtime.executor import Executor
from ard.eval.experiment import ExperimentRunner
from ard.eval.judge import LLMJudge
from ard.context.pack import ContextPack

BGE_DIM = 1024

config = Config(
    db_path='data/ard_bge.db',
    vector_index_path='data/ard_bge.faiss',
    embedding_dim=BGE_DIM,
)
db = Database(config.db_path); db.init_schema()

print('Loading BGE-M3 (1024-dim, local CPU)...')
embed_fn = create_bge_embed_fn()

vi = VectorIndex(dim=BGE_DIM, index_path=config.vector_index_path)
store = KnowledgeStore(db, vi, embed_fn, config.file_store_path)

# Comprehensive knowledge base
docs = [
    'MemGPT (2023) two-tier memory: Main Context (RAM) with system instructions and FIFO conversation queue, External Context (disk) with recall storage and archival storage via vector DB. LLM controls paging via function calls like core_memory_replace().',
    'Letta framework evolved from MemGPT Sep 2024. Memory Blocks: structured labeled editable units within context window. Achieved 74% on LoCoMo benchmark beating Mem0 68.5%. Sleeptime compute enables idle reflection.',
    'RAPTOR (Stanford 2024) hierarchical tree datastore via recursive GMM clustering with UMAP dimensionality reduction, SBERT embedding, and LLM summarization. 20% absolute improvement on QuALITY benchmark over prior SOTA.',
    'LongMem (NeurIPS 2023) decoupled network: frozen backbone LLM encodes to KV cache (prevents staleness), trainable SideNet adapter retrieves and fuses via joint-attention. Up to 65K token memory bank.',
    'MemLong (2024) extends context from 4K to 80K tokens on single NVIDIA 3090 GPU via external retriever (ret-mem module) with fine-grained controllable retrieval attention. Only partial model training needed.',
    'ARD Context MMU applies deterministic 6-step pipeline: 1.RETRIEVE (multi-strategy hybrid), 2.FILTER (deduplication + trust-level filtering), 3.RANK (priority-based ordering), 4.COMPRESS (token budget compression), 5.ASSEMBLE (build ContextPack with sections), 6.BUDGET (explicit allocation by section ratio).',
    'ARD TransactionManager implements optimistic locking: begin() starts transaction, read_set records keys with their seq_num at read time, verify() checks no concurrent writes, commit() atomically appends all events to EventStore, rollback() on conflict detection.',
    'ARD EventStore is immutable write-ahead log — the sole source of truth. Each event has seq_num (monotonic, serves as MVCC version), stream, stream_key, event_type, payload. Events can never be modified or deleted. replay(after_seq) reconstructs state at any historical point.',
    'ARD scoring formula: Score = 0.35*semantic_similarity + 0.20*keyword_match + 0.15*entity_relevance + 0.10*recency + 0.10*importance + 0.10*structural_relevance - 0.10*token_cost - 0.20*trust_penalty. Eight factors combined for final relevance ranking.',
    'Self-RAG (ICLR 2024) trains LLM with special reflection tokens — Retrieve token signals need for retrieval, Critique token evaluates answer quality. Requires custom training data with reflection annotations.',
    'HyDE generates hypothetical answer document from query via LLM, then embeds the hypothetical to retrieve semantically similar real documents. Improves retrieval relevance for abstract queries but adds latency from extra LLM generation step.',
    'PostgreSQL MVCC implements tuple versioning: each row version has xmin (creation transaction ID) and xmax (deletion transaction ID). Visibility determined by snapshot comparison with transaction ID. VACUUM process reclaims storage from dead tuples.',
    'CockroachDB is distributed SQL database using Raft consensus protocol for replication. Automatically shards data across nodes via range partitioning. Uses serializable snapshot isolation by default, stronger than PostgreSQL default read committed.',
    'Write-Ahead Logging (WAL) records all changes to sequential log file BEFORE applying modifications to data files. Essential for crash recovery — database replays WAL to restore consistency after failure. Enables point-in-time recovery and streaming replication.',
    'ACID properties: Atomicity ensures all-or-nothing execution via rollback. Consistency guarantees valid state transitions through constraints. Isolation prevents interference between concurrent transactions via MVCC or locking. Durability ensures committed data survives crashes via WAL.',
    'B-Tree storage engine organizes data in balanced multi-level tree pages (typically 4-16KB) matching disk block sizes. B+Tree variant places all data in leaf nodes with linked list for efficient range scans. Query optimizer uses table statistics for plan selection.',
    'Optimistic locking assumes conflicts are rare — checks version at commit time and retries on conflict. Pessimistic locking acquires locks upfront preventing concurrent access. Optimistic yields higher throughput under low contention but can cause starvation.',
    'Snapshot isolation gives each transaction a consistent database view as of its start time using transaction ID comparison. Prevents dirty reads and non-repeatable reads but may allow write skew anomalies.',
    'Python asyncio event loop implements single-threaded cooperative multitasking. Coroutines explicitly yield control at await points. Uses non-blocking I/O via epoll/kqueue. TaskGroup ensures structured concurrency. Best suited for I/O-bound workloads.',
    'Rust async/await provides zero-cost futures compiled to state machines with no heap allocation unless Box::pin is used. Requires external executors like tokio or async-std. Send and Sync traits guarantee compile-time thread safety.',
    'Actor model: each actor maintains private state and communicates exclusively through asynchronous message passing. Messages processed sequentially within each actor. Supervision trees provide fault tolerance in Erlang/OTP and Akka.',
    'Deadlock occurs when multiple tasks form circular resource dependency — each holds a resource and waits for another. Prevention strategies: consistent lock acquisition ordering, timeout-based acquisition, lock-free data structures, deadlock detection algorithms.',
    'Structured concurrency ensures child tasks cannot outlive parent scope — when parent exits all children are guaranteed complete. Prevents resource leaks and orphaned tasks. Python asyncio.TaskGroup, Kotlin coroutineScope, Swift async let implement this pattern.',
    'Tokio work-stealing scheduler: multi-threaded runtime where each worker thread maintains local task queue. Idle workers steal tasks from busy workers queues. Provides automatic load balancing without central coordinator. Based on Go runtime design.',
    'MVCC with Event Sourcing: EventStore provides immutable sequential log (WAL), seq_num enables snapshot isolation (MVCC), TransactionManager provides concurrency control (optimistic locking). Together they form complete ACID-compliant state management for AI systems.',
]

for i, doc in enumerate(docs):
    store.index_chunks(
        [{'text': doc, 'source_type': 'text', 'file_name': f'kb_{i}.txt', 'trust_level': 'user_provided_data'}],
        f'src_{uuid.uuid4().hex[:6]}'
    )
print(f'Knowledge: {store.count_chunks()} chunks, FAISS: {vi.count} vectors')

# Real LLM pipeline
llm_fn = create_llm_fn(provider='deepseek', model='deepseek-chat')
executor = Executor(llm_fn)
hybrid = HybridRetriever(store, QueryPlanner(), Reranker(config))
mmu = ContextMMU(TokenBudgeter(config), config)

# Judge with real LLM
def judge_call(prompt):
    p = ContextPack('j', 'j', 'j', 1000)
    return create_llm_fn(provider='deepseek', model='deepseek-chat')(p, prompt)
judge = LLMJudge(judge_call, 'deepseek-chat')
runner = ExperimentRunner(store, hybrid, mmu, executor, judge, config)

with open('eval_data/benchmark_v2.json', encoding='utf-8') as f:
    queries = json.load(f)['queries']

NQ = min(len(queries), 30)  # Use 30 queries for BGE experiment
print(f'\nBGE-M3 Experiment: {NQ} queries x 10 conditions = {NQ*10} LLM calls')
print(f'Estimated: {NQ*10*4/60:.0f} min (~${NQ*10*4000*0.14/1e6:.3f})\n')

t0 = time.time()
report = runner.run(queries[:NQ], quiet=True)
elapsed = time.time() - t0

print(f'\nCompleted in {elapsed/60:.1f} min, {len(report.runs)} runs\n')

# Print results table
print(f'{"Condition":20s} | {"Tokens In":>9s} | {"Judge":>5s} | {"Correct":>7s} | {"Halluc":>6s} | {"Citation":>8s}')
print('-' * 70)
for cond in report.conditions:
    cr = report.condition_runs(cond)
    toks = [r.tokens_input for r in cr]
    if not toks:
        continue
    info = report.judge_summary.get(cond, {})
    print(f'{cond:20s} | {sum(toks)/len(toks):9.0f} | '
          f'{info.get("overall",0):5.2f} | {info.get("correctness",0):7.2f} | '
          f'{info.get("hallucination",0):6.2f} | {info.get("citation_accuracy",0):8.2f}')

# Save results
with open('eval_data/experiment_bge_results.json', 'w', encoding='utf-8') as f:
    json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
print('\nResults: eval_data/experiment_bge_results.json')

# Statistical report
print(report.statistical_tests[:800])
