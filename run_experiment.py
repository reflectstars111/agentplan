"""Run the full ARD E1 experiment: 48 queries x 10 conditions.

REQUIRE: set DEEPSEEK_API_KEY environment variable before running.
"""
import os, json, time, uuid, sys

if not os.environ.get("DEEPSEEK_API_KEY"):
    print("ERROR: DEEPSEEK_API_KEY not set. Run: set DEEPSEEK_API_KEY=sk-...")
    exit(1)

from ard.infra.config import Config
from ard.infra.db import Database
from ard.store.knowledge_store import KnowledgeStore
from ard.retriever.vector_index import VectorIndex
from src.embedding import create_mock_embed_fn
from src.llm.llm_factory import create_llm_fn
from ard.retriever.reranker import Reranker
from ard.retriever.query_planner import QueryPlanner
from ard.retriever.hybrid import HybridRetriever
from ard.context.token_budgeter import TokenBudgeter
from ard.context.mmu import ContextMMU
from ard.runtime.executor import Executor
from ard.eval.experiment import ExperimentRunner
from ard.eval.judge import LLMJudge

config = Config(db_path='data/ard_exp.db', vector_index_path='data/ard_exp.faiss', embedding_dim=1536)
db = Database(config.db_path); db.init_schema()
embed_fn = create_mock_embed_fn(dim=1536)
vi = VectorIndex(dim=1536, index_path=config.vector_index_path)
store = KnowledgeStore(db, vi, embed_fn, config.file_store_path)

# Compact knowledge base covering all 3 domains
docs = [
    'MemGPT (2023) two-tier memory: Main Context (RAM) with system instructions and FIFO conversation queue, External Context (disk) with recall storage and archival storage via vector DB. LLM controls paging via function calls.',
    'Letta evolved from MemGPT Sep 2024. Memory Blocks: structured labeled editable units within context window. Achieved 74% on LoCoMo benchmark beating Mem0 68.5%. Sleeptime compute for idle reflection.',
    'RAPTOR (Stanford 2024) hierarchical tree datastore via recursive GMM clustering, SBERT embedding, LLM summarization. 20% improvement on QuALITY. Tree traversal or collapsed tree retrieval.',
    'LongMem (NeurIPS 2023) decoupled: frozen backbone LLM encodes to KV cache, trainable SideNet adapter retrieves and fuses. Solves memory staleness. Up to 65K token cache.',
    'MemLong (2024) extends context 4K to 80K on single 3090 GPU via external retriever with fine-grained retrieval attention. Partial training only.',
    'ARD Context MMU 6 steps: RETRIEVE (hybrid), FILTER (dedup+trust), RANK (priority), COMPRESS (budget), ASSEMBLE (ContextPack), BUDGET (token allocation). Scoring: 0.35*semantic+0.20*keyword+0.15*entity+0.10*recency+0.10*importance+0.10*structure-0.10*token-0.20*trust.',
    'ARD TransactionManager optimistic locking: begin/read_set/verify/commit/rollback. Records keys+seq_num at read, verifies no concurrent writes at commit.',
    'ARD EventStore immutable write-ahead log. seq_num = MVCC version. replay() reconstructs state at any point. Events never modified or deleted.',
    'Self-RAG (ICLR 2024) trains LLM with reflection tokens for retrieval decisions and critique. HyDE generates hypothetical document via LLM then retrieves real documents.',
    'PostgreSQL MVCC: xmin/xmax transaction IDs, VACUUM reclaims dead tuples. CockroachDB: distributed SQL, Raft consensus, serializable snapshot isolation.',
    'WAL records changes sequentially to log before applying to data files. Crash recovery via replay. Point-in-time recovery. Streaming replication.',
    'ACID: Atomicity (rollback), Consistency (constraints), Isolation (MVCC/locking), Durability (WAL). B-Tree stores data in balanced pages matching disk blocks.',
    'Python asyncio single-threaded cooperative scheduler. Coroutines yield at await. TaskGroup structured concurrency. Rust async zero-cost state machines with tokio work-stealing.',
    'Actor model: private state, async message passing, sequential processing. Erlang/OTP supervision trees. Deadlock prevention via lock ordering and timeouts.',
    'Optimistic locking checks at commit via version comparison. Pessimistic locking acquires locks upfront. MVCC snapshot isolation prevents dirty reads.',
]

for i, doc in enumerate(docs):
    store.index_chunks([{'text': doc, 'source_type': 'text', 'file_name': f'kb_{i}.txt', 'trust_level': 'user_provided_data'}], f'src_{uuid.uuid4().hex[:6]}')
print(f'Knowledge: {store.count_chunks()} chunks, {len(store.list_sources())} sources')

# Real LLM pipeline
llm_fn = create_llm_fn(provider='deepseek', model='deepseek-chat')
executor = Executor(llm_fn)
hybrid = HybridRetriever(store, QueryPlanner(), Reranker(config))
mmu = ContextMMU(TokenBudgeter(config), config)

# Judge
from ard.context.pack import ContextPack, ContextSection
def judge_call(prompt):
    p = ContextPack('j', 'j', 'j', 1000)
    return create_llm_fn(provider='deepseek', model='deepseek-chat')(p, prompt)
judge = LLMJudge(judge_call, 'deepseek-chat')
runner = ExperimentRunner(store, hybrid, mmu, executor, judge, config)

with open('eval_data/benchmark_v2.json', encoding='utf-8') as f:
    queries = json.load(f)['queries']

NQ = min(len(queries), 48)
print(f'Running: {NQ} queries x 10 conditions = {NQ*10} LLM calls (~{NQ*10*4/60:.0f} min)')
t0 = time.time()
report = runner.run(queries[:NQ], quiet=True)
elapsed = time.time() - t0

print(f'\nCompleted in {elapsed/60:.1f} min, {len(report.runs)} runs')

# Print results
for cond in report.conditions:
    cr = report.condition_runs(cond)
    scores = report.condition_scores(cond, 'overall')
    toks = [r.tokens_input for r in cr]
    if scores and toks:
        sc = sum(scores)/len(scores)
        tk = sum(toks)/len(toks)
        print(f'  {cond:20s} | tokens={tk:6.0f} | judge_overall={sc:.2f}')

# Save
with open('eval_data/experiment_results_v1.json', 'w', encoding='utf-8') as f:
    json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

# Statistical report
print(report.statistical_tests[:500])
print('\nResults saved.')
