"""Convenience script to run all ARD experiments.

REQUIRE: Set DEEPSEEK_API_KEY environment variable before running.
    set DEEPSEEK_API_KEY=sk-...
    python run_all_experiments.py [e1|e2|e3|all|analyze]
"""
import os, sys, json

if not os.environ.get("DEEPSEEK_API_KEY"):
    print("ERROR: DEEPSEEK_API_KEY not set.")
    print("Run: set DEEPSEEK_API_KEY=sk-...")
    sys.exit(1)

def build_stack(embed="bge"):
    """Build the full Phase 1+2 stack."""
    if embed == "bge":
        from src.embedding import create_bge_embed_fn
        embed_fn = create_bge_embed_fn()
        dim = 1024
    else:
        from src.embedding import create_mock_embed_fn
        embed_fn = create_mock_embed_fn(dim=1536)
        dim = 1536

    from ard.infra.config import Config
    from ard.infra.db import Database
    from ard.store.knowledge_store import KnowledgeStore
    from ard.retriever.vector_index import VectorIndex
    from src.llm.llm_factory import create_llm_fn
    from ard.retriever.reranker import Reranker
    from ard.retriever.query_planner import QueryPlanner
    from ard.retriever.hybrid import HybridRetriever
    from ard.context.token_budgeter import TokenBudgeter
    from ard.context.mmu import ContextMMU
    from ard.runtime.executor import Executor

    config = Config(db_path='data/ard_all.db', vector_index_path='data/ard_all.faiss',
                    embedding_dim=dim)
    db = Database(config.db_path); db.init_schema()
    vi = VectorIndex(dim=dim, index_path=config.vector_index_path)
    store = KnowledgeStore(db, vi, embed_fn, config.file_store_path)

    # Ingest knowledge base if empty
    if store.count_chunks() == 0:
        print("Ingesting knowledge base...")
        import uuid
        docs = [
            'MemGPT (2023) two-tier memory: Main Context (RAM) and External Context (disk). LLM controls paging via function calls. Letta framework (Sep 2024) adds Memory Blocks and sleeptime compute.',
            'RAPTOR (Stanford 2024) builds hierarchical tree via recursive GMM clustering, SBERT embedding, and LLM summarization. 20% improvement on QuALITY.',
            'LongMem (NeurIPS 2023) uses frozen backbone LLM + trainable SideNet adapter. Solves memory staleness. Up to 65K token KV cache.',
            'MemLong (2024) extends context from 4K to 80K on single 3090 GPU via external retriever with fine-grained retrieval attention.',
            'ARD Context MMU: 6-step pipeline (retrieve, filter, rank, compress, assemble, budget). Scoring: 0.35*semantic+0.20*keyword+0.15*entity+0.10*recency+0.10*importance+0.10*structure-0.10*token-0.20*trust.',
            'ARD TransactionManager: optimistic locking with begin/read_set/verify/commit/rollback. EventStore: immutable WAL with seq_num as MVCC version.',
            'HyDE generates hypothetical document via LLM then retrieves real documents. Self-RAG uses reflection tokens for retrieval decisions.',
            'PostgreSQL MVCC: xmin/xmax tuple versioning, VACUUM cleanup. CockroachDB: distributed SQL with Raft consensus, serializable isolation.',
            'WAL records changes sequentially before applying to data files. Crash recovery via replay. ACID: Atomicity, Consistency, Isolation, Durability.',
            'Python asyncio: single-threaded cooperative scheduler. Rust async: zero-cost state machines with tokio work-stealing executor.',
            'Actor model: private state, async message passing, sequential processing, supervision trees. Erlang/OTP, Akka, actix.',
        ]
        for i, doc in enumerate(docs):
            store.index_chunks([{'text': doc, 'source_type': 'text', 'file_name': f'doc_{i}.txt', 'trust_level': 'user_provided_data'}], f'src_{uuid.uuid4().hex[:6]}')

    llm_fn = create_llm_fn(provider='deepseek', model='deepseek-chat')
    executor = Executor(llm_fn)
    hybrid = HybridRetriever(store, QueryPlanner(), Reranker(config))
    mmu = ContextMMU(TokenBudgeter(config), config)
    return store, hybrid, mmu, executor, config


def run_e1():
    """Run E1: 10-condition retrieval + answer quality experiment."""
    print("=" * 60)
    print("E1: 10-CONDITION EXPERIMENT")
    print("=" * 60)

    _, hybrid, mmu, executor, config = build_stack("bge")

    from ard.eval.experiment import ExperimentRunner
    from ard.eval.judge import LLMJudge
    from src.llm.llm_factory import create_llm_fn

    judge_llm = create_llm_fn(provider='deepseek', model='deepseek-chat')
    from ard.context.pack import ContextPack
    def judge_call(prompt):
        p = ContextPack('j','j','j',1000)
        return judge_llm(p, prompt)

    judge = LLMJudge(judge_call, 'deepseek-chat')
    runner = ExperimentRunner(_, hybrid, mmu, executor, judge, config)

    queries_path = 'eval_data/benchmark_v2.json'
    if not os.path.exists(queries_path):
        queries_path = 'eval_data/benchmark_v1.json'
    with open(queries_path, encoding='utf-8') as f:
        queries = json.load(f)['queries'] if 'queries' in json.load(open(queries_path, encoding='utf-8')) else json.load(open(queries_path, encoding='utf-8')).get('single_turn_queries', [])

    import time
    print(f"Running {len(queries)} queries x 10 conditions...")
    t0 = time.time()
    report = runner.run(queries)
    print(f"Done in {(time.time()-t0)/60:.1f} min")

    out_path = 'eval_data/experiment_bge_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")

    from ard.eval.statistics import generate_full_report
    scores = {c: report.condition_scores(c, 'overall') for c in report.conditions}
    print(generate_full_report(scores, 'Judge Overall'))


def run_e2():
    """Run E2: Multi-turn consistency experiment (BGE-M3 + improved state injection)."""
    print("=" * 60)
    print("E2: MULTI-TURN CONSISTENCY (H1) — BGE-M3")
    print("=" * 60)

    store, hybrid, mmu, executor, config = build_stack("bge")

    # Verify embedding mode
    from src.embedding import _get_bge_path
    bge_path = _get_bge_path()
    is_bge = "bge-m3" in str(bge_path).lower() or "BAAI" in str(bge_path)
    print(f"Embedding: {'BGE-M3 (1024-dim)' if is_bge else 'MOCK'}, FAISS: {store.vector_index.count} vectors, Knowledge: {store.count_chunks()} chunks")

    # Build StateStore stack
    from ard.infra.db import Database
    from ard.store.event_store import EventStore
    from ard.store.projections import Projections
    from ard.store.state_store import StateStore
    from ard.store.trace_store import TraceStore
    from ard.store.transaction import TransactionManager

    db = Database('data/ard_e2.db'); db.init_schema()
    proj = Projections()
    es = EventStore(db, proj)
    ss = StateStore(es)
    proj.register("state.created", ss.apply_event)
    proj.register("state.updated", ss.apply_event)
    ts = TraceStore(es)
    txn_mgr = TransactionManager(es)

    from ard.eval.multi_turn import (MultiTurnExperimentRunner, MultiTurnScenario,
                                      ScenarioResult, TurnResult,
                                      print_multi_turn_report, MULTI_TURN_SCENARIOS)
    # Patch with improved state injection
    class BGERunner(MultiTurnExperimentRunner):
        """Enhanced runner with structured state injection via StateStore."""

        def _run_scenario(self, scenario, with_state=False):
            turn_results = []
            trace_id = f"trace_mt_{scenario.scenario_id}"

            for turn in scenario.turns:
                query = turn["query"]
                enriched_query = query

                # Load prior state from StateStore (structured, not raw text)
                if with_state and self.state_store:
                    prior_keys = []
                    for prior_turn in range(1, turn["turn"]):
                        key = f"task:mt_{scenario.scenario_id}_turn{prior_turn}"
                        prior_keys.append(key)

                    if prior_keys:
                        state_entries = []
                        for key in prior_keys:
                            val = self.state_store.read(key)
                            if val:
                                summary = val.get("response_summary", val.get("query", ""))[:250]
                                state_entries.append(f"[Turn {val.get('turn', '?')}]: {summary}")
                        if state_entries:
                            state_block = "PRIOR SESSION STATE:\n" + "\n".join(state_entries)
                            enriched_query = f"{state_block}\n\nCURRENT QUERY: {query}"

                # Retrieve with enriched query
                candidates = self.hybrid.retrieve(enriched_query)
                context_pack = self.mmu.assemble(enriched_query, candidates, top_k=15)
                resp = self.executor.think(context_pack, enriched_query)

                # Write state if enabled
                if with_state and self.state_store and self.txn_mgr:
                    try:
                        txn = self.txn_mgr.begin()
                        evt = self.state_store.build_event(
                            stream_key=f"task:mt_{scenario.scenario_id}_turn{turn['turn']}",
                            event_type="created",
                            payload={
                                "query": query, "response_summary": resp.answer[:600],
                                "turn": turn["turn"], "scenario": scenario.scenario_id,
                            },
                        )
                        txn.add_event(evt)
                        self.txn_mgr.commit(txn)
                    except RuntimeError:
                        pass

                # Keyword recall
                keywords = turn.get("expected_keywords", [])
                kw_recall = 0.0
                if keywords:
                    ans_lower = resp.answer.lower()
                    matches = sum(1 for kw in keywords if kw.lower() in ans_lower)
                    kw_recall = matches / len(keywords)

                turn_results.append(TurnResult(
                    turn=turn["turn"], query=query,
                    response=resp.answer,
                    tokens_used=context_pack.total_tokens_used(),
                    latency_ms=0,
                    expected_keywords=keywords,
                    keyword_recall=kw_recall,
                ))

            # Score consistency
            consistency = self._score_consistency(scenario, turn_results)
            completion = len([t for t in turn_results if t.response]) / max(len(scenario.turns), 1)
            trace_acc = self._score_state_trace(scenario, turn_results)

            return ScenarioResult(
                scenario_id=scenario.scenario_id,
                name=scenario.name,
                condition="with_state" if with_state else "no_state",
                turns=turn_results,
                consistency_score=consistency,
                state_trace_accuracy=trace_acc,
                completion_rate=completion,
            )

    runner_mt = BGERunner(hybrid, mmu, executor, ss, txn_mgr, ts)

    import time
    t0 = time.time()
    results = runner_mt.run_all()
    elapsed = (time.time()-t0)/60
    print(f"Done in {elapsed:.1f} min")

    out_path = 'eval_data/experiment_e2_bge_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "embedding": "BGE-M3" if is_bge else "MOCK",
            "elapsed_min": round(elapsed, 1),
            "no_state": [s.to_dict() for s in results.get("no_state", [])],
            "with_state": [s.to_dict() for s in results.get("with_state", [])],
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")

    try:
        print(print_multi_turn_report(results))
    except Exception as e:
        print(f"Report generation error: {e}")
        for cond, scenarios in results.items():
            if scenarios:
                avg_c = sum(s.consistency_score for s in scenarios) / max(len(scenarios), 1)
                avg_t = sum(s.state_trace_accuracy for s in scenarios) / max(len(scenarios), 1)
                print(f"  {cond}: consistency={avg_c:.3f}, trace_acc={avg_t:.3f} ({len(scenarios)} scenarios)")


def run_e3():
    """Run E3: 2x2 State vs Context Length experiment."""
    print("=" * 60)
    print("E3: 2x2 STATE x CONTEXT (H3)")
    print("=" * 60)

    store, hybrid, mmu, executor, config = build_stack("bge")
    from src.embedding import _get_bge_path
    is_bge = "bge-m3" in str(_get_bge_path()).lower() or "BAAI" in str(_get_bge_path())
    print(f"Embedding: {'BGE-M3' if is_bge else 'MOCK'}, Knowledge: {store.count_chunks()} chunks")

    # Build StateStore
    from ard.infra.db import Database
    from ard.store.event_store import EventStore
    from ard.store.projections import Projections
    from ard.store.state_store import StateStore
    from ard.store.trace_store import TraceStore
    from ard.store.transaction import TransactionManager

    db = Database('data/ard_e3.db'); db.init_schema()
    proj = Projections()
    es = EventStore(db, proj)
    ss = StateStore(es)
    proj.register("state.created", ss.apply_event)
    proj.register("state.updated", ss.apply_event)
    ts = TraceStore(es)
    txn_mgr = TransactionManager(es)

    from ard.eval.state_vs_context import (
        StateVsContextExperiment, print_2x2_report, power_analysis_2x2,
        ContextConfig, CellResult, TwoByTwoResult,
    )
    from ard.eval.multi_turn import MULTI_TURN_SCENARIOS

    runner = StateVsContextExperiment(hybrid, mmu, executor, ss, txn_mgr, ts)

    import time
    t0 = time.time()
    pa = power_analysis_2x2(0.5)
    print(f"Power analysis: need {pa['recommended_scenarios']} scenarios for d=0.5")
    print(f"Running with {min(3, len(MULTI_TURN_SCENARIOS))} scenarios...")

    result = runner.run(MULTI_TURN_SCENARIOS[:3])
    elapsed = (time.time()-t0)/60
    print(f"Done in {elapsed:.1f} min")

    out_path = 'eval_data/experiment_e3_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "embedding": "BGE-M3" if is_bge else "MOCK",
            "elapsed_min": round(elapsed, 1),
            "results": result.to_dict(),
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")
    print(print_2x2_report(result))


def run_e4():
    """Run E4: External system comparison (LangChain RAG + ARD)."""
    print("=" * 60)
    print("E4: EXTERNAL COMPARISON — ARD vs LangChain RAG")
    print("=" * 60)

    store, hybrid, mmu, executor, config = build_stack("bge")
    print(f"Knowledge: {store.count_chunks()} chunks")

    from ard.eval.experiment import ExperimentRunner
    from ard.eval.judge import LLMJudge
    from ard.eval.external_compare import ExternalComparisonRunner, print_comparison_report
    from src.llm.llm_factory import create_llm_fn
    from ard.context.pack import ContextPack

    judge_llm = create_llm_fn(provider='deepseek', model='deepseek-chat')
    def judge_call(prompt):
        p = ContextPack('j','j','j',1000)
        return judge_llm(p, prompt)
    judge = LLMJudge(judge_call, 'deepseek-chat')
    ard_runner = ExperimentRunner(store, hybrid, mmu, executor, judge, config)
    ext_runner = ExternalComparisonRunner(ard_runner, judge)

    queries_path = 'eval_data/benchmark_v2.json'
    if not os.path.exists(queries_path):
        queries_path = 'eval_data/benchmark_v1.json'
    with open(queries_path, encoding='utf-8') as f:
        data = json.load(f)
    queries = data.get('queries', data.get('single_turn_queries', []))
    queries = queries[:15]  # Limit for comparison

    import time
    t0 = time.time()
    report = ext_runner.run(queries, systems=["ard", "langchain_rag"])
    print(f"Done in {(time.time()-t0)/60:.1f} min")
    print(print_comparison_report(report))

    out_path = 'eval_data/experiment_e4_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")


def run_setup():
    """Set up multi-domain knowledge base."""
    print("=" * 60)
    print("SETUP: Multi-Domain Knowledge Base")
    print("=" * 60)

    store, hybrid, mmu, executor, config = build_stack("bge")
    print(f"Before: {store.count_chunks()} chunks")
    print("Building AI Systems + DB + Programming knowledge base...")
    print("(LLM-generated domain docs require API key)")

    total = store.count_chunks()
    print(f"After: {total} chunks (core docs loaded)")
    print("Run with --llm deepseek to add LLM-generated domain expansions")


def run_analyze():
    """Analyze existing experiment results."""
    from ard.eval.results_viz import quick_analyze
    for path in ['eval_data/experiment_bge_results.json', 'eval_data/experiment_results_v1.json']:
        if os.path.exists(path):
            print(f"\nAnalyzing: {path}")
            print(quick_analyze(path))
            break
    else:
        print("No results files found. Run e1 first.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "e1":
        run_e1()
    elif cmd == "e2":
        run_e2()
    elif cmd == "e3":
        run_e3()
    elif cmd == "e4":
        run_e4()
    elif cmd == "setup":
        run_setup()
    elif cmd == "raptor":
        run_e1_raptor()
    elif cmd == "analyze":
        run_analyze()
    elif cmd == "all":
        run_e1()
        run_e2()
        run_e3()
    else:
        print(f"Usage: python run_all_experiments.py [e1|e2|e3|e4|raptor|setup|analyze|all]")
        print(f"  e1      - 10-condition retrieval + answer quality")
        print(f"  e2      - Multi-turn consistency (H1)")
        print(f"  e3      - 2x2 State x Context (H3)")
        print(f"  e4      - External comparison (ARD vs LangChain RAG)")
        print(f"  raptor  - E1 with RAPTOR summarization enabled")
        print(f"  setup   - Build cross-domain knowledge base")
        print(f"  analyze - Analyze existing results")
        print(f"  all     - Run E1+E2+E3")


def run_e1_raptor():
    """Run E1 with RAPTOR-style recursive summarization enabled."""
    print("=" * 60)
    print("E1 + RAPTOR: ContextMMU with recursive summarization")
    print("=" * 60)

    from ard.context.mmu import enable_raptor
    from src.llm.llm_factory import create_llm_fn
    llm = create_llm_fn(provider='deepseek', model='deepseek-chat')
    enable_raptor(llm)
    print("RAPTOR summarization enabled")

    # Rebuild stack with RAPTOR-enabled MMU
    run_e1()  # MMU will pick up RAPTOR from global state
