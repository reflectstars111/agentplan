"""ARD Phase 1 — Context MMU Verification Prototype.

Usage:
    python -m ard ingest --dir eval_data/       # index knowledge base
    python -m ard benchmark                      # run experiments
    python -m ard query "your question"          # single query test
    python -m ard serve --port 8000              # start API (minimal)

Modes:
    ingest:    Parse and index documents into the knowledge base.
    benchmark: Run Baseline 1, Baseline 2, and Proposed experiments.
    query:     Run a single query through the full ARD pipeline.
    serve:     Start a minimal FastAPI server.
"""

import argparse
import os
import sys
from datetime import datetime, timezone


def build_phase1(llm_provider: str = "mock", embed_provider: str = "bge",
                  api_key: str = ""):
    """Build the Phase 1 stack: KnowledgeStore + HybridRetriever + ContextMMU + Executor."""
    if api_key:
        import os
        os.environ["DEEPSEEK_API_KEY"] = api_key
        os.environ["OPENAI_API_KEY"] = api_key  # LLM factory checks this too
    else:
        api_key = None  # Let LLM factory fall back to env vars
    from ard.infra.config import Config
    from ard.infra.db import Database
    from ard.store.knowledge_store import KnowledgeStore
    from ard.retriever.vector_index import VectorIndex
    from ard.retriever.query_planner import QueryPlanner
    from ard.retriever.reranker import Reranker
    from ard.retriever.hybrid import HybridRetriever
    from ard.context.token_budgeter import TokenBudgeter
    from ard.context.mmu import ContextMMU
    from ard.runtime.executor import Executor

    config = Config()
    if embed_provider == "bge":
        config.embedding_dim = 1024
    else:
        config.embedding_dim = 1536

    db = Database(config.db_path)
    db.init_schema()

    # Embedding
    if embed_provider == "bge":
        from src.embedding import create_bge_embed_fn
        try:
            embed_fn = create_bge_embed_fn()
            print(f"Embedding: BGE-M3 (1024-dim)")
        except Exception as e:
            print(f"BGE-M3 failed ({e}), using mock")
            from src.embedding import create_mock_embed_fn
            embed_fn = create_mock_embed_fn(dim=config.embedding_dim)
    else:
        from src.embedding import create_mock_embed_fn
        embed_fn = create_mock_embed_fn(dim=config.embedding_dim)

    # Vector index
    vector_index = VectorIndex(dim=config.embedding_dim, index_path=config.vector_index_path)

    # Knowledge store
    knowledge_store = KnowledgeStore(
        db=db, vector_index=vector_index, embed_fn=embed_fn,
        data_dir=config.file_store_path,
    )

    # Rebuild FAISS from SQLite if needed
    if vector_index.count == 0 and knowledge_store.count_chunks() > 0:
        chunks = knowledge_store.get_chunks("") or []
        all_chunks = []
        seen = set()
        for src in knowledge_store.list_sources():
            for c in knowledge_store.get_chunks(src["source_id"]):
                if c["chunk_id"] not in seen:
                    seen.add(c["chunk_id"])
                    all_chunks.append(c)
        if all_chunks:
            vector_index.rebuild_from_db(all_chunks, embed_fn)
            vector_index.persist()
            print(f"Rebuilt FAISS index: {len(all_chunks)} chunks")

    # Retriever
    reranker = Reranker(config)
    query_planner = QueryPlanner()
    hybrid = HybridRetriever(knowledge_store, query_planner, reranker)

    # Context MMU
    budgeter = TokenBudgeter(config)
    mmu = ContextMMU(budgeter, config)

    # LLM
    if llm_provider == "mock":
        llm_fn = None
    else:
        from src.llm.llm_factory import create_llm_fn
        model = config.llm_model
        llm_fn = create_llm_fn(provider=llm_provider, model=model, api_key=api_key)

    executor = Executor(llm_fn)

    return knowledge_store, hybrid, mmu, executor, config


def cmd_ingest(args):
    """Ingest documents into the knowledge base."""
    import glob

    store, _, _, _, config = build_phase1(args.llm, args.embed, args.api_key)

    data_dir = args.dir or "eval_data"
    if not os.path.isdir(data_dir):
        print(f"Error: directory not found: {data_dir}")
        sys.exit(1)

    # Find all ingestible files
    patterns = ["*.md", "*.txt", "*.pdf", "*.py", "*.docx"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(data_dir, "**", pat), recursive=True))

    if not files:
        print(f"No files found in {data_dir}")
        return

    print(f"Found {len(files)} files to ingest.")

    for filepath in files:
        try:
            _ingest_file(filepath, store)
        except Exception as e:
            print(f"  SKIP {filepath}: {e}")

    print(f"Ingestion complete. {store.count_chunks()} chunks indexed.")


def _ingest_file(filepath: str, store) -> None:
    """Ingest a single file: read → chunk → index."""
    import uuid
    from pathlib import Path

    path = Path(filepath)
    ext = path.suffix.lower()
    source_id = f"src_{uuid.uuid4().hex[:8]}"

    # Read content
    if ext in (".md", ".txt", ".py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        source_type = {"md": "markdown", "txt": "text", "py": "code"}.get(ext, "text")
    elif ext == ".pdf":
        from src.parsing.pdf_parser import parse_pdf
        text = parse_pdf(str(path)) or path.read_text(encoding="utf-8", errors="replace")
        source_type = "pdf"
    elif ext == ".docx":
        from src.parsing.word_parser import parse_docx
        text = parse_docx(str(path)) or path.read_text(encoding="utf-8", errors="replace")
        source_type = "docx"
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        source_type = "text"

    if not text or not text.strip():
        print(f"  EMPTY: {filepath}")
        return

    # Simple chunking: split by paragraphs, max ~500 chars per chunk
    paragraphs = text.split("\n\n")
    chunks = []
    buffer = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(buffer) + len(p) < 2000:
            buffer += p + "\n\n"
        else:
            if buffer.strip():
                chunks.append(buffer.strip())
            buffer = p + "\n\n"
    if buffer.strip():
        chunks.append(buffer.strip())

    # If only 1 chunk, try line-based splitting
    if len(chunks) <= 1:
        lines = text.split("\n")
        chunks = []
        buffer = ""
        for line in lines:
            if len(buffer) + len(line) < 2000:
                buffer += line + "\n"
            else:
                if buffer.strip():
                    chunks.append(buffer.strip())
                buffer = line + "\n"
        if buffer.strip():
            chunks.append(buffer.strip())

    chunk_dicts = [
        {
            "text": c,
            "summary": c[:200] + "..." if len(c) > 200 else c,
            "location": {"file": path.name},
            "keywords": [],
            "source_type": source_type,
            "file_name": path.name,
            "trust_level": "external_untrusted",
        }
        for c in chunks
    ]

    count = store.index_chunks(chunk_dicts, source_id)
    print(f"  OK  {filepath} → {count} chunks")


def cmd_benchmark(args):
    """Run the full ARD experiment suite.

    Supports:
      --dataset PATH    JSON benchmark dataset (default: eval_data/benchmark_v1.json)
      --export PATH     Export results to JSON for charting
      --full            Run ablation + multi-turn (requires StateStore)
      --llm PROVIDER    LLM provider for real experiments
    """
    store, hybrid, mmu, executor, config = build_phase1(args.llm, args.embed, args.api_key)

    if store.count_chunks() == 0:
        print("No chunks indexed. Run 'python -m ard ingest' first.")
        sys.exit(1)

    from ard.eval.benchmark import (load_dataset, Benchmark, print_comparison,
                                    export_results)

    dataset_path = getattr(args, "dataset", "") or "eval_data/benchmark_v1.json"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Falling back to sample queries.")
        from ard.eval.benchmark import EvalQuery
        queries = []
        for i in range(15):
            queries.append(EvalQuery(
                query_id=f"sample_{i}", query=f"Test query {i}",
                expected_keywords=["test"], category="factoid",
            ))
        scenarios = []
    else:
        queries, scenarios = load_dataset(dataset_path)
        print(f"Dataset: {dataset_path}")
        print(f"  Single-turn queries: {len(queries)}")
        print(f"  Multi-turn scenarios: {len(scenarios)}")

    print(f"Knowledge base: {len(store.list_sources())} sources, {store.count_chunks()} chunks")

    # Build benchmark
    bench = Benchmark(store, hybrid, mmu, executor, config=config)

    do_full = getattr(args, "full", False)

    if do_full:
        # Full experiment suite
        print("\nRunning full experiment suite...")
        results = bench.run_all(queries, scenarios if scenarios else None)

        for report in results["single_turn"].values():
            print(f"\n{report.summary()}")

        print_comparison(results["single_turn"])

        if "ablation" in results and results["ablation"]:
            print("\n=== ABLATION STUDY ===")
            for label, report in results["ablation"].items():
                print(f"  {label}: TE={report.avg_token_efficiency:.4f}")

        if "multi_turn" in results and results["multi_turn"]:
            print("\n=== MULTI-TURN SCENARIOS ===")
            for mt in results["multi_turn"]:
                print(f"  {mt.name} [{mt.condition}]: consistency={mt.consistency_score:.4f}, "
                      f"completion={mt.completion_rate:.2f}")
    else:
        # Quick single-turn only
        print("\nRunning single-turn experiments...")
        reports = bench.run_single_turn(queries)
        for report in reports.values():
            print(f"\n{report.summary()}")
        print_comparison(reports)
        results = {"single_turn": reports}

    # Export if requested
    export_path = getattr(args, "export", "") or ""
    if export_path:
        export_results(results, export_path)
        print(f"\nResults exported to: {export_path}")
    elif do_full:
        # Auto-export full results
        auto_path = f"eval_data/results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        export_results(results, auto_path)
        print(f"\nResults auto-exported to: {auto_path}")


def cmd_query(args):
    """Run a single query through the full ARD pipeline."""
    store, hybrid, mmu, executor, config = build_phase1(args.llm, args.embed, args.api_key)
    import time

    t0 = time.time()
    candidates = hybrid.retrieve(args.query)
    t1 = time.time()

    context_pack = mmu.assemble(
        query=args.query,
        retrieval_results=candidates,
        system_instruction=(
            "You are a research assistant. Answer questions based ONLY on "
            "the provided context. Cite source references when possible."
        ),
    )

    response = executor.think(context_pack)
    t2 = time.time()

    print(f"\n{'='*60}")
    print(f"Query: {args.query}")
    print(f"{'='*60}")
    print(f"\nRetrieved: {len(candidates)} candidates ({1000*(t1-t0):.0f}ms)")
    print(f"Context: {context_pack.total_tokens_used()} tokens, {len(context_pack.sections)} sections")
    print(f"Sources: {context_pack.source_refs}")
    print(f"\n--- Answer ---")
    print(response.answer)
    print(f"\n--- Stats ---")
    print(f"Latency: {1000*(t2-t0):.0f}ms (retrieve {1000*(t1-t0):.0f}ms + assemble+llm {1000*(t2-t1):.0f}ms)")


def cmd_serve(args):
    """Start a minimal FastAPI server."""
    store, hybrid, mmu, executor, config = build_phase1(args.llm, args.embed, args.api_key)

    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="ARD Phase 1", version="0.1.0")

    class QueryRequest(BaseModel):
        query: str

    class UploadRequest(BaseModel):
        text: str
        source_name: str = "uploaded_text"

    @app.get("/health")
    async def health():
        return {"status": "ok", "chunks": store.count_chunks()}

    @app.post("/query")
    async def query(req: QueryRequest):
        candidates = hybrid.retrieve(req.query)
        context_pack = mmu.assemble(req.query, candidates)
        response = executor.think(context_pack)
        return {
            "answer": response.answer,
            "sources": context_pack.source_refs,
            "tokens_used": context_pack.total_tokens_used(),
        }

    @app.post("/upload")
    async def upload(req: UploadRequest):
        import uuid
        source_id = f"src_{uuid.uuid4().hex[:8]}"
        chunks = [{
            "text": req.text,
            "source_type": "text",
            "file_name": req.source_name,
            "trust_level": "user_provided_data",
        }]
        count = store.index_chunks(chunks, source_id)
        return {"status": "indexed", "source_id": source_id, "chunks": count}

    @app.get("/sources")
    async def list_sources():
        return {"sources": store.list_sources()}

    print(f"Server: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser(description="ARD Phase 1 — Context MMU Verification")
    parser.add_argument("--llm", default="mock", choices=["mock", "openai", "deepseek", "anthropic"])
    parser.add_argument("--model", default="", help="Model name override")
    parser.add_argument("--embed", default="bge", choices=["bge", "mock"])
    parser.add_argument("--api-key", default="", help="LLM API key")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")

    subparsers = parser.add_subparsers(dest="command", help="Command")

    ingest_p = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_p.add_argument("--dir", default="eval_data", help="Data directory")

    bench_p = subparsers.add_parser("benchmark", help="Run experiments")
    bench_p.add_argument("--dataset", default="eval_data/benchmark_v1.json",
                         help="Path to benchmark dataset JSON")
    bench_p.add_argument("--export", default="",
                         help="Export results to JSON file")
    bench_p.add_argument("--full", action="store_true",
                         help="Run full suite (ablation + multi-turn)")

    query_p = subparsers.add_parser("query", help="Single query")
    query_p.add_argument("query", help="The query text")

    serve_p = subparsers.add_parser("serve", help="Start API server")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
