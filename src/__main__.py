"""Agent-OS startup: build full stack and start server.

Usage:
    python -m src                          # start with mock LLM
    python -m src --llm openai             # use OpenAI
    python -m src --llm deepseek           # use DeepSeek
    python -m src --port 8000 --host 0.0.0.0
"""

import argparse
import os

def build_runtime(llm_provider="mock", llm_model=""):
    """Build the full AgentRuntime with all components wired."""
    from src.config import Config, config
    from src.db import Database
    from src.storage.file_store import FileStore
    from src.storage.memory_store import MemoryStore
    from src.index.vector_index import VectorIndex
    from src.index.keyword_index import KeywordIndex
    from src.index.hybrid_retriever import HybridRetriever
    from src.context.token_budgeter import TokenBudgeter
    from src.context.mmu import ContextMMU
    from src.runtime.verifier import Verifier
    from src.runtime.writeback_gate import WritebackGate
    from src.runtime.trace_logger import TraceLogger
    from src.runtime.agent_runtime import AgentRuntime

    db = Database(config.db_path)
    db.init_schema()

    # Embedding function
    from src.embedding import create_mock_embed_fn, create_openai_embed_fn
    if llm_provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        embed_fn = create_openai_embed_fn(api_key=api_key) if api_key else create_mock_embed_fn(dim=1536)
        if not api_key:
            print("Warning: OPENAI_API_KEY not set, using mock embeddings")
    else:
        embed_fn = create_mock_embed_fn(dim=1536)
        if llm_provider == "deepseek":
            print("Note: DeepSeek has no embeddings API. Using mock embeddings (deterministic hash).")

    # LLM function
    from src.llm.llm_factory import create_llm_fn
    if llm_provider == "mock":
        llm_fn = None  # use default mock
    else:
        model = llm_model or {"openai": "gpt-4o", "deepseek": "deepseek-chat"}.get(llm_provider, "gpt-4o")
        llm_fn = create_llm_fn(provider=llm_provider, model=model)

    runtime = AgentRuntime(
        file_store=FileStore(db),
        memory_store=MemoryStore(db),
        retriever=HybridRetriever(VectorIndex(dim=config.embedding_dim), KeywordIndex(db), db, config),
        mmu=ContextMMU(TokenBudgeter(), config),
        verifier=Verifier(),
        writeback_gate=WritebackGate(),
        trace_logger=TraceLogger(db),
        config=config,
        embed_fn=embed_fn,
    )
    if llm_fn:
        runtime.llm_fn = llm_fn

    return runtime


def main():
    parser = argparse.ArgumentParser(description="Agent-OS Runtime")
    parser.add_argument("--llm", default="mock", choices=["mock", "openai", "deepseek", "anthropic"])
    parser.add_argument("--model", default="", help="Model name override")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"Agent-OS starting with LLM: {args.llm}")
    runtime = build_runtime(args.llm, args.model)

    from src.api.main import create_app
    app = create_app(runtime)

    import uvicorn
    print(f"Server: http://{args.host}:{args.port}")
    print(f"GUI:    http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
