"""Agent-OS startup: build full stack and start server.

Usage:
    python -m src                          # start with BGE-M3 embedding + mock LLM
    python -m src --llm deepseek           # BGE-M3 embedding + DeepSeek LLM
    python -m src --llm openai             # BGE-M3 embedding + OpenAI LLM
    python -m src --embed mock             # use mock embeddings (fast startup)
    python -m src --port 8000 --host 0.0.0.0
"""

import argparse
import os

BGE_DIM = 1024  # BGE-M3 output dimension


def initialize_vector_index(config, file_store, embed_fn):
    """Load a persisted FAISS index or rebuild it from SQLite chunks."""
    from src.index.vector_index import VectorIndex

    try:
        vector_index = VectorIndex(
            dim=config.embedding_dim,
            index_path=config.vector_index_path,
        )
    except (OSError, ValueError):
        vector_index = VectorIndex(dim=config.embedding_dim)
        vector_index.index_path = config.vector_index_path

    if vector_index.count == 0 and file_store.count_chunks() > 0:
        vector_index.rebuild_from_db(file_store.list_chunks(), embed_fn)
        vector_index.persist()
    return vector_index


def build_runtime(
    llm_provider="mock",
    llm_model="",
    embed_provider="bge",
    config=None,
):
    """Build the full AgentRuntime with all components wired."""
    from src.config import Config
    if config is None:
        config = Config(
            embedding_dim=BGE_DIM if embed_provider == "bge" else 1536
        )
    elif embed_provider == "bge":
        config.embedding_dim = BGE_DIM

    from src.db import Database
    from src.storage.file_store import FileStore
    from src.storage.memory_store import MemoryStore
    from src.storage.conversation_cache import ConversationCache
    from src.storage.dependency_graph import DependencyGraph
    from src.index.keyword_index import KeywordIndex
    from src.index.hybrid_retriever import HybridRetriever
    from src.index.entity_index import EntityIndex
    from src.index.query_planner import QueryPlanner
    from src.index.reranker import Reranker
    from src.index.structure_index import StructureIndex
    from src.context.token_budgeter import TokenBudgeter
    from src.context.mmu import ContextMMU
    from src.context.page_fault import ContextPageFault
    from src.runtime.verifier import Verifier
    from src.runtime.writeback_gate import WritebackGate
    from src.runtime.trace_logger import TraceLogger
    from src.runtime.permission_checker import PermissionChecker
    from src.runtime.input_sanitizer import InputSanitizer
    from src.runtime.audit_log import AuditLog
    from src.runtime.tool_router import ToolRegistry, ToolRouter
    from src.runtime.agent_runtime import AgentRuntime

    db = Database(config.db_path)
    db.init_schema()

    # Embedding: BGE-M3 (local, default) or mock (fast)
    from src.embedding import create_mock_embed_fn, create_bge_embed_fn
    if embed_provider == "bge":
        try:
            embed_fn = create_bge_embed_fn()
            print(f"Embedding: BGE-M3 (local CPU, {BGE_DIM}-dim, 100+ languages)")
        except Exception as e:
            print(f"Warning: BGE-M3 load failed ({e}), falling back to mock")
            config.embedding_dim = 1536
            embed_fn = create_mock_embed_fn(dim=1536)
    else:
        embed_fn = create_mock_embed_fn(dim=config.embedding_dim)
        print("Embedding: mock (deterministic hash)")

    # LLM function
    from src.llm.llm_factory import create_llm_fn
    if llm_provider == "mock":
        llm_fn = None  # use default mock
    else:
        model = llm_model or {"openai": "gpt-4o", "deepseek": "deepseek-chat"}.get(llm_provider, "gpt-4o")
        llm_fn = create_llm_fn(provider=llm_provider, model=model)

    # Build core index/context components first (shared dependencies)
    file_store = FileStore(db, config.file_store_path)
    vector_index = initialize_vector_index(config, file_store, embed_fn)
    structure_index = StructureIndex(db)
    retriever = HybridRetriever(
        vector_index,
        KeywordIndex(db),
        db,
        config,
        structure_index=structure_index,
        reranker=Reranker(),
        query_planner=QueryPlanner(),
    )

    # P1-P3 components: activate entity/dependency/persistence/permission/page-fault
    entity_index = EntityIndex(db)
    dep_graph = DependencyGraph(db)
    conv_cache = ConversationCache(max_turns=20)
    perm_checker = PermissionChecker()
    trace_logger = TraceLogger(db)
    input_sanitizer = InputSanitizer()
    audit_log = AuditLog(db, trace_logger)
    tool_router = ToolRouter(
        ToolRegistry(),
        perm_checker,
        trace_logger=trace_logger,
    )
    page_fault = ContextPageFault(
        retriever=retriever,
        mmu=ContextMMU(TokenBudgeter(), config),
    )

    runtime = AgentRuntime(
        file_store=file_store,
        memory_store=MemoryStore(db),
        retriever=retriever,
        mmu=ContextMMU(TokenBudgeter(), config),
        verifier=Verifier(),
        writeback_gate=WritebackGate(),
        trace_logger=trace_logger,
        config=config,
        embed_fn=embed_fn,
        entity_index=entity_index,
        dependency_graph=dep_graph,
        conversation_cache=conv_cache,
        permission_checker=perm_checker,
        page_fault=page_fault,
        input_sanitizer=input_sanitizer,
        audit_log=audit_log,
        tool_router=tool_router,
    )
    if llm_fn:
        runtime.llm_fn = llm_fn

    return runtime


def build_controller(runtime):
    """Build the task controller with shared multi-agent coordination."""
    from src.models.agent import AgentProcess, AgentRole, AgentStatus
    from src.models.blackboard import SharedBlackboard
    from src.runtime.agent_registry import AgentRegistry
    from src.runtime.agent_runtime import AgentRuntime
    from src.runtime.controller import Controller
    from src.runtime.intent_decoder import IntentDecoder
    from src.runtime.interrupt_handler import InterruptHandler
    from src.runtime.merger import Merger
    from src.runtime.planner import Planner
    from src.runtime.scheduler import Scheduler

    verifier_runtime = AgentRuntime(
        file_store=runtime.file_store,
        memory_store=runtime.memory_store,
        retriever=runtime.retriever,
        mmu=runtime.mmu,
        verifier=runtime.verifier,
        writeback_gate=runtime.writeback_gate,
        trace_logger=runtime.trace_logger,
        config=runtime.config,
        embed_fn=runtime.embed_fn,
        llm_fn=runtime.llm_fn,
        agent_id="agent_verifier_001",
        role="verifier",
        memory_scope=runtime.memory_scope,
        page_fault=runtime.page_fault,
        entity_index=runtime.entity_index,
        permission_checker=runtime.permission_checker,
        input_sanitizer=runtime.input_sanitizer,
        audit_log=runtime.audit_log,
        tool_router=runtime.tool_router,
    )

    registry = AgentRegistry()
    registry.register(
        "worker",
        AgentProcess(
            agent_id=runtime.agent_id,
            role=AgentRole.WORKER,
            status=AgentStatus.READY,
        ),
        runtime,
    )
    registry.register(
        "verifier",
        AgentProcess(
            agent_id=verifier_runtime.agent_id,
            role=AgentRole.VERIFIER,
            status=AgentStatus.READY,
        ),
        verifier_runtime,
    )
    blackboard = SharedBlackboard()
    scheduler = Scheduler(
        agent_runtime=runtime,
        agent_registry=registry,
        blackboard=blackboard,
        trace_logger=runtime.trace_logger,
        page_fault=runtime.page_fault,
    )
    return Controller(
        agent_runtime=runtime,
        intent_decoder=IntentDecoder(),
        planner=Planner(),
        scheduler=scheduler,
        trace_logger=runtime.trace_logger,
        agent_registry=registry,
        blackboard=blackboard,
        merger=Merger(),
        interrupt_handler=InterruptHandler(),
    )


def main():
    parser = argparse.ArgumentParser(description="Agent-OS Runtime")
    parser.add_argument("--llm", default="mock", choices=["mock", "openai", "deepseek", "anthropic"])
    parser.add_argument("--model", default="", help="Model name override")
    parser.add_argument("--embed", default="bge", choices=["bge", "mock"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"Agent-OS: LLM={args.llm}, Embed={args.embed}")
    runtime = build_runtime(args.llm, args.model, args.embed)

    controller = build_controller(runtime)

    from src.api.main import create_app
    app = create_app(runtime, controller=controller)

    import uvicorn
    print(f"Server: http://{args.host}:{args.port}")
    print(f"GUI:    http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
