# Agent-OS

A Von Neumann-inspired Multi-Agent Runtime System — built with Python, SQLite, FAISS, and React.

> 面向多智能体系统的类冯诺依曼运行时架构，通过分层记忆、上下文虚拟化、多级索引、任务调度和结果验证机制，使 Agent 能够在有限上下文窗口内稳定调用长期历史与大规模外部资料。

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                   Agent-OS Runtime                      │
│                                                         │
│  POST /query ─→ AgentRuntime (simple mode)              │
│  POST /task  ─→ Controller → Scheduler (task graph)     │
│                                                         │
│  ┌─ Control Layer ────────────────────────────────┐    │
│  │ IntentDecoder → Planner → TaskGraph(DAG)        │    │
│  │ Scheduler (topological + parallel exec)         │    │
│  │ AgentRegistry + SharedBlackboard                │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─ Index Layer ──────────────────────────────────┐    │
│  │ HybridRetriever (8-component scoring)           │    │
│  │   ├─ VectorIndex  (FAISS + BGE-M3, 1024-dim)   │    │
│  │   ├─ KeywordIndex (SQLite FTS5, BM25)           │    │
│  │   └─ StructureIndex (hierarchical tree)         │    │
│  │ CodeParser (tree-sitter: Python/JS/TS)          │    │
│  │ PDFParser  (OpenDataLoader + PyMuPDF fallback)  │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─ Context Layer ────────────────────────────────┐    │
│  │ TokenBudgeter (tiktoken) + ContextMMU           │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─ Security ─────────────────────────────────────┐    │
│  │ InputSanitizer + PermissionChecker + AuditLog   │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  SQLite (12 tables)  ·  FAISS  ·  React GUI             │
└────────────────────────────────────────────────────────┘
```

## Features

| Category | Capability |
|----------|------------|
| **Multi-Level Memory** | L0(input) → L1(dialog cache) → L2(working) → L3(long-term) → L4(files) |
| **Hybrid Retrieval** | FAISS vector search + FTS5 keyword search + 8-component weighted scoring |
| **Code Understanding** | tree-sitter AST parsing for Python/JS/TS — function/class/method extraction |
| **PDF Parsing** | OpenDataLoader (structured Markdown/JSON) with PyMuPDF fallback |
| **Context MMU** | Dedup → sort → token budget → source annotation → ContextPack |
| **Task Graph** | DAG-based task decomposition with topological scheduling |
| **Multi-Agent** | AgentRegistry routing + SharedBlackboard + 6-stage Merger |
| **Real LLM** | OpenAI / DeepSeek / Anthropic via unified factory (model selector in GUI) |
| **Local Embedding** | BGE-M3 (1024-dim, 100+ languages, pure CPU) |
| **Security** | Prompt injection detection, Agent permission model, audit logging |
| **GitHub Indexing** | Clone + auto-index repos via GitPython |
| **Observability** | Full execution trace per request, structured audit queries |
| **Evaluation** | precision@k, recall@k, MRR, nDCG, hit@k against 5 scenarios |

## Quick Start

### Prerequisites

- Python ≥ 3.10
- Java ≥ 11 (for OpenDataLoader PDF parsing; optional, PyMuPDF fallback works without it)
- Node.js ≥ 18 (for GUI build)

### Install

```bash
git clone https://github.com/reflectstars111/agentplan.git
cd agentplan
pip install -r requirements.txt

# Build GUI
cd gui && npm install && npm run build && cd ..
```

### Run

```bash
# Mock mode (no API key needed, for testing)
python -m src

# DeepSeek + BGE-M3 (recommended)
set DEEPSEEK_API_KEY=sk-your-key
python -m src --llm deepseek --model deepseek-chat

# OpenAI + BGE-M3
set OPENAI_API_KEY=sk-your-key
python -m src --llm openai --model gpt-4o

# Open http://127.0.0.1:8000
```

### CLI Options

```
python -m src [options]

  --llm    mock|openai|deepseek|anthropic  (default: mock)
  --model  Model name override             (default: deepseek-chat / gpt-4o)
  --embed  bge|mock                        (default: bge)
  --port   Server port                     (default: 8000)
  --host   Bind address                    (default: 127.0.0.1)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload text content for indexing |
| `POST` | `/upload/file` | Upload file (PDF, code, markdown) |
| `POST` | `/upload/github` | Clone and index a GitHub repository |
| `POST` | `/query` | Simple Q&A (single-agent pipeline) |
| `POST` | `/task` | Task graph execution (multi-agent pipeline) |
| `GET` | `/trace/{id}` | Retrieve execution trace |
| `GET` | `/health` | Health check |

### Example

```bash
# Upload knowledge
curl -X POST http://localhost:8000/upload \
  -H "Content-Type: application/json" \
  -d '{"content": "FastAPI is a modern Python web framework...", "source_name": "fastapi.txt"}'

# Index a GitHub repo
curl -X POST http://localhost:8000/upload/github \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/user/repo"}'

# Simple query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FastAPI?", "model": "deepseek-chat"}'

# Task graph mode
curl -X POST http://localhost:8000/task \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze the authentication system in this codebase", "model": "deepseek-v4-pro"}'
```

## Project Structure

```
agentplan/
├── src/
│   ├── __main__.py           # Entry point (python -m src)
│   ├── config.py             # Global configuration
│   ├── embedding.py          # BGE-M3 + OpenAI + mock embed functions
│   ├── api/                  # FastAPI application + routes
│   ├── models/               # 10 data models (Memory, Chunk, Context, Task, Intent, Agent, etc.)
│   ├── db/                   # SQLite connection + migrations (12 tables)
│   ├── storage/              # FileStore, MemoryStore, Chunker, CodeChunker
│   ├── index/                # VectorIndex, KeywordIndex, StructureIndex, HybridRetriever
│   ├── context/              # TokenBudgeter, ContextMMU
│   ├── runtime/              # AgentRuntime, Controller, Scheduler, Verifier, WritebackGate, etc.
│   ├── llm/                  # LLM factory (OpenAI/DeepSeek/Anthropic) + prompt templates
│   ├── parsing/              # CodeParser (tree-sitter), PDFParser (OpenDataLoader)
│   └── sources/              # GithubSource (clone + index)
├── gui/                      # React + Vite + TypeScript frontend
├── models/bge-m3/            # BGE-M3 model files (local)
├── eval/                     # Evaluation scenarios + metrics
├── tests/                    # 34 test files, 273 tests
├── agent_os_initial_plan.md  # Original architectural plan
└── requirements.txt
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| Database | SQLite + FTS5 |
| Vector Search | FAISS (IndexFlatIP + IndexIDMap) |
| Embedding | BGE-M3 (1024-dim, local CPU via FlagEmbedding) |
| Code Parsing | tree-sitter (Python, JavaScript, TypeScript) |
| PDF Parsing | OpenDataLoader PDF + PyMuPDF fallback |
| LLM API | OpenAI / DeepSeek / Anthropic (unified factory) |
| API Server | FastAPI + Uvicorn |
| Frontend | React 18 + TypeScript + Vite |
| Testing | pytest (273 tests, 100% pass) |
| Git Integration | GitPython |

## Configuration

Key settings in `src/config.py`:

```python
# Embedding
embedding_dim: int = 1024          # BGE-M3 output dimension

# Retrieval
weight_semantic: float = 0.35
weight_keyword: float = 0.20
weight_entity: float = 0.15
weight_recency: float = 0.10
weight_importance: float = 0.10
weight_structural: float = 0.10
penalty_token_cost: float = 0.10
penalty_trust: float = 0.20

# Context
default_token_budget: int = 24000
top_k_after_rerank: int = 15

# Task execution
task_max_retries: int = 2
parallel_enabled: bool = False
max_parallel_agents: int = 4

# Write-back gate
writeback_min_score: float = 0.5
```

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_hybrid_retriever.py -v
python -m pytest tests/parsing/test_code_parser.py -v
```

273 tests, 100% pass rate.

## License

MIT
