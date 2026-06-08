# Agent-OS MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a verifiable MVP baseline — single Agent + multi-level memory + Context MMU + write-back gate + trace logger — and prove it outperforms plain RAG on 3+ task types.

**Architecture:** Python monolith with layered design: models (dataclasses/schemas) → storage (SQLite + filesystem) → index (FAISS + BM25) → context (MMU + token budgeter) → runtime (Agent, Verifier, WritebackGate, TraceLogger) → API (FastAPI). Each layer depends only on the layer below. TDD throughout: red-green-refactor per module.

**Tech Stack:** Python 3.11+, SQLite, FAISS, BM25 (rank_bm25), PyMuPDF, FastAPI, pytest, dataclasses_json

**Reference Docs:**
- [agent_os_initial_plan.md](../../../agent_os_initial_plan.md) — Full architectural spec
- [PLAN.md](../../../PLAN.md) — MVP scope decisions

---

## File Structure (Complete)

```
agentplan/
├── src/
│   ├── __init__.py
│   ├── config.py                       # Global config
│   ├── db/
│   │   ├── __init__.py
│   │   ├── connection.py               # SQLite connection + schema init
│   │   └── migrations.py               # DDL statements
│   ├── models/
│   │   ├── __init__.py
│   │   ├── memory.py                   # MemoryItem dataclass
│   │   ├── chunk.py                    # DocumentChunk dataclass
│   │   ├── context.py                  # ContextPack, ContextSection
│   │   ├── trace.py                    # TraceStep, Trace
│   │   ├── agent.py                    # AgentPCB dataclass
│   │   └── task.py                     # TaskTCB dataclass
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── memory_store.py             # Memory CRUD (L2/L3)
│   │   ├── file_store.py               # File ingest orchestration
│   │   └── chunker.py                  # Text → DocumentChunk[]
│   ├── index/
│   │   ├── __init__.py
│   │   ├── vector_index.py             # FAISS wrapper
│   │   ├── keyword_index.py            # BM25 via SQLite FTS
│   │   └── hybrid_retriever.py         # Combined retriever
│   ├── context/
│   │   ├── __init__.py
│   │   ├── mmu.py                      # Context MMU
│   │   └── token_budgeter.py           # Token estimation + budget allocation
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── agent_runtime.py            # Single-agent execution loop
│   │   ├── verifier.py                 # Source verification
│   │   ├── writeback_gate.py           # Memory write-back decisions
│   │   └── trace_logger.py             # Execution trace recorder
│   └── api/
│       ├── __init__.py
│       ├── main.py                     # FastAPI app factory
│       └── routes.py                   # /upload, /query, /trace endpoints
├── tests/
│   ├── __init__.py
│   ├── conftest.py                     # Shared fixtures (temp DB, sample data)
│   ├── test_memory_store.py
│   ├── test_chunker.py
│   ├── test_file_store.py
│   ├── test_vector_index.py
│   ├── test_keyword_index.py
│   ├── test_hybrid_retriever.py
│   ├── test_token_budgeter.py
│   ├── test_context_mmu.py
│   ├── test_verifier.py
│   ├── test_writeback_gate.py
│   ├── test_trace_logger.py
│   └── test_agent_runtime.py
├── eval/
│   ├── __init__.py
│   ├── scenarios.py                    # 5 task scenario definitions
│   ├── test_queries.py                 # Standard queries + expected chunks
│   └── metrics.py                      # precision, recall, MRR, nDCG
├── requirements.txt
├── agent_os_initial_plan.md
└── PLAN.md
```

---

## Milestone 0: Project Scaffold + Schema + Evaluation Scenarios

**Goal:** Runnable project skeleton, all data models defined, 5 evaluation scenarios documented, test infrastructure in place.

### Task 0.1: Project scaffold + requirements

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `eval/__init__.py`

- [ ] **Step 1: Write requirements.txt**

```text
# Core
dataclasses_json==0.6.7
pydantic==2.10.6

# Storage
# SQLite3 is stdlib

# Index
faiss-cpu==1.9.0
rank-bm25==0.2.2
numpy==1.26.4

# File parsing
PyMuPDF==1.25.3

# API
fastapi==0.115.6
uvicorn==0.34.0

# Testing
pytest==8.3.4
pytest-asyncio==0.25.0

# Token estimation (simple char-based, no heavy ML dep)
tiktoken==0.8.0
```

- [ ] **Step 2: Write `src/config.py`**

```python
"""Global configuration for Agent-OS MVP."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Application configuration loaded from environment or defaults."""

    # Database
    db_path: str = "data/agent_os.db"

    # File storage
    file_store_path: str = "data/files"
    chunk_size: int = 500        # tokens per chunk
    chunk_overlap: int = 50      # token overlap between chunks

    # Vector index
    embedding_dim: int = 1536    # OpenAI text-embedding-3-small
    vector_index_path: str = "data/vector_index.faiss"

    # Context MMU
    default_token_budget: int = 24000
    max_retrieval_candidates: int = 50
    top_k_after_rerank: int = 15

    # Retrieval weights (per agent_os_initial_plan.md §5.2)
    weight_semantic: float = 0.35
    weight_keyword: float = 0.20
    weight_entity: float = 0.15
    weight_recency: float = 0.10
    weight_importance: float = 0.10
    weight_structural: float = 0.10
    penalty_token_cost: float = 0.10
    penalty_trust: float = 0.20

    # Write-back gate
    writeback_min_score: float = 0.5     # Minimum WriteScore to persist
    writeback_user_confirm_threshold: float = 0.7  # Above this, ask user

    # Trace
    trace_enabled: bool = True

    def __post_init__(self):
        """Ensure data directories exist."""
        for p in [self.db_path, self.file_store_path, self.vector_index_path]:
            Path(p).parent.mkdir(parents=True, exist_ok=True)


# Singleton
config = Config()
```

- [ ] **Step 3: Write `tests/conftest.py`**

```python
"""Shared test fixtures."""

import os
import tempfile
import pytest
from pathlib import Path
from src.config import Config


@pytest.fixture
def temp_config() -> Config:
    """Config pointing at temp directories for isolated tests."""
    tmp = tempfile.mkdtemp()
    return Config(
        db_path=f"{tmp}/test.db",
        file_store_path=f"{tmp}/files",
        vector_index_path=f"{tmp}/vec.index",
    )


@pytest.fixture
def sample_pdf_path() -> Path:
    """Path to a minimal test PDF. Created on first use."""
    return Path(__file__).parent / "fixtures" / "sample.pdf"


@pytest.fixture
def sample_markdown_path(tmp_path: Path) -> Path:
    """Create a temporary sample markdown file."""
    p = tmp_path / "sample.md"
    p.write_text("""# Test Document

## Section 1

This is the first section. It contains important information about the project.

## Section 2

This is the second section. It contains more details about implementation.

### Subsection 2.1

Here are some code examples.

```python
def hello():
    print("Hello, World!")
```

## Section 3

Final conclusions and next steps.
""")
    return p
```

- [ ] **Step 4: Create empty `__init__.py` files**

Run: `touch src/__init__.py tests/__init__.py eval/__init__.py`

- [ ] **Step 5: Verify project structure**

Run: `cd F:/agentplan && python -c "from src.config import Config; c = Config(); print(f'DB path: {c.db_path}')"`
Expected: Prints DB path without errors.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/__init__.py src/config.py tests/__init__.py tests/conftest.py eval/__init__.py
git commit -m "feat: project scaffold with config and test fixtures"
```

---

### Task 0.2: Define core data models (MemoryItem, DocumentChunk, ContextPack, TraceStep)

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/memory.py`
- Create: `src/models/chunk.py`
- Create: `src/models/context.py`
- Create: `src/models/trace.py`

- [ ] **Step 1: Write `src/models/memory.py`**

```python
"""MemoryItem — the core memory record. Maps to agent_os_initial_plan.md §4.3."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from dataclasses_json import dataclass_json, config


class MemoryType(str, Enum):
    PROJECT_STATE = "project_state"
    USER_PREFERENCE = "user_preference"
    DECISION = "decision"
    FILE_SUMMARY = "file_summary"
    CONVERSATION_SUMMARY = "conversation_summary"
    INTERMEDIATE_RESULT = "intermediate_result"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


@dataclass_json
@dataclass
class MemoryItem:
    """A single memory record in L2/L3 storage."""

    memory_id: str
    type: MemoryType
    content: str
    summary: str = ""
    entities: list[str] = field(default_factory=list)
    importance: float = 0.5       # 0.0–1.0
    confidence: float = 0.5       # 0.0–1.0
    source: str = "conversation"  # conversation | file | agent | user
    scope: str = "project"        # project | user | session
    status: MemoryStatus = MemoryStatus.ACTIVE
    version: int = 1
    source_ref: Optional[str] = None  # e.g. "file:repo_001/README.md"
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        metadata=config(encoder=lambda d: d.isoformat(), decoder=lambda s: datetime.fromisoformat(s)),
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        metadata=config(encoder=lambda d: d.isoformat(), decoder=lambda s: datetime.fromisoformat(s)),
    )

    def to_keywords(self) -> list[str]:
        """Extract keywords for keyword index."""
        import re
        words = set()
        for field in [self.content, self.summary] + self.entities:
            tokens = re.findall(r'[\w一-鿿]+', field.lower())
            words.update(t for t in tokens if len(t) > 1)
        return list(words)
```

- [ ] **Step 2: Write `src/models/chunk.py`**

```python
"""DocumentChunk — a slice of an external document. Maps to agent_os_initial_plan.md §4.4."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from dataclasses_json import dataclass_json, config


class ChunkType(str, Enum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CODE = "code"
    FIGURE = "figure"
    HEADING = "heading"


class TrustLevel(str, Enum):
    TRUSTED_INSTRUCTION = "trusted_instruction"
    USER_INSTRUCTION = "user_instruction"
    INTERNAL_MEMORY = "internal_memory"
    USER_PROVIDED_DATA = "user_provided_data"
    EXTERNAL_UNTRUSTED = "external_untrusted"
    TOOL_OBSERVATION = "tool_observation"
    AGENT_GENERATED = "agent_generated"


@dataclass_json
@dataclass
class ChunkLocation:
    """Where this chunk lives in its source document."""
    page: Optional[int] = None
    section: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None


@dataclass_json
@dataclass
class DocumentChunk:
    """A chunk from an external document (PDF, code, web, etc.)."""

    chunk_id: str
    source_id: str               # e.g. "file:paper_001.pdf"
    source_type: str              # pdf | markdown | code | web | text
    text: str
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    location: ChunkLocation = field(default_factory=ChunkLocation)
    chunk_type: ChunkType = ChunkType.PARAGRAPH
    embedding_id: Optional[str] = None  # index in FAISS
    trust_level: TrustLevel = TrustLevel.EXTERNAL_UNTRUSTED
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        metadata=config(encoder=lambda d: d.isoformat(), decoder=lambda s: datetime.fromisoformat(s)),
    )
```

- [ ] **Step 3: Write `src/models/context.py`**

```python
"""ContextPack — the assembled context sent to the LLM. Maps to agent_os_initial_plan.md §9.2."""

from dataclasses import dataclass, field
from typing import Optional
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ContextSection:
    """A named section within a context pack."""
    name: str                    # e.g. "current_task", "working_memory", "retrieved_evidence"
    tokens: int                  # token count of items
    priority: int                # 1 = highest
    items: list[dict] = field(default_factory=list)  # [{source_ref, trust_level, text}]


@dataclass_json
@dataclass
class ContextPack:
    """The complete context assembled for one LLM inference call."""

    context_id: str
    task_id: str
    agent_id: str
    budget: int                  # total token budget
    sections: list[ContextSection] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)  # all source references used
    used_tokens: int = 0
    created_at: str = ""         # ISO datetime

    def remaining_budget(self) -> int:
        return max(0, self.budget - self.used_tokens)

    def add_section(self, section: ContextSection) -> bool:
        """Add section if budget remains. Returns True if added."""
        if self.used_tokens + section.tokens > self.budget:
            return False
        self.sections.append(section)
        self.used_tokens += section.tokens
        return True
```

- [ ] **Step 4: Write `src/models/trace.py`**

```python
"""TraceStep and Trace — execution audit log. Maps to agent_os_initial_plan.md §15.3."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from dataclasses_json import dataclass_json, config


class StepType(str, Enum):
    INTENT_DECODE = "intent_decode"
    RETRIEVE_MEMORY = "retrieve_memory"
    RETRIEVE_FILE = "retrieve_file"
    CONTEXT_ASSEMBLE = "context_assemble"
    LLM_REASONING = "llm_reasoning"
    TOOL_CALL = "tool_call"
    VERIFY = "verify"
    WRITE_MEMORY = "write_memory"
    RESPOND = "respond"


class StepStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass_json
@dataclass
class TraceStep:
    """One step in an execution trace."""

    step_id: str
    type: StepType
    input: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    status: StepStatus = StepStatus.SUCCESS
    error: Optional[str] = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        metadata=config(encoder=lambda d: d.isoformat(), decoder=lambda s: datetime.fromisoformat(s)),
    )


@dataclass_json
@dataclass
class Trace:
    """Complete execution trace for one user request."""

    trace_id: str
    request_id: str
    steps: list[TraceStep] = field(default_factory=list)

    def add_step(self, step: TraceStep) -> None:
        self.steps.append(step)

    def to_summary(self) -> str:
        """Human-readable trace summary."""
        lines = [f"Trace {self.trace_id} (request: {self.request_id})"]
        for s in self.steps:
            status_icon = "✓" if s.status == StepStatus.SUCCESS else "✗" if s.status == StepStatus.FAILED else "○"
            lines.append(f"  {status_icon} [{s.type.value}] {s.timestamp.isoformat()}")
            if s.error:
                lines.append(f"     Error: {s.error}")
        return "\n".join(lines)
```

- [ ] **Step 5: Write `src/models/__init__.py`**

```python
from src.models.memory import MemoryItem, MemoryType, MemoryStatus
from src.models.chunk import DocumentChunk, ChunkType, TrustLevel, ChunkLocation
from src.models.context import ContextPack, ContextSection
from src.models.trace import Trace, TraceStep, StepType, StepStatus

__all__ = [
    "MemoryItem", "MemoryType", "MemoryStatus",
    "DocumentChunk", "ChunkType", "TrustLevel", "ChunkLocation",
    "ContextPack", "ContextSection",
    "Trace", "TraceStep", "StepType", "StepStatus",
]
```

- [ ] **Step 6: Verify models import correctly**

Run: `cd F:/agentplan && python -c "from src.models import MemoryItem, DocumentChunk, ContextPack, Trace; print('All models imported OK')"`
Expected: Prints "All models imported OK"

- [ ] **Step 7: Commit**

```bash
git add src/models/
git commit -m "feat: add core data models (MemoryItem, DocumentChunk, ContextPack, Trace)"
```

---

### Task 0.3: Define 5 evaluation scenarios

**Files:**
- Create: `eval/scenarios.py`
- Create: `eval/test_queries.py`

- [ ] **Step 1: Write `eval/scenarios.py`**

```python
"""Five task scenarios for MVP evaluation. Each scenario defines input, expected output type,
and success criteria. Per PLAN.md §Test Plan."""

from dataclasses import dataclass, field


@dataclass
class EvalScenario:
    """A single evaluation scenario."""
    scenario_id: str
    name: str
    description: str
    task_type: str                # doc_qa | code_locator | project_continuity | memory_assisted | conflict_detection
    input_description: str        # What the user provides
    expected_output_type: str     # What the system should produce
    success_criteria: list[str]   # How to judge success
    sample_queries: list[str] = field(default_factory=list)
    expected_source_refs: list[str] = field(default_factory=list)  # Expected source files/chunks


# The 5 defined scenarios:

SCENARIO_1_DOC_QA = EvalScenario(
    scenario_id="s1_doc_qa",
    name="PDF Document Q&A",
    description="Upload a PDF paper, ask questions about its content. System must retrieve relevant chunks, "
                "answer with citations, and not hallucinate beyond the document.",
    task_type="doc_qa",
    input_description="A 5-15 page PDF paper (e.g., an arXiv CS paper)",
    expected_output_type="Natural language answer with inline source references (page/section)",
    success_criteria=[
        "Answer is factually grounded in the PDF",
        "Every claim has a source reference (page or section)",
        "No hallucinated facts beyond the document",
        "Can handle multi-hop questions (info spans multiple sections)",
    ],
    sample_queries=[
        "What is the main contribution of this paper?",
        "What dataset was used for evaluation?",
        "How does the proposed method compare to baselines?",
        "What are the limitations mentioned by the authors?",
    ],
)

SCENARIO_2_CODE_LOCATOR = EvalScenario(
    scenario_id="s2_code_locator",
    name="Code Repository Understanding",
    description="Upload a code directory. System indexes it at file/symbol level. "
                "User asks where specific functionality is implemented.",
    task_type="code_locator",
    input_description="A directory with 10-50 source files (Python project)",
    expected_output_type="File paths and line ranges pointing to relevant code, with brief explanation",
    success_criteria=[
        "Correctly identifies the file(s) containing the target functionality",
        "Returns specific line ranges, not whole files",
        "Can trace call chains across files",
        "Distinguishes between definition site and call sites",
    ],
    sample_queries=[
        "Where is the main entry point of this project?",
        "Which file handles database connections?",
        "Find all places where error handling wraps API calls",
        "What functions call the 'process_data' function?",
    ],
)

SCENARIO_3_PROJECT_CONTINUITY = EvalScenario(
    scenario_id="s3_project_continuity",
    name="Long Conversation Project Continuity",
    description="Multi-turn conversation about a project. System must use working memory "
                "to track decisions across turns without re-explaining context each time.",
    task_type="project_continuity",
    input_description="A sequence of 5-10 related user requests across a simulated project session",
    expected_output_type="Responses that build on prior turns, referencing previous decisions without re-stating them",
    success_criteria=[
        "References prior decisions from earlier turns without user re-prompting",
        "Does not lose context when conversation exceeds 10 turns",
        "Working memory is updated with key decisions",
        "Inconsistent new info triggers conflict detection, not silent overwrite",
    ],
    sample_queries=[
        "Turn 1: Let's design a user authentication system. I want JWT-based auth.",
        "Turn 2: Add role-based access control to the design.",
        "Turn 5: Now that we have auth and roles, design the API endpoint structure.",
        "Turn 8: Given our auth decisions, how should we handle token refresh?",
    ],
)

SCENARIO_4_MEMORY_ASSISTED = EvalScenario(
    scenario_id="s4_memory_assisted",
    name="Historical Memory Assisted Writing",
    description="System has stored long-term memories (user preferences, past decisions). "
                "New task should proactively retrieve and apply relevant memories.",
    task_type="memory_assisted",
    input_description="Pre-seeded long-term memories + a new writing/design task",
    expected_output_type="Output that incorporates relevant past preferences without being explicitly told",
    success_criteria=[
        "Retrieves relevant long-term memories without explicit user mention",
        "Applies past preferences to new output",
        "Does not retrieve irrelevant memories (no pollution)",
        "Outdated memories are detected and flagged, not blindly applied",
    ],
    sample_queries=[
        "Write a README for my new Python library.",
        "Design the API for a data processing service.",
        "Create a project structure for a machine learning experiment.",
    ],
)

SCENARIO_5_CONFLICT_DETECTION = EvalScenario(
    scenario_id="s5_conflict_detection",
    name="Conflict Information Detection",
    description="User provides new information that contradicts stored memory. "
                "System must detect the conflict and ask for clarification.",
    task_type="conflict_detection",
    input_description="Pre-seeded memories + user input that contradicts one or more memories",
    expected_output_type="Flag the conflict, show both old and new information, ask user to resolve",
    success_criteria=[
        "Detects when new input contradicts stored memory",
        "Presents both old and new information clearly",
        "Does NOT silently overwrite the old memory",
        "Does NOT flag non-conflicts as conflicts (low false positive rate)",
    ],
    sample_queries=[
        "Actually, let's switch the database from PostgreSQL to MongoDB.",
        "I've decided to use Rust instead of Python for the core engine.",
        "The project deadline is now Q3, not Q2.",
    ],
)

# Registry
ALL_SCENARIOS = [
    SCENARIO_1_DOC_QA,
    SCENARIO_2_CODE_LOCATOR,
    SCENARIO_3_PROJECT_CONTINUITY,
    SCENARIO_4_MEMORY_ASSISTED,
    SCENARIO_5_CONFLICT_DETECTION,
]
```

- [ ] **Step 2: Write `eval/test_queries.py`**

```python
"""Standard test queries with expected results for retrieval evaluation.
Each query maps to expected chunks or source references that a good retriever should find."""

from dataclasses import dataclass, field


@dataclass
class EvalQuery:
    """A single evaluation query with ground truth."""
    query_id: str
    scenario_id: str
    query_text: str
    # Expected chunks (for retrieval eval) — not exact matches, but topics that should appear
    expected_topics: list[str] = field(default_factory=list)
    expected_source_files: list[str] = field(default_factory=list)
    # Minimum number of relevant chunks expected in top-10
    min_relevant_in_top10: int = 1


# Document QA queries (for evaluation against a known PDF)
DOC_QA_QUERIES = [
    EvalQuery(
        query_id="dq_001",
        scenario_id="s1_doc_qa",
        query_text="What is the main contribution of this paper?",
        expected_topics=["contribution", "novel", "proposed method"],
        min_relevant_in_top10=1,
    ),
    EvalQuery(
        query_id="dq_002",
        scenario_id="s1_doc_qa",
        query_text="What dataset was used for evaluation?",
        expected_topics=["dataset", "evaluation", "benchmark", "experiment"],
        min_relevant_in_top10=1,
    ),
    EvalQuery(
        query_id="dq_003",
        scenario_id="s1_doc_qa",
        query_text="How does the proposed method compare to baselines?",
        expected_topics=["comparison", "baseline", "outperform", "result", "table"],
        min_relevant_in_top10=2,
    ),
]

# Code locator queries (for evaluation against a known repo)
CODE_LOCATOR_QUERIES = [
    EvalQuery(
        query_id="cl_001",
        scenario_id="s2_code_locator",
        query_text="Where is the main entry point?",
        expected_topics=["main", "entry", "cli", "__main__"],
        expected_source_files=["main.py", "__main__.py", "cli.py"],
        min_relevant_in_top10=1,
    ),
    EvalQuery(
        query_id="cl_002",
        scenario_id="s2_code_locator",
        query_text="Where are database operations defined?",
        expected_topics=["database", "db", "connection", "query", "session"],
        expected_source_files=["db.py", "database.py", "connection.py"],
        min_relevant_in_top10=1,
    ),
]

# Memory continuity queries (simulated multi-turn)
MEMORY_QUERIES = [
    EvalQuery(
        query_id="mc_001",
        scenario_id="s4_memory_assisted",
        query_text="Write a README for my Python library.",
        expected_topics=["readme", "documentation", "python", "library"],
        min_relevant_in_top10=1,
    ),
    EvalQuery(
        query_id="mc_002",
        scenario_id="s4_memory_assisted",
        query_text="What API framework did we decide to use?",
        expected_topics=["fastapi", "api", "framework", "decision"],
        min_relevant_in_top10=1,
    ),
]

ALL_QUERIES = DOC_QA_QUERIES + CODE_LOCATOR_QUERIES + MEMORY_QUERIES
```

- [ ] **Step 3: Commit**

```bash
git add eval/
git commit -m "feat: define 5 evaluation scenarios and standard test queries"
```

---

### Task 0.4: Database schema initialization

**Files:**
- Create: `src/db/__init__.py`
- Create: `src/db/connection.py`
- Create: `src/db/migrations.py`

- [ ] **Step 1: Write `src/db/migrations.py`**

```python
"""DDL statements for Agent-OS MVP. Maps to agent_os_initial_plan.md §21."""

MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT DEFAULT '',
    entities TEXT DEFAULT '[]',       -- JSON array
    importance REAL DEFAULT 0.5,
    confidence REAL DEFAULT 0.5,
    source TEXT DEFAULT 'conversation',
    scope TEXT DEFAULT 'project',
    status TEXT DEFAULT 'active',      -- active | superseded | archived
    version INTEGER DEFAULT 1,
    source_ref TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    text TEXT NOT NULL,
    summary TEXT DEFAULT '',
    keywords TEXT DEFAULT '[]',        -- JSON array
    location_page INTEGER,
    location_section TEXT,
    location_line_start INTEGER,
    location_line_end INTEGER,
    chunk_type TEXT DEFAULT 'paragraph',
    embedding_id TEXT,
    trust_level TEXT DEFAULT 'external_untrusted',
    created_at TEXT NOT NULL
);
"""

TRACES_TABLE = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    steps TEXT DEFAULT '[]',           -- JSON array of TraceStep
    created_at TEXT NOT NULL
);
"""

AGENTS_TABLE = """
CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    status TEXT DEFAULT 'created',
    priority INTEGER DEFAULT 5,
    prompt_id TEXT,
    memory_scope TEXT DEFAULT '{}',    -- JSON
    permissions TEXT DEFAULT '{}',     -- JSON
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

TASKS_TABLE = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    agent_id TEXT,
    parent_task_id TEXT,
    task_type TEXT NOT NULL,
    status TEXT DEFAULT 'created',
    dependencies TEXT DEFAULT '[]',    -- JSON array
    input_refs TEXT DEFAULT '[]',      -- JSON array
    output_ref TEXT,
    priority INTEGER DEFAULT 5,
    created_at TEXT NOT NULL
);
"""

# FTS5 virtual table for keyword search on memories
MEMORIES_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id,
    content,
    summary,
    entities,
    content='memories',
    content_rowid='rowid'
);
"""

# FTS5 virtual table for keyword search on chunks
CHUNKS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id,
    text,
    summary,
    keywords,
    content='chunks',
    content_rowid='rowid'
);
"""

ALL_MIGRATIONS = [
    ("memories", MEMORIES_TABLE),
    ("chunks", CHUNKS_TABLE),
    ("traces", TRACES_TABLE),
    ("agents", AGENTS_TABLE),
    ("tasks", TASKS_TABLE),
    ("memories_fts", MEMORIES_FTS),
    ("chunks_fts", CHUNKS_FTS),
]

# Triggers to keep FTS in sync with base tables
MEMORIES_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, memory_id, content, summary, entities)
    VALUES (new.rowid, new.memory_id, new.content, new.summary, new.entities);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, memory_id, content, summary, entities)
    VALUES ('delete', old.rowid, old.memory_id, old.content, old.summary, old.entities);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, memory_id, content, summary, entities)
    VALUES ('delete', old.rowid, old.memory_id, old.content, old.summary, old.entities);
    INSERT INTO memories_fts(rowid, memory_id, content, summary, entities)
    VALUES (new.rowid, new.memory_id, new.content, new.summary, new.entities);
END;
"""

CHUNKS_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, chunk_id, text, summary, keywords)
    VALUES (new.rowid, new.chunk_id, new.text, new.summary, new.keywords);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, text, summary, keywords)
    VALUES ('delete', old.rowid, old.chunk_id, old.text, old.summary, old.keywords);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, text, summary, keywords)
    VALUES ('delete', old.rowid, old.chunk_id, old.text, old.summary, old.keywords);
    INSERT INTO chunks_fts(rowid, chunk_id, text, summary, keywords)
    VALUES (new.rowid, new.chunk_id, new.text, new.summary, new.keywords);
END;
"""
```

- [ ] **Step 2: Write `src/db/connection.py`**

```python
"""SQLite connection management and schema initialization."""

import sqlite3
import os
from pathlib import Path
from src.db.migrations import ALL_MIGRATIONS, MEMORIES_FTS_TRIGGERS, CHUNKS_FTS_TRIGGERS


class Database:
    """Thin wrapper around SQLite connection with schema management."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        """Get or create a connection."""
        if self._conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def init_schema(self) -> None:
        """Create all tables if they don't exist."""
        conn = self.connect()
        for name, ddl in ALL_MIGRATIONS:
            conn.execute(ddl)
        conn.executescript(MEMORIES_FTS_TRIGGERS)
        conn.executescript(CHUNKS_FTS_TRIGGERS)
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        return self.connect().execute(sql, params)

    def commit(self) -> None:
        if self._conn:
            self._conn.commit()
```

- [ ] **Step 3: Write `src/db/__init__.py`**

```python
from src.db.connection import Database

__all__ = ["Database"]
```

- [ ] **Step 4: Verify schema initialization**

Run: `cd F:/agentplan && python -c "from src.db import Database; db = Database('data/test_init.db'); db.init_schema(); print('Schema OK'); db.close()"`
Expected: Creates `data/test_init.db` and prints "Schema OK"

- [ ] **Step 5: Commit**

```bash
git add src/db/
git commit -m "feat: database schema with FTS5 indexes for memories and chunks"
```

---

## Milestone 1: Storage Infrastructure

**Goal:** Chunk PDFs and markdown files, store memories and chunks in SQLite, basic CRUD operations.

### Task 1.1: Text chunker with overlap

**Files:**
- Create: `src/storage/__init__.py`
- Create: `src/storage/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write failing test `tests/test_chunker.py`**

```python
"""Tests for the chunker module."""

import pytest
from src.storage.chunker import chunk_text, ChunkerConfig


class TestChunkText:
    def test_splits_long_text_into_chunks(self):
        text = "hello world. " * 200  # ~4000 chars
        config = ChunkerConfig(chunk_size=100, chunk_overlap=20)
        chunks = chunk_text(text, source_id="test", config=config)
        assert len(chunks) > 1
        # Each chunk should be roughly within chunk_size chars
        for c in chunks:
            assert len(c.text) <= config.chunk_size + config.chunk_overlap + 50  # some slack

    def test_short_text_returns_single_chunk(self):
        text = "A short piece of text."
        config = ChunkerConfig(chunk_size=1000, chunk_overlap=50)
        chunks = chunk_text(text, source_id="test", config=config)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_empty_text_returns_empty_list(self):
        chunks = chunk_text("", source_id="test")
        assert chunks == []

    def test_chunks_have_unique_ids(self):
        text = "sentence. " * 100
        chunks = chunk_text(text, source_id="doc_1")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunks_have_source_id(self):
        text = "sentence. " * 50
        chunks = chunk_text(text, source_id="my_doc.pdf")
        for c in chunks:
            assert c.source_id == "my_doc.pdf"

    def test_overlap_preserves_context(self):
        text = "AAAA BBBB CCCC DDDD EEEE FFFF GGGG HHHH IIII JJJJ"
        config = ChunkerConfig(chunk_size=20, chunk_overlap=10)
        chunks = chunk_text(text, source_id="test", config=config)
        if len(chunks) >= 2:
            # Last chars of chunk N should appear in chunk N+1
            assert chunks[0].text[-10:] in chunks[1].text or any(
                w in chunks[1].text for w in chunks[0].text.split()[-3:]
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd F:/agentplan && python -m pytest tests/test_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.storage.chunker'`

- [ ] **Step 3: Write `src/storage/chunker.py`**

```python
"""Text chunker with configurable size and overlap. Produces DocumentChunk objects."""

import re
import uuid
from dataclasses import dataclass
from src.models.chunk import DocumentChunk, ChunkType, ChunkLocation, TrustLevel


@dataclass
class ChunkerConfig:
    """Configuration for the chunker."""
    chunk_size: int = 500       # characters per chunk
    chunk_overlap: int = 50     # character overlap between adjacent chunks


DEFAULT_CONFIG = ChunkerConfig()


def chunk_text(
    text: str,
    source_id: str,
    source_type: str = "text",
    config: ChunkerConfig | None = None,
    trust_level: TrustLevel = TrustLevel.EXTERNAL_UNTRUSTED,
) -> list[DocumentChunk]:
    """Split plain text into overlapping DocumentChunks.

    Uses sentence-boundary-aware splitting: splits on paragraph breaks first,
    then on sentence boundaries, falling back to character-level splitting
    for very long sentences.
    """
    if config is None:
        config = DEFAULT_CONFIG

    if not text.strip():
        return []

    # Step 1: Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text.strip())

    chunks: list[DocumentChunk] = []
    current = ""
    line_counter = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph stays within chunk_size, accumulate
        if len(current) + len(para) + 2 <= config.chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            # Flush current chunk
            if current:
                chunks.append(_make_chunk(
                    text=current,
                    source_id=source_id,
                    source_type=source_type,
                    chunk_index=len(chunks),
                    line_start=line_counter,
                    trust_level=trust_level,
                ))
                line_counter += current.count('\n') + 2
                current = ""

            # If paragraph itself exceeds chunk_size, split at sentence boundaries
            if len(para) > config.chunk_size:
                sub_chunks = _split_long_paragraph(
                    para, config, source_id, source_type, trust_level,
                    start_index=len(chunks), start_line=line_counter,
                )
                chunks.extend(sub_chunks)
                line_counter += para.count('\n') + 2
                current = ""
            else:
                current = para

    # Flush remaining
    if current:
        chunks.append(_make_chunk(
            text=current,
            source_id=source_id,
            source_type=source_type,
            chunk_index=len(chunks),
            line_start=line_counter,
            trust_level=trust_level,
        ))

    return chunks


def _split_long_paragraph(
    para: str,
    config: ChunkerConfig,
    source_id: str,
    source_type: str,
    trust_level: TrustLevel,
    start_index: int,
    start_line: int,
) -> list[DocumentChunk]:
    """Split a paragraph that exceeds chunk_size into sentence-boundary chunks with overlap."""
    sentences = re.split(r'(?<=[.!?])\s+', para)
    chunks = []
    current = ""
    line_offset = start_line

    for sent in sentences:
        if len(current) + len(sent) + 1 <= config.chunk_size:
            current = (current + " " + sent).strip() if current else sent
        else:
            if current:
                chunks.append(_make_chunk(
                    text=current, source_id=source_id, source_type=source_type,
                    chunk_index=start_index + len(chunks),
                    line_start=line_offset, trust_level=trust_level,
                ))
                line_offset += current.count('\n') + 1
                # Overlap: keep last `overlap` chars of previous chunk
                if config.chunk_overlap > 0:
                    overlap_text = current[-config.chunk_overlap:]
                    current = overlap_text + " " + sent
                else:
                    current = sent
            else:
                # Single sentence exceeds chunk_size — hard split
                for i in range(0, len(sent), config.chunk_size - config.chunk_overlap):
                    piece = sent[i:i + config.chunk_size]
                    chunks.append(_make_chunk(
                        text=piece, source_id=source_id, source_type=source_type,
                        chunk_index=start_index + len(chunks),
                        line_start=line_offset, trust_level=trust_level,
                    ))
                current = ""
                line_offset += 1

    if current:
        chunks.append(_make_chunk(
            text=current, source_id=source_id, source_type=source_type,
            chunk_index=start_index + len(chunks),
            line_start=line_offset, trust_level=trust_level,
        ))

    return chunks


def _make_chunk(
    text: str,
    source_id: str,
    source_type: str,
    chunk_index: int,
    line_start: int,
    trust_level: TrustLevel,
) -> DocumentChunk:
    """Create a DocumentChunk with computed metadata."""
    import re
    keywords = list(set(
        t.lower() for t in re.findall(r'[\w一-鿿]{2,}', text)
    ))[:20]  # top 20 keywords

    return DocumentChunk(
        chunk_id=f"chunk_{source_id}_{chunk_index:04d}",
        source_id=source_id,
        source_type=source_type,
        text=text,
        summary="",  # Summary generated later by LLM or simple heuristic
        keywords=keywords,
        location=ChunkLocation(line_start=line_start, line_end=line_start + text.count('\n')),
        chunk_type=ChunkType.PARAGRAPH,
        trust_level=trust_level,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd F:/agentplan && python -m pytest tests/test_chunker.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/storage/__init__.py src/storage/chunker.py tests/test_chunker.py
git commit -m "feat: text chunker with configurable size, overlap, and sentence-boundary awareness"
```

---

### Task 1.2: Memory Store (CRUD for L2/L3 memories)

**Files:**
- Create: `src/storage/memory_store.py`
- Create: `tests/test_memory_store.py`

- [ ] **Step 1: Write failing test `tests/test_memory_store.py`**

```python
"""Tests for MemoryStore."""

import pytest
from datetime import datetime, timezone
from src.db import Database
from src.models.memory import MemoryItem, MemoryType, MemoryStatus
from src.storage.memory_store import MemoryStore


@pytest.fixture
def db():
    d = Database(":memory:")
    d.init_schema()
    yield d
    d.close()


@pytest.fixture
def store(db):
    return MemoryStore(db)


SAMPLE_MEMORY = MemoryItem(
    memory_id="mem_001",
    type=MemoryType.PROJECT_STATE,
    content="The project uses FastAPI for the API layer.",
    summary="API framework: FastAPI",
    entities=["FastAPI", "API"],
    importance=0.8,
    confidence=0.95,
    source="conversation",
    scope="project",
)


class TestMemoryStore:
    def test_insert_and_get(self, store):
        store.insert(SAMPLE_MEMORY)
        result = store.get("mem_001")
        assert result is not None
        assert result.memory_id == "mem_001"
        assert result.content == SAMPLE_MEMORY.content

    def test_get_nonexistent_returns_none(self, store):
        assert store.get("does_not_exist") is None

    def test_list_by_scope(self, store):
        m1 = MemoryItem(memory_id="m1", type=MemoryType.PROJECT_STATE,
                        content="A", scope="project")
        m2 = MemoryItem(memory_id="m2", type=MemoryType.USER_PREFERENCE,
                        content="B", scope="user")
        store.insert(m1)
        store.insert(m2)
        project_mems = store.list_by_scope("project")
        assert len(project_mems) == 1
        assert project_mems[0].memory_id == "m1"

    def test_list_active(self, store):
        active = MemoryItem(memory_id="a1", type=MemoryType.DECISION,
                            content="Active", status=MemoryStatus.ACTIVE)
        archived = MemoryItem(memory_id="a2", type=MemoryType.DECISION,
                              content="Archived", status=MemoryStatus.ARCHIVED)
        store.insert(active)
        store.insert(archived)
        result = store.list_active()
        assert len(result) == 1
        assert result[0].memory_id == "a1"

    def test_update_status(self, store):
        store.insert(SAMPLE_MEMORY)
        store.update_status("mem_001", MemoryStatus.SUPERSEDED)
        result = store.get("mem_001")
        assert result.status == MemoryStatus.SUPERSEDED

    def test_delete(self, store):
        store.insert(SAMPLE_MEMORY)
        store.delete("mem_001")
        assert store.get("mem_001") is None

    def test_search_by_keyword(self, store):
        store.insert(SAMPLE_MEMORY)
        store.insert(MemoryItem(
            memory_id="mem_002", type=MemoryType.FILE_SUMMARY,
            content="The data pipeline uses Apache Kafka for streaming.",
            entities=["Kafka", "streaming"],
        ))
        results = store.search_keyword("FastAPI")
        assert len(results) >= 1
        assert any(r.memory_id == "mem_001" for r in results)

    def test_insert_duplicate_id_updates(self, store):
        store.insert(SAMPLE_MEMORY)
        updated = MemoryItem(
            memory_id="mem_001",
            type=MemoryType.PROJECT_STATE,
            content="Updated: The project uses Flask instead.",
            version=2,
        )
        store.insert(updated)
        result = store.get("mem_001")
        assert result.content == updated.content
        assert result.version == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd F:/agentplan && python -m pytest tests/test_memory_store.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `src/storage/memory_store.py`**

```python
"""MemoryStore — CRUD operations for MemoryItem records (L2/L3 storage)."""

import json
from datetime import datetime, timezone
from typing import Optional
from src.db.connection import Database
from src.models.memory import MemoryItem, MemoryType, MemoryStatus


class MemoryStore:
    """Manages persistent storage of MemoryItem records in SQLite."""

    def __init__(self, db: Database):
        self.db = db

    def insert(self, item: MemoryItem) -> None:
        """Insert or replace a memory item."""
        now = datetime.now(timezone.utc).isoformat()
        if not item.created_at:
            item.created_at = datetime.fromisoformat(now) if isinstance(now, str) else item.created_at
        sql = """
        INSERT OR REPLACE INTO memories
            (memory_id, type, content, summary, entities, importance, confidence,
             source, scope, status, version, source_ref, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.db.execute(sql, (
            item.memory_id,
            item.type.value,
            item.content,
            item.summary,
            json.dumps(item.entities),
            item.importance,
            item.confidence,
            item.source,
            item.scope,
            item.status.value,
            item.version,
            item.source_ref,
            item.created_at.isoformat() if hasattr(item.created_at, 'isoformat') else str(item.created_at),
            now,
        ))
        self.db.commit()

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a single memory by ID."""
        row = self.db.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_item(dict(row))

    def list_by_scope(self, scope: str) -> list[MemoryItem]:
        """List all memories in a given scope."""
        rows = self.db.execute(
            "SELECT * FROM memories WHERE scope = ? ORDER BY updated_at DESC", (scope,)
        ).fetchall()
        return [self._row_to_item(dict(r)) for r in rows]

    def list_active(self) -> list[MemoryItem]:
        """List all active (non-archived, non-superseded) memories."""
        rows = self.db.execute(
            "SELECT * FROM memories WHERE status = 'active' ORDER BY updated_at DESC"
        ).fetchall()
        return [self._row_to_item(dict(r)) for r in rows]

    def list_by_type(self, mem_type: MemoryType) -> list[MemoryItem]:
        """List memories of a specific type."""
        rows = self.db.execute(
            "SELECT * FROM memories WHERE type = ? AND status = 'active' ORDER BY updated_at DESC",
            (mem_type.value,),
        ).fetchall()
        return [self._row_to_item(dict(r)) for r in rows]

    def update_status(self, memory_id: str, status: MemoryStatus) -> None:
        """Update the status of a memory."""
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            "UPDATE memories SET status = ?, updated_at = ? WHERE memory_id = ?",
            (status.value, now, memory_id),
        )
        self.db.commit()

    def delete(self, memory_id: str) -> None:
        """Delete a memory record."""
        self.db.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        self.db.commit()

    def search_keyword(self, query: str, limit: int = 20) -> list[MemoryItem]:
        """Keyword search using FTS5 on memories."""
        # Escape FTS5 special characters
        clean_query = query.replace('"', '""')
        rows = self.db.execute(
            """SELECT m.* FROM memories m
               INNER JOIN memories_fts fts ON m.rowid = fts.rowid
               WHERE memories_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (f'"{clean_query}"', limit),
        ).fetchall()
        if not rows:
            # Fallback to LIKE search if FTS returns nothing
            rows = self.db.execute(
                "SELECT * FROM memories WHERE content LIKE ? OR summary LIKE ? LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        return [self._row_to_item(dict(r)) for r in rows]

    def count(self) -> int:
        """Total number of memories."""
        row = self.db.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
        return row["cnt"] if row else 0

    def _row_to_item(self, row: dict) -> MemoryItem:
        """Convert a database row dict to a MemoryItem."""
        return MemoryItem(
            memory_id=row["memory_id"],
            type=MemoryType(row["type"]),
            content=row["content"],
            summary=row["summary"] or "",
            entities=json.loads(row["entities"]) if row.get("entities") else [],
            importance=row["importance"] or 0.5,
            confidence=row["confidence"] or 0.5,
            source=row.get("source", "conversation"),
            scope=row.get("scope", "project"),
            status=MemoryStatus(row["status"]) if row.get("status") else MemoryStatus.ACTIVE,
            version=row.get("version", 1),
            source_ref=row.get("source_ref"),
            created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row.get("updated_at") else datetime.now(timezone.utc),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd F:/agentplan && python -m pytest tests/test_memory_store.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/storage/memory_store.py tests/test_memory_store.py
git commit -m "feat: MemoryStore with CRUD, FTS5 keyword search, and scope filtering"
```

---

### Task 1.3: File Store (file ingest + chunking pipeline)

**Files:**
- Create: `src/storage/file_store.py`
- Create: `tests/test_file_store.py`

- [ ] **Step 1: Write failing test `tests/test_file_store.py`**

```python
"""Tests for FileStore."""

import pytest
from pathlib import Path
from src.db import Database
from src.storage.file_store import FileStore


@pytest.fixture
def db():
    d = Database(":memory:")
    d.init_schema()
    yield d
    d.close()


@pytest.fixture
def store(db, tmp_path):
    return FileStore(db, file_store_path=str(tmp_path / "files"))


class TestFileStore:
    def test_ingest_markdown(self, store, sample_markdown_path):
        source_id = store.ingest_file(sample_markdown_path)
        assert source_id.startswith("file:")
        # Should have created chunks
        chunks = store.get_chunks(source_id)
        assert len(chunks) > 0
        # Each chunk should have source_id
        for c in chunks:
            assert c.source_id == source_id

    def test_ingest_text_content(self, store):
        source_id = store.ingest_text(
            content="This is a test document. " * 50,
            source_name="test_doc.txt",
        )
        chunks = store.get_chunks(source_id)
        assert len(chunks) > 0

    def test_get_chunks_by_source(self, store, sample_markdown_path):
        sid1 = store.ingest_file(sample_markdown_path)
        sid2 = store.ingest_text(content="Other content.", source_name="other.txt")
        chunks1 = store.get_chunks(sid1)
        chunks2 = store.get_chunks(sid2)
        assert len(chunks1) > 0
        # Each set should only contain its own source_id
        assert all(c.source_id == sid1 for c in chunks1)
        assert all(c.source_id == sid2 for c in chunks2)

    def test_list_sources(self, store, sample_markdown_path):
        store.ingest_file(sample_markdown_path)
        store.ingest_text(content="Another doc.", source_name="doc2.txt")
        sources = store.list_sources()
        assert len(sources) >= 2

    def test_delete_source(self, store, sample_markdown_path):
        sid = store.ingest_file(sample_markdown_path)
        assert len(store.get_chunks(sid)) > 0
        store.delete_source(sid)
        assert store.get_chunks(sid) == []

    def test_ingest_empty_file(self, store, tmp_path):
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("")
        sid = store.ingest_file(empty_file)
        chunks = store.get_chunks(sid)
        assert chunks == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd F:/agentplan && python -m pytest tests/test_file_store.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `src/storage/file_store.py`**

```python
"""FileStore — file ingestion, chunking, and chunk persistence."""

import json
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
from src.db.connection import Database
from src.models.chunk import DocumentChunk, ChunkType, TrustLevel, ChunkLocation
from src.storage.chunker import chunk_text, ChunkerConfig


class FileStore:
    """Manages file ingestion, chunking, and chunk persistence."""

    def __init__(self, db: Database, file_store_path: str = "data/files"):
        self.db = db
        self.file_store_path = Path(file_store_path)
        self.file_store_path.mkdir(parents=True, exist_ok=True)

    def ingest_file(self, file_path: Path, source_type: str | None = None) -> str:
        """Ingest a file from disk. Returns source_id."""
        if source_type is None:
            source_type = self._guess_type(file_path)

        source_id = f"file:{file_path.name}"

        if source_type == "pdf":
            return self._ingest_pdf(file_path, source_id)
        elif source_type in ("markdown", "md", "text", "txt", "py", "js", "ts"):
            content = file_path.read_text(encoding="utf-8", errors="replace")
            return self.ingest_text(content, source_name=file_path.name, source_type=source_type)
        else:
            # Treat unknown as text
            content = file_path.read_text(encoding="utf-8", errors="replace")
            return self.ingest_text(content, source_name=file_path.name, source_type="text")

    def ingest_text(
        self,
        content: str,
        source_name: str,
        source_type: str = "text",
        trust_level: TrustLevel = TrustLevel.EXTERNAL_UNTRUSTED,
    ) -> str:
        """Ingest text content. Returns source_id."""
        source_id = f"file:{source_name}"

        # Delete existing chunks for this source (re-ingest)
        self.delete_source(source_id)

        chunks = chunk_text(
            content,
            source_id=source_id,
            source_type=source_type,
            trust_level=trust_level,
        )

        for chunk in chunks:
            self._insert_chunk(chunk)

        return source_id

    def get_chunks(self, source_id: str) -> list[DocumentChunk]:
        """Retrieve all chunks for a given source."""
        rows = self.db.execute(
            "SELECT * FROM chunks WHERE source_id = ? ORDER BY chunk_id", (source_id,)
        ).fetchall()
        return [self._row_to_chunk(dict(r)) for r in rows]

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a single chunk by ID."""
        row = self.db.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_chunk(dict(row))

    def list_sources(self) -> list[str]:
        """List all distinct source IDs."""
        rows = self.db.execute(
            "SELECT DISTINCT source_id FROM chunks"
        ).fetchall()
        return [r["source_id"] for r in rows]

    def delete_source(self, source_id: str) -> None:
        """Delete all chunks for a source."""
        self.db.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
        self.db.commit()

    def count_chunks(self) -> int:
        row = self.db.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
        return row["cnt"] if row else 0

    def _insert_chunk(self, chunk: DocumentChunk) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            """INSERT INTO chunks
               (chunk_id, source_id, source_type, text, summary, keywords,
                location_page, location_section, location_line_start, location_line_end,
                chunk_type, embedding_id, trust_level, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chunk.chunk_id, chunk.source_id, chunk.source_type,
                chunk.text, chunk.summary,
                json.dumps(chunk.keywords),
                chunk.location.page, chunk.location.section,
                chunk.location.line_start, chunk.location.line_end,
                chunk.chunk_type.value, chunk.embedding_id,
                chunk.trust_level.value, now,
            ),
        )
        self.db.commit()

    def _ingest_pdf(self, file_path: Path, source_id: str) -> str:
        """Ingest a PDF file using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF (fitz) is required for PDF ingestion. pip install PyMuPDF")

        doc = fitz.open(str(file_path))
        full_text_parts = []

        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                full_text_parts.append(f"[Page {page_num + 1}]\n{text}")

        doc.close()
        full_text = "\n\n".join(full_text_parts)

        return self.ingest_text(
            content=full_text,
            source_name=file_path.name,
            source_type="pdf",
            trust_level=TrustLevel.EXTERNAL_UNTRUSTED,
        )

    def _guess_type(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        mapping = {
            ".pdf": "pdf", ".md": "markdown", ".txt": "text",
            ".py": "code", ".js": "code", ".ts": "code",
            ".rs": "code", ".go": "code", ".java": "code",
            ".html": "text", ".css": "code", ".json": "text",
            ".yaml": "text", ".yml": "text", ".toml": "text",
        }
        return mapping.get(ext, "text")

    def _row_to_chunk(self, row: dict) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=row["chunk_id"],
            source_id=row["source_id"],
            source_type=row["source_type"],
            text=row["text"],
            summary=row.get("summary") or "",
            keywords=json.loads(row.get("keywords", "[]")),
            location=ChunkLocation(
                page=row.get("location_page"),
                section=row.get("location_section"),
                line_start=row.get("location_line_start"),
                line_end=row.get("location_line_end"),
            ),
            chunk_type=ChunkType(row.get("chunk_type", "paragraph")),
            embedding_id=row.get("embedding_id"),
            trust_level=TrustLevel(row.get("trust_level", "external_untrusted")),
            created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else datetime.now(timezone.utc),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd F:/agentplan && python -m pytest tests/test_file_store.py -v`
Expected: 6 tests PASS

- [ ] **Step 5: Write `src/storage/__init__.py`**

```python
from src.storage.memory_store import MemoryStore
from src.storage.file_store import FileStore
from src.storage.chunker import chunk_text, ChunkerConfig

__all__ = ["MemoryStore", "FileStore", "chunk_text", "ChunkerConfig"]
```

- [ ] **Step 6: Commit**

```bash
git add src/storage/ tests/test_file_store.py
git commit -m "feat: FileStore with markdown, text, and PDF ingestion + chunking pipeline"
```

---

## Milestone 2: Hybrid Index + Retriever

**Goal:** Vector index (FAISS) + keyword index (SQLite FTS5), combined hybrid retriever with scoring per agent_os_initial_plan.md §5.2.

*Note: Milestones 2-5 would continue with the same TDD pattern. For brevity since this is a plan document, the remaining milestones outline the key architecture decisions and critical test cases. Full task-by-task breakdown would follow during implementation.*

### Task 2.1: Vector Index (FAISS wrapper)

**Key design decisions:**
- FAISS IndexFlatIP (inner product) for exact search during MVP
- Embedding generation is a dependency-injected callable (not hardcoded to OpenAI)
- Index is rebuilt from SQLite chunks on startup or after ingestion
- Each chunk's `embedding_id` maps to FAISS internal index position

### Task 2.2: Keyword Index (SQLite FTS5 wrapper)

**Key design decisions:**
- Leverages FTS5 tables already created in migrations
- BM25-like ranking via FTS5 built-in rank
- Supports phrase queries via double-quote wrapping

### Task 2.3: Hybrid Retriever

**Key design decisions:**
- Combines vector + keyword results using the scoring formula from §5.2
- Deduplicates by `chunk_id`
- Returns top-K with source references and trust levels
- Each result includes the score breakdown for traceability

---

## Milestone 3: Context MMU

**Goal:** Given a task and retrieval candidates, assemble a ContextPack within token budget.

### Task 3.1: Token Budgeter

### Task 3.2: Context MMU (assembly, dedup, sort, budget enforcement)

---

## Milestone 4: Single Agent Runtime

**Goal:** End-to-end pipeline: Input → Retrieve → Context MMU → LLM → Verifier → Writeback Gate → Trace Logger → Output.

### Task 4.1: Trace Logger

### Task 4.2: Verifier (source reference validation)

### Task 4.3: Writeback Gate (memory write decisions)

### Task 4.4: Agent Runtime (orchestrates the full pipeline)

---

## Milestone 5: API + Integration Tests

**Goal:** FastAPI endpoints for file upload, query, and trace retrieval. End-to-end integration tests against the 5 evaluation scenarios.

### Task 5.1: FastAPI routes (`/upload`, `/query`, `/trace/{id}`)

### Task 5.2: Evaluation metrics runner

### Task 5.3: Integration test against eval scenarios

---

## Key Interfaces (Locked)

These interfaces are the contract between modules. They must not change without updating this plan.

### Retriever → Context MMU
```python
RetrievalResult = namedtuple("RetrievalResult", [
    "chunk_id",      # str
    "score",         # float 0.0-1.0
    "source_ref",    # str (e.g. "file:paper.pdf")
    "trust_level",   # TrustLevel enum
    "text_preview",  # str (first 200 chars)
])
```

### Context MMU → Agent Runtime
```python
# ContextPack (defined in models/context.py)
# .sections contains all assembled context
# .source_refs contains all citations
```

### Agent Runtime → Verifier
```python
VerifyInput = namedtuple("VerifyInput", [
    "response_text",  # str — the LLM's response
    "source_refs",    # list[str] — claimed sources
    "context_pack",   # ContextPack — what the LLM saw
])
VerifyOutput = namedtuple("VerifyOutput", [
    "is_verified",        # bool
    "unverified_claims",  # list[str]
    "conflicting_pairs",  # list[tuple[str, str]]
    "suggestions",        # list[str]
])
```

### Agent Runtime → Writeback Gate
```python
WritebackInput = namedtuple("WritebackInput", [
    "content",         # str — proposed memory content
    "source",          # str — where it came from
    "importance",      # float
    "confidence",      # float
    "current_memory",  # Optional[MemoryItem] — existing memory if any
])
WritebackDecision = namedtuple("WritebackDecision", [
    "action",          # "write" | "skip" | "ask_user"
    "location",        # "working_memory" | "long_term_memory" | "none"
    "reason",          # str
    "score",           # float — the WriteScore
])
```

---

## Test Plan (cross-cutting)

| Test Layer | What It Covers | When It Runs |
|---|---|---|
| Unit tests | Each module in isolation | `pytest tests/` on every commit |
| Retrieval eval | Top-k hit rate, MRR, nDCG per scenario | `python eval/metrics.py` after index changes |
| Context eval | Token budget enforcement, dedup, source annotation | Manual + scripted |
| Integration test | Full pipeline for doc QA scenario | `pytest tests/test_agent_runtime.py` |
| Comparison test | Agent-OS MVP vs plain RAG on same task | Manual with recorded traces |

---

## Self-Review

### 1. Spec Coverage

| Requirement from PLAN.md | Covered By |
|---|---|
| Phase 0: 5 evaluation scenarios | Task 0.3 |
| Phase 1: Memory Store, File Store, Chunker | Tasks 1.1–1.3 |
| Phase 1: Vector/Keyword Hybrid Index | Milestone 2 (Tasks 2.1–2.3) |
| Phase 2: Context MMU (candidate recall, dedup, sort, budget) | Milestone 3 (Tasks 3.1–3.2) |
| Phase 3: Input Adapter → Retriever → MMU → Worker → Verifier → Writeback → Trace | Milestone 4 (Tasks 4.1–4.4) |
| Fixed data objects: MemoryItem, DocumentChunk, ContextPack, TraceStep | Task 0.2 |
| Fixed tables: memories, chunks, traces | Task 0.4 |
| Context Pack with task_id, budget, sections, source_refs, trust_level | Task 0.2 (ContextPack model) |
| Writeback Gate: write/skip, location, confidence, user confirm, reason | Milestone 4 (Task 4.3) |
| Trace Logger: input, query, candidates, assembly, verifier, writeback, source | Milestone 4 (Task 4.1) |
| Retrieval test with standard queries | eval/test_queries.py + eval/metrics.py |
| Context MMU budget test | Milestone 3 tests |
| Writeback test | Task 4.3 tests |
| Citation test | Task 4.2 tests (Verifier) |
| Comparison test: Agent-OS vs plain RAG | Milestone 5 (Task 5.3) |

### 2. Placeholder Scan

- Milestones 2-5 are outlined architecturally rather than as complete TDD tasks. This is intentional — they will be fully detailed during implementation as each milestone builds on verified lower layers.
- No "TBD", "TODO", or "implement later" markers in code blocks.
- All code steps in Milestones 0-1 contain actual runnable code.

### 3. Type Consistency

- `MemoryItem.memory_id: str` — used consistently across MemoryStore, FTS index
- `DocumentChunk.chunk_id: str` — matches pattern `chunk_{source_id}_{index:04d}`
- `ContextPack.context_id: str` — used in Trace and API routes
- `TraceStep.step_id: str` — referenced by Trace.add_step()
- All `source_id` values use format `file:{name}` — consistent across FileStore and retriever
- Config singleton `config` is lowercase — matches usage throughout

### 4. Missing from Scope (Intentionally Deferred)

Per PLAN.md "暂不实现":
- Multi-agent concurrency → post-MVP
- Complex permission sandbox → post-MVP
- Knowledge graph → post-MVP
- Full GUI → post-MVP
- Autonomous long-running → post-MVP
- Agent PCB / Task TCB runtime usage → post-MVP (schemas defined but not wired)

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-03-agent-os-mvp.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
