"""Global configuration for Agent-OS MVP."""

from dataclasses import dataclass
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

    # Task execution (Phase 2)
    task_max_retries: int = 2
    task_default_priority: int = 5
    parallel_enabled: bool = False       # Phase 5: enable parallel agent execution
    max_parallel_agents: int = 4         # max concurrent agents

    # LLM API (Post-MVP)
    llm_provider: str = "mock"           # "openai" | "deepseek" | "anthropic" | "mock"
    llm_model: str = "gpt-4o"            # or "deepseek-chat", "claude-sonnet-4-6"
    llm_api_key_env: str = "OPENAI_API_KEY"  # env var name
    llm_base_url: str = ""               # override API base URL (empty = use default)
    llm_temperature: float = 0.3

    # Multi-agent (Phase 3)
    agent_default_context_budget: int = 24000
    merge_confidence_threshold: float = 0.5

    # Trace
    trace_enabled: bool = True

    def __post_init__(self):
        """Ensure data directories exist."""
        for p in [self.db_path, self.file_store_path, self.vector_index_path]:
            Path(p).parent.mkdir(parents=True, exist_ok=True)


# Singleton
config = Config()
