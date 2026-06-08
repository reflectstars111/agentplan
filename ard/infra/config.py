"""Phase-1 minimal configuration for ARD."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Application configuration for Phase 1 (Context MMU verification)."""

    # Database
    db_path: str = "data/ard_phase1.db"

    # File storage
    file_store_path: str = "data/files"
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Vector index
    embedding_dim: int = 1024  # BGE-M3 default
    vector_index_path: str = "data/ard_vector_index.faiss"

    # Context MMU
    default_token_budget: int = 8000  # smaller for controlled experiments
    max_retrieval_candidates: int = 50
    top_k_after_rerank: int = 15

    # Retrieval weights (ARD §7 scoring formula)
    weight_semantic: float = 0.35
    weight_keyword: float = 0.20
    weight_entity: float = 0.15
    weight_recency: float = 0.10
    weight_importance: float = 0.10
    weight_structural: float = 0.10
    penalty_token_cost: float = 0.10
    penalty_trust: float = 0.20

    # LLM
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    llm_api_key_env: str = "DEEPSEEK_API_KEY"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_temperature: float = 0.3

    # Evaluation
    eval_data_dir: str = "eval_data"
    eval_queries_path: str = "eval_data/queries.json"

    def __post_init__(self):
        # Create parent dirs for file paths
        for p in [self.db_path, self.vector_index_path]:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
        # Create file_store_path as a directory itself
        Path(self.file_store_path).mkdir(parents=True, exist_ok=True)
