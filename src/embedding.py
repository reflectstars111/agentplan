"""Embedding functions for Agent-OS.

Supports both mock (deterministic hash-based) and real OpenAI embeddings.
"""

import numpy as np


def create_mock_embed_fn(dim: int = 1536):
    """Create a deterministic mock embedding function for testing/demo.

    Uses character-hash based embeddings. Not semantically meaningful
    but consistent (same text always produces the same vector).
    """
    def mock_embed(texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for j, ch in enumerate(text):
                result[i, j % dim] += (ord(ch) / 256.0)
            norm = np.linalg.norm(result[i])
            if norm > 0:
                result[i] /= norm
        return result
    return mock_embed


def create_openai_embed_fn(api_key: str | None = None, model: str = "text-embedding-3-small"):
    """Create an OpenAI embedding function.

    Args:
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        model: Embedding model name.

    Returns:
        Callable that takes list[str] and returns np.ndarray.
    """
    import os
    from openai import OpenAI

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def openai_embed(texts: list[str]) -> np.ndarray:
        response = client.embeddings.create(model=model, input=texts)
        embeddings = [d.embedding for d in response.data]
        return np.array(embeddings, dtype=np.float32)

    return openai_embed


_bge_model = None
_BGE_LOCAL_PATH = None


def _get_bge_path() -> str:
    """Find BGE-M3 model: local path first, then HF hub download."""
    from pathlib import Path
    global _BGE_LOCAL_PATH
    if _BGE_LOCAL_PATH:
        return _BGE_LOCAL_PATH
    # Check local project directory
    local = Path(__file__).parent.parent / "models" / "bge-m3"
    if local.exists() and (local / "config.json").exists():
        _BGE_LOCAL_PATH = str(local)
        return _BGE_LOCAL_PATH
    # Fallback: let HuggingFace download
    return "BAAI/bge-m3"


def create_bge_embed_fn(model_name: str | None = None):
    """Create a BGE-M3 embedding function (local CPU inference).

    Uses models/bge-m3/ if available, otherwise downloads from HuggingFace.
    BGE-M3: 100+ languages, 8192 token input, 1024-dim dense vectors.

    Returns:
        Callable that takes list[str] and returns np.ndarray (1024-dim).
    """
    global _bge_model
    import numpy as np

    path = model_name or _get_bge_path()

    def bge_embed(texts: list[str]) -> np.ndarray:
        global _bge_model
        if _bge_model is None:
            from FlagEmbedding import BGEM3FlagModel
            print(f"Loading BGE-M3 from {path}...")
            _bge_model = BGEM3FlagModel(path, use_fp16=False)
            print("BGE-M3 loaded.")
        embeddings = _bge_model.encode(
            texts, batch_size=8, max_length=8192,
        )["dense_vecs"]
        return np.array(embeddings, dtype=np.float32)

    return bge_embed
