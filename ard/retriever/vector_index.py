"""FAISS-based vector index for ARD."""

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class VectorIndex:
    """FAISS flat-L2 index with id mapping and disk persistence."""

    def __init__(self, dim: int, index_path: str = ""):
        if faiss is None:
            raise ImportError("faiss-cpu is required for VectorIndex")
        self.dim = dim
        self.index_path = index_path
        self._id_to_chunk: dict[int, str] = {}
        self._next_id: int = 0
        self._index: faiss.IndexFlatL2 | None = None
        self._load_or_create()

    def _load_or_create(self) -> None:
        """Load existing index from disk or create a new one."""
        if self.index_path and os.path.exists(self.index_path):
            try:
                self._index = faiss.read_index(self.index_path)
                meta_path = self.index_path + ".meta.json"
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    self._id_to_chunk = {int(k): v for k, v in meta.get("id_map", {}).items()}
                    self._next_id = meta.get("next_id", 0)
                return
            except Exception:
                pass
        self._index = faiss.IndexFlatL2(self.dim)
        self._id_to_chunk = {}
        self._next_id = 0

    @property
    def count(self) -> int:
        return self._index.ntotal if self._index else 0

    def add(self, vectors: np.ndarray, chunk_ids: Sequence[str]) -> None:
        """Add vectors with their chunk IDs to the index."""
        if len(vectors) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        for chunk_id in chunk_ids:
            self._id_to_chunk[self._next_id] = chunk_id
            self._next_id += 1
        self._index.add(vectors)

    def search(self, query_vec: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, indices) for top-k matches."""
        query_vec = np.asarray(query_vec, dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        k = min(k, self._index.ntotal)
        if k == 0:
            empty_dist = np.array([[-1.0]], dtype=np.float32)
            empty_idx = np.array([[-1]], dtype=np.int64)
            return empty_dist, empty_idx
        return self._index.search(query_vec, k)

    def get_id(self, internal_id: int) -> str | None:
        return self._id_to_chunk.get(internal_id)

    def persist(self) -> None:
        """Save index and id map to disk."""
        if not self.index_path:
            return
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, self.index_path)
        meta = {
            "dim": self.dim,
            "id_map": {str(k): v for k, v in self._id_to_chunk.items()},
            "next_id": self._next_id,
            "total": self._index.ntotal,
        }
        with open(self.index_path + ".meta.json", "w") as f:
            json.dump(meta, f)

    def rebuild_from_db(self, chunks: list[dict], embed_fn: callable) -> None:
        """Rebuild the entire FAISS index from chunk data."""
        self._index = faiss.IndexFlatL2(self.dim)
        self._id_to_chunk = {}
        self._next_id = 0

        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.get("text", "") for c in batch]
            chunk_ids = [c.get("chunk_id", "") for c in batch]
            embeddings = embed_fn(texts)
            self.add(embeddings, chunk_ids)
