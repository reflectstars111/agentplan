"""VectorIndex — FAISS wrapper for semantic similarity search."""

import numpy as np
from pathlib import Path


class VectorIndex:
    """FAISS-based vector index with custom string IDs.

    Uses IndexFlatIP (inner product) for exact search. Embeddings should be
    normalized to unit length so that inner product equals cosine similarity.
    """

    def __init__(self, dim: int = 1536, index_path: str | None = None):
        import faiss

        self.dim = dim
        self.index_path = index_path
        # IndexFlatIP for inner product; normalize vectors for cosine similarity
        self._quantizer = faiss.IndexFlatIP(dim)
        # IndexIDMap allows custom integer IDs and removal
        self._index = faiss.IndexIDMap(self._quantizer)
        # Map chunk_id (str) <-> FAISS internal ID (int)
        self._id_to_chunk: dict[int, str] = {}
        self._chunk_to_id: dict[str, int] = {}
        self._next_id: int = 0

        if (
            index_path
            and Path(index_path).exists()
            and Path(index_path + ".meta.json").exists()
        ):
            self.load(index_path)

    @property
    def count(self) -> int:
        return self._index.ntotal

    def add(self, chunk_id: str, embedding: np.ndarray) -> None:
        """Add or replace an embedding for a chunk_id."""
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero
        embedding = embedding / norms

        if chunk_id in self._chunk_to_id:
            # Remove old embedding first
            faiss_id = self._chunk_to_id[chunk_id]
            self._index.remove_ids(np.array([faiss_id], dtype=np.int64))

        faiss_id = self._next_id
        self._next_id += 1
        self._index.add_with_ids(embedding, np.array([faiss_id], dtype=np.int64))
        self._id_to_chunk[faiss_id] = chunk_id
        self._chunk_to_id[chunk_id] = faiss_id

    def search(self, query_embedding: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """Search for k most similar chunks. Returns [(chunk_id, score), ...].

        Score is inner product (cosine similarity for normalized vectors).
        """
        if self.count == 0:
            return []

        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize query
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        k = min(k, self.count)
        scores, ids = self._index.search(query_embedding, k)

        results = []
        seen = set()
        for score, faiss_id in zip(scores[0], ids[0]):
            if faiss_id == -1:
                continue
            chunk_id = self._id_to_chunk.get(int(faiss_id))
            if chunk_id and chunk_id not in seen:
                results.append((chunk_id, float(score)))
                seen.add(chunk_id)

        return results

    def remove(self, chunk_id: str) -> None:
        """Remove a chunk from the index."""
        if chunk_id not in self._chunk_to_id:
            return
        faiss_id = self._chunk_to_id.pop(chunk_id)
        self._id_to_chunk.pop(faiss_id, None)
        self._index.remove_ids(np.array([faiss_id], dtype=np.int64))

    def save(self, path: str) -> None:
        """Persist index and ID mappings to disk."""
        import json
        import faiss

        faiss.write_index(self._index, path)
        # Save ID mappings alongside the index
        meta_path = path + ".meta.json"
        meta = {
            "id_to_chunk": {str(k): v for k, v in self._id_to_chunk.items()},
            "chunk_to_id": self._chunk_to_id,
            "next_id": self._next_id,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def persist(self) -> None:
        """Persist to the configured index path, if one was supplied."""
        if not self.index_path:
            return
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        self.save(self.index_path)

    def load(self, path: str) -> None:
        """Load index and ID mappings from disk."""
        import json
        import faiss

        loaded = faiss.read_index(path)
        if loaded.d != self.dim:
            raise ValueError(
                f"Vector index dimension {loaded.d} does not match {self.dim}"
            )
        self._index = loaded
        # Load ID mappings
        meta_path = path + ".meta.json"
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._id_to_chunk = {int(k): v for k, v in meta["id_to_chunk"].items()}
            self._chunk_to_id = meta["chunk_to_id"]
            self._next_id = meta["next_id"]
        except FileNotFoundError:
            self._id_to_chunk = {}
            self._chunk_to_id = {}
            self._next_id = self._index.ntotal * 2

    def rebuild_from_db(
        self, chunks: list, embed_fn, batch_size: int = 64
    ) -> None:
        """Rebuild the entire index from a list of chunks with embeddings.

        Args:
            chunks: List of DocumentChunk objects (must have chunk_id and text).
            embed_fn: Callable that takes list[str] and returns np.ndarray of embeddings.
            batch_size: Number of texts to embed at once.
        """
        import faiss

        # Reset index
        self._quantizer = faiss.IndexFlatIP(self.dim)
        self._index = faiss.IndexIDMap(self._quantizer)
        self._id_to_chunk = {}
        self._chunk_to_id = {}
        self._next_id = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]
            embeddings = embed_fn(texts)
            for chunk, emb in zip(batch, embeddings):
                self.add(chunk.chunk_id, emb)
