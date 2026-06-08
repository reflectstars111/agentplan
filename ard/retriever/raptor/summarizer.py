"""RAPTOR-style recursive summarization for ContextMMU's COMPRESS step.

Clusters similar retrieved chunks and generates LLM summaries at multiple
levels of abstraction, building a lightweight hierarchy for context assembly.

Key insight: Instead of truncating long chunks (current COMPRESS), replace
groups of related chunks with a concise LLM-generated summary.
"""

import re
import numpy as np
from collections import defaultdict


class RaptorSummarizer:
    """Lightweight RAPTOR-style recursive summarization.

    Two modes:
    - cluster_only: Group similar chunks, pick representative (no LLM needed)
    - llm_summarize: Cluster + LLM summarization per cluster
    """

    def __init__(self, llm_fn=None, max_clusters: int = 5, similarity_threshold: float = 0.3):
        self.llm_fn = llm_fn
        self.max_clusters = max_clusters
        self.similarity_threshold = similarity_threshold

    def compress(self, items: list[dict], token_budget: int,
                 text_key: str = "text") -> list[dict]:
        """Compress a list of context items into fewer, higher-quality items.

        Args:
            items: List of dicts with 'text' key (and optional 'source_ref', 'score').
            token_budget: Maximum total tokens for output.
            text_key: Key for the text field in items.

        Returns:
            Compressed list of items fitting within token_budget.
        """
        if len(items) <= 3:
            return items  # too few to cluster meaningfully

        # 1. Cluster similar items
        clusters = self._cluster_by_overlap(items, text_key)

        # 2. For each cluster: either summarize (LLM) or pick best (heuristic)
        compressed = []
        for cluster in clusters:
            if len(cluster) == 1:
                compressed.append(cluster[0])
            elif self.llm_fn and len(cluster) >= 3:
                summary_item = self._llm_summarize_cluster(cluster, text_key)
                compressed.append(summary_item)
            else:
                # Pick the highest-scoring or longest item as representative
                best = max(cluster, key=lambda x: (
                    x.get("score", 0) * 0.5 + len(x.get(text_key, "")) / 1000 * 0.5
                ))
                compressed.append(best)

        # 3. Fit to token budget
        return self._fit_budget(compressed, token_budget, text_key)

    def _cluster_by_overlap(self, items: list[dict], text_key: str) -> list[list[dict]]:
        """Cluster items by keyword overlap (simple greedy)."""
        if len(items) <= self.max_clusters:
            return [[item] for item in items]

        # Build keyword sets
        item_keywords = []
        for item in items:
            words = set(re.findall(r'\w+', item.get(text_key, "").lower()))
            # Keep only meaningful words (>3 chars)
            meaningful = {w for w in words if len(w) > 3}
            item_keywords.append(meaningful)

        # Greedy clustering: assign each item to the best-matching cluster
        clusters = []
        assigned = set()

        for i in range(min(self.max_clusters, len(items))):
            # Find unassigned item with the most "central" keywords
            best_i = -1
            best_centrality = -1
            for j in range(len(items)):
                if j in assigned:
                    continue
                # Centrality = average overlap with all other unassigned items
                overlaps = []
                for k in range(len(items)):
                    if k in assigned or k == j:
                        continue
                    ik = item_keywords[j]
                    kk = item_keywords[k]
                    if ik and kk:
                        overlap = len(ik & kk) / max(len(ik | kk), 1)
                        overlaps.append(overlap)
                centrality = np.mean(overlaps) if overlaps else 0
                if centrality > best_centrality:
                    best_centrality = centrality
                    best_i = j

            if best_i < 0:
                break

            # Create cluster centered on best_i
            cluster = [items[best_i]]
            assigned.add(best_i)

            for j in range(len(items)):
                if j in assigned:
                    continue
                overlap = 0
                if item_keywords[best_i] and item_keywords[j]:
                    overlap = len(item_keywords[best_i] & item_keywords[j]) / max(
                        len(item_keywords[best_i] | item_keywords[j]), 1)
                if overlap >= self.similarity_threshold:
                    cluster.append(items[j])
                    assigned.add(j)

            clusters.append(cluster)

        # Add any remaining unassigned items
        for j in range(len(items)):
            if j not in assigned:
                # Add to closest cluster
                best_cluster = 0
                best_overlap = -1
                for ci, cluster in enumerate(clusters):
                    for member in cluster:
                        idx = items.index(member)
                        if item_keywords[idx] and item_keywords[j]:
                            overlap = len(item_keywords[idx] & item_keywords[j]) / max(
                                len(item_keywords[idx] | item_keywords[j]), 1)
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_cluster = ci
                clusters[best_cluster].append(items[j])

        return clusters

    def _llm_summarize_cluster(self, cluster: list[dict], text_key: str) -> dict:
        """Use LLM to produce a summary of a cluster of related chunks."""
        if not self.llm_fn:
            return cluster[0]

        texts = []
        for item in cluster:
            text = item.get(text_key, "")
            src = item.get("source_ref", "unknown")
            texts.append(f"[{src}]: {text[:300]}")

        prompt = f"""Summarize these related passages into a single coherent paragraph (max 200 words). Preserve key facts, numbers, and source references.

Passages:
{chr(10).join(texts)}

Summary:"""

        try:
            summary = self.llm_fn(prompt)
            if isinstance(summary, str) and len(summary) > 20:
                return {
                    text_key: summary.strip(),
                    "source_ref": ", ".join(set(
                        item.get("source_ref", "unknown") for item in cluster
                    )),
                    "trust_level": min(
                        (item.get("trust_level", "external_untrusted") for item in cluster),
                        key=lambda x: {"internal_memory": 0, "user_provided_data": 1,
                                      "agent_generated": 2, "tool_observation": 3,
                                      "external_untrusted": 4}.get(x, 4)
                    ),
                    "score": max(item.get("score", 0) for item in cluster),
                    "raptor_compressed": True,
                    "original_chunk_count": len(cluster),
                }
        except Exception:
            pass

        return cluster[0]  # fallback

    @staticmethod
    def _fit_budget(items: list[dict], budget: int, text_key: str) -> list[dict]:
        """Fit compressed items into token budget (chars/4 ≈ tokens)."""
        result = []
        tokens_used = 0
        for item in items:
            text = item.get(text_key, "")
            item_tokens = max(1, len(text) // 4)
            if tokens_used + item_tokens <= budget:
                result.append(item)
                tokens_used += item_tokens
            elif tokens_used < budget:
                # Truncate last item to fit
                remaining = budget - tokens_used
                max_chars = remaining * 4
                if max_chars > 20:
                    truncated = {**item, text_key: text[:max_chars] + "...",
                                "truncated": True}
                    result.append(truncated)
                break  # no more room
        return result
