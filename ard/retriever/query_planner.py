"""QueryPlanner — analyzes a query and produces a RetrievalPlan."""

import re

from ard.retriever import RetrievalPlan, QueryPlannerProtocol


class QueryPlanner(QueryPlannerProtocol):
    """Analyzes query intent to select retrieval strategies and weights.

    Phase 1: Simple heuristic-based analysis (no LLM).
    Future: Could use LLM for more sophisticated query decomposition.
    """

    def plan(self, query: str) -> RetrievalPlan:
        """Produce a retrieval plan based on query characteristics.

        Heuristics:
        - Short factual queries → keyword heavy
        - Long descriptive queries → vector heavy
        - Questions with "where"/"find"/"locate" → structure boost
        - Questions with "when"/"recent" → temporal boost
        """

        query_lower = query.lower().strip()
        word_count = len(query.split())

        strategies = ["vector", "keyword"]
        weights = {}

        # Detect structural intent
        structure_terms = {"where", "find", "locate", "file", "section", "chapter", "page",
                           "class", "function", "method", "module", "path", "folder"}
        has_structure = bool(set(self._tokenize(query_lower)) & structure_terms)

        # Detect temporal intent
        temporal_terms = {"when", "recent", "latest", "last", "before", "after",
                          "version", "update", "change", "history"}
        has_temporal = bool(set(self._tokenize(query_lower)) & temporal_terms)

        # Detect factoid vs. conceptual
        factoid_patterns = ["what is", "who is", "define", "definition", "how many",
                            "when did", "where is"]
        is_factoid = any(p in query_lower for p in factoid_patterns)

        if has_structure:
            strategies.append("structure")
            weights = {"vector": 0.3, "keyword": 0.3, "structure": 0.4}
        elif is_factoid and word_count <= 10:
            weights = {"vector": 0.3, "keyword": 0.7}
        elif has_temporal:
            weights = {"vector": 0.4, "keyword": 0.4}
            strategies.append("temporal")
            weights["temporal"] = 0.2
        elif word_count > 20:
            # Long conceptual query → vector heavy
            weights = {"vector": 0.7, "keyword": 0.3}
        else:
            weights = {"vector": 0.5, "keyword": 0.5}

        return RetrievalPlan(strategies=strategies, weights=weights)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'\w+', text)
