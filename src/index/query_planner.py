"""QueryPlanner — decompose user query into sub-queries for multi-index retrieval.

Maps to agent_os_initial_plan.md §5.3 (retrieval flow diagram: Query Planner dispatches
to vector/keyword/structure/time retrieval).
"""

import re
from dataclasses import dataclass, field


@dataclass
class QueryPlan:
    original_query: str
    vector_query: str = ""        # query for semantic search
    keyword_query: str = ""       # query for FTS5 keyword search
    structure_query: str = ""     # query for structure index
    time_filter: str = ""         # ISO date for recency filter (empty = no filter)
    entity_filters: list[str] = field(default_factory=list)


class QueryPlanner:
    """Decompose a user query into sub-queries for different index types.

    The plan (§5.3) shows: User Question → QueryPlanner →
      ├─ Vector Search
      ├─ Keyword Search
      ├─ Structure Search
      └─ Time Filter
    """

    def plan(self, query: str) -> QueryPlan:
        """Generate a retrieval plan from the user's query."""
        query_lower = query.lower()

        plan = QueryPlan(
            original_query=query,
            vector_query=query,           # semantic search uses full query
            keyword_query=query,          # keyword search uses full query
            structure_query="",
            time_filter="",
            entity_filters=[],
        )

        # Extract entity filters (quoted phrases or capitalized terms)
        entities = re.findall(r'"([^"]+)"', query) + \
                   re.findall(r'\b([A-Z][A-Za-z0-9_]{2,}(?:\s+[A-Z][a-z]+)*)\b', query)
        plan.entity_filters = list(set(entities))[:5]

        # Detect structure queries (section/page/file references)
        if re.search(r'\b(section|chapter|heading|page)\s+\d+', query_lower):
            plan.structure_query = query

        # Detect time/recency signals
        time_patterns = [
            (r'\b(recent|latest|newest|last week|this month)\b', "recent"),
            (r'\b(old|archived|historical|last year|202[0-9])\b', "historical"),
        ]
        for pattern, _ in time_patterns:
            if re.search(pattern, query_lower):
                plan.time_filter = "recent"  # simplified for MVP
                break

        return plan
