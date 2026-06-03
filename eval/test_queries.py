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
