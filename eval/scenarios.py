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
