"""Tests for ContextMMU."""

import pytest
from src.config import Config
from src.models.memory import MemoryItem, MemoryType
from src.models.context import ContextPack, ContextSection
from src.context.token_budgeter import TokenBudgeter
from src.index.hybrid_retriever import RetrievalResult
from src.context.mmu import ContextMMU


@pytest.fixture
def budgeter():
    return TokenBudgeter()


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def mmu(budgeter, config):
    return ContextMMU(budgeter, config)


@pytest.fixture
def sample_results():
    return [
        RetrievalResult(
            chunk_id="chunk_1",
            score=0.85,
            source_ref="file:paper.pdf",
            trust_level="external_untrusted",
            text_preview="The FastAPI framework provides async support for Python APIs.",
            score_breakdown={"semantic": 0.8, "keyword": 0.7, "combined": 0.85},
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            score=0.72,
            source_ref="file:guide.md",
            trust_level="user_provided_data",
            text_preview="FastAPI uses Starlette for the web parts and Pydantic for data validation.",
            score_breakdown={"semantic": 0.65, "keyword": 0.6, "combined": 0.72},
        ),
        RetrievalResult(
            chunk_id="chunk_3",
            score=0.50,
            source_ref="file:notes.txt",
            trust_level="external_untrusted",
            text_preview="Python web frameworks include Django, Flask, and FastAPI.",
            score_breakdown={"semantic": 0.4, "keyword": 0.5, "combined": 0.50},
        ),
    ]


@pytest.fixture
def sample_memories():
    return [
        MemoryItem(
            memory_id="mem_1",
            type=MemoryType.DECISION,
            content="Use FastAPI for all new API services.",
            summary="API framework decision: FastAPI",
            importance=0.9,
            confidence=0.95,
        ),
        MemoryItem(
            memory_id="mem_2",
            type=MemoryType.PROJECT_STATE,
            content="Project is a multi-agent runtime system inspired by Von Neumann architecture.",
            importance=0.8,
        ),
    ]


class TestContextMMU:
    def test_assemble_returns_context_pack(self, mmu, sample_results):
        pack = mmu.assemble(
            query="What is FastAPI?",
            retrieval_results=sample_results,
            task_id="task_001",
            agent_id="agent_001",
        )
        assert isinstance(pack, ContextPack)
        assert pack.context_id.startswith("ctx_")
        assert pack.task_id == "task_001"
        assert pack.used_tokens > 0

    def test_assemble_includes_retrieval_evidence(self, mmu, sample_results):
        pack = mmu.assemble(
            query="Tell me about Python web frameworks.",
            retrieval_results=sample_results,
        )
        sections = {s.name for s in pack.sections}
        assert "retrieved_evidence" in sections

        # Find the evidence section
        evidence = next(s for s in pack.sections if s.name == "retrieved_evidence")
        assert len(evidence.items) > 0

    def test_assemble_includes_working_memory(self, mmu, sample_results, sample_memories):
        pack = mmu.assemble(
            query="What framework should we use?",
            retrieval_results=sample_results,
            working_memories=sample_memories,
        )
        sections = {s.name for s in pack.sections}
        assert "working_memory" in sections

    def test_assemble_respects_token_budget(self, mmu, sample_results):
        small_config = Config(default_token_budget=500)
        small_mmu = ContextMMU(TokenBudgeter(), small_config)

        pack = small_mmu.assemble(
            query="What is FastAPI? " * 20,  # long query
            retrieval_results=sample_results,
        )
        assert pack.used_tokens <= 500

    def test_assemble_collects_source_refs(self, mmu, sample_results):
        pack = mmu.assemble(
            query="FastAPI features",
            retrieval_results=sample_results,
        )
        assert len(pack.source_refs) > 0
        assert "file:paper.pdf" in pack.source_refs
        assert "file:guide.md" in pack.source_refs

    def test_assemble_deduplicates_content(self, mmu, sample_results):
        # Add duplicate results (same text)
        dup_results = sample_results + [
            RetrievalResult(
                chunk_id="chunk_1",  # same as first
                score=0.80,
                source_ref="file:paper.pdf",
                trust_level="external_untrusted",
                text_preview="The FastAPI framework provides async support for Python APIs.",
            ),
        ]
        pack = mmu.assemble(
            query="FastAPI",
            retrieval_results=dup_results,
        )
        # Should not have duplicate items in evidence section
        evidence = next((s for s in pack.sections if s.name == "retrieved_evidence"), None)
        if evidence:
            texts = [item.get("text", "") for item in evidence.items]
            unique_texts = set(texts)
            assert len(texts) == len(unique_texts), "Duplicate texts found"

    def test_assemble_handles_empty_inputs(self, mmu):
        pack = mmu.assemble(
            query="Hello",
            retrieval_results=[],
        )
        assert isinstance(pack, ContextPack)
        assert pack.used_tokens >= 0

    def test_assemble_system_instruction_included(self, mmu, sample_results):
        pack = mmu.assemble(
            query="What is FastAPI?",
            retrieval_results=sample_results,
            system_instruction="You are a helpful coding assistant.",
        )
        sections = {s.name for s in pack.sections}
        assert "system_instruction" in sections

    def test_assemble_sections_ordered_by_priority(self, mmu, sample_results):
        pack = mmu.assemble(
            query="FastAPI",
            retrieval_results=sample_results,
            system_instruction="Be helpful.",
        )
        priorities = [s.priority for s in pack.sections]
        assert priorities == sorted(priorities), f"Sections not ordered by priority: {priorities}"
