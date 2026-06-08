"""Tests for Executor."""

import pytest

from ard.runtime.executor import Executor, ExecutorResponse
from ard.context.pack import ContextPack, ContextSection


class TestExecutor:
    @pytest.fixture
    def executor(self):
        return Executor()

    @pytest.fixture
    def context_pack(self):
        pack = ContextPack(
            context_id="ctx_test", task_id="task_test",
            agent_id="agent_test", budget=1000,
        )
        pack.sections = [
            ContextSection(name="current_query", tokens=10, priority=2,
                          items=[{"text": "What is AI?", "trust_level": "user_instruction"}]),
            ContextSection(name="retrieved_evidence", tokens=50, priority=6,
                          items=[{"text": "AI is artificial intelligence.", "trust_level": "user_provided_data"}]),
        ]
        return pack

    def test_think_returns_response(self, executor, context_pack):
        resp = executor.think(context_pack)
        assert isinstance(resp, ExecutorResponse)
        assert resp.answer
        assert resp.context_id == "ctx_test"

    def test_mock_llm_returns_text(self, executor, context_pack):
        result = executor._mock_llm(context_pack, query="What is AI?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_custom_llm_fn(self, context_pack):
        def my_llm(prompt, system=""):
            return f"Custom: {len(prompt)} chars, system={bool(system)}"

        executor = Executor(my_llm)
        resp = executor.think(context_pack)
        assert resp.answer.startswith("Custom:")

    def test_tokens_used_in_response(self, executor, context_pack):
        resp = executor.think(context_pack)
        assert resp.tokens_used > 0
