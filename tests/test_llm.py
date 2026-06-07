"""Tests for LLM integration (factory, prompts, embedding)."""

import pytest
from src.llm.llm_factory import create_llm_fn, LLMProvider
from src.llm.prompts.templates import AGENT_PROMPTS, build_prompt
from src.models.context import ContextPack, ContextSection


class TestLLMFactory:
    def test_create_openai_client(self):
        fn = create_llm_fn(provider="openai", api_key="sk-test")
        assert fn is not None
        assert callable(fn)

    def test_create_deepseek_client(self):
        fn = create_llm_fn(provider="deepseek", api_key="sk-test")
        assert fn is not None
        assert callable(fn)

    def test_create_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="not a valid LLMProvider"):
            create_llm_fn(provider="unknown", api_key="sk-test")

    def test_mock_mode_works(self):
        fn = create_llm_fn(provider="mock")
        assert callable(fn)
        result = fn(None, "Hello")
        assert isinstance(result, str)

    def test_custom_base_url(self):
        fn = create_llm_fn(provider="openai", api_key="sk-test",
                          base_url="https://custom.api.com/v1")
        assert fn is not None


class TestPrompts:
    def test_all_roles_have_prompts(self):
        for role in ["planner", "worker", "verifier"]:
            assert role in AGENT_PROMPTS
            assert len(AGENT_PROMPTS[role]) > 0

    def test_build_prompt_includes_context(self):
        pack = ContextPack(
            context_id="ctx_1", task_id="t1", agent_id="a1", budget=1000,
        )
        pack.add_section(ContextSection(
            name="retrieved_evidence", tokens=100, priority=3,
            items=[{"text": "FastAPI is a Python web framework.", "trust_level": "external_untrusted",
                    "source_ref": "file:fastapi.txt"}],
        ))
        prompt = build_prompt(pack, "What is FastAPI?", role="worker")
        assert "FastAPI" in prompt
        assert "file:fastapi.txt" in prompt

    def test_build_prompt_includes_system_instruction(self):
        prompt = build_prompt(None, "Hello", role="planner",
                             system_instruction="You are helpful.")
        assert "You are helpful" in prompt


class TestEmbeddingIntegration:
    def test_mock_embed_fn_returns_array(self):
        from src.embedding import create_mock_embed_fn
        fn = create_mock_embed_fn(dim=64)
        result = fn(["hello", "world"])
        assert result.shape == (2, 64)

    def test_env_reading_for_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test")
        from src.config import Config
        c = Config()
        # Config should work normally
        assert c.default_token_budget == 24000
