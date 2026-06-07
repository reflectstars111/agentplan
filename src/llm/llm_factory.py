"""LLM factory — creates callable LLM functions for OpenAI, DeepSeek, Anthropic."""

import os
from enum import Enum
from typing import Callable


class LLMProvider(str, Enum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"
    MOCK = "mock"


# Default base URLs
DEFAULT_BASE_URLS = {
    LLMProvider.OPENAI: "https://api.openai.com/v1",
    LLMProvider.DEEPSEEK: "https://api.deepseek.com",
}


def create_llm_fn(
    provider: str = "openai",
    model: str = "gpt-4o",
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.3,
) -> Callable:
    """Create a callable LLM function.

    Args:
        provider: "openai" | "deepseek" | "anthropic" | "mock"
        model: Model name (e.g. "gpt-4o", "deepseek-chat", "claude-sonnet-4-6")
        api_key: API key. Falls back to env var: OPENAI_API_KEY, DEEPSEEK_API_KEY, ANTHROPIC_API_KEY
        base_url: Override API base URL.
        temperature: Response temperature (0.0-1.0).

    Returns:
        Callable[[context_pack, query], str] — the LLM function.
    """
    provider = LLMProvider(provider)

    if provider == LLMProvider.MOCK:
        return _create_mock_llm()

    # Resolve API key from env if not provided
    env_key_map = {
        LLMProvider.OPENAI: "OPENAI_API_KEY",
        LLMProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
        LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    }
    if not api_key:
        api_key = os.environ.get(env_key_map.get(provider, ""))

    # Resolve base URL
    if not base_url:
        base_url = DEFAULT_BASE_URLS.get(provider)

    if provider == LLMProvider.ANTHROPIC:
        return _create_anthropic_llm(model, api_key, temperature)
    else:
        # OpenAI and DeepSeek both use the OpenAI client
        return _create_openai_compatible_llm(model, api_key, base_url, temperature)


def _create_openai_compatible_llm(
    model: str, api_key: str, base_url: str | None, temperature: float
) -> Callable:
    """Create LLM function using OpenAI-compatible API.

    Client is created lazily on first call, so startup doesn't fail
    if the API key env var hasn't been read yet.
    """
    _client = None

    def _get_client():
        nonlocal _client
        if _client is None:
            from openai import OpenAI
            key = api_key
            if not key:
                key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
            if not key:
                raise ValueError(
                    "No API key found. Set environment variable: "
                    "OPENAI_API_KEY, DEEPSEEK_API_KEY, or ANTHROPIC_API_KEY"
                )
            _client = OpenAI(api_key=key, base_url=base_url)
        return _client

    def llm_fn(context_pack, query: str) -> str:
        from src.llm.prompts.templates import build_prompt
        prompt = build_prompt(context_pack, query, role="worker")
        try:
            response = _get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"[LLM Error: {e}]"

    return llm_fn


def _create_anthropic_llm(
    model: str, api_key: str, temperature: float
) -> Callable:
    """Create LLM function using Anthropic API."""
    def llm_fn(context_pack, query: str) -> str:
        try:
            import anthropic
            from src.llm.prompts.templates import build_prompt
            prompt = build_prompt(context_pack, query, role="worker")
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except ImportError:
            return "[Error: anthropic package not installed. pip install anthropic]"
        except Exception as e:
            return f"[LLM Error: {e}]"

    return llm_fn


def _create_mock_llm() -> Callable:
    """Mock LLM that concatenates context evidence (for testing)."""
    def llm_fn(context_pack, query: str) -> str:
        if context_pack:
            for section in context_pack.sections:
                if section.name == "retrieved_evidence" and section.items:
                    parts = []
                    for item in section.items[:3]:
                        src = item.get("source_ref", "unknown")
                        text = item.get("text", "")
                        if text:
                            parts.append(f"According to {src}: {text}")
                    if parts:
                        return " ".join(parts)
        return f"No relevant information found for: {query}"
    return llm_fn
