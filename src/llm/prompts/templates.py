"""Prompt templates for each AgentRole."""

from src.models.context import ContextPack

AGENT_PROMPTS = {
    "planner": (
        "You are a task planning agent. Your job is to analyze user requests "
        "and break them down into structured subtasks. For each subtask, specify "
        "the task type, required inputs, and dependencies. Be precise and concise."
    ),
    "worker": (
        "You are a research and execution agent. Answer the user's question "
        "based ONLY on the provided context below. If the context does not contain "
        "enough information, say so clearly. Always cite your sources using "
        "[source:...] notation. Do not fabricate information."
    ),
    "verifier": (
        "You are a verification agent. Your job is to check whether the provided "
        "response is factually grounded in the source references. Flag any claims "
        "that cannot be traced to a source. Identify contradictions and "
        "low-confidence language. Be strict — prefer false positives over "
        "letting unverified claims through."
    ),
}


def build_prompt(
    context_pack: ContextPack | None,
    query: str,
    role: str = "worker",
    system_instruction: str = "",
) -> str:
    """Build a full prompt from system instruction + context + query.

    Args:
        context_pack: The assembled context pack (or None for simple prompts).
        query: The user's query or task description.
        role: Agent role for system prompt selection.
        system_instruction: Override system instruction (overrides role default).

    Returns:
        Full prompt string ready for LLM consumption.
    """
    # System prompt
    if system_instruction:
        sys_prompt = system_instruction
    else:
        sys_prompt = AGENT_PROMPTS.get(role, AGENT_PROMPTS["worker"])

    parts = [sys_prompt]

    # Context sections
    if context_pack and context_pack.sections:
        parts.append("\n\n--- CONTEXT ---\n")
        for section in context_pack.sections:
            parts.append(f"\n[{section.name.upper()}]")
            for item in section.items:
                src = item.get("source_ref", "unknown")
                text = item.get("text", "")
                if text:
                    parts.append(f"[source:{src}] {text}")

    # Query
    parts.append(f"\n\n--- QUERY ---\n{query}")
    parts.append("\n\n--- RESPONSE ---\n")

    return "\n".join(parts)
