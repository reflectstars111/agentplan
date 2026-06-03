"""FastAPI application factory for Agent-OS MVP."""

from fastapi import FastAPI
from src.runtime.agent_runtime import AgentRuntime
from src.api.routes import create_router


def create_app(runtime: AgentRuntime) -> FastAPI:
    """Create a FastAPI app wired to an AgentRuntime instance.

    Args:
        runtime: A fully configured AgentRuntime instance.

    Returns:
        FastAPI application ready to serve.
    """
    app = FastAPI(
        title="Agent-OS MVP",
        description="Von Neumann-inspired Agent Runtime with multi-level memory, "
                    "hybrid retrieval, context assembly, and trace observability.",
        version="0.1.0",
    )

    router = create_router(runtime)
    app.include_router(router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    return app
