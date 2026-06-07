"""FastAPI application factory for Agent-OS MVP."""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

    # CORS for GUI dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    router = create_router(runtime)
    app.include_router(router)

    # Serve built GUI if available
    gui_path = Path(__file__).parent.parent.parent / "gui" / "dist"
    if gui_path.exists():
        app.mount("/", StaticFiles(directory=str(gui_path), html=True), name="gui")

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.2.0"}

    return app
