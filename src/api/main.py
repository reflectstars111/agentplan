"""FastAPI application factory for Agent-OS MVP."""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from src.runtime.agent_runtime import AgentRuntime
from src.api.routes import create_router


def create_app(
    runtime: AgentRuntime,
    controller=None,
) -> FastAPI:
    app = FastAPI(
        title="Agent-OS MVP",
        description="Von Neumann-inspired Agent Runtime with multi-level memory, "
                    "hybrid retrieval, context assembly, and trace observability.",
        version="0.2.0",
    )
    app.state.runtime = runtime
    app.state.controller = controller
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    router = create_router(runtime)
    app.include_router(router)

    # Wire task execution endpoints if Controller is provided
    if controller is not None:
        from src.api.task_routes import create_task_router
        task_router = create_task_router(controller)
        app.include_router(task_router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.2.0"}

    # Serve GUI index.html at root if built
    gui_dist = Path(__file__).parent.parent.parent / "gui" / "dist"
    if gui_dist.exists():
        @app.get("/")
        async def gui_index():
            index_path = gui_dist / "index.html"
            return HTMLResponse(index_path.read_text(encoding="utf-8"))

        from fastapi.staticfiles import StaticFiles
        app.mount("/assets", StaticFiles(directory=str(gui_dist / "assets")), name="gui_assets")

    return app
