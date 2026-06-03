"""Task-mode API routes for Phase 2 Controller."""

from fastapi import APIRouter
from pydantic import BaseModel
from src.runtime.controller import Controller


class TaskRequest(BaseModel):
    query: str


class TaskResponse(BaseModel):
    response: str
    status: str
    intent: dict
    task_graph_summary: dict
    trace_ids: list[str] = []


def create_task_router(controller: Controller) -> APIRouter:
    router = APIRouter(prefix="/task", tags=["task"])

    @router.post("", response_model=TaskResponse)
    async def execute_task(req: TaskRequest):
        """Execute a full task cycle: IntentDecode → Plan → Schedule → Execute."""
        result = controller.process(req.query)
        return TaskResponse(
            response=result["response"],
            status=result["status"],
            intent=result["intent"],
            task_graph_summary=result["task_graph_summary"],
            trace_ids=result.get("trace_ids", []),
        )

    return router
