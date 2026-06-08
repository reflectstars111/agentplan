"""Task-mode API routes for Phase 2 Controller."""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from src.runtime.controller import Controller


class TaskRequest(BaseModel):
    query: str
    model: str = ""


class TaskResponse(BaseModel):
    response: str
    trace_id: str
    status: str
    intent: dict
    task_graph_summary: dict
    trace_ids: list[str] = Field(default_factory=list)
    conflicting_pairs: list[tuple[str, str]] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    writeback: dict = Field(default_factory=dict)
    writeback_confirmation_required: bool = False
    security: dict = Field(default_factory=dict)


def create_task_router(controller: Controller) -> APIRouter:
    router = APIRouter(prefix="/task", tags=["task"])

    @router.post("", response_model=TaskResponse)
    async def execute_task(req: TaskRequest):
        """Execute a full task cycle: IntentDecode → Plan → Schedule → Execute."""
        kwargs = {"model": req.model} if req.model else {}
        result = controller.process(req.query, **kwargs)
        return TaskResponse(
            response=result["response"],
            trace_id=result["trace_id"],
            status=result["status"],
            intent=result["intent"],
            task_graph_summary=result["task_graph_summary"],
            trace_ids=result.get("trace_ids", []),
            conflicting_pairs=result.get("conflicting_pairs", []),
            suggestions=result.get("suggestions", []),
            writeback=result.get("writeback", {}),
            writeback_confirmation_required=result.get(
                "writeback_confirmation_required", False
            ),
            security=result.get("security", {}),
        )

    return router
