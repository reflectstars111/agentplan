"""InterruptHandler — HALT/PAUSE/RESUME for task execution.

Maps to agent_os_initial_plan.md §6.1 #8 (Interrupt Handler), §6.3 (HALT).
"""

from src.models.task import Task


class InterruptHandler:
    """Supports HALT (stop all), PAUSE (suspend task), RESUME (continue task)."""

    def __init__(self):
        self._halted = False
        self._paused: dict[str, Task] = {}

    def halt(self, reason: str = "") -> None:
        self._halted = True
        self._halt_reason = reason

    def is_halted(self) -> bool:
        return self._halted

    def pause(self, task_id: str, task: Task) -> None:
        self._paused[task_id] = task

    def resume(self, task_id: str) -> Task | None:
        return self._paused.pop(task_id, None)

    def get_paused_tasks(self) -> list[str]:
        return list(self._paused.keys())
