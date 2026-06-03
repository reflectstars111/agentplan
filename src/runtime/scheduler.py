"""Scheduler — topological task execution with retry and failure handling.

Executes a TaskGraph by walking nodes in dependency order, delegating each
node's execution to AgentRuntime.process_query().

Maps to agent_os_initial_plan.md §6.2 (Controller execution cycle) and §8.3.
"""

from typing import Any
from src.models.task import Task, TaskStatus, TaskGraph
from src.runtime.agent_runtime import AgentRuntime


class Scheduler:
    """Execute a TaskGraph respecting topological ordering.

    MVP: Synchronous execution (one node at a time). No concurrency.
    Handles retries (max 2 per node) and failure cascading.
    """

    def __init__(self, agent_runtime: AgentRuntime):
        self.agent_runtime = agent_runtime

    def execute(
        self, task_graph: TaskGraph, request_id: str = ""
    ) -> dict[str, Any]:
        """Execute all tasks in topological order.

        Args:
            task_graph: The TaskGraph to execute.
            request_id: Optional request ID for trace linking.

        Returns:
            Dict with:
              - results: dict[task_id -> output_dict]
              - status: "completed" | "partial_failure"
              - trace_ids: list of all trace IDs generated
              - failed_tasks: list of failed task_ids
        """
        completed: set[str] = set()
        failed: set[str] = set()
        results: dict[str, dict] = {}
        trace_ids: list[str] = []

        # Main execution loop
        while not task_graph.all_completed():
            ready = task_graph.get_ready_nodes(completed)

            if not ready:
                # Remaining nodes have unsatisfiable dependencies due to failures
                self._skip_remaining(task_graph, completed, failed)
                break

            for task_id in ready:
                task = task_graph.get_node(task_id)
                self._execute_one(task, task_graph, completed, failed, results, trace_ids)

        # Determine final status
        status = "completed" if not failed else "partial_failure"

        return {
            "results": results,
            "status": status,
            "trace_ids": trace_ids,
            "failed_tasks": list(failed),
        }

    def _execute_one(
        self,
        task: Task,
        task_graph: TaskGraph,
        completed: set[str],
        failed: set[str],
        results: dict,
        trace_ids: list[str],
    ) -> None:
        """Execute a single task node with retry logic."""
        task.status = TaskStatus.RUNNING

        for attempt in range(task.max_retries + 1):
            try:
                # Build query: combine the task input with the task type
                query = task.input.get("query", task.input.get("task", ""))
                result = self.agent_runtime.process_query(
                    query, request_id=task.task_id
                )

                task.status = TaskStatus.COMPLETED
                task.output = result
                task.trace_id = result.get("trace_id", "")
                results[task.task_id] = result
                completed.add(task.task_id)
                trace_ids.append(task.trace_id or "")
                return

            except Exception as e:
                task.retry_count = attempt + 1
                if attempt >= task.max_retries:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    failed.add(task.task_id)
                    self._propagate_failure(task_graph, task.task_id)
                    return
                # Else: retry

    def _propagate_failure(self, task_graph: TaskGraph, failed_id: str) -> None:
        """Mark all transitive dependents of a failed task as SKIPPED."""
        to_skip = set()
        queue = [failed_id]
        while queue:
            current = queue.pop(0)
            for succ in task_graph.adj_out.get(current, set()):
                if succ not in to_skip:
                    to_skip.add(succ)
                    queue.append(succ)

        for tid in to_skip:
            try:
                node = task_graph.get_node(tid)
                if node.status in (TaskStatus.CREATED, TaskStatus.READY):
                    node.status = TaskStatus.SKIPPED
                    node.error = f"Skipped: dependency '{failed_id}' failed"
            except KeyError:
                pass

    def _skip_remaining(
        self,
        task_graph: TaskGraph,
        completed: set[str],
        failed: set[str],
    ) -> None:
        """Skip any remaining CREATED/READY nodes that can't be executed."""
        for tid, task in task_graph.nodes.items():
            if task.status in (TaskStatus.CREATED, TaskStatus.READY):
                task.status = TaskStatus.SKIPPED
                task.error = "Skipped: dependencies could not be satisfied"
