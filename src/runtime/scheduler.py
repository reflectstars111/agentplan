"""Scheduler — topological task execution with retry and failure handling.

Executes a TaskGraph by walking nodes in dependency order, delegating each
node's execution to AgentRuntime.process_query().

Phase 3: Supports AgentRegistry-based routing and SharedBlackboard output.
Post-MVP: Supports parallel execution of independent nodes via ThreadPoolExecutor.

Maps to agent_os_initial_plan.md §6.2 (Controller execution cycle) and §8.3.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from src.models.task import Task, TaskStatus, TaskGraph
from src.models.blackboard import BlackboardEntry
from src.runtime.agent_runtime import AgentRuntime


class Scheduler:
    """Execute a TaskGraph respecting topological ordering.

    MVP: Synchronous execution (one node at a time). No concurrency.
    Handles retries (max 2 per node) and failure cascading.

    Phase 3: Accepts AgentRegistry for type-based routing and
    SharedBlackboard for inter-agent result exchange.
    """

    def __init__(
        self,
        agent_runtime: AgentRuntime,
        agent_registry=None,
        blackboard=None,
        trace_logger=None,
        page_fault=None,
    ):
        self.agent_runtime = agent_runtime
        self.agent_registry = agent_registry
        self.blackboard = blackboard
        self.trace_logger = trace_logger
        self.page_fault = page_fault

    def execute(
        self, task_graph: TaskGraph, request_id: str = "",
        parallel: bool = False,
    ) -> dict[str, Any]:
        """Execute all tasks in topological order.

        Args:
            task_graph: The TaskGraph to execute.
            request_id: Optional request ID for trace linking.
            parallel: If True, execute independent nodes in parallel via ThreadPoolExecutor.
                      Default False for backward compatibility.

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
                self._skip_remaining(task_graph, completed, failed)
                break

            if parallel and len(ready) > 1:
                max_workers = min(len(ready), getattr(self, '_max_workers', 4))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    for task_id in ready:
                        task = task_graph.get_node(task_id)
                        f = executor.submit(
                            self._execute_one, task, task_graph,
                            completed, failed, results, trace_ids,
                        )
                        futures[f] = task_id
                    for f in as_completed(futures):
                        f.result()  # Will raise if the task function raised
            else:
                for task_id in ready:
                    task = task_graph.get_node(task_id)
                    self._execute_one(task, task_graph, completed, failed, results, trace_ids)

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
        """Execute a single task node with retry logic.

        Routes to the correct agent if AgentRegistry is configured,
        otherwise falls back to the default agent_runtime.
        Writes output to SharedBlackboard if output_ref is set.
        """
        task.status = TaskStatus.RUNNING

        # Route to correct agent
        runtime = self._resolve_runtime(task)

        for attempt in range(task.max_retries + 1):
            try:
                # Build query: combine the task input with the task type
                query = task.input.get("query", task.input.get("task", ""))
                # Use page-fault-aware execution if configured
                if self.page_fault and hasattr(runtime, 'process_query_with_page_fault'):
                    result = runtime.process_query_with_page_fault(
                        query, request_id=task.task_id,
                    )
                else:
                    result = runtime.process_query(
                        query, request_id=task.task_id,
                    )

                task.status = TaskStatus.COMPLETED
                task.output = result
                task.trace_id = result.get("trace_id", "")
                results[task.task_id] = result
                completed.add(task.task_id)
                trace_ids.append(task.trace_id or "")

                # Record per-task trace step if trace_logger configured
                if self.trace_logger:
                    from src.models.trace import TraceStep, StepType, StepStatus
                    self.trace_logger.add_step(task.trace_id or f"trace_{task.task_id}", TraceStep(
                        step_id=f"step_{task.task_id}",
                        type=StepType.LLM_REASONING,
                        input={"task_type": task.task_type, "query": task.input.get("query", task.input.get("task", ""))},
                        output={
                            "response_length": len(result.get("response", "")),
                            "verified": result.get("verified", False),
                        },
                        status=StepStatus.SUCCESS,
                    ))

                # Write to shared blackboard if configured
                if self.blackboard and task.output_ref:
                    self.blackboard.write(BlackboardEntry(
                        key=task.output_ref,
                        value=result.get("response", ""),
                        created_by=task.agent_id or task.agent_type,
                        confidence=0.8,
                        source_refs=result.get("context_pack_source_refs", []),
                    ))

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

    def _resolve_runtime(self, task: Task) -> AgentRuntime:
        """Find the correct runtime for this task's agent_type.

        Returns the registered runtime or the default fallback.
        """
        if self.agent_registry and self.agent_registry.has_agent(task.agent_type):
            _, runtime = self.agent_registry.get_agent(task.agent_type)
            if runtime is not None:
                return runtime
        return self.agent_runtime

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
