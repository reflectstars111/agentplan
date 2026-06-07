"""Controller — orchestrates the full task execution cycle.

Wires IntentDecoder → Planner → TaskGraph → Scheduler into a coherent
control loop. Wraps AgentRuntime for backward-compatible simple queries.

Maps to agent_os_initial_plan.md §6 (Control System) and §19 (Phase 2).
"""

import uuid
from typing import Any
from src.config import Config
from src.models.trace import TraceStep, StepType, StepStatus
from src.runtime.agent_runtime import AgentRuntime
from src.runtime.intent_decoder import IntentDecoder
from src.runtime.planner import Planner
from src.runtime.scheduler import Scheduler
from src.runtime.trace_logger import TraceLogger


class Controller:
    """Task execution orchestrator.

    Process (task mode): IntentDecode → Plan → Schedule → Execute → Response
    Process_query (simple mode): passthrough to AgentRuntime.process_query()
    """

    def __init__(
        self,
        agent_runtime: AgentRuntime,
        intent_decoder: IntentDecoder,
        planner: Planner,
        scheduler: Scheduler,
        trace_logger: TraceLogger,
        config: Config | None = None,
        agent_registry=None,
        blackboard=None,
        merger=None,
        interrupt_handler=None,
    ):
        self.agent_runtime = agent_runtime
        self.intent_decoder = intent_decoder
        self.planner = planner
        self.scheduler = scheduler
        self.trace_logger = trace_logger
        self.config = config or Config()
        self.agent_registry = agent_registry
        self.blackboard = blackboard
        self.merger = merger
        self.interrupt_handler = interrupt_handler

    def process(
        self, query: str, request_id: str = "", model: str = ""
    ) -> dict[str, Any]:
        """Full task execution cycle.

        Args:
            query: User's natural language request.
            request_id: Optional request ID (auto-generated if empty).
            model: Optional LLM model override.

        Returns:
            Dict with: response, intent, task_graph_summary, results, status, trace_ids
        """
        if not request_id:
            request_id = f"req_{uuid.uuid4().hex[:12]}"

        # Start a controller-level trace
        trace = self.trace_logger.start_trace(request_id)

        # Phase 1: Intent Decode
        intent = self.intent_decoder.decode(query, request_id)
        self.trace_logger.add_step(trace.trace_id, TraceStep(
            step_id="step_intent",
            type=StepType.INTENT_DECODE,
            input={"query": query},
            output={
                "intent_type": intent.intent_type.value,
                "confidence": intent.confidence,
                "entities": intent.entities,
            },
        ))

        # Check for halt before planning
        if self.interrupt_handler and self.interrupt_handler.is_halted():
            return {"response": "Task halted by interrupt.", "status": "halted",
                    "intent": intent.to_dict(), "results": {}, "trace_ids": []}

        # Phase 2: Plan
        task_graph = self.planner.plan(intent)

        self.trace_logger.add_step(trace.trace_id, TraceStep(
            step_id="step_plan",
            type=StepType.PLAN,
            input={"intent_type": intent.intent_type.value, "entities": intent.entities},
            output={
                "node_count": task_graph.node_count(),
                "nodes": list(task_graph.nodes.keys()),
            },
        ))

        # Check for halt before execution
        if self.interrupt_handler and self.interrupt_handler.is_halted():
            return {"response": "Task halted before execution.", "status": "halted",
                    "intent": intent.to_dict(), "task_graph_summary": {
                        "node_count": task_graph.node_count()}, "results": {}, "trace_ids": []}

        # Phase 3: Schedule + Execute
        exec_result = self.scheduler.execute(task_graph, request_id)

        self.trace_logger.add_step(trace.trace_id, TraceStep(
            step_id="step_schedule",
            type=StepType.SCHEDULE,
            input={"node_count": task_graph.node_count()},
            output={
                "completed": len(exec_result["results"]),
                "failed": len(exec_result.get("failed_tasks", [])),
                "status": exec_result["status"],
            },
        ))

        # Phase 4: Assemble response
        final_response = self._assemble_response(task_graph, exec_result, intent)

        self.trace_logger.add_step(trace.trace_id, TraceStep(
            step_id="step_respond",
            type=StepType.RESPOND,
            input={"intent_type": intent.intent_type.value},
            output={
                "status": exec_result["status"],
                "num_tasks": task_graph.node_count(),
                "num_completed": len(exec_result["results"]),
            },
        ))

        return {
            "response": final_response,
            "trace_id": trace.trace_id,
            "intent": intent.to_dict(),
            "task_graph_summary": {
                "node_count": task_graph.node_count(),
                "completed": len(exec_result["results"]),
                "failed": len(exec_result.get("failed_tasks", [])),
            },
            "results": exec_result["results"],
            "status": exec_result["status"],
            "trace_ids": exec_result.get("trace_ids", []),
        }

    def process_query(
        self, query: str, request_id: str = ""
    ) -> dict[str, Any]:
        """Simple mode: pass-through to AgentRuntime.process_query().

        Preserved for backward compatibility with the existing /query API.
        """
        return self.agent_runtime.process_query(query, request_id)

    def _assemble_response(self, task_graph, exec_result, intent) -> str:
        """Assemble a final response from task execution results."""
        results = exec_result.get("results", {})

        # Collect outputs from all completed nodes in order
        parts = []
        for tid in task_graph.topological_sort():
            if tid in results:
                node_result = results[tid]
                response = node_result.get("response", "")
                if response and len(response) > 10:
                    parts.append(response)

        if not parts:
            if intent.intent_type.value == "general":
                return f"I processed: {intent.original_query}"
            return f"Task execution {exec_result['status']}: {len(results)} nodes completed."

        # For simple graphs (1-2 nodes), return the last response directly
        if len(parts) <= 2:
            return parts[-1]

        # For complex graphs, join with separators
        return "\n\n".join(parts)
