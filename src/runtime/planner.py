"""Planner — template-based task decomposition.

Converts an Intent into a TaskGraph using predefined DAG templates.
Each IntentType maps to a template function that builds a 1-5 node DAG.

Maps to agent_os_initial_plan.md §6 (Control System) and §19 (Phase 2).
"""

import uuid
from src.models.intent import Intent, IntentType
from src.models.task import Task, TaskGraph


class Planner:
    """Decompose an Intent into a TaskGraph using templates.

    MVP: Each IntentType maps to a predefined DAG template.
    Future: swap in LLM-based planner behind the same plan() interface.
    """

    def plan(self, intent: Intent) -> TaskGraph:
        """Decompose an Intent into an executable TaskGraph.

        Args:
            intent: The structured intent from IntentDecoder.

        Returns:
            A TaskGraph with ready-to-execute Task nodes.
        """
        template_fn = {
            IntentType.DOCUMENT_QA: self._plan_doc_qa,
            IntentType.CODE_ANALYSIS: self._plan_code_analysis,
            IntentType.MULTI_TURN: self._plan_multi_turn,
            IntentType.MEMORY_QUERY: self._plan_memory_query,
            IntentType.GENERAL: self._plan_general,
        }.get(intent.intent_type, self._plan_general)

        graph = TaskGraph(intent_id=intent.intent_id)
        template_fn(graph, intent)
        return graph

    # ── Templates ──────────────────────────────────────────────

    def _plan_doc_qa(self, g: TaskGraph, intent: Intent) -> None:
        """3-node linear: retrieve -> reason -> verify."""
        t_retrieve = self._make_task("retrieve", {
            "query": intent.original_query,
            "task": "retrieve relevant document chunks",
        })
        t_reason = self._make_task("reason", {
            "query": intent.original_query,
            "task": "answer the question based on retrieved evidence",
        }, deps=[t_retrieve.task_id])
        t_verify = self._make_task("verify", {
            "query": intent.original_query,
            "task": "verify the answer against source references",
        }, deps=[t_reason.task_id])

        for t in [t_retrieve, t_reason, t_verify]:
            g.add_node(t)
        g.add_edge(t_retrieve.task_id, t_reason.task_id)
        g.add_edge(t_reason.task_id, t_verify.task_id)

    def _plan_code_analysis(self, g: TaskGraph, intent: Intent) -> None:
        """3-node linear: retrieve -> analyze -> verify."""
        t_retrieve = self._make_task("retrieve", {
            "query": intent.original_query,
            "entities": intent.entities,
            "task": "retrieve relevant code files",
        })
        t_analyze = self._make_task("analyze", {
            "query": intent.original_query,
            "entities": intent.entities,
            "task": "analyze the code and locate the target functionality",
        }, deps=[t_retrieve.task_id])
        t_verify = self._make_task("verify", {
            "query": intent.original_query,
            "task": "verify code locations against the codebase",
        }, deps=[t_analyze.task_id])

        for t in [t_retrieve, t_analyze, t_verify]:
            g.add_node(t)
        g.add_edge(t_retrieve.task_id, t_analyze.task_id)
        g.add_edge(t_analyze.task_id, t_verify.task_id)

    def _plan_multi_turn(self, g: TaskGraph, intent: Intent) -> None:
        """5-node diamond: retrieve_memory + retrieve_chunks -> merge -> reason -> verify -> writeback."""
        t_mem = self._make_task("retrieve_memory", {
            "query": intent.original_query,
            "task": "retrieve relevant working memories and past decisions",
        })
        t_chunks = self._make_task("retrieve_chunks", {
            "query": intent.original_query,
            "task": "retrieve relevant document chunks",
        })
        t_merge = self._make_task("merge", {
            "query": intent.original_query,
            "task": "merge memory and chunk retrieval results",
        }, deps=[t_mem.task_id, t_chunks.task_id])
        t_reason = self._make_task("reason", {
            "query": intent.original_query,
            "task": "generate response incorporating memories and evidence",
        }, deps=[t_merge.task_id])
        t_verify = self._make_task("verify", {
            "query": intent.original_query,
            "task": "verify response against sources and memories",
        }, deps=[t_reason.task_id])
        t_writeback = self._make_task("writeback", {
            "query": intent.original_query,
            "task": "evaluate if decisions should be written to memory",
        }, deps=[t_verify.task_id])

        for t in [t_mem, t_chunks, t_merge, t_reason, t_verify, t_writeback]:
            g.add_node(t)
        g.add_edge(t_mem.task_id, t_merge.task_id)
        g.add_edge(t_chunks.task_id, t_merge.task_id)
        g.add_edge(t_merge.task_id, t_reason.task_id)
        g.add_edge(t_reason.task_id, t_verify.task_id)
        g.add_edge(t_verify.task_id, t_writeback.task_id)

    def _plan_memory_query(self, g: TaskGraph, intent: Intent) -> None:
        """2-node: retrieve_memory -> reason."""
        t_mem = self._make_task("retrieve_memory", {
            "query": intent.original_query,
            "task": "retrieve relevant memories",
        })
        t_reason = self._make_task("reason", {
            "query": intent.original_query,
            "task": "answer based on memory context",
        }, deps=[t_mem.task_id])

        for t in [t_mem, t_reason]:
            g.add_node(t)
        g.add_edge(t_mem.task_id, t_reason.task_id)

    def _plan_general(self, g: TaskGraph, intent: Intent) -> None:
        """1-node: simple retrieve+reason."""
        t = self._make_task("general", {
            "query": intent.original_query,
            "task": "retrieve and answer",
        })
        g.add_node(t)

    # ── Helpers ────────────────────────────────────────────────

    # Mapping from task_type to agent_type
    _AGENT_TYPE_MAP: dict[str, str] = {
        "retrieve": "worker",
        "retrieve_memory": "worker",
        "retrieve_chunks": "worker",
        "reason": "worker",
        "analyze": "worker",
        "merge": "worker",
        "general": "worker",
        "writeback": "worker",
        "verify": "verifier",
    }

    def _make_task(
        self,
        task_type: str,
        input_data: dict,
        deps: list[str] | None = None,
    ) -> Task:
        """Create a Task with consistent defaults."""
        agent_type = self._AGENT_TYPE_MAP.get(task_type, "worker")
        return Task(
            task_id=f"task_{uuid.uuid4().hex[:12]}",
            task_type=task_type,
            agent_type=agent_type,
            dependencies=deps or [],
            input=input_data,
            priority=5,
            max_retries=2,
        )
