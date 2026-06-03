"""Task and TaskGraph — DAG-based task decomposition.

Maps to agent_os_initial_plan.md §8 (Task Thread Model) and §19 (Phase 2).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from dataclasses_json import dataclass_json


class TaskStatus(str, Enum):
    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass_json
@dataclass
class Task:
    """A single task node in a TaskGraph.

    Maps to the tasks DB table (agent_os_initial_plan.md §21.2).
    """

    task_id: str
    task_type: str                      # e.g. "retrieve", "reason", "verify", "writeback"
    agent_type: str = "worker"           # which agent executes (Phase 2: all "worker")
    status: TaskStatus = TaskStatus.CREATED
    dependencies: list[str] = field(default_factory=list)
    input: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    priority: int = 5
    retry_count: int = 0
    max_retries: int = 2
    error: Optional[str] = None
    trace_id: Optional[str] = None
    created_at: str = ""
    completed_at: Optional[str] = None


class TaskGraph:
    """In-memory adjacency-list DAG of Task nodes.

    Supports cycle detection (DFS), topological sort (Kahn), and
    ready-node queries for scheduler-driven execution.
    """

    def __init__(self, intent_id: str):
        self.intent_id = intent_id
        self.nodes: dict[str, Task] = {}
        self.adj_in: dict[str, set[str]] = {}    # node -> prerequisites
        self.adj_out: dict[str, set[str]] = {}   # node -> successors

    def add_node(self, task: Task) -> None:
        """Add a task node to the graph. Replaces if task_id already exists."""
        tid = task.task_id
        self.nodes[tid] = task
        if tid not in self.adj_in:
            self.adj_in[tid] = set()
        if tid not in self.adj_out:
            self.adj_out[tid] = set()

    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add a dependency edge: from_id must complete before to_id can start."""
        if from_id not in self.nodes or to_id not in self.nodes:
            raise ValueError(
                f"Both nodes must exist: '{from_id}' and '{to_id}'"
            )
        self.adj_in.setdefault(to_id, set()).add(from_id)
        self.adj_out.setdefault(from_id, set()).add(to_id)

    def get_node(self, task_id: str) -> Task:
        """Get a node by ID. Raises KeyError if not found."""
        if task_id not in self.nodes:
            raise KeyError(f"Task '{task_id}' not found in graph")
        return self.nodes[task_id]

    def node_count(self) -> int:
        return len(self.nodes)

    def all_completed(self) -> bool:
        """True if every node is COMPLETED, FAILED, or SKIPPED."""
        terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED}
        return all(n.status in terminal for n in self.nodes.values())

    def get_ready_nodes(self, completed: set[str]) -> list[str]:
        """Return task_ids whose dependencies are all in `completed`.

        Only returns nodes in CREATED or READY status.
        """
        ready = []
        for tid, task in self.nodes.items():
            if task.status not in (TaskStatus.CREATED, TaskStatus.READY):
                continue
            deps = self.adj_in.get(tid, set())
            if deps.issubset(completed):
                ready.append(tid)
        return ready

    def validate_acyclic(self) -> bool:
        """Returns True if the graph has no directed cycles (DFS tricolor)."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {tid: WHITE for tid in self.nodes}

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for succ in self.adj_out.get(node, set()):
                if color[succ] == GRAY:
                    return False  # back edge -> cycle
                if color[succ] == WHITE and not dfs(succ):
                    return False
            color[node] = BLACK
            return True

        for tid in self.nodes:
            if color[tid] == WHITE:
                if not dfs(tid):
                    return False
        return True

    def topological_sort(self) -> list[str]:
        """Return task_ids in topological order (Kahn's algorithm).

        Raises ValueError if a cycle is detected.
        """
        in_degree = {
            tid: len(self.adj_in.get(tid, set())) for tid in self.nodes
        }
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for succ in self.adj_out.get(node, set()):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(result) != len(self.nodes):
            raise ValueError(
                f"Cycle detected in task graph '{self.intent_id}': "
                f"only {len(result)}/{len(self.nodes)} nodes sorted"
            )
        return result

    def to_dict(self) -> dict:
        """Serialize the graph to a dict (for debugging/API responses)."""
        return {
            "intent_id": self.intent_id,
            "node_count": self.node_count(),
            "nodes": {
                tid: {
                    "task_type": t.task_type,
                    "status": t.status.value,
                    "dependencies": list(self.adj_in.get(tid, set())),
                }
                for tid, t in self.nodes.items()
            },
        }
