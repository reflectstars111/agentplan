from semstate.models import (
    AnomalyType,
    DependencyEdge,
    EdgeKind,
    NodeStatus,
    StateNode,
    StateWrite,
    TransactionEnvelope,
)
from semstate.validation import ValidationIssue


def _seed(runtime, key, version, task, **value):
    runtime.seed_node(StateNode(
        key=key,
        version=version,
        value=value,
        producer_task=task,
    ))


def test_occ_passes_but_semantic_constraint_rejects(runtime):
    _seed(runtime, "deploy:database", 1, "task:db", max_connections=100, reserve=10)
    _seed(runtime, "deploy:service", 1, "task:service", connection_pool=40)
    envelope = TransactionEnvelope(
        agent_id="agent",
        task_id="task:resize",
        domain="deployment",
        read_set={"deploy:database": 1, "deploy:service": 1},
        write_set={
            "deploy:service": StateWrite(
                key="deploy:service",
                value={"connection_pool": 95},
            )
        },
    )

    assert all(
        runtime.store.current_version(key) == version
        for key, version in envelope.read_set.items()
    )
    result = runtime.commit(envelope)

    assert not result.committed
    assert result.decision.anomaly_type == AnomalyType.CROSS_KEY_CONSTRAINT
    assert runtime.store.get_node("deploy:service").value["connection_pool"] == 40


def test_hard_and_soft_invalidation_and_selective_repair(runtime):
    _seed(runtime, "source", 1, "task:source", value="v1")
    _seed(runtime, "hard-child", 1, "task:hard", value="derived")
    _seed(runtime, "soft-child", 1, "task:soft", value="summary")
    _seed(runtime, "grandchild", 1, "task:grand", value="artifact")
    _seed(runtime, "unrelated", 1, "task:other", value="stable")
    runtime.store.replace_dependencies("hard-child", [
        DependencyEdge("source", "hard-child", 1, kind=EdgeKind.HARD)
    ])
    runtime.store.replace_dependencies("soft-child", [
        DependencyEdge("source", "soft-child", 1, kind=EdgeKind.SOFT)
    ])
    runtime.store.replace_dependencies("grandchild", [
        DependencyEdge("hard-child", "grandchild", 1, kind=EdgeKind.HARD)
    ])

    result = runtime.commit(TransactionEnvelope(
        agent_id="agent",
        task_id="task:update-source",
        read_set={"source": 1},
        write_set={"source": StateWrite("source", {"value": "v2"})},
    ))

    assert result.committed
    assert runtime.store.get_node("hard-child").status == NodeStatus.STALE
    assert runtime.store.get_node("grandchild").status == NodeStatus.STALE
    assert runtime.store.get_node("soft-child").status == NodeStatus.NEEDS_VERIFICATION
    assert runtime.store.get_node("unrelated").status == NodeStatus.VALID

    plan = runtime.repair(result.conflict_id)
    assert plan.topological_order.index("hard-child") < plan.topological_order.index(
        "grandchild"
    )
    assert set(plan.rerun_tasks) == {"task:hard", "task:soft", "task:grand"}
    assert "task:other" not in plan.rerun_tasks


def test_revalidated_nodes_are_removed_from_repair_closure(runtime):
    _seed(runtime, "source", 1, "task:source", value="v1")
    _seed(runtime, "child", 1, "task:child", value="derived")
    runtime.store.replace_dependencies("child", [
        DependencyEdge("source", "child", 1, kind=EdgeKind.HARD)
    ])
    result = runtime.commit(TransactionEnvelope(
        agent_id="agent",
        task_id="task:update",
        read_set={"source": 1},
        write_set={"source": StateWrite("source", {"value": "v2"})},
    ))

    runtime.mark_revalidated(result.conflict_id, ["child"])
    plan = runtime.repair(result.conflict_id)

    assert plan.invalid_nodes == []
    assert plan.rerun_tasks == []


def test_commit_persists_dependencies_with_new_node(runtime):
    _seed(runtime, "input", 2, "task:input", value="ready")
    result = runtime.commit(TransactionEnvelope(
        agent_id="agent",
        task_id="task:output",
        read_set={"input": 2},
        write_set={"output": StateWrite("output", {"value": "built"})},
        dependencies=[
            DependencyEdge("input", "output", 2, origin="task_input")
        ],
    ))

    assert result.committed
    assert result.committed_versions == {"output": 1}
    assert runtime.store.incoming("output")[0].source == "input"


def test_executable_check_rejects_invalid_candidate(runtime):
    def health_check(state, envelope):
        candidate = state.get("service")
        if candidate and candidate.get("healthcheck") is False:
            return [ValidationIssue(
                anomaly_type=AnomalyType.EXECUTABLE_CHECK_FAILED,
                message="healthcheck failed",
                affected_states=["service"],
                details={"exit_code": 1},
            )]
        return []

    runtime.validator.register_executable_check(
        "deployment",
        "service_healthcheck",
        health_check,
    )
    result = runtime.commit(TransactionEnvelope(
        agent_id="agent",
        task_id="task:deploy",
        domain="deployment",
        write_set={
            "service": StateWrite("service", {"healthcheck": False})
        },
    ))

    assert not result.committed
    assert result.decision.anomaly_type == AnomalyType.EXECUTABLE_CHECK_FAILED
