from semstate.dependencies import DependencyCollector
from semstate.models import EdgeKind


class ContextPack:
    source_refs = ["state:context"]


def test_collects_observable_dependencies_and_deduplicates():
    collector = DependencyCollector()
    edges = collector.collect(
        target="state:output",
        read_set={"state:input": 2},
        task_inputs=[{"state_key": "state:task", "version": 3}],
        context_pack=ContextPack(),
        tool_args={
            "config": {
                "state_key": "state:input",
                "version": 2,
                "kind": "soft",
            }
        },
        source_refs=[{"state_key": "state:source", "version": 4}],
        current_versions={"state:context": 5},
    )

    by_source = {edge.source: edge for edge in edges}
    assert set(by_source) == {
        "state:input",
        "state:task",
        "state:context",
        "state:source",
    }
    assert by_source["state:input"].kind == EdgeKind.HARD
    assert by_source["state:context"].kind == EdgeKind.SOFT
    assert by_source["state:context"].source_version == 5
