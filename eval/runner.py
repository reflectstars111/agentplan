"""Evaluation runner — runs sample queries against the Agent-OS runtime
and computes retrieval+verification metrics per scenario.

Usage:
    python -m eval.runner              # run all 5 scenarios
    python -m eval.runner --scenario s1_doc_qa  # run a single scenario
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.scenarios import ALL_SCENARIOS, EvalScenario
from eval.metrics import precision_at_k, recall_at_k, hit_at_k


def run_eval(scenario_filter: str = "", k: int = 10) -> dict:
    """Run evaluation across all scenarios and return results keyed by scenario_id.

    Uses mock embedding + mock LLM for deterministic, fast evaluation.
    """
    from src.__main__ import build_runtime

    runtime = build_runtime(llm_provider="mock", embed_provider="mock")

    scenarios = ALL_SCENARIOS
    if scenario_filter:
        scenarios = [s for s in scenarios if s.scenario_id == scenario_filter]
        if not scenarios:
            print(f"Scenario '{scenario_filter}' not found. Available: "
                  f"{[s.scenario_id for s in ALL_SCENARIOS]}")
            return {}

    results: dict[str, dict] = {}

    for scenario in scenarios:
        query_results = []

        for query_text in scenario.sample_queries:
            # Upload a placeholder document so there is something to retrieve
            if scenario.task_type in ("doc_qa", "code_locator"):
                source_id = runtime.upload_text(
                    _make_placeholder_content(scenario), "eval_doc.txt"
                )

            result = runtime.process_query(query_text)

            verified = result.get("verified", False)
            unverified_count = len(result.get("unverified_claims", []))
            trace_id = result.get("trace_id", "")

            # Extract retrieval info from trace
            trace = runtime.trace_logger.get_trace(trace_id)
            num_retrieved = 0
            for step in (trace.steps if trace else []):
                if step.type.value == "retrieve_file":
                    num_retrieved = step.output.get("num_results", 0)
                    break

            query_results.append({
                "query": query_text[:80],
                "verified": verified,
                "unverified_claims": unverified_count,
                "num_retrieved": num_retrieved,
            })

        # Compute aggregate metrics for the scenario
        total = len(query_results)
        verified_count = sum(1 for r in query_results if r["verified"])
        avg_retrieved = (
            sum(r["num_retrieved"] for r in query_results) / total
            if total > 0 else 0
        )

        results[scenario.scenario_id] = {
            "name": scenario.name,
            "task_type": scenario.task_type,
            "num_queries": total,
            "queries": query_results,
            "verified_rate": verified_count / total if total > 0 else 0,
            "avg_retrieved": round(avg_retrieved, 1),
            "success_criteria": scenario.success_criteria,
        }

    return results


def print_report(results: dict) -> None:
    """Print evaluation results as a Markdown report."""
    if not results:
        print("No results to report.")
        return

    print("\n# Agent-OS Evaluation Report\n")

    # Summary table
    print("## Summary\n")
    print("| Scenario | Type | Queries | Verified Rate | Avg Retrieved |")
    print("|----------|------|---------|---------------|---------------|")
    for sid, data in results.items():
        print(f"| {data['name']} | {data['task_type']} | {data['num_queries']} "
              f"| {data['verified_rate']:.0%} | {data['avg_retrieved']} |")
    print()

    # Per-scenario detail
    for sid, data in results.items():
        print(f"## {data['name']} (`{sid}`)\n")
        print(f"**Type:** {data['task_type']}")
        print(f"**Verified Rate:** {data['verified_rate']:.1%}")
        print(f"**Avg Retrieved Chunks:** {data['avg_retrieved']}")
        print()

        print("### Sample Query Results\n")
        print("| Query | Verified | Unverified Claims | Retrieved |")
        print("|-------|----------|-------------------|-----------|")
        for qr in data["queries"]:
            v_icon = "PASS" if qr["verified"] else "FAIL"
            print(f"| {qr['query']} | {v_icon} | {qr['unverified_claims']} | {qr['num_retrieved']} |")
        print()

        print("### Success Criteria\n")
        for crit in data["success_criteria"]:
            print(f"- {crit}")
        print()


def _make_placeholder_content(scenario: EvalScenario) -> str:
    """Generate placeholder content so retrieval has something to find."""
    lines = [
        f"# {scenario.name}",
        "",
        scenario.description,
        "",
    ]
    for i, query in enumerate(scenario.sample_queries):
        lines.append(f"## Answer for: {query}")
        lines.append(f"This is a placeholder answer for evaluation query {i+1}. ")
        lines.append(f"It contains relevant information about {scenario.task_type}. ")
        lines.append(f"The expected output type is: {scenario.expected_output_type}. ")
        lines.append("")
    return "\n".join(lines)


def main():
    import io
    # Force UTF-8 on Windows to avoid GBK encoding errors
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(description="Agent-OS Evaluation Runner")
    parser.add_argument("--scenario", "-s", default="",
                       help="Run a single scenario by ID (e.g. s1_doc_qa)")
    parser.add_argument("--top-k", "-k", type=int, default=10,
                       help="Top-K for retrieval metrics")
    args = parser.parse_args()

    print("Agent-OS Evaluation Runner")
    print(f"Running with mock embedding + mock LLM (deterministic)")
    print()

    results = run_eval(scenario_filter=args.scenario, k=args.top_k)
    print_report(results)


if __name__ == "__main__":
    main()
