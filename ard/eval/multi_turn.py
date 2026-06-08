"""E2: Multi-turn consistency experiment — validates H1.

Tests whether StateStore + TransactionManager enable cross-turn
consistency that stateless RAG cannot achieve.

Scoring modes:
  - heuristic (default): 3-signal heuristic (keyword + reference + coherence)
  - llm_judge: LLM-as-Judge evaluates cross-turn consistency on 0-5 scale

5 scenarios × 5 turns × 2 conditions (with_state / no_state) = 50 evaluations.
"""

import json
import time
from dataclasses import dataclass, field

from ard.infra.logging import log
from ard.eval.judge import LLMJudge


@dataclass
class MultiTurnScenario:
    """A multi-turn conversation scenario."""
    scenario_id: str
    name: str
    description: str
    turns: list[dict]  # [{turn, query, expected_keywords, state_to_carry, requires_previous}]


@dataclass
class TurnResult:
    """Result of a single turn."""
    turn: int
    query: str
    response: str
    tokens_used: int
    latency_ms: float
    expected_keywords: list[str] = field(default_factory=list)
    keyword_recall: float = 0.0

    def to_dict(self) -> dict:
        return {
            "turn": self.turn, "query": self.query,
            "response": self.response[:300],
            "tokens_used": self.tokens_used,
            "latency_ms": round(self.latency_ms, 1),
            "keyword_recall": round(self.keyword_recall, 3),
        }


@dataclass
class ScenarioResult:
    """Complete result for one scenario under one condition."""
    scenario_id: str
    name: str
    condition: str  # "with_state" | "no_state"
    turns: list[TurnResult] = field(default_factory=list)
    consistency_score: float = 0.0      # cross-turn consistency (0-5, judge-scored)
    state_trace_accuracy: float = 0.0    # final turn recall of decision chain
    completion_rate: float = 0.0         # fraction of turns with valid responses

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id, "name": self.name,
            "condition": self.condition,
            "consistency_score": round(self.consistency_score, 3),
            "state_trace_accuracy": round(self.state_trace_accuracy, 3),
            "completion_rate": round(self.completion_rate, 3),
            "turns": [t.to_dict() for t in self.turns],
        }


# ── 5 multi-turn scenarios ──────────────────────────────────

MULTI_TURN_SCENARIOS = [
    MultiTurnScenario("mt_s1", "Project Development",
        "Multi-turn project development with accumulating decisions",
        [
            {"turn": 1, "query": "Analyze the core architecture of our system. What are its key components and how do they interact?",
             "expected_keywords": ["architecture", "component", "interact", "retrieve", "context", "store"],
             "state_to_carry": ["architecture_analysis"]},
            {"turn": 2, "query": "Based on the architecture from before, which component is the most innovative and why? Compare it with traditional approaches.",
             "expected_keywords": ["innovative", "compare", "traditional", "context mmu", "transaction"],
             "state_to_carry": ["innovation_analysis"], "requires_previous_turn": True},
            {"turn": 3, "query": "Now that we identified the key innovation, propose a concrete improvement to its pipeline. What would you change and what impact would it have?",
             "expected_keywords": ["improvement", "pipeline", "change", "impact", "step", "filter", "budget"],
             "state_to_carry": ["improvement_proposal"], "requires_previous_turn": True},
            {"turn": 4, "query": "Given your improvement from the previous step, how would it affect the transaction manager's consistency guarantees? Are there any new edge cases?",
             "expected_keywords": ["transaction", "consistency", "edge", "case", "verification", "rollback"],
             "state_to_carry": ["impact_analysis"], "requires_previous_turn": True},
            {"turn": 5, "query": "Summarize our entire discussion: the architecture, the innovation, your proposed improvement, and its impact. Trace the full decision chain. What key decisions did we make?",
             "expected_keywords": ["summary", "architecture", "innovation", "improvement", "decision", "chain", "trace"],
             "state_to_carry": ["final_summary"], "requires_previous_turn": True},
        ]
    ),
    MultiTurnScenario("mt_s2", "Paper Deep Analysis",
        "Progressive deepening of research paper understanding",
        [
            {"turn": 1, "query": "What is the main research problem addressed by state management for AI systems? Frame it as a formal research question.",
             "expected_keywords": ["research", "problem", "state", "management", "long-horizon", "context"],
             "state_to_carry": ["research_problem"]},
            {"turn": 2, "query": "Based on the problem you identified, what are the three core hypotheses (H1, H2, H3) that would prove the solution works? Explain each with its falsification condition.",
             "expected_keywords": ["hypothesis", "H1", "H2", "H3", "state", "context", "workspace", "falsify"],
             "state_to_carry": ["hypotheses"], "requires_previous_turn": True},
            {"turn": 3, "query": "For H2 specifically (Context Window as Execution Workspace), design an experiment that would provide the strongest possible evidence. What metrics, baselines, and sample sizes?",
             "expected_keywords": ["experiment", "H2", "evidence", "metric", "baseline", "sample", "token", "efficiency"],
             "state_to_carry": ["experimental_design"], "requires_previous_turn": True},
            {"turn": 4, "query": "How does the transactional runtime (Transaction + MVCC) provide evidence for H1? What specific mechanism proves state management matters more than raw reasoning?",
             "expected_keywords": ["transaction", "MVCC", "H1", "state", "reasoning", "mechanism", "consistency"],
             "state_to_carry": ["h1_evidence"], "requires_previous_turn": True},
            {"turn": 5, "query": "Design a single comprehensive experiment that would simultaneously validate H1, H2, and H3. Be specific about conditions, metrics, and expected outcomes.",
             "expected_keywords": ["comprehensive", "experiment", "H1", "H2", "H3", "validate", "condition", "outcome"],
             "state_to_carry": ["experiment_design"], "requires_previous_turn": True},
        ]
    ),
    MultiTurnScenario("mt_s3", "Design Evolution with Reversals",
        "System design with requirements changes and decision reversals",
        [
            {"turn": 1, "query": "We need to add real-time collaborative editing to our system. Propose an architecture extension that supports multiple concurrent writers.",
             "expected_keywords": ["collaborative", "real-time", "concurrent", "writer", "extend", "architecture"],
             "state_to_carry": ["feature_proposal"]},
            {"turn": 2, "query": "Wait, collaborative editing might cause too many transaction conflicts with optimistic locking. What alternative concurrency control strategy could we use instead?",
             "expected_keywords": ["conflict", "alternative", "concurrency", "pessimistic", "locking", "strategy"],
             "state_to_carry": ["alternative"], "requires_previous_turn": True},
            {"turn": 3, "query": "Actually, revert to the original proposal. Can we make collaborative editing work with MVCC snapshot isolation instead? How?",
             "expected_keywords": ["MVCC", "snapshot", "isolation", "revert", "original", "collaborative"],
             "state_to_carry": ["revisited_proposal"], "requires_previous_turn": True},
            {"turn": 4, "query": "New constraint: we now need to support mobile clients with intermittent connectivity and local-first editing. How does this change the architecture?",
             "expected_keywords": ["mobile", "offline", "local-first", "sync", "conflict", "resolution", "CRDT"],
             "state_to_carry": ["new_constraint"], "requires_previous_turn": True},
            {"turn": 5, "query": "Given all the decisions, reversals, and constraints we discussed, what is the final recommended architecture? Trace the full evolution of our thinking.",
             "expected_keywords": ["final", "architecture", "decision", "evolution", "reversal", "constraint", "trace"],
             "state_to_carry": ["final_architecture"], "requires_previous_turn": True},
        ]
    ),
    MultiTurnScenario("mt_s4", "Debug Session",
        "Systematic debugging with root cause analysis",
        [
            {"turn": 1, "query": "Users report that the system occasionally loses state after long conversations. What are the most likely causes? List at least three hypotheses.",
             "expected_keywords": ["lose", "state", "cause", "hypothesis", "conversation", "bug"],
             "state_to_carry": ["hypotheses_list"]},
            {"turn": 2, "query": "For your top hypothesis from the list, design a diagnostic test that would confirm or rule it out. What would you instrument?",
             "expected_keywords": ["diagnostic", "test", "instrument", "hypothesis", "confirm", "rule out"],
             "state_to_carry": ["diagnostic_plan"], "requires_previous_turn": True},
            {"turn": 3, "query": "The diagnostic reveals the issue is in the writeback step — TransactionManager.commit() sometimes fails silently. Propose a fix with proper error handling.",
             "expected_keywords": ["writeback", "commit", "error", "handling", "fix", "transaction", "silent"],
             "state_to_carry": ["fix_proposal"], "requires_previous_turn": True},
            {"turn": 4, "query": "Implementing your fix introduces a new issue: duplicate events in the EventStore. How do we prevent this while keeping the fix?",
             "expected_keywords": ["duplicate", "event", "idempotent", "prevent", "fix", "EventStore"],
             "state_to_carry": ["fix_revision"], "requires_previous_turn": True},
            {"turn": 5, "query": "Summarize: the original bug, your diagnostic findings, the fix you proposed, the new issue it introduced, and the final resolution. What did we learn?",
             "expected_keywords": ["summary", "bug", "diagnostic", "fix", "learn", "resolution", "root", "cause"],
             "state_to_carry": ["final_report"], "requires_previous_turn": True},
        ]
    ),
    MultiTurnScenario("mt_s5", "Cross-Domain Knowledge Synthesis",
        "Synthesizing insights across AI, database, and programming domains",
        [
            {"turn": 1, "query": "What design patterns from database systems (MVCC, WAL, ACID) are most applicable to AI state management? Explain each with mapping.",
             "expected_keywords": ["MVCC", "WAL", "ACID", "database", "pattern", "mapping", "AI", "state"],
             "state_to_carry": ["db_patterns"]},
            {"turn": 2, "query": "How do Rust's ownership model and Python's asyncio inform the design of concurrent state management for AI agents? Draw specific parallels.",
             "expected_keywords": ["Rust", "ownership", "Python", "asyncio", "concurrent", "parallel", "agent"],
             "state_to_carry": ["lang_insights"], "requires_previous_turn": False},
            {"turn": 3, "query": "Synthesize the database patterns from earlier and the programming language insights into a unified set of design principles for AI state management.",
             "expected_keywords": ["synthesis", "database", "programming", "unified", "principle", "design"],
             "state_to_carry": ["synthesis"], "requires_previous_turn": True},
            {"turn": 4, "query": "Apply your unified design principles to critique the current ARD architecture. What would you change and why?",
             "expected_keywords": ["critique", "ARD", "architecture", "change", "principle", "apply"],
             "state_to_carry": ["critique"], "requires_previous_turn": True},
            {"turn": 5, "query": "Write a comprehensive design manifesto for state management in AI systems, drawing on all three domains we discussed: databases, programming languages, and AI systems.",
             "expected_keywords": ["manifesto", "design", "database", "programming", "AI", "state", "management", "principle"],
             "state_to_carry": ["manifesto"], "requires_previous_turn": True},
        ]
    ),
    MultiTurnScenario("mt_s6", "Code Review with Iterations",
        "Iterative code review with accumulating improvement suggestions",
        [
            {"turn": 1, "query": "Review this system design: 'The system uses a single-threaded event loop with async I/O for handling concurrent requests'. What are potential issues?",
             "expected_keywords": ["single", "thread", "event", "loop", "concurrent", "issue", "bottleneck"],
             "state_to_carry": ["review_findings"]},
            {"turn": 2, "query": "Based on your review findings, propose a new architecture that addresses the concurrency bottleneck. Be specific about the threading model.",
             "expected_keywords": ["architecture", "concurrency", "thread", "model", "worker", "pool"],
             "state_to_carry": ["architecture_proposal"], "requires_previous_turn": True},
            {"turn": 3, "query": "The team rejects your proposal because it adds too much complexity. Can you find a middle ground — something simpler that still addresses the main bottleneck?",
             "expected_keywords": ["simpler", "middle", "ground", "complexity", "tradeoff", "bottleneck"],
             "state_to_carry": ["revised_proposal"], "requires_previous_turn": True},
            {"turn": 4, "query": "Now the team asks: how would your revised proposal handle error recovery when a worker crashes mid-request?",
             "expected_keywords": ["error", "recovery", "crash", "worker", "retry", "supervisor"],
             "state_to_carry": ["error_handling"], "requires_previous_turn": True},
            {"turn": 5, "query": "Summarize the entire design evolution: the original issue, your first proposal, the simplification, and the error handling addition. Trace the decision chain.",
             "expected_keywords": ["summary", "evolution", "original", "proposal", "simplify", "error", "decision"],
             "state_to_carry": ["final_summary"], "requires_previous_turn": True},
        ]
    ),
    MultiTurnScenario("mt_s7", "Database Schema Migration Planning",
        "Planning a complex database migration with rollback strategy",
        [
            {"turn": 1, "query": "We need to migrate a PostgreSQL database from single-node to CockroachDB distributed. What are the key differences in transaction isolation we need to consider?",
             "expected_keywords": ["PostgreSQL", "CockroachDB", "isolation", "distributed", "migration", "transaction"],
             "state_to_carry": ["isolation_analysis"]},
            {"turn": 2, "query": "Given those isolation differences, design a migration strategy that ensures zero data loss. Include the sequence of steps.",
             "expected_keywords": ["migration", "strategy", "zero", "loss", "sequence", "step", "data"],
             "state_to_carry": ["migration_plan"], "requires_previous_turn": True},
            {"turn": 3, "query": "What if the migration fails halfway through? Design a rollback plan that can restore the original PostgreSQL state.",
             "expected_keywords": ["rollback", "fail", "restore", "PostgreSQL", "snapshot", "backup"],
             "state_to_carry": ["rollback_plan"], "requires_previous_turn": True},
            {"turn": 4, "query": "The business requires < 5 minutes of downtime. Can your migration + rollback plan meet this? What would you change?",
             "expected_keywords": ["downtime", "minute", "five", "online", "blue", "green", "change"],
             "state_to_carry": ["downtime_optimization"], "requires_previous_turn": True},
            {"turn": 5, "query": "Present the final migration plan: isolation considerations, step-by-step migration, rollback strategy, and downtime optimization. What risks remain?",
             "expected_keywords": ["final", "plan", "isolation", "step", "rollback", "downtime", "risk"],
             "state_to_carry": ["final_plan"], "requires_previous_turn": True},
        ]
    ),
    MultiTurnScenario("mt_s8", "Research Paper Replication Analysis",
        "Analyzing whether a paper's claims can be replicated",
        [
            {"turn": 1, "query": "A paper claims that 'optimistic locking outperforms pessimistic locking by 3x under low contention'. What factors could confound this result?",
             "expected_keywords": ["optimistic", "pessimistic", "locking", "contention", "confound", "factor"],
             "state_to_carry": ["confound_analysis"]},
            {"turn": 2, "query": "Design a replication experiment that controls for the confounding factors you identified. What hardware, workload, and metrics?",
             "expected_keywords": ["replication", "experiment", "control", "hardware", "workload", "metric"],
             "state_to_carry": ["experiment_design"], "requires_previous_turn": True},
            {"turn": 3, "query": "The replication results show only 1.5x improvement, not 3x. What could explain the discrepancy? List possible causes.",
             "expected_keywords": ["discrepancy", "explain", "cause", "difference", "1.5x", "3x"],
             "state_to_carry": ["discrepancy_analysis"], "requires_previous_turn": True},
            {"turn": 4, "query": "Given the discrepancy, should we still recommend optimistic locking for production systems? What caveats would you add to the recommendation?",
             "expected_keywords": ["recommend", "production", "caveat", "tradeoff", "suitable"],
             "state_to_carry": ["recommendation"], "requires_previous_turn": True},
            {"turn": 5, "query": "Write a one-paragraph summary for an engineering blog: the original claim, the replication attempt, the findings, and the practical recommendation.",
             "expected_keywords": ["summary", "blog", "claim", "replication", "finding", "recommendation", "practical"],
             "state_to_carry": ["blog_summary"], "requires_previous_turn": True},
        ]
    ),
    MultiTurnScenario("mt_s9", "API Design with Evolving Requirements",
        "Designing an API that evolves as requirements change",
        [
            {"turn": 1, "query": "Design a REST API for a task management system. Define the endpoints, request/response formats, and error codes.",
             "expected_keywords": ["REST", "API", "endpoint", "request", "response", "error", "task"],
             "state_to_carry": ["api_design"]},
            {"turn": 2, "query": "New requirement: the API must support bulk operations (create/update/delete multiple tasks atomically). How does this change your design?",
             "expected_keywords": ["bulk", "atomic", "batch", "transaction", "multiple", "change"],
             "state_to_carry": ["bulk_design"], "requires_previous_turn": True},
            {"turn": 3, "query": "Another requirement: add real-time notifications via WebSocket when tasks change. How do you integrate this with the REST API?",
             "expected_keywords": ["WebSocket", "real-time", "notification", "event", "subscribe", "integrate"],
             "state_to_carry": ["websocket_design"], "requires_previous_turn": True},
            {"turn": 4, "query": "The team wants to version the API (v1 and v2) with backward compatibility. How do you structure this across REST and WebSocket?",
             "expected_keywords": ["version", "v1", "v2", "backward", "compatible", "deprecate"],
             "state_to_carry": ["versioning"], "requires_previous_turn": True},
            {"turn": 5, "query": "Write the complete API specification covering: REST endpoints (v1+v2), bulk operations, WebSocket events, versioning strategy, and error handling.",
             "expected_keywords": ["specification", "complete", "REST", "WebSocket", "version", "error", "bulk"],
             "state_to_carry": ["final_spec"], "requires_previous_turn": True},
        ]
    ),
    MultiTurnScenario("mt_s10", "Performance Optimization Chain",
        "Chained performance optimizations with measurement at each step",
        [
            {"turn": 1, "query": "A system processes 100 requests/second but needs to handle 1000/sec. Profile the likely bottlenecks in a Python async web service with PostgreSQL.",
             "expected_keywords": ["bottleneck", "profile", "Python", "async", "PostgreSQL", "100", "1000"],
             "state_to_carry": ["bottleneck_analysis"]},
            {"turn": 2, "query": "Based on the bottleneck analysis, implement optimization #1: connection pooling. How much improvement do you expect and why?",
             "expected_keywords": ["connection", "pool", "optimization", "improvement", "expect", "pooling"],
             "state_to_carry": ["opt1_pooling"], "requires_previous_turn": True},
            {"turn": 3, "query": "Optimization #1 only got us to 250/sec. What's the next bottleneck? Propose optimization #2 with expected gains.",
             "expected_keywords": ["250", "next", "bottleneck", "optimization", "cache", "query", "index"],
             "state_to_carry": ["opt2_caching"], "requires_previous_turn": True},
            {"turn": 4, "query": "Optimization #2 got us to 600/sec. We still need 1000/sec. What architectural change (not just tuning) could get us there?",
             "expected_keywords": ["600", "architectural", "change", "scale", "horizontal", "replicate"],
             "state_to_carry": ["opt3_architecture"], "requires_previous_turn": True},
            {"turn": 5, "query": "Summarize the optimization journey: original state (100/sec), each optimization, its impact, and the final architecture. What was the most impactful change and why?",
             "expected_keywords": ["summary", "journey", "100", "250", "600", "1000", "impact", "final"],
             "state_to_carry": ["optimization_summary"], "requires_previous_turn": True},
        ]
    ),
]


class MultiTurnExperimentRunner:
    """Runs multi-turn scenarios with and without state management."""

    def __init__(self, hybrid_retriever, context_mmu, executor,
                 state_store=None, txn_mgr=None, trace_store=None, judge=None):
        self.hybrid = hybrid_retriever
        self.mmu = context_mmu
        self.executor = executor
        self.state_store = state_store
        self.txn_mgr = txn_mgr
        self.trace_store = trace_store
        self.judge = judge or LLMJudge()

    def run_all(self, scenarios: list[MultiTurnScenario] | None = None,
                quiet: bool = False) -> dict[str, list[ScenarioResult]]:
        """Run all scenarios under both conditions."""
        if scenarios is None:
            scenarios = MULTI_TURN_SCENARIOS

        results = {"no_state": [], "with_state": []}

        for scenario in scenarios:
            if not quiet:
                print(f"  Scenario: {scenario.name}")

            # Without state
            r_no = self._run_scenario(scenario, with_state=False)
            results["no_state"].append(r_no)

            # With state (requires StateStore)
            if self.state_store and self.txn_mgr:
                r_with = self._run_scenario(scenario, with_state=True)
                results["with_state"].append(r_with)
            else:
                if not quiet:
                    print("    (StateStore not available, skipping with_state)")

        return results

    def _run_scenario(self, scenario: MultiTurnScenario,
                      with_state: bool = False) -> ScenarioResult:
        """Execute one scenario under one condition."""
        turn_results = []
        accumulated_state: dict[str, str] = {}  # key → response_summary
        trace_id = f"trace_mt_{scenario.scenario_id}"

        for turn in scenario.turns:
            query = turn["query"]
            if with_state and accumulated_state and turn.get("requires_previous_turn", False):
                state_text = "\n".join(
                    f"[{k}]: {v[:300]}" for k, v in accumulated_state.items()
                )
                enriched_query = f"Previous session state:\n{state_text}\n\nNew question: {query}"
            else:
                enriched_query = query

            t0 = time.time()
            candidates = self.hybrid.retrieve(enriched_query)
            context_pack = self.mmu.assemble(enriched_query, candidates, top_k=15)
            response = self.executor.think(context_pack, enriched_query)
            latency = (time.time() - t0) * 1000

            # Write state if enabled
            if with_state and self.state_store and self.txn_mgr:
                try:
                    txn = self.txn_mgr.begin()
                    evt = self.state_store.build_event(
                        stream_key=f"task:mt_{scenario.scenario_id}_turn{turn['turn']}",
                        event_type="created",
                        payload={
                            "query": query, "response_summary": response.answer[:500],
                            "turn": turn["turn"], "scenario": scenario.scenario_id,
                        },
                    )
                    txn.add_event(evt)
                    self.txn_mgr.commit(txn)
                except RuntimeError:
                    pass  # conflict — skip write for this turn

            # Track keyword recall
            keywords = turn.get("expected_keywords", [])
            kw_recall = 0.0
            if keywords:
                ans_lower = response.answer.lower()
                matches = sum(1 for kw in keywords if kw.lower() in ans_lower)
                kw_recall = matches / len(keywords)

            turn_results.append(TurnResult(
                turn=turn["turn"], query=query,
                response=response.answer,
                tokens_used=context_pack.total_tokens_used(),
                latency_ms=latency,
                expected_keywords=keywords,
                keyword_recall=kw_recall,
            ))

            # Accumulate state for next turn
            for key in turn.get("state_to_carry", []):
                accumulated_state[key] = response.answer[:300]

        # Score consistency
        consistency = self._score_consistency(scenario, turn_results)
        completion = len([t for t in turn_results if t.response]) / max(len(scenario.turns), 1)
        trace_acc = self._score_state_trace(scenario, turn_results)

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            name=scenario.name,
            condition="with_state" if with_state else "no_state",
            turns=turn_results,
            consistency_score=consistency,
            state_trace_accuracy=trace_acc,
            completion_rate=completion,
        )

    def _score_consistency(self, scenario: MultiTurnScenario,
                           results: list[TurnResult]) -> float:
        """Score cross-turn consistency via keyword continuity + semantic signals.

        Uses 3 signals:
        1. State key recall (40%): do prior turn's state_to_carry keys appear?
        2. Reference density (30%): does response reference prior turns by number?
        3. Response coherence (30%): is response length reasonable (not too short)?
        """
        if len(results) < 2:
            return 1.0
        scores = []
        for i in range(1, len(results)):
            response = results[i].response.lower()
            prev_keys = set(scenario.turns[i - 1].get("state_to_carry", []))

            # Signal 1: State key recall
            key_score = 0.0
            if prev_keys:
                matched = sum(
                    1 for k in prev_keys
                    if k.replace("_", " ") in response
                )
                key_score = matched / len(prev_keys)

            # Signal 2: Reference density (mentions of prior/previous/before/earlier)
            ref_signals = ["prior", "previous", "before", "earlier", "above",
                          "as discussed", "as mentioned", "the last", "turn 1",
                          "turn 2", "turn 3", "turn 4", "first", "second", "third"]
            ref_count = sum(1 for s in ref_signals if s in response)
            ref_score = min(1.0, ref_count / 3.0)

            # Signal 3: Response coherence (length-based proxy)
            length = len(results[i].response)
            if length > 500:
                coh_score = 1.0
            elif length > 200:
                coh_score = 0.7
            elif length > 100:
                coh_score = 0.4
            else:
                coh_score = 0.2

            # Weighted combination
            combined = 0.40 * key_score + 0.30 * ref_score + 0.30 * coh_score
            scores.append(combined)

        return sum(scores) / len(scores) if scores else 0.0

    def _score_state_trace(self, scenario: MultiTurnScenario,
                           results: list[TurnResult]) -> float:
        """Score the final turn's ability to trace the decision chain."""
        if len(results) < 2 or not results[-1].response:
            return 0.0
        # Collect all state_to_carry keys from earlier turns
        all_keys = set()
        for turn in scenario.turns[:-1]:
            all_keys.update(turn.get("state_to_carry", []))
        if not all_keys:
            return 1.0
        final_response = results[-1].response.lower()
        matched = sum(1 for k in all_keys if k.replace("_", " ") in final_response)
        return matched / len(all_keys)


def print_multi_turn_report(results: dict[str, list[ScenarioResult]]) -> str:
    """Generate a formatted report for multi-turn experiments."""
    lines = ["\n" + "=" * 70,
             "MULTI-TURN CONSISTENCY REPORT (E2 — H1)",
             "=" * 70]

    for condition, scenarios in results.items():
        if not scenarios:
            continue
        lines.append(f"\n--- {condition.upper()} ---")
        lines.append(f"{'Scenario':30s} | Consistency | Trace Acc | Completion")
        lines.append("-" * 60)

        avg_cons = []
        avg_trace = []
        avg_comp = []
        for sr in scenarios:
            lines.append(f"{sr.name:30s} | {sr.consistency_score:.3f}       | "
                        f"{sr.state_trace_accuracy:.3f}     | {sr.completion_rate:.2f}")
            avg_cons.append(sr.consistency_score)
            avg_trace.append(sr.state_trace_accuracy)
            avg_comp.append(sr.completion_rate)

        if avg_cons:
            lines.append("-" * 60)
            lines.append(f"{'AVERAGE':30s} | {sum(avg_cons)/len(avg_cons):.3f}       | "
                        f"{sum(avg_trace)/len(avg_trace):.3f}     | {sum(avg_comp)/len(avg_comp):.2f}")

    # H1 comparison
    no_s = results.get("no_state", [])
    with_s = results.get("with_state", [])
    if no_s and with_s:
        no_avg = sum(s.consistency_score for s in no_s) / max(len(no_s), 1)
        with_avg = sum(s.consistency_score for s in with_s) / max(len(with_s), 1)
        diff = with_avg - no_avg
        lines.append(f"\nH1 ANALYSIS: With State consistency = {with_avg:.3f} vs No State = {no_avg:.3f}")
        if abs(no_avg) > 0.0001:
            lines.append(f"  Difference: {diff:+.3f} ({diff/no_avg*100:+.1f}%)")
        else:
            lines.append(f"  Difference: {diff:+.3f} (baseline is ~0)")
        if diff > 0.01:
            lines.append("  *** H1 SUPPORTED: State management improves cross-turn consistency ***")
        else:
            lines.append("  --- H1 NOT SUPPORTED: no significant improvement ---")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# P2: LLM Judge for Multi-Turn Consistency
# ═══════════════════════════════════════════════════════════

MULTI_TURN_JUDGE_PROMPT = """You are evaluating an AI assistant's performance in a multi-turn conversation. Your task is to score how well the assistant maintains consistency across turns.

## Conversation History
{history}

## Current Turn Question
{current_query}

## Assistant's Response
{response}

## Evaluation
Score the response on these dimensions (0-5 each):

1. **cross_turn_reference** (0-5): Does the response reference or build upon information from PREVIOUS turns? Does it show awareness of the conversation history?
   - 5: Explicitly references and builds on prior turns with specific details
   - 3: Shows general awareness of prior context
   - 0: No reference to prior conversation; treats query as standalone

2. **state_consistency** (0-5): Is the response CONSISTENT with decisions, facts, and conclusions from previous turns?
   - 5: Fully consistent with all prior decisions and facts
   - 3: Mostly consistent, minor discrepancies
   - 0: Contradicts or ignores prior decisions/facts

3. **progressive_understanding** (0-5): Does the response show DEEPER understanding compared to earlier turns?
   - 5: Builds on prior insights, adds new depth
   - 3: Maintains same level of understanding
   - 0: Regresses or repeats without adding value

Output ONLY a JSON object:
```json
{{"cross_turn_reference": <0-5>, "state_consistency": <0-5>, "progressive_understanding": <0-5>, "explanation": "<brief>"}}
```"""


class LLMJudgeMultiTurn:
    """Evaluate multi-turn consistency using LLM-as-Judge.

    Provides more nuanced scoring than heuristic keyword matching,
    capturing semantic consistency, progressive understanding, and
    cross-turn reference quality.
    """

    def __init__(self, llm_fn=None):
        """Args:
            llm_fn: callable(prompt) -> str. If None, uses heuristic fallback.
        """
        self.llm_fn = llm_fn

    def evaluate_scenario(self, scenario_result) -> dict:
        """Evaluate all turns in a scenario for cross-turn consistency.

        Args:
            scenario_result: ScenarioResult from MultiTurnExperimentRunner.

        Returns:
            Dict with per-turn judge scores and aggregated metrics.
        """
        turns = scenario_result.turns
        if len(turns) < 2:
            return {"turns": [], "avg_cross_turn_reference": 0, "avg_state_consistency": 0, "avg_progressive_understanding": 0}

        turn_scores = []
        for i in range(1, len(turns)):
            history = self._build_history(turns[:i])
            scores = self._judge_turn(
                history=history,
                current_query=turns[i].query,
                response=turns[i].response,
            )
            turn_scores.append(scores)

        return {
            "turns": turn_scores,
            "avg_cross_turn_reference": sum(s["cross_turn_reference"] for s in turn_scores) / len(turn_scores),
            "avg_state_consistency": sum(s["state_consistency"] for s in turn_scores) / len(turn_scores),
            "avg_progressive_understanding": sum(s["progressive_understanding"] for s in turn_scores) / len(turn_scores),
        }

    def compare_conditions(self, no_state_results, with_state_results) -> dict:
        """Compare LLM Judge scores between No State and With State conditions.

        Returns:
            Dict with comparative analysis suitable for H1 evaluation.
        """
        no_scores = []
        with_scores = []
        for sr in no_state_results:
            eval_result = self.evaluate_scenario(sr)
            if eval_result["turns"]:
                no_scores.append(eval_result["avg_state_consistency"])
        for sr in with_state_results:
            eval_result = self.evaluate_scenario(sr)
            if eval_result["turns"]:
                with_scores.append(eval_result["avg_state_consistency"])

        import numpy as np
        return {
            "no_state": {"mean": float(np.mean(no_scores)) if no_scores else 0, "scores": no_scores},
            "with_state": {"mean": float(np.mean(with_scores)) if with_scores else 0, "scores": with_scores},
            "delta": float(np.mean(with_scores) - np.mean(no_scores)) if (no_scores and with_scores) else 0,
            "h1_supported": bool(np.mean(with_scores) > np.mean(no_scores)) if (no_scores and with_scores) else False,
        }

    def _judge_turn(self, history: str, current_query: str, response: str) -> dict:
        """Judge one turn in the context of its history."""
        if not self.llm_fn:
            return self._heuristic_judge(history, current_query, response)

        prompt = MULTI_TURN_JUDGE_PROMPT.format(
            history=history, current_query=current_query, response=response[:2000]
        )

        try:
            raw = self.llm_fn(prompt)
            import json, re
            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                result = json.loads(match.group(0))
                return {
                    "cross_turn_reference": result.get("cross_turn_reference", 2),
                    "state_consistency": result.get("state_consistency", 2),
                    "progressive_understanding": result.get("progressive_understanding", 2),
                    "explanation": result.get("explanation", ""),
                }
        except Exception:
            pass
        return self._heuristic_judge(history, current_query, response)

    @staticmethod
    def _heuristic_judge(history: str, current_query: str, response: str) -> dict:
        """Heuristic fallback when LLM judge unavailable."""
        # Cross-turn reference: does response mention prior topics?
        ref_signals = ["prior", "previous", "before", "earlier", "above",
                      "as discussed", "as mentioned", "the last"]
        ref_count = sum(1 for s in ref_signals if s in response.lower())
        cross_ref = min(5, ref_count * 2)

        # State consistency: keyword overlap with history
        hist_words = set(history.lower().split())
        resp_words = set(response.lower().split())
        overlap = len(hist_words & resp_words) / max(len(resp_words), 1) if resp_words else 0
        consistency = min(5, int(overlap * 10))

        # Progressive understanding: response length as proxy
        prog = min(5, len(response) // 300)

        return {"cross_turn_reference": cross_ref, "state_consistency": consistency,
                "progressive_understanding": prog, "explanation": "Heuristic fallback"}

    @staticmethod
    def _build_history(turns) -> str:
        """Build conversation history string from previous turns."""
        parts = []
        for t in turns:
            parts.append(f"Turn {t.turn}:\nQuestion: {t.query}\nAnswer: {t.response[:300]}")
        return "\n\n".join(parts)
