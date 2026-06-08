"""Evaluation benchmark for ARD experimental validation.

Supports:
- JSON dataset loading (single-turn + multi-turn scenarios)
- 3 experimental conditions (Baseline 1: vector RAG, Baseline 2: hybrid RAG, Proposed: Context MMU)
- Multi-turn conversation evaluation with state continuity
- Ablation studies (disable individual Context MMU steps)
- Statistical analysis (bootstrap confidence intervals)
- Results export for paper-ready charts

Maps to State_Management_Research_Plan.md Track B.
"""

import json
import math
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ard.infra.config import Config
from ard.infra.logging import log


# ═══════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EvalQuery:
    """A single-turn evaluation query."""
    query_id: str
    query: str
    category: str = "factoid"
    expected_keywords: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    relevant_chunk_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalQuery":
        return cls(
            query_id=d.get("query_id", ""),
            query=d.get("query", ""),
            category=d.get("category", "factoid"),
            expected_keywords=d.get("expected_keywords", []),
            difficulty=d.get("difficulty", "medium"),
            relevant_chunk_ids=d.get("relevant_chunk_ids", []),
        )


@dataclass
class EvalResult:
    """Results for a single evaluation."""
    query_id: str
    condition: str
    response: str
    latency_ms: float
    tokens_input: int
    tokens_output: int = 0
    source_refs: list[str] = field(default_factory=list)
    context_items: int = 0
    strategy_labels: list[str] = field(default_factory=list)

    # Computed metrics
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    keyword_recall: float = 0.0
    token_efficiency: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "condition": self.condition,
            "response": self.response[:500],
            "latency_ms": self.latency_ms,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "precision_at_k": round(self.precision_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "keyword_recall": round(self.keyword_recall, 4),
            "token_efficiency": round(self.token_efficiency, 4),
        }


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results."""
    condition: str
    total_queries: int
    avg_latency_ms: float
    avg_tokens_input: float
    avg_precision: float
    avg_recall: float
    avg_mrr: float
    avg_keyword_recall: float
    avg_token_efficiency: float
    token_efficiency_ci: tuple[float, float] = (0.0, 0.0)  # 95% bootstrap CI
    keyword_recall_ci: tuple[float, float] = (0.0, 0.0)
    per_query: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        te_lo, te_hi = self.token_efficiency_ci
        kr_lo, kr_hi = self.keyword_recall_ci
        return (
            f"=== {self.condition} (n={self.total_queries}) ===\n"
            f"  Latency:          {self.avg_latency_ms:7.1f} ms\n"
            f"  Tokens Input:     {self.avg_tokens_input:7.0f}\n"
            f"  Precision@K:      {self.avg_precision:7.4f}\n"
            f"  Recall@K:         {self.avg_recall:7.4f}\n"
            f"  MRR:              {self.avg_mrr:7.4f}\n"
            f"  Keyword Recall:   {self.avg_keyword_recall:7.4f}  [95% CI: {kr_lo:.4f}, {kr_hi:.4f}]\n"
            f"  Token Efficiency: {self.avg_token_efficiency:7.4f}  [95% CI: {te_lo:.4f}, {te_hi:.4f}]"
        )

    def to_dict(self) -> dict:
        te_lo, te_hi = self.token_efficiency_ci
        return {
            "condition": self.condition,
            "total_queries": self.total_queries,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_tokens_input": round(self.avg_tokens_input, 0),
            "avg_precision": round(self.avg_precision, 4),
            "avg_recall": round(self.avg_recall, 4),
            "avg_mrr": round(self.avg_mrr, 4),
            "avg_keyword_recall": round(self.avg_keyword_recall, 4),
            "avg_token_efficiency": round(self.avg_token_efficiency, 4),
            "token_efficiency_ci_95": [round(te_lo, 4), round(te_hi, 4)],
            "keyword_recall_ci_95": [round(self.keyword_recall_ci[0], 4), round(self.keyword_recall_ci[1], 4)],
            "per_query": self.per_query,
        }


@dataclass
class MultiTurnResult:
    """Results for a multi-turn scenario."""
    scenario_id: str
    name: str
    condition: str
    turns: list[dict] = field(default_factory=list)
    consistency_score: float = 0.0
    completion_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "condition": self.condition,
            "consistency_score": round(self.consistency_score, 4),
            "completion_rate": round(self.completion_rate, 4),
            "turns": self.turns,
        }


@dataclass
class AblationStep:
    """Configuration for one ablation experiment."""
    label: str
    disable_steps: list[str]  # ContextMMU steps to skip


# ═══════════════════════════════════════════════════════════════════
# Statistics Helpers
# ═══════════════════════════════════════════════════════════════════

def bootstrap_ci(values: list[float], n_bootstrap: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a list of values."""
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) < 5:
        return float(np.mean(values)), float(np.mean(values))

    arr = np.array(values)
    means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))

    alpha = (1.0 - ci) / 2.0
    lo = np.percentile(means, alpha * 100)
    hi = np.percentile(means, (1 - alpha) * 100)
    return float(lo), float(hi)


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Cohen's d effect size."""
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = math.sqrt((v1 + v2) / 2)
    if pooled_sd == 0:
        return 0.0
    return (m1 - m2) / pooled_sd


# ═══════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════

class Benchmark:
    """ARD experimental benchmark framework.

    Runs 3 conditions (Baseline 1, Baseline 2, Proposed) plus
    ablation studies and multi-turn scenarios.
    """

    def __init__(
        self,
        knowledge_store,
        hybrid_retriever,
        context_mmu,
        executor,
        state_store=None,
        trace_store=None,
        transaction_manager=None,
        config: Config | None = None,
    ):
        self.store = knowledge_store
        self.hybrid = hybrid_retriever
        self.mmu = context_mmu
        self.executor = executor
        self.state_store = state_store
        self.trace_store = trace_store
        self.txn_mgr = transaction_manager
        self.config = config or Config()

    # ══════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════

    def run_single_turn(self, queries: list[EvalQuery]) -> dict[str, BenchmarkReport]:
        """Run all 3 conditions on single-turn queries."""
        reports = {}
        reports["baseline1"] = self._run_condition(queries, "baseline1")
        reports["baseline2"] = self._run_condition(queries, "baseline2")
        reports["proposed"] = self._run_condition(queries, "proposed")
        return reports

    def run_multi_turn(self, scenarios: list[dict]) -> list[MultiTurnResult]:
        """Run multi-turn scenarios with and without state management."""
        results = []
        for scenario in scenarios:
            results.append(self._run_multi_turn(scenario, with_state=False))
            results.append(self._run_multi_turn(scenario, with_state=True))
        return results

    def run_ablation(self, queries: list[EvalQuery]) -> dict[str, BenchmarkReport]:
        """Run ablation study: disable ContextMMU steps one at a time."""
        ablation_configs = [
            AblationStep("full_mmu", []),
            AblationStep("no_compress", ["compress"]),
            AblationStep("no_filter", ["filter"]),
            AblationStep("no_budget", ["budget"]),
            AblationStep("bare_retrieve_assemble", ["filter", "rank", "compress", "budget"]),
        ]
        reports = {}
        for ab in ablation_configs:
            reports[ab.label] = self._run_condition_ablated(queries, ab)
        return reports

    def run_all(self, queries: list[EvalQuery],
                scenarios: list[dict] | None = None) -> dict:
        """Run full experiment suite."""
        results = {
            "single_turn": self.run_single_turn(queries),
            "ablation": {},
        }

        if scenarios:
            results["multi_turn"] = self.run_multi_turn(scenarios)

        # Run ablation only if we have enough queries
        if len(queries) >= 20:
            results["ablation"] = self.run_ablation(queries)

        return results

    # ══════════════════════════════════════════════════════════
    # Condition Implementations
    # ══════════════════════════════════════════════════════════

    def _run_condition(self, queries: list[EvalQuery],
                       condition: str) -> BenchmarkReport:
        """Run one experimental condition over all queries."""
        results: list[EvalResult] = []

        for q in queries:
            t0 = time.time()

            if condition == "baseline1":
                result = self._baseline1(q)
            elif condition == "baseline2":
                result = self._baseline2(q)
            else:
                result = self._proposed(q)

            result.latency_ms = (time.time() - t0) * 1000
            self._compute_metrics(result, q)
            results.append(result)
            log.debug("query_done", condition=condition, query_id=q.query_id,
                      te=round(result.token_efficiency, 4))

        return self._aggregate(condition, results)

    def _baseline1(self, q: EvalQuery) -> EvalResult:
        """Baseline 1: Vector → Top-K → LLM"""
        candidates = self.store._vector_search(q.query, top_k=15)
        ctx_text = "\n\n".join(r.text_preview for r in candidates)
        response = self.executor.think(self._text_context(ctx_text, q.query))
        return EvalResult(
            query_id=q.query_id, condition="baseline1",
            response=response.answer,
            latency_ms=0.0,
            tokens_input=len(ctx_text) // 4,
            tokens_output=len(response.answer) // 4,
            source_refs=[r.source_ref for r in candidates],
            context_items=len(candidates),
            strategy_labels=["vector"],
        )

    def _baseline2(self, q: EvalQuery) -> EvalResult:
        """Baseline 2: Vector + Keyword → Rerank → LLM"""
        candidates = self.hybrid.retrieve(q.query)
        ctx_text = "\n\n".join(r.text_preview for r in candidates[:15])
        response = self.executor.think(self._text_context(ctx_text, q.query))
        return EvalResult(
            query_id=q.query_id, condition="baseline2",
            response=response.answer,
            latency_ms=0.0,
            tokens_input=len(ctx_text) // 4,
            tokens_output=len(response.answer) // 4,
            source_refs=[r.source_ref for r in candidates],
            context_items=len(candidates),
            strategy_labels=list(set(r.strategy for r in candidates)),
        )

    def _proposed(self, q: EvalQuery) -> EvalResult:
        """Proposed: Hybrid → Context MMU → LLM"""
        candidates = self.hybrid.retrieve(q.query)
        context_pack = self.mmu.assemble(
            query=q.query, retrieval_results=candidates,
            system_instruction="Answer based ONLY on provided context. Cite sources.",
            top_k=15,
        )
        response = self.executor.think(context_pack)
        return EvalResult(
            query_id=q.query_id, condition="proposed",
            response=response.answer,
            latency_ms=0.0,
            tokens_input=context_pack.total_tokens_used(),
            tokens_output=len(response.answer) // 4,
            source_refs=context_pack.source_refs,
            context_items=len(context_pack.sections),
            strategy_labels=list(set(r.strategy for r in candidates)),
        )

    def _run_condition_ablated(self, queries: list[EvalQuery],
                               ablation: AblationStep) -> BenchmarkReport:
        """Run with specific ContextMMU steps disabled."""
        results = []
        for q in queries:
            candidates = self.hybrid.retrieve(q.query)
            context_pack = self.mmu.assemble(
                query=q.query, retrieval_results=candidates,
                system_instruction="Answer based ONLY on provided context.",
                top_k=15,
                disabled_steps=set(ablation.disable_steps),
            )
            response = self.executor.think(context_pack)
            result = EvalResult(
                query_id=q.query_id, condition=ablation.label,
                response=response.answer,
                latency_ms=0,
                tokens_input=context_pack.total_tokens_used(),
                tokens_output=len(response.answer) // 4,
                source_refs=context_pack.source_refs,
                context_items=len(context_pack.sections),
            )
            self._compute_metrics(result, q)
            results.append(result)
        return self._aggregate(ablation.label, results)

    # ══════════════════════════════════════════════════════════
    # Multi-Turn
    # ══════════════════════════════════════════════════════════

    def _run_multi_turn(self, scenario: dict,
                        with_state: bool = False) -> MultiTurnResult:
        """Run a multi-turn scenario."""
        turns_data = scenario.get("turns", [])
        condition_label = "state" if with_state else "no_state"
        turn_results = []
        accumulated_state: dict = {}

        for turn in turns_data:
            query = turn["query"]
            turn_num = turn["turn"]

            # Inject prior state into query
            enriched_query = query
            if with_state and accumulated_state and turn.get("requires_previous_turn"):
                state_summary = "\n".join(
                    f"[Prior {k}]: {v}" for k, v in accumulated_state.items()
                )
                enriched_query = f"Previous session state:\n{state_summary}\n\nNew question: {query}"

            # Run through proposed pipeline
            candidates = self.hybrid.retrieve(enriched_query)
            context_pack = self.mmu.assemble(enriched_query, candidates, top_k=15)
            response = self.executor.think(context_pack)

            # Write state if enabled
            if with_state and self.state_store and self.txn_mgr:
                txn = self.txn_mgr.begin()
                evt = self.state_store.build_event(
                    stream_key=f"task:mt_{scenario['scenario_id']}_turn{turn_num}",
                    event_type="created",
                    payload={"query": query, "response": response.answer[:800],
                             "turn": turn_num},
                )
                txn.add_event(evt)
                try:
                    self.txn_mgr.commit(txn)
                except RuntimeError:
                    pass

            # Store state for next turn
            accumulated_state[f"turn_{turn_num}_summary"] = response.answer[:300]

            turn_results.append({
                "turn": turn_num,
                "query": query,
                "response": response.answer[:500],
                "tokens_used": context_pack.total_tokens_used(),
                "expected_keywords": turn.get("expected_keywords", []),
            })

        # Compute consistency
        consistency = self._compute_consistency(turns_data, turn_results)
        completion = len([t for t in turn_results if t["response"]]) / max(len(turns_data), 1)

        return MultiTurnResult(
            scenario_id=scenario["scenario_id"],
            name=scenario["name"],
            condition=condition_label,
            turns=turn_results,
            consistency_score=consistency,
            completion_rate=completion,
        )

    @staticmethod
    def _compute_consistency(turns_data: list[dict],
                             turn_results: list[dict]) -> float:
        """Score cross-turn consistency by keyword continuity."""
        if len(turns_data) < 2:
            return 1.0

        carry_keys: list[set] = []
        for turn in turns_data:
            carry_keys.append(set(turn.get("state_to_carry", [])))

        consistency_scores = []
        for i in range(1, len(turns_data)):
            if not turn_results[i]["response"]:
                consistency_scores.append(0.0)
                continue
            # Check if keywords from prior state appear in current response
            prior_keys = {k.lower().replace("_", " ")
                          for k in turns_data[i-1].get("state_to_carry", [])}
            response_lower = turn_results[i]["response"].lower()
            if prior_keys:
                match_count = sum(1 for k in prior_keys if k in response_lower)
                consistency_scores.append(match_count / len(prior_keys))
            else:
                consistency_scores.append(1.0)

        return float(np.mean(consistency_scores)) if consistency_scores else 0.0

    # ══════════════════════════════════════════════════════════
    # Metrics
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _compute_metrics(result: EvalResult, query: EvalQuery) -> None:
        """Compute retrieval and answer quality metrics."""
        # Keyword recall
        if query.expected_keywords:
            answer_lower = result.response.lower()
            matches = sum(
                1 for kw in query.expected_keywords if kw.lower() in answer_lower
            )
            result.keyword_recall = matches / len(query.expected_keywords)

        # Token efficiency: keywords recalled per 100 input tokens
        if result.tokens_input > 0:
            result.token_efficiency = (
                result.keyword_recall * 100 / max(result.tokens_input, 1)
            )

    @staticmethod
    def _aggregate(condition: str, results: list[EvalResult]) -> BenchmarkReport:
        """Aggregate results with bootstrap confidence intervals."""
        n = max(len(results), 1)

        te_values = [r.token_efficiency for r in results]
        kr_values = [r.keyword_recall for r in results]

        return BenchmarkReport(
            condition=condition,
            total_queries=len(results),
            avg_latency_ms=sum(r.latency_ms for r in results) / n,
            avg_tokens_input=sum(r.tokens_input for r in results) / n,
            avg_precision=sum(r.precision_at_k for r in results) / n,
            avg_recall=sum(r.recall_at_k for r in results) / n,
            avg_mrr=sum(r.mrr for r in results) / n,
            avg_keyword_recall=float(np.mean(kr_values)) if kr_values else 0.0,
            avg_token_efficiency=float(np.mean(te_values)) if te_values else 0.0,
            token_efficiency_ci=bootstrap_ci(te_values),
            keyword_recall_ci=bootstrap_ci(kr_values),
            per_query=[r.to_dict() for r in results],
        )

    @staticmethod
    def _text_context(text: str, query: str):
        """Build minimal ContextPack-compatible from raw text."""
        from ard.context.pack import ContextPack, ContextSection
        pack = ContextPack("eval", "eval", "eval", len(text) // 4)
        pack.sections = [
            ContextSection("current_query", len(query)//4, 2,
                          [{"text": query, "trust_level": "user_instruction"}]),
            ContextSection("retrieved_evidence", len(text)//4, 6,
                          [{"text": text, "trust_level": "external_untrusted"}]),
        ]
        return pack


# ═══════════════════════════════════════════════════════════════════
# I/O Helpers
# ═══════════════════════════════════════════════════════════════════

def generate_sample_queries() -> list[EvalQuery]:
    """Generate sample queries for quick testing."""
    return [
        EvalQuery(query_id=qid, query=q,
                  expected_keywords=["test"], category="factoid")
        for qid, q in [
            ("q001", "What is the main contribution of this system?"),
            ("q002", "How does the system handle memory management?"),
            ("q003", "What architecture does the system use?"),
            ("q004", "What are the key evaluation metrics?"),
            ("q005", "How does this compare to previous approaches?"),
        ]
    ]


def load_dataset(path: str) -> tuple[list[EvalQuery], list[dict]]:
    """Load benchmark dataset from JSON file.

    Returns:
        (single_turn_queries, multi_turn_scenarios)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = [EvalQuery.from_dict(q) for q in data.get("single_turn_queries", [])]
    scenarios = data.get("multi_turn_scenarios", [])

    log.info("dataset_loaded", path=path,
             single_turn=len(queries),
             multi_turn=len(scenarios))
    return queries, scenarios


def print_comparison(reports: dict[str, BenchmarkReport]) -> None:
    """Print a formatted comparison between conditions."""
    proposed = reports.get("proposed")
    baseline1 = reports.get("baseline1")
    baseline2 = reports.get("baseline2")

    if not proposed or not baseline1:
        return

    print("\n" + "=" * 70)
    print("COMPARISON REPORT")
    print("=" * 70)

    for label, report in reports.items():
        print(f"\n{report.summary()}")

    # Statistical comparison
    print("\n" + "-" * 70)
    print("STATISTICAL ANALYSIS")
    print("-" * 70)

    te_b1 = [r["token_efficiency"] for r in baseline1.per_query]
    te_b2 = [r["token_efficiency"] for r in baseline2.per_query]
    te_p = [r["token_efficiency"] for r in proposed.per_query]

    d_vs_b1 = cohens_d(te_p, te_b1)
    d_vs_b2 = cohens_d(te_p, te_b2)

    print(f"  Proposed vs Baseline1:  Cohen's d = {d_vs_b1:+.3f}")
    print(f"  Proposed vs Baseline2:  Cohen's d = {d_vs_b2:+.3f}")

    te_gain_b1 = (proposed.avg_token_efficiency - baseline1.avg_token_efficiency)
    te_pct_b1 = (te_gain_b1 / max(baseline1.avg_token_efficiency, 0.0001)) * 100
    print(f"  Token Efficiency gain vs B1: {te_pct_b1:+.1f}%")

    if te_pct_b1 >= 15:
        print(f"\n  *** H2 THRESHOLD MET: +{te_pct_b1:.1f}% >= 15% ***")
    else:
        print(f"\n  --- H2 THRESHOLD NOT MET: +{te_pct_b1:.1f}% < 15% ---")

    effect_label = "large" if abs(d_vs_b1) > 0.8 else ("medium" if abs(d_vs_b1) > 0.5 else "small")
    print(f"  Effect size vs B1: {effect_label}")


def export_results(results: dict, path: str) -> None:
    """Export all results to JSON for charting."""
    export = {}

    for key, reports in results.items():
        if isinstance(reports, dict):
            export[key] = {k: r.to_dict() for k, r in reports.items()}
        elif isinstance(reports, list):
            export[key] = [r.to_dict() for r in reports]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)

    log.info("results_exported", path=path)
