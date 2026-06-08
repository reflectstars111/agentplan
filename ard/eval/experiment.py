"""E1: Provenance causal chain experiment — 9 controlled conditions.

Design: Incrementally add SourceID → TrustLevel → ContextPack → MMU
while keeping retrieval, ranking, top-k, prompt, model, and temperature identical.

9 conditions:
  1. bm25              — BM25 keyword only (classic baseline)
  2. vector            — BGE-M3 vector only (semantic baseline)
  3. hybrid            — BM25+Vector+Rerank (mixed baseline)
  4. hybrid_source_id  — Same as hybrid + source_id labels injected
  5. hybrid_provenance — Same as hybrid + source_id + trust_level labels
  6. ard_minimal       — Same retrieval + ContextPack (structured sections, FILTER/RANK/COMPRESS/BUDGET disabled)
  7. ard_full          — Same retrieval + full 6-step MMU ★
  8. ard_no_filter     — Same retrieval + MMU without FILTER step
  9. ard_no_budget     — Same retrieval + MMU without BUDGET step
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ard.infra.config import Config
from ard.infra.logging import log
from ard.store import RetrievalResult
from ard.eval.judge import LLMJudge, JudgeScores
from ard.eval.statistics import (
    cohens_d, paired_ttest, bonferroni_correct,
    required_sample_size, power_analysis_paired,
    full_pairwise_report,
)


@dataclass
class ExperimentCondition:
    """Definition of one experimental condition."""
    name: str
    description: str
    is_baseline: bool = True
    is_ard: bool = False
    disabled_steps: set[str] | None = None
    # Provenance chain attributes
    has_source_id: bool = False
    has_trust_level: bool = False
    has_context_pack: bool = False


# The 9-condition provenance chain
CONDITIONS = [
    # Pure baselines (no provenance)
    ExperimentCondition("bm25", "BM25 keyword → LLM",
                        has_source_id=False, has_trust_level=False, has_context_pack=False),
    ExperimentCondition("vector", "BGE-M3 vector → LLM",
                        has_source_id=False, has_trust_level=False, has_context_pack=False),
    ExperimentCondition("hybrid", "Vector + BM25 + Rerank → LLM",
                        has_source_id=False, has_trust_level=False, has_context_pack=False),
    # Provenance chain: controlled increments
    ExperimentCondition("hybrid_source_id", "Hybrid + source_id labels injected",
                        has_source_id=True, has_trust_level=False, has_context_pack=False),
    ExperimentCondition("hybrid_provenance", "Hybrid + source_id + trust_level labels injected",
                        has_source_id=True, has_trust_level=True, has_context_pack=False),
    ExperimentCondition("ard_minimal", "Hybrid + structured ContextPack (MMU steps disabled)",
                        is_baseline=False, is_ard=True,
                        disabled_steps={"filter", "rank", "compress", "budget"},
                        has_source_id=True, has_trust_level=True, has_context_pack=True),
    # Full ARD
    ExperimentCondition("ard_full", "★ Hybrid + full 6-step MMU",
                        is_baseline=False, is_ard=True,
                        has_source_id=True, has_trust_level=True, has_context_pack=True),
    # Ablation variants
    ExperimentCondition("ard_no_filter", "ARD without FILTER step",
                        is_baseline=False, is_ard=True,
                        disabled_steps={"filter"},
                        has_source_id=True, has_trust_level=True, has_context_pack=True),
    ExperimentCondition("ard_no_budget", "ARD without BUDGET step",
                        is_baseline=False, is_ard=True,
                        disabled_steps={"budget"},
                        has_source_id=True, has_trust_level=True, has_context_pack=True),
]


@dataclass
class ExperimentRun:
    """Complete result for one condition on one query."""
    query_id: str
    condition: str
    query: str
    answer: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    source_refs: list[str] = field(default_factory=list)
    strategy_labels: list[str] = field(default_factory=list)
    judge_scores: JudgeScores | None = None


@dataclass
class ExperimentReport:
    """Aggregated report for all conditions."""
    conditions: list[str]
    n_queries: int
    runs: list[ExperimentRun]
    judge_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    statistical_tests: str = ""

    def condition_runs(self, condition: str) -> list[ExperimentRun]:
        return [r for r in self.runs if r.condition == condition]

    def condition_scores(self, condition: str, metric: str = "overall") -> list[float]:
        scores = []
        for r in self.condition_runs(condition):
            if r.judge_scores:
                scores.append(getattr(r.judge_scores, metric, 0))
        return scores

    def to_dict(self) -> dict:
        result = {
            "conditions": self.conditions,
            "n_queries": self.n_queries,
            "judge_summary": self.judge_summary,
            "statistical_tests": self.statistical_tests,
            "per_condition": {},
        }
        for cond in self.conditions:
            cr = self.condition_runs(cond)
            result["per_condition"][cond] = {
                "runs": [{
                    "query_id": r.query_id,
                    "answer": r.answer[:200],
                    "tokens_input": r.tokens_input,
                    "tokens_output": r.tokens_output,
                    "latency_ms": r.latency_ms,
                    "judge_scores": r.judge_scores.to_dict() if r.judge_scores else None,
                } for r in cr],
                "avg_tokens_input": sum(r.tokens_input for r in cr) / max(len(cr), 1),
                "avg_latency": sum(r.latency_ms for r in cr) / max(len(cr), 1),
            }
        return result


class ExperimentRunner:
    """Runs the 9-condition provenance chain experiment.

    Key design: For conditions 3-7 (hybrid through ard_full), the SAME
    retrieval results are used. Only the provenance metadata and context
    assembly differ, enabling causal attribution of quality gains.
    """

    def __init__(self, knowledge_store, hybrid_retriever, context_mmu,
                 executor, judge=None, config=None):
        self.store = knowledge_store
        self.hybrid = hybrid_retriever
        self.mmu = context_mmu
        self.executor = executor
        self.judge = judge or LLMJudge()
        self.config = config or Config()

    def run(self, queries: list[dict], conditions=None,
            quiet: bool = False) -> ExperimentReport:
        """Run the experiment.

        For provenance chain conditions (hybrid → ard_full), retrieval
        is done ONCE per query and shared across all conditions in the chain.
        """
        if conditions is None:
            conditions = CONDITIONS

        condition_names = [c.name for c in conditions]
        all_runs: list[ExperimentRun] = []

        # Pre-compute retrieval for provenance chain conditions
        # Conditions 3-7 share the same retrieval results
        shared_retrieval_cache: dict[str, list[RetrievalResult]] = {}
        chain_conditions = [c for c in conditions if c.name in
                          ("hybrid", "hybrid_source_id", "hybrid_provenance",
                           "ard_minimal", "ard_full", "ard_no_filter", "ard_no_budget")]

        for ci, cond in enumerate(conditions):
            if not quiet:
                print(f"[{ci+1}/{len(conditions)}] {cond.name:25s} — {cond.description}")

            for qi, q in enumerate(queries):
                # For chain conditions, use shared retrieval
                if cond in chain_conditions and cond.name != "hybrid":
                    # Use cached retrieval from hybrid baseline
                    if q["query_id"] not in shared_retrieval_cache:
                        shared_retrieval_cache[q["query_id"]] = self.hybrid.retrieve(q["query"])
                    run = self._run_with_cached_retrieval(q, cond, shared_retrieval_cache[q["query_id"]])
                else:
                    run = self._run_single(q, cond)

                all_runs.append(run)

            if not quiet and len(queries) >= 10:
                cr = [r for r in all_runs if r.condition == cond.name]
                avg_tok = sum(r.tokens_input for r in cr) / max(len(cr), 1)
                print(f"  Done: avg tokens={avg_tok:.0f}")

        report = ExperimentReport(conditions=condition_names, n_queries=len(queries),
                                   runs=all_runs)
        report = self._judge_all(report, queries, quiet)
        report.statistical_tests = self._statistical_report(report)
        return report

    def _run_single(self, query: dict, cond: ExperimentCondition) -> ExperimentRun:
        """Execute one query under one condition."""
        q_text = query["query"]
        t0 = time.time()

        # Dispatch to condition-specific handler
        if cond.name == "bm25":
            answer, ctx = self._condition_bm25(q_text)
        elif cond.name == "vector":
            answer, ctx = self._condition_vector(q_text)
        elif cond.name == "hybrid":
            answer, ctx = self._condition_hybrid_baseline(q_text)
        elif cond.name == "hybrid_source_id":
            answer, ctx = self._condition_provenance_chain(q_text, "source_id")
        elif cond.name == "hybrid_provenance":
            answer, ctx = self._condition_provenance_chain(q_text, "provenance")
        elif cond.name == "ard_minimal":
            answer, ctx = self._condition_ard_with_steps(q_text, {"filter", "rank", "compress", "budget"})
        elif cond.name == "ard_full":
            answer, ctx = self._condition_ard_with_steps(q_text, set())
        elif cond.name == "ard_no_filter":
            answer, ctx = self._condition_ard_with_steps(q_text, {"filter"})
        elif cond.name == "ard_no_budget":
            answer, ctx = self._condition_ard_with_steps(q_text, {"budget"})
        else:
            answer, ctx = self._condition_hybrid_baseline(q_text)  # fallback

        latency = (time.time() - t0) * 1000
        return ExperimentRun(
            query_id=query["query_id"], condition=cond.name, query=q_text,
            answer=answer, tokens_input=ctx.get("tokens_input", 0),
            tokens_output=len(answer) // 4, latency_ms=latency,
            source_refs=ctx.get("source_refs", []),
            strategy_labels=ctx.get("strategy_labels", []),
        )

    def _run_with_cached_retrieval(self, query: dict, cond: ExperimentCondition,
                                   candidates: list[RetrievalResult]) -> ExperimentRun:
        """Execute using pre-computed retrieval results (provenance chain)."""
        q_text = query["query"]
        t0 = time.time()

        if cond.name == "hybrid_source_id":
            answer, ctx = self._provenance_chain_from_candidates(
                q_text, candidates, "source_id")
        elif cond.name == "hybrid_provenance":
            answer, ctx = self._provenance_chain_from_candidates(
                q_text, candidates, "provenance")
        elif cond.name == "ard_minimal":
            answer, ctx = self._ard_from_candidates(
                q_text, candidates, {"filter", "rank", "compress", "budget"})
        elif cond.name == "ard_full":
            answer, ctx = self._ard_from_candidates(q_text, candidates, set())
        elif cond.name == "ard_no_filter":
            answer, ctx = self._ard_from_candidates(q_text, candidates, {"filter"})
        elif cond.name == "ard_no_budget":
            answer, ctx = self._ard_from_candidates(q_text, candidates, {"budget"})
        else:
            answer, ctx = self._condition_hybrid_baseline(q_text)

        latency = (time.time() - t0) * 1000
        return ExperimentRun(
            query_id=query["query_id"], condition=cond.name, query=q_text,
            answer=answer, tokens_input=ctx.get("tokens_input", 0),
            tokens_output=len(answer) // 4, latency_ms=latency,
            source_refs=ctx.get("source_refs", []),
            strategy_labels=ctx.get("strategy_labels", []),
        )

    # ── Baseline condition handlers ─────────────────────────

    def _condition_bm25(self, query: str) -> tuple[str, dict]:
        candidates = self.store._keyword_search(query, top_k=15)
        ctx_text = "\n\n".join(r.text_preview for r in candidates)
        resp = self.executor.think(self._text_context(ctx_text, query), query)
        return resp.answer, {"tokens_input": len(ctx_text)//4,
                             "source_refs": [], "strategy_labels": ["bm25"]}

    def _condition_vector(self, query: str) -> tuple[str, dict]:
        candidates = self.store._vector_search(query, top_k=15)
        ctx_text = "\n\n".join(r.text_preview for r in candidates)
        resp = self.executor.think(self._text_context(ctx_text, query), query)
        return resp.answer, {"tokens_input": len(ctx_text)//4,
                             "source_refs": [], "strategy_labels": ["vector"]}

    def _condition_hybrid_baseline(self, query: str) -> tuple[str, dict]:
        """Hybrid retrieval → flat text → LLM (NO provenance)."""
        candidates = self.hybrid.retrieve(query)
        ctx_text = "\n\n".join(r.text_preview for r in candidates[:15])
        resp = self.executor.think(self._text_context(ctx_text, query), query)
        return resp.answer, {
            "tokens_input": len(ctx_text)//4,
            "source_refs": [],  # No source tracking
            "strategy_labels": list(set(r.strategy for r in candidates)),
        }

    # ── Provenance chain handlers ───────────────────────────

    def _condition_provenance_chain(self, query: str, level: str) -> tuple[str, dict]:
        """Hybrid retrieval + provenance labels injected into text.

        level="source_id": [src:X] prefix before each chunk
        level="provenance": [src:X|trust:Y] prefix before each chunk
        """
        candidates = self.hybrid.retrieve(query)
        return self._provenance_chain_from_candidates(query, candidates, level)

    def _provenance_chain_from_candidates(self, query: str,
                                          candidates: list[RetrievalResult],
                                          level: str) -> tuple[str, dict]:
        """Build provenance-labeled text from cached candidates."""
        top = candidates[:15]
        parts = []
        for r in top:
            if level == "source_id":
                parts.append(f"[source:{r.source_ref}] {r.text_preview}")
            elif level == "provenance":
                parts.append(f"[source:{r.source_ref}|trust:{r.trust_level}] {r.text_preview}")
            else:
                parts.append(r.text_preview)
        ctx_text = "\n\n".join(parts)
        resp = self.executor.think(self._text_context(ctx_text, query), query)
        source_refs = [r.source_ref for r in top] if level in ("source_id", "provenance") else []
        return resp.answer, {
            "tokens_input": len(ctx_text)//4,
            "source_refs": source_refs,
            "strategy_labels": list(set(r.strategy for r in candidates)),
        }

    def _condition_ard_with_steps(self, query: str,
                                  disabled_steps: set[str]) -> tuple[str, dict]:
        """ARD with specific MMU steps disabled."""
        candidates = self.hybrid.retrieve(query)
        return self._ard_from_candidates(query, candidates, disabled_steps)

    def _ard_from_candidates(self, query: str, candidates: list[RetrievalResult],
                             disabled_steps: set[str]) -> tuple[str, dict]:
        """Build ARD context from cached candidates."""
        context_pack = self.mmu.assemble(
            query=query, retrieval_results=candidates,
            system_instruction="Answer based ONLY on provided context. Cite sources when possible.",
            top_k=15, disabled_steps=disabled_steps,
        )
        resp = self.executor.think(context_pack, query)
        return resp.answer, {
            "tokens_input": context_pack.total_tokens_used(),
            "source_refs": context_pack.source_refs,
            "strategy_labels": list(set(r.strategy for r in candidates)),
        }

    # ── Judge ────────────────────────────────────────────────

    def _judge_all(self, report: ExperimentReport, queries: list[dict],
                   quiet: bool = False) -> ExperimentReport:
        gt_map = {q["query_id"]: q.get("ground_truth_answer", "") for q in queries}
        for run in report.runs:
            gt = gt_map.get(run.query_id, "")
            scores = self.judge.evaluate(
                query=run.query, answer=run.answer, ground_truth=gt,
                context_sources=run.source_refs, condition=run.condition,
            )
            run.judge_scores = scores

        dims = ["correctness", "completeness", "conciseness", "citation_accuracy", "groundedness", "overall"]
        judge_summary = {}
        for cond in report.conditions:
            runs = report.condition_runs(cond)
            scored = [r for r in runs if r.judge_scores]
            if not scored:
                continue
            judge_summary[cond] = {}
            for dim in dims:
                values = [getattr(r.judge_scores, dim) for r in scored]
                judge_summary[cond][dim] = sum(values) / max(len(values), 1)
            judge_summary[cond]["n"] = len(scored)
        report.judge_summary = judge_summary

        if not quiet:
            print(f"\nJudge Summary ({len(report.runs)} answers):")
            for cond, scores in sorted(judge_summary.items()):
                print(f"  {cond:25s}: overall={scores.get('overall',0):.2f}, "
                      f"citation={scores.get('citation_accuracy',0):.2f}")
        return report

    def _statistical_report(self, report: ExperimentReport) -> str:
        condition_scores = {}
        for cond in report.conditions:
            scores = report.condition_scores(cond, "overall")
            if scores:
                condition_scores[cond] = scores
        if not condition_scores:
            return "No scores"
        return full_pairwise_report(condition_scores, "Judge Overall Score")

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _text_context(text: str, query: str):
        from ard.context.pack import ContextPack, ContextSection
        pack = ContextPack("eval", "eval", "eval", max(len(text)//4, 1))
        pack.sections = [
            ContextSection("current_query", len(query)//4, 2,
                          [{"text": query, "trust_level": "user_instruction"}]),
            ContextSection("retrieved_evidence", len(text)//4, 6,
                          [{"text": text, "trust_level": "external_untrusted"}]),
        ]
        return pack
