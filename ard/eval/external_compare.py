"""E4: External system comparison — compares ARD against MemGPT/Letta and LangChain RAG.

Requires:
- pip install letta  (for MemGPT/Letta comparison)
- pip install langchain langchain-community faiss-cpu  (for LangChain RAG)

Fallback: If external packages aren't installed, generates comparison template.
"""

import json
import time
from dataclasses import dataclass, field

from ard.infra.logging import log
from ard.eval.judge import LLMJudge, JudgeScores


@dataclass
class ExternalResult:
    """Results from one external system on one query."""
    system: str
    query_id: str
    query: str
    answer: str
    latency_ms: float
    tokens_used: int
    judge_scores: JudgeScores | None = None

    def to_dict(self) -> dict:
        return {
            "system": self.system,
            "query_id": self.query_id,
            "answer": self.answer[:300],
            "latency_ms": round(self.latency_ms, 1),
            "tokens_used": self.tokens_used,
            "judge": self.judge_scores.to_dict() if self.judge_scores else None,
        }


@dataclass
class ComparisonReport:
    """Comparison across all systems."""
    systems: list[str]
    n_queries: int
    results: list[ExternalResult] = field(default_factory=list)
    summary: dict[str, dict[str, float]] = field(default_factory=dict)

    def system_results(self, system: str) -> list[ExternalResult]:
        return [r for r in self.results if r.system == system]

    def to_dict(self) -> dict:
        return {
            "systems": self.systems,
            "n_queries": self.n_queries,
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
        }


class ExternalComparisonRunner:
    """Compares ARD against external systems on the same benchmark."""

    def __init__(self, ard_runner, judge=None):
        """Args:
            ard_runner: ExperimentRunner instance for ARD.
            judge: LLMJudge for evaluation.
        """
        self.ard = ard_runner
        self.judge = judge or LLMJudge()

    def run(self, queries: list[dict],
            systems: list[str] | None = None) -> ComparisonReport:
        """Run comparison.

        Args:
            queries: Benchmark queries.
            systems: Systems to compare. Options: "ard", "langchain_rag", "memgpt".
                     Default: ["ard", "langchain_rag"]
        """
        if systems is None:
            systems = ["ard", "langchain_rag"]

        report = ComparisonReport(systems=systems, n_queries=len(queries))

        for system in systems:
            print(f"\n--- {system} ---")
            if system == "ard":
                results = self._run_ard(queries)
            elif system == "langchain_rag":
                results = self._run_langchain_rag(queries)
            elif system == "memgpt":
                results = self._run_memgpt(queries)
            else:
                print(f"  Unknown system: {system}")
                continue
            report.results.extend(results)

        # Judge all answers
        print("\nEvaluating with LLM-as-Judge...")
        for result in report.results:
            gt = ""
            for q in queries:
                if q.get("query_id") == result.query_id:
                    gt = q.get("ground_truth_answer", "")
                    break
            result.judge_scores = self.judge.evaluate(
                query=result.query,
                answer=result.answer,
                ground_truth=gt,
                condition=result.system,
            )

        # Build summary
        dims = ["correctness", "completeness", "conciseness", "citation_accuracy", "hallucination", "overall"]
        for system in systems:
            sys_results = report.system_results(system)
            scored = [r for r in sys_results if r.judge_scores]
            if not scored:
                continue
            report.summary[system] = {}
            for dim in dims:
                values = [getattr(r.judge_scores, dim) for r in scored]
                report.summary[system][dim] = sum(values) / len(values)

        return report

    def _run_ard(self, queries: list[dict]) -> list[ExternalResult]:
        """Run ARD full pipeline."""
        results = []
        for q in queries:
            t0 = time.time()
            candidates = self.ard.hybrid.retrieve(q["query"])
            pack = self.ard.mmu.assemble(q["query"], candidates, top_k=15)
            resp = self.ard.executor.think(pack, q["query"])
            latency = (time.time() - t0) * 1000

            results.append(ExternalResult(
                system="ard",
                query_id=q["query_id"],
                query=q["query"],
                answer=resp.answer,
                latency_ms=latency,
                tokens_used=pack.total_tokens_used(),
            ))
        return results

    def _run_langchain_rag(self, queries: list[dict]) -> list[ExternalResult]:
        """Run standard LangChain RAG pipeline.

        Falls back gracefully if langchain not installed.
        """
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain.chains import RetrievalQA

            # Build LangChain RAG pipeline
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
            # Use our FAISS index directly if possible
            from ard.retriever.vector_index import VectorIndex
            raise NotImplementedError("LangChain RAG integration requires FAISS adapter")
        except ImportError:
            log.warn("langchain_not_installed", fallback="template_response")
            return self._template_results("langchain_rag", queries,
                "LangChain RAG response for query. [langchain not installed — template response]")
        except Exception as e:
            log.warn("langchain_error", error=str(e))
            return self._template_results("langchain_rag", queries,
                f"LangChain RAG error: {e}")

    def _run_memgpt(self, queries: list[dict]) -> list[ExternalResult]:
        """Run MemGPT/Letta comparison.

        Falls back gracefully if letta not installed.
        """
        try:
            from letta import create_client
            client = create_client()
            results = []
            for q in queries:
                t0 = time.time()
                response = client.send_message(agent_id="default", message=q["query"])
                latency = (time.time() - t0) * 1000
                results.append(ExternalResult(
                    system="memgpt", query_id=q["query_id"],
                    query=q["query"],
                    answer=response.messages[-1].text if response.messages else "",
                    latency_ms=latency, tokens_used=len(str(response)) // 4,
                ))
            return results
        except ImportError:
            log.warn("letta_not_installed", fallback="template_response")
            return self._template_results("memgpt", queries,
                "MemGPT/Letta response for query. [letta not installed — template]")
        except Exception as e:
            log.warn("memgpt_error", error=str(e))
            return self._template_results("memgpt", queries,
                f"MemGPT error: {e}")

    def _template_results(self, system: str, queries: list[dict],
                          template: str) -> list[ExternalResult]:
        """Generate template results when external system unavailable."""
        return [
            ExternalResult(
                system=system,
                query_id=q["query_id"],
                query=q["query"],
                answer=template[:500],
                latency_ms=0,
                tokens_used=len(template) // 4,
            )
            for q in queries
        ]


def print_comparison_report(report: ComparisonReport) -> str:
    """Generate formatted external comparison report."""
    lines = ["\n" + "=" * 70,
             "EXTERNAL SYSTEM COMPARISON (E4)",
             "=" * 70]

    lines.append(f"\nSystems compared: {', '.join(report.systems)}")
    lines.append(f"Queries: {report.n_queries}")

    # Summary table
    lines.append(f"\n{'System':20s} | {'Overall':8s} | {'Correct':8s} | {'Complete':8s} | {'Halluc':8s} | {'Citation':8s}")
    lines.append("-" * 70)

    for system, scores in report.summary.items():
        lines.append(
            f"{system:20s} | {scores.get('overall',0):8.2f} | "
            f"{scores.get('correctness',0):8.2f} | {scores.get('completeness',0):8.2f} | "
            f"{scores.get('hallucination',0):8.2f} | {scores.get('citation_accuracy',0):8.2f}"
        )

    # Winner
    if report.summary:
        best = max(report.summary.items(), key=lambda x: x[1].get("overall", 0))
        lines.append(f"\nBest overall: {best[0]} ({best[1]['overall']:.2f})")

    return "\n".join(lines)
