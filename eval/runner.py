"""Labeled retrieval and continuity benchmark for Agent-OS."""

import argparse
import json
import tempfile
import time
from pathlib import Path

from eval.metrics import mrr, precision_at_k


THRESHOLDS = {
    "precision@k": 0.30,
    "mrr": 0.60,
    "continuity_hit_rate": 2 / 3,
}


def run_benchmark(work_dir: str | Path, k: int = 5) -> dict:
    """Run real PDF, code, and memory cases against Agent-OS and baselines."""
    from src.__main__ import build_runtime
    from src.config import Config
    from src.models.memory import MemoryItem, MemoryType

    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = Config(
        db_path=str(root / "benchmark.db"),
        file_store_path=str(root / "files"),
        vector_index_path=str(root / "benchmark.faiss"),
        embedding_dim=64,
        default_token_budget=4000,
        top_k_after_rerank=k,
    )
    runtime = build_runtime(embed_provider="mock", config=config)

    pdf_path = root / "raptor_eval.pdf"
    _write_pdf(pdf_path)
    pdf_source = runtime.file_store.ingest_pdf(
        pdf_path,
        parser_mode="pymupdf",
    )
    runtime.index_source(pdf_source)

    code_dir = root / "sample_project"
    code_paths = _write_code_project(code_dir)
    for path in code_paths:
        runtime.upload_file(path)

    retrieval_cases = [
        ("Which benchmarks evaluate RAPTOR?", {"file:raptor_eval.pdf"}),
        ("What limitation does RAPTOR report?", {"file:raptor_eval.pdf"}),
        ("Where is the SQLite connection created?", {"file:database.py"}),
        ("Which file starts the application entry point?", {"file:main.py"}),
        ("Where is JWT token refresh implemented?", {"file:auth.py"}),
    ]

    agent_rankings, agent_latency, agent_tokens = _run_hybrid(
        runtime, retrieval_cases, k, rerank=True
    )
    ablation_rankings, ablation_latency, ablation_tokens = _run_hybrid(
        runtime, retrieval_cases, k, rerank=False
    )
    baseline_rankings, baseline_latency, baseline_tokens = _run_keyword_baseline(
        runtime, retrieval_cases, k
    )
    relevant_sets = [relevant for _, relevant in retrieval_cases]

    memories = [
        MemoryItem(
            memory_id="memory_api",
            type=MemoryType.DECISION,
            content="The project selected FastAPI as its API framework.",
            importance=0.9,
        ),
        MemoryItem(
            memory_id="memory_auth",
            type=MemoryType.PROJECT_STATE,
            content="JWT refresh tokens rotate after every successful refresh.",
            importance=0.9,
        ),
        MemoryItem(
            memory_id="memory_db",
            type=MemoryType.DECISION,
            content="The project selected PostgreSQL as the primary database.",
            importance=0.9,
        ),
        MemoryItem(
            memory_id="memory_noise",
            type=MemoryType.FILE_SUMMARY,
            content="Kafka retains event logs for seven days.",
        ),
    ]
    for item in memories:
        runtime.memory_store.insert(item)

    continuity_cases = [
        ("Which API framework did we decide to use?", "memory_api"),
        ("How should token refresh work?", "memory_auth"),
        ("What database did the project choose?", "memory_db"),
    ]
    continuity_rankings = []
    continuity_hits = 0
    for query, expected_id in continuity_cases:
        selection = runtime.memory_retriever.retrieve(query, limit=3)
        ranked_ids = [item.memory_id for item in selection.all]
        continuity_rankings.append(ranked_ids)
        continuity_hits += int(expected_id in ranked_ids)

    modes = {
        "agent_os": _metrics(
            agent_rankings,
            relevant_sets,
            k,
            agent_latency,
            agent_tokens,
            continuity_hits / len(continuity_cases),
        ),
        "hybrid_no_rerank": _metrics(
            ablation_rankings,
            relevant_sets,
            k,
            ablation_latency,
            ablation_tokens,
            0.0,
        ),
        "keyword_only": _metrics(
            baseline_rankings,
            relevant_sets,
            k,
            baseline_latency,
            baseline_tokens,
            0.0,
        ),
    }
    passed = (
        modes["agent_os"][f"precision@{k}"] >= THRESHOLDS["precision@k"]
        and modes["agent_os"]["mrr"] >= THRESHOLDS["mrr"]
        and modes["agent_os"]["continuity_hit_rate"]
        >= THRESHOLDS["continuity_hit_rate"]
    )

    report = {
        "dataset": {
            "pdf_pages": 2,
            "code_files": len(code_paths),
            "retrieval_queries": len(retrieval_cases),
            "continuity_queries": len(continuity_cases),
        },
        "modes": modes,
        "delta_vs_keyword": {
            "precision": round(
                modes["agent_os"][f"precision@{k}"]
                - modes["keyword_only"][f"precision@{k}"],
                4,
            ),
            "mrr": round(
                modes["agent_os"]["mrr"] - modes["keyword_only"]["mrr"],
                4,
            ),
            "continuity_hit_rate": round(
                modes["agent_os"]["continuity_hit_rate"]
                - modes["keyword_only"]["continuity_hit_rate"],
                4,
            ),
        },
        "thresholds": {
            f"precision@{k}": THRESHOLDS["precision@k"],
            "mrr": THRESHOLDS["mrr"],
            "continuity_hit_rate": THRESHOLDS["continuity_hit_rate"],
        },
        "passed": passed,
    }
    runtime.close()
    return report


def run_eval(
    scenario_filter: str = "",
    k: int = 5,
    work_dir: str | Path | None = None,
) -> dict:
    """Backward-compatible evaluation entry point."""
    if scenario_filter:
        raise ValueError(
            "Scenario filters are no longer used; the labeled benchmark "
            "always runs all acceptance cases."
        )
    if work_dir is not None:
        return run_benchmark(work_dir, k)
    with tempfile.TemporaryDirectory(prefix="agent_os_eval_") as tmp:
        return run_benchmark(tmp, k)


def _run_hybrid(runtime, cases, k, rerank):
    rankings = []
    latencies = []
    token_costs = []
    for query, _ in cases:
        start = time.perf_counter()
        if rerank:
            results = runtime.retriever.retrieve_and_rerank(
                query, runtime.embed_fn, k=k
            )
        else:
            results = runtime.retriever.retrieve(
                query, runtime.embed_fn, k=k
            )
        latencies.append((time.perf_counter() - start) * 1000)
        rankings.append([result.source_ref for result in results])
        token_costs.append(
            sum(max(1, len(result.text_preview) // 4) for result in results)
        )
    return rankings, latencies, token_costs


def _run_keyword_baseline(runtime, cases, k):
    rankings = []
    latencies = []
    token_costs = []
    for query, _ in cases:
        start = time.perf_counter()
        hits = runtime.retriever.keyword_index.search_chunks(query, k=k)
        latencies.append((time.perf_counter() - start) * 1000)
        sources = []
        tokens = 0
        for chunk_id, _ in hits:
            chunk = runtime.file_store.get_chunk(chunk_id)
            if chunk is not None:
                sources.append(chunk.source_id)
                tokens += max(1, len(chunk.text) // 4)
        rankings.append(sources)
        token_costs.append(tokens)
    return rankings, latencies, token_costs


def _metrics(
    rankings,
    relevant_sets,
    k,
    latencies,
    token_costs,
    continuity_hit_rate,
):
    precisions = [
        precision_at_k(ranking, relevant, k)
        for ranking, relevant in zip(rankings, relevant_sets)
    ]
    return {
        f"precision@{k}": round(sum(precisions) / len(precisions), 4),
        "mrr": round(mrr(rankings, relevant_sets), 4),
        "continuity_hit_rate": round(continuity_hit_rate, 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3),
        "avg_context_tokens": round(sum(token_costs) / len(token_costs), 1),
    }


def _write_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()
    page_one = doc.new_page()
    page_one.insert_textbox(
        fitz.Rect(50, 50, 550, 760),
        (
            "RAPTOR Evaluation Paper\n\n"
            "RAPTOR combines dense retrieval with chain-of-thought prompts. "
            "The method is evaluated on GLUE, SuperGLUE, and RAFT benchmarks."
        ),
        fontsize=12,
    )
    page_two = doc.new_page()
    page_two.insert_textbox(
        fitz.Rect(50, 50, 550, 760),
        (
            "Limitations\n\n"
            "RAPTOR depends on a high-quality retrieval corpus and performance "
            "degrades on out-of-domain tasks."
        ),
        fontsize=12,
    )
    doc.save(path)
    doc.close()


def _write_code_project(root: Path) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "main.py": (
            "from database import connect\n\n"
            "def main():\n"
            "    connection = connect()\n"
            "    return connection\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
        "database.py": (
            "import sqlite3\n\n"
            "def connect():\n"
            "    return sqlite3.connect('agent.db')\n"
        ),
        "auth.py": (
            "def refresh_jwt_token(refresh_token):\n"
            "    \"\"\"Rotate a JWT refresh token and return a new access token.\"\"\"\n"
            "    return {'access_token': refresh_token + '-rotated'}\n"
        ),
    }
    paths = []
    for name, content in files.items():
        path = root / name
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


def print_report(report: dict) -> None:
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent-OS labeled benchmark")
    parser.add_argument("--top-k", "-k", type=int, default=5)
    parser.add_argument("--work-dir", default="")
    args = parser.parse_args()
    report = run_eval(k=args.top_k, work_dir=args.work_dir or None)
    print_report(report)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
