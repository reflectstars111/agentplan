"""Integration test for the labeled Agent-OS benchmark."""

from eval.runner import run_benchmark


def test_benchmark_uses_real_artifacts_and_fixed_thresholds(tmp_path):
    report = run_benchmark(tmp_path, k=3)

    assert report["dataset"]["pdf_pages"] == 2
    assert report["dataset"]["code_files"] == 3
    assert report["dataset"]["retrieval_queries"] >= 5
    assert report["dataset"]["continuity_queries"] == 3

    for mode in ("agent_os", "hybrid_no_rerank", "keyword_only"):
        metrics = report["modes"][mode]
        assert "precision@3" in metrics
        assert "mrr" in metrics
        assert "avg_latency_ms" in metrics
        assert "avg_context_tokens" in metrics

    assert report["modes"]["agent_os"]["continuity_hit_rate"] >= 2 / 3
    assert report["thresholds"]["mrr"] > 0
    assert report["thresholds"]["continuity_hit_rate"] > 0
    assert report["passed"] is True
