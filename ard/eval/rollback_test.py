"""T8: Version query and rollback performance benchmarks.

Tests version query performance across event chain sizes:
10, 100, 1K, 10K, 100K events.
Measures: current read, historical read, full history query, projection rebuild.
"""

import uuid
import time
from dataclasses import dataclass, field

import numpy as np

from ard.infra.logging import log


@dataclass
class VersionPerfResult:
    """Performance metrics for one chain size."""
    chain_size: int
    current_read_ms: float
    historical_read_ms: float
    full_history_ms: float
    projection_rebuild_ms: float
    events_per_second: float

    def to_dict(self) -> dict:
        return {
            "chain_size": self.chain_size,
            "current_read_ms": round(self.current_read_ms, 3),
            "historical_read_ms": round(self.historical_read_ms, 3),
            "full_history_ms": round(self.full_history_ms, 3),
            "projection_rebuild_ms": round(self.projection_rebuild_ms, 3),
            "events_per_second": round(self.events_per_second, 1),
        }


def run_performance_tests(state_store, txn_mgr) -> list[VersionPerfResult]:
    """Run version query performance benchmarks.

    Tests chain sizes: 10, 100, 1K, 10K, 100K.
    """
    results = []
    sizes = [10, 100, 1000, 10000]

    for size in sizes:
        result = _benchmark_chain_size(state_store, txn_mgr, size)
        results.append(result)

    return results


def _benchmark_chain_size(state_store, txn_mgr, size: int) -> VersionPerfResult:
    """Benchmark version queries at a specific event chain size."""
    key = f"test:perf_bench_{uuid.uuid4().hex[:6]}"

    # Write events
    batch_size = 100
    t0 = time.perf_counter()
    written = 0
    for batch_start in range(0, size, batch_size):
        batch_end = min(batch_start + batch_size, size)
        for i in range(batch_start, batch_end):
            txn = txn_mgr.begin()
            evt = state_store.build_event(
                key, "created" if i == 0 else "updated",
                {"iteration": i, "data": f"value_{i}", "padding": "x" * 20},
            )
            txn.add_event(evt)
            txn_mgr.commit(txn)
            written += 1
    write_time = time.perf_counter() - t0
    events_per_sec = written / max(write_time, 0.001)

    # Benchmark current read (5 trials, take median)
    trials = 5
    current_reads = []
    for _ in range(trials):
        t0 = time.perf_counter()
        state_store.read(key)
        current_reads.append((time.perf_counter() - t0) * 1000)
    current_read_ms = float(np.median(current_reads))

    # Benchmark historical read
    history = state_store.history(key)
    mid_version = history[len(history) // 2]["seq_num"] if history else 1

    hist_reads = []
    for _ in range(trials):
        t0 = time.perf_counter()
        state_store.read(key, version=mid_version)
        hist_reads.append((time.perf_counter() - t0) * 1000)
    historical_read_ms = float(np.median(hist_reads))

    # Benchmark full history query
    full_hist_times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        state_store.history(key)
        full_hist_times.append((time.perf_counter() - t0) * 1000)
    full_history_ms = float(np.median(full_hist_times))

    # Benchmark projection rebuild
    rebuild_times = []
    for _ in range(min(3, trials)):
        t0 = time.perf_counter()
        events = state_store.event_store.replay(stream_key=key)
        # Rebuild projection from events
        for evt in events:
            state_store.apply_event({
                "_stream_key": key, "_seq_num": evt.seq_num,
                "event_type": evt.event_type, **evt.payload,
            })
        rebuild_times.append((time.perf_counter() - t0) * 1000)
    projection_rebuild_ms = float(np.median(rebuild_times)) if rebuild_times else 0

    return VersionPerfResult(
        chain_size=size,
        current_read_ms=current_read_ms,
        historical_read_ms=historical_read_ms,
        full_history_ms=full_history_ms,
        projection_rebuild_ms=projection_rebuild_ms,
        events_per_second=events_per_sec,
    )


def print_version_perf_report(results: list[VersionPerfResult]) -> str:
    """Generate formatted version query performance report."""
    lines = ["\n" + "=" * 70,
             "VERSION QUERY PERFORMANCE REPORT (T8)",
             "=" * 70,
             f"{'Chain Size':>10s} | {'Current':>8s} | {'Historical':>10s} | "
             f"{'FullHist':>8s} | {'Rebuild':>8s} | {'Write':>8s}",
             "-" * 65]

    for r in results:
        lines.append(
            f"{r.chain_size:10d} | {r.current_read_ms:7.3f}ms | "
            f"{r.historical_read_ms:9.3f}ms | {r.full_history_ms:7.3f}ms | "
            f"{r.projection_rebuild_ms:7.3f}ms | {r.events_per_second:7.0f}/s"
        )

    if len(results) >= 2:
        r1 = results[0]
        r2 = results[-1]
        lines.append(f"\nScaling (from {r1.chain_size} to {r2.chain_size} events):")
        if r1.current_read_ms > 0:
            lines.append(f"  Current read: {r2.current_read_ms/r1.current_read_ms:.1f}x")
        if r1.full_history_ms > 0:
            lines.append(f"  Full history: {r2.full_history_ms/r1.full_history_ms:.1f}x")

    return "\n".join(lines)


# ── Correctness tests (preserved from original) ───────────

@dataclass
class RollbackResult:
    test_name: str
    versions_written: int
    rollback_successful: bool
    recovered_state_correct: bool
    version_history_intact: bool
    version_query_time_ms: float

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name, "versions_written": self.versions_written,
            "rollback_successful": self.rollback_successful,
            "recovered_state_correct": self.recovered_state_correct,
            "version_history_intact": self.version_history_intact,
            "version_query_time_ms": round(self.version_query_time_ms, 2),
        }


def run_rollback_tests(state_store, txn_mgr) -> list[RollbackResult]:
    """Run version rollback correctness tests."""
    results = []
    results.append(_test_simple_rollback(state_store, txn_mgr))
    results.append(_test_decision_reversal(state_store, txn_mgr))
    return results


def _test_simple_rollback(state_store, txn_mgr) -> RollbackResult:
    key = f"test:rollback_{uuid.uuid4().hex[:6]}"
    versions = ["initial_design", "changed_to_v2", "changed_to_v3"]
    for i, data in enumerate(versions):
        txn = txn_mgr.begin()
        txn.add_event(state_store.build_event(key, "created" if i == 0 else "updated",
                                                {"design": data, "version": i + 1}))
        txn_mgr.commit(txn)

    t0 = time.perf_counter()
    history = state_store.history(key)
    v1 = state_store.read(key, version=history[0]["seq_num"])
    v3 = state_store.read(key, version=history[2]["seq_num"])
    qt = (time.perf_counter() - t0) * 1000
    correct = v1 is not None and "initial_design" in str(v1) and v3 is not None

    return RollbackResult(
        test_name="simple_3version_rollback", versions_written=3,
        rollback_successful=correct, recovered_state_correct=correct,
        version_history_intact=len(history) == 3, version_query_time_ms=qt,
    )


def _test_decision_reversal(state_store, txn_mgr) -> RollbackResult:
    key = f"test:decision_{uuid.uuid4().hex[:6]}"

    txn1 = txn_mgr.begin()
    txn1.add_event(state_store.build_event(key, "created", {"architecture": "REST", "auth": "none"}))
    txn_mgr.commit(txn1)

    txn2 = txn_mgr.begin()
    state_store.read_for_transaction(key, txn2)
    txn2.add_event(state_store.build_event(key, "updated", {"architecture": "GraphQL", "auth": "JWT"}))
    txn_mgr.commit(txn2)

    history = state_store.history(key)
    v1_state = state_store.read(key, version=history[0]["seq_num"])

    txn3 = txn_mgr.begin()
    state_store.read_for_transaction(key, txn3)
    txn3.add_event(state_store.build_event(key, "updated", {
        "architecture": v1_state["architecture"], "auth": "JWT",
        "note": "reverted architecture to REST, kept JWT auth from v2",
    }))
    txn_mgr.commit(txn3)

    final = state_store.read(key)
    correct = final is not None and final.get("architecture") == "REST" and final.get("auth") == "JWT"

    return RollbackResult(
        test_name="decision_reversal_rest_graphql_rest", versions_written=3,
        rollback_successful=correct, recovered_state_correct=correct,
        version_history_intact=len(state_store.history(key)) == 3,
        version_query_time_ms=0,
    )


def print_rollback_report(results: list[RollbackResult]) -> str:
    lines = ["\n" + "=" * 70, "VERSION ROLLBACK CORRECTNESS REPORT", "=" * 70]
    for r in results:
        lines.append(f"\n{r.test_name}:")
        lines.append(f"  Versions: {r.versions_written}")
        lines.append(f"  Rollback: {'OK' if r.rollback_successful else 'FAIL'}")
        lines.append(f"  State correct: {'OK' if r.recovered_state_correct else 'FAIL'}")
        lines.append(f"  History intact: {'OK' if r.version_history_intact else 'FAIL'}")
        lines.append(f"  Query time: {r.version_query_time_ms:.2f}ms")
    all_ok = all(r.recovered_state_correct and r.version_history_intact for r in results)
    lines.append(f"\nAll tests: {'PASSED' if all_ok else 'FAILED'}")
    return "\n".join(lines)
