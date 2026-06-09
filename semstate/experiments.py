"""Recoverable JSONL experiment execution with a small manifest."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable


class ExperimentRunner:
    def __init__(self, output_path: str, manifest_path: str):
        self.output_path = Path(output_path)
        self.manifest_path = Path(manifest_path)
        self._write_lock = Lock()

    def run(
        self,
        cases: Iterable[dict],
        evaluator: Callable[[dict], dict],
        *,
        max_workers: int = 1,
    ) -> list[dict]:
        manifest = self._load_manifest()
        completed = set(manifest.get("completed", []))
        pending = [case for case in cases if case["case_id"] not in completed]
        results = []

        if max_workers <= 1:
            for case in pending:
                results.append(self._execute(case, evaluator, completed))
            return results

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(evaluator, case): case
                for case in pending
            }
            for future in as_completed(futures):
                case = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    with self._write_lock:
                        self._append_jsonl({
                            "case_id": case["case_id"],
                            "status": "failed",
                            "error": str(exc),
                            "input": case,
                        })
                    raise
                results.append(self._record_success(case, result, completed))
        return results

    def _execute(
        self,
        case: dict,
        evaluator: Callable[[dict], dict],
        completed: set[str],
    ) -> dict:
        try:
            result = evaluator(case)
        except Exception as exc:
            self._append_jsonl({
                "case_id": case["case_id"],
                "status": "failed",
                "error": str(exc),
                "input": case,
            })
            raise
        return self._record_success(case, result, completed)

    def _record_success(
        self,
        case: dict,
        result: dict,
        completed: set[str],
    ) -> dict:
        record = {
            "case_id": case["case_id"],
            "status": "completed",
            "input": case,
            "result": result,
        }
        with self._write_lock:
            self._append_jsonl(record)
            completed.add(case["case_id"])
            self._append_manifest(case["case_id"])
        return record

    def _append_jsonl(self, record: dict) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {"completed": []}
        completed = set()
        text = self.manifest_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if "completed" in record:
                completed.update(record["completed"])
            elif record.get("status") == "completed":
                completed.add(record["case_id"])
        return {"completed": sorted(completed)}

    def _append_manifest(self, case_id: str) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(
                {"case_id": case_id, "status": "completed"},
                ensure_ascii=False,
                sort_keys=True,
            ) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
