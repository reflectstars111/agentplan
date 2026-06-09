# SemState Research Package

SemState studies commit-time semantic validity for versioned shared artifacts.
It is intentionally separate from the Agent-OS Web, GUI, and general runtime.

## Implemented

- Atomic ARD event, projection, and transaction-status commits.
- Versioned state nodes and observable dependency edges.
- Ordered validation: read versions, dependency versions, schema, domain
  constraints, executable checks, and evidence versions.
- Hard and soft invalidation propagation.
- Selective repair plans over the affected downstream closure.
- Twelve hand-authored G0 cases across deployment, migration, and pipelines.
- A canonical generator for 40 scenarios and six schedules (240 histories).
- Eight deterministic baselines and paired cluster bootstrap.
- Independent Ground Truth reconstruction and dependency-noise curves.
- Append-only JSONL experiment results and a crash-safe manifest journal.

## Generate Benchmark Assets

```powershell
python -m semstate
```

This writes:

- `eval_data/semstate_benchmark_v1.json`
- `eval_data/semstate_baselines_v1.json`

The current report is a deterministic synthetic sanity check. It is not an
LLM experiment and must not be reported as the paper's main empirical result.
