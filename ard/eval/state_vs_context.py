"""E3: 2x2 State vs Context Length experiment — validates H3.

Hypothesis: State management can compensate for smaller context windows.
H3 predicts: 8K+State >= 32K+No State (non-inferiority)

Design: 2 (State: on/off) x 2 (Context: 8K/32K) = 4 conditions.
Each condition runs multi-turn scenarios and measures consistency.
"""

import time
from dataclasses import dataclass, field

from ard.infra.logging import log
from ard.eval.statistics import (
    non_inferiority_test, cohens_d, paired_ttest,
    power_analysis_paired, required_sample_size,
)
from ard.eval.multi_turn import (
    MultiTurnScenario, ScenarioResult, TurnResult,
    MULTI_TURN_SCENARIOS, print_multi_turn_report,
)


@dataclass
class ContextConfig:
    """Configuration for one context length condition."""
    label: str
    token_budget: int
    description: str


CONTEXT_CONFIGS = [
    ContextConfig("small_8k", 8000, "8K tokens (constrained)"),
    ContextConfig("large_32k", 32000, "32K tokens (abundant)"),
]


@dataclass
class CellResult:
    """Result for one cell in the 2x2 matrix."""
    state: bool     # True = with state, False = no state
    context: str    # "small_8k" or "large_32k"
    scenarios: list[ScenarioResult] = field(default_factory=list)
    avg_consistency: float = 0.0
    avg_tokens: float = 0.0
    avg_completion: float = 0.0

    def label(self) -> str:
        state_label = "State" if self.state else "NoState"
        ctx_label = "8K" if self.context == "small_8k" else "32K"
        return f"{state_label}_{ctx_label}"


@dataclass
class TwoByTwoResult:
    """Complete 2x2 experiment results."""
    cells: dict[str, CellResult] = field(default_factory=dict)
    scenarios: list[str] = field(default_factory=list)
    n_turns: int = 5
    h3_supported: bool = False
    non_inferiority_p: float = 1.0
    interaction_effect: float = 0.0

    def to_dict(self) -> dict:
        return {
            "h3_supported": self.h3_supported,
            "non_inferiority_p": round(self.non_inferiority_p, 4),
            "interaction_effect": round(self.interaction_effect, 3),
            "cells": {
                label: {
                    "state": cell.state,
                    "context": cell.context,
                    "avg_consistency": round(cell.avg_consistency, 4),
                    "avg_tokens": round(cell.avg_tokens, 0),
                    "avg_completion": round(cell.avg_completion, 4),
                    "scenarios": [s.to_dict() for s in cell.scenarios],
                }
                for label, cell in self.cells.items()
            },
        }


class StateVsContextExperiment:
    """Runs the 2x2 State x Context Length experiment.

    Tests H3: good state management compensates for limited context.
    """

    def __init__(self, hybrid_retriever, mmu, executor,
                 state_store=None, txn_mgr=None, trace_store=None):
        self.hybrid = hybrid_retriever
        self.mmu = mmu
        self.executor = executor
        self.state_store = state_store
        self.txn_mgr = txn_mgr
        self.trace_store = trace_store

    def run(self, scenarios: list[MultiTurnScenario] | None = None,
            quiet: bool = False) -> TwoByTwoResult:
        """Run all 4 cells of the 2x2 matrix."""
        if scenarios is None:
            scenarios = MULTI_TURN_SCENARIOS[:3]  # Use 3 for efficiency

        result = TwoByTwoResult(scenarios=[s.name for s in scenarios])

        # Run all 4 cells
        for state_enabled in [False, True]:
            for ctx_config in CONTEXT_CONFIGS:
                cell_label = f"state_{state_enabled}_{ctx_config.label}"
                if not quiet:
                    state_str = "With State" if state_enabled else "No State"
                    print(f"\n  Cell: {state_str} x {ctx_config.label} ({ctx_config.token_budget} tokens)")

                # Temporarily override token budget
                original_budget = self.mmu.config.default_token_budget
                self.mmu.config.default_token_budget = ctx_config.token_budget

                cell = CellResult(
                    state=state_enabled,
                    context=ctx_config.label,
                )

                for scenario in scenarios:
                    sr = self._run_scenario_with_config(
                        scenario, with_state=state_enabled,
                        token_budget=ctx_config.token_budget,
                    )
                    cell.scenarios.append(sr)
                    if not quiet:
                        print(f"    {scenario.name}: consistency={sr.consistency_score:.3f}")

                # Restore original budget
                self.mmu.config.default_token_budget = original_budget

                # Compute cell averages
                if cell.scenarios:
                    cell.avg_consistency = sum(s.consistency_score for s in cell.scenarios) / len(cell.scenarios)
                    cell.avg_tokens = sum(
                        sum(t.tokens_used for t in s.turns) / max(len(s.turns), 1)
                        for s in cell.scenarios
                    ) / len(cell.scenarios)
                    cell.avg_completion = sum(s.completion_rate for s in cell.scenarios) / len(cell.scenarios)

                result.cells[cell_label] = cell

        # H3 analysis: non-inferiority of C (8K+State) vs B (32K+No State)
        cell_c = result.cells.get("state_True_small_8k")
        cell_b = result.cells.get("state_False_large_32k")

        if cell_c and cell_b and cell_c.scenarios and cell_b.scenarios:
            c_cons = [s.consistency_score for s in cell_c.scenarios]
            b_cons = [s.consistency_score for s in cell_b.scenarios]

            # Pad to equal length
            min_len = min(len(c_cons), len(b_cons))
            c_cons = c_cons[:min_len]
            b_cons = b_cons[:min_len]

            test = non_inferiority_test(c_cons, b_cons, margin=0.5,
                                        name="H3: 8K+State vs 32K+NoState")
            result.h3_supported = test.significant
            result.non_inferiority_p = test.p_value

            # Interaction effect
            cell_a = result.cells.get("state_False_small_8k")
            cell_d = result.cells.get("state_True_large_32k")

            if cell_a and cell_d and cell_a.scenarios and cell_d.scenarios:
                # Interaction = (D-A) - ((B-A) + (C-A))
                # Simplified: (D-C) - (B-A)
                a_avg = cell_a.avg_consistency
                state_effect_small = (cell_c.avg_consistency - a_avg) if a_avg else 0
                state_effect_large = (cell_d.avg_consistency - cell_b.avg_consistency) if cell_b.avg_consistency else 0
                result.interaction_effect = state_effect_large - state_effect_small

        return result

    def _run_scenario_with_config(self, scenario: MultiTurnScenario,
                                  with_state: bool, token_budget: int) -> ScenarioResult:
        """Run one scenario under given state and context config."""
        from ard.eval.multi_turn import MultiTurnExperimentRunner
        runner = MultiTurnExperimentRunner(
            self.hybrid, self.mmu, self.executor,
            self.state_store, self.txn_mgr, self.trace_store,
        )
        return runner._run_scenario(scenario, with_state=with_state)


def print_2x2_report(result: TwoByTwoResult) -> str:
    """Generate formatted 2x2 experiment report."""
    lines = ["\n" + "=" * 70,
             "2x2 STATE x CONTEXT EXPERIMENT (E3 — H3)",
             "=" * 70]

    # Matrix
    lines.append("\nCONSISTENCY SCORE MATRIX:")
    lines.append(f"{'':25s} | {'Small Context (8K)':20s} | {'Large Context (32K)':20s}")
    lines.append("-" * 70)

    for state_label, state_key in [("No State", "_False_"), ("With State", "_True_")]:
        small_key = f"state{state_key}small_8k"
        large_key = f"state{state_key}large_32k"
        small_val = result.cells.get(small_key)
        large_val = result.cells.get(large_key)

        s_str = f"{small_val.avg_consistency:.3f}" if small_val else "N/A"
        l_str = f"{large_val.avg_consistency:.3f}" if large_val else "N/A"
        lines.append(f"{state_label:25s} | {s_str:20s} | {l_str:20s}")

    # H3 test
    lines.append(f"\nH3 NON-INFERIORITY TEST: 8K+State vs 32K+NoState")
    lines.append(f"  Non-inferiority p-value: {result.non_inferiority_p:.4f}")
    lines.append(f"  Interaction effect: {result.interaction_effect:+.3f}")

    if result.h3_supported:
        lines.append("  *** H3 SUPPORTED: 8K+State is non-inferior to 32K+NoState ***")
        lines.append("  State management can compensate for 4x context reduction.")
    else:
        lines.append("  --- H3 NOT SUPPORTED: cannot claim non-inferiority ---")

    # Cell details
    for label, cell in result.cells.items():
        lines.append(f"\n{cell.label()}: consistency={cell.avg_consistency:.3f}, "
                    f"tokens={cell.avg_tokens:.0f}, completion={cell.avg_completion:.2f}")

    return "\n".join(lines)


def power_analysis_2x2(min_effect: float = 0.5) -> dict:
    """Power analysis for 2x2 factorial design."""
    # For interaction effect detection in 2x2, we need ~4x the sample size of main effects
    n_main = required_sample_size(min_effect, target_power=0.8)
    n_interaction = required_sample_size(min_effect / 2, target_power=0.8)

    return {
        "design": "2x2 factorial (State x Context)",
        "min_detectable_effect": min_effect,
        "required_n_per_cell_main_effect": n_main,
        "required_n_per_cell_interaction": n_interaction,
        "recommended_scenarios": max(3, min(10, n_main)),
    }
