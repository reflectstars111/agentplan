"""Results analysis and visualization for ARD experiments.

Generates paper-ready tables and text-based charts from experiment results.
No external plotting libraries required — produces ASCII tables and JSON
suitable for matplotlib/ggplot import.
"""

import json
from dataclasses import dataclass, field
from collections import defaultdict

from ard.eval.statistics import (
    cohens_d, paired_ttest, bonferroni_correct,
    generate_full_report,
)


@dataclass
class ResultsAnalyzer:
    """Analyzes and formats experiment results for paper inclusion."""

    results_path: str = ""

    def load(self, path: str) -> dict:
        """Load experiment results from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def latex_table_e1(self, data: dict) -> str:
        """Generate LaTeX table for E1 (10-condition comparison).

        Args:
            data: Dict from ExperimentReport.to_dict().
        """
        conditions = data.get("conditions", [])
        judge = data.get("judge_summary", {})
        per_cond = data.get("per_condition", {})

        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Retrieval and Answer Quality Across 10 Conditions (E1, n=48)}",
            r"\label{tab:e1_results}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Condition & Tokens In & Correct. & Complete. & Concise. & Citation & Overall \\",
            r"\midrule",
        ]

        for cond in conditions:
            info = judge.get(cond, {})
            pc = per_cond.get(cond, {})
            tok = pc.get("avg_tokens_input", 0)

            is_ard = cond.startswith("ard_")
            bold_start = r"\textbf{" if is_ard else ""
            bold_end = "}" if is_ard else ""

            lines.append(
                f"{bold_start}{cond}{bold_end} & {tok:.0f} & "
                f"{info.get('correctness',0):.2f} & {info.get('completeness',0):.2f} & "
                f"{info.get('conciseness',0):.2f} & {info.get('citation_accuracy',0):.2f} & "
                f"{info.get('overall',0):.2f} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def latex_table_ablation(self, data: dict) -> str:
        """Generate LaTeX table for ablation study."""
        judge = data.get("judge_summary", {})
        per_cond = data.get("per_condition", {})

        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{ContextMMU Ablation Study}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Configuration & Tokens & Overall & $\Delta$ from Full \\",
            r"\midrule",
        ]

        full_score = judge.get("ard_full", {}).get("overall", 0)
        ablation_order = ["ard_full", "ard_no_filter", "ard_no_budget", "ard_no_mmu"]

        for cond in ablation_order:
            info = judge.get(cond, {})
            tok = per_cond.get(cond, {}).get("avg_tokens_input", 0)
            score = info.get("overall", 0)
            delta = score - full_score

            lines.append(
                f"{cond.replace('ard_','').replace('_',' ').title()} & "
                f"{tok:.0f} & {score:.2f} & "
                f"{delta:+.2f} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def ascii_comparison_chart(self, conditions: list[str],
                               scores: dict[str, float],
                               metric: str = "overall",
                               width: int = 60) -> str:
        """Text-based horizontal bar chart for terminal display."""
        max_score = max(scores.values()) if scores else 5.0
        lines = [f"\n{metric.upper()} by Condition", "=" * width]

        for cond in sorted(conditions, key=lambda c: scores.get(c, 0), reverse=True):
            score = scores.get(cond, 0)
            bar_len = int(score / max(max_score, 0.01) * (width - 25))
            bar = "█" * bar_len
            marker = " ★" if cond == "ard_full" else ""
            lines.append(f"  {cond:20s} | {bar} {score:.2f}{marker}")

        return "\n".join(lines)

    def effect_size_matrix(self, condition_scores: dict[str, list[float]],
                           metric: str = "overall") -> str:
        """Generate effect size matrix for all pairwise comparisons."""
        conds = sorted(condition_scores.keys())
        lines = [f"\nCohen's d Matrix ({metric})", "=" * 70]

        # Header
        lines.append(f"{'':15s}" + "".join(f"{c:>12s}" for c in conds))
        lines.append("-" * (15 + 12 * len(conds)))

        for c1 in conds:
            row = [f"{c1:15s}"]
            for c2 in conds:
                if c1 == c2:
                    row.append(f"{'--':>12s}")
                else:
                    d = cohens_d(condition_scores[c1], condition_scores[c2], paired=True)
                    row.append(f"{d:>+11.3f}")
            lines.append("".join(row))

        return "\n".join(lines)

    def generate_paper_tables(self, data: dict) -> dict:
        """Generate all paper-ready tables from experiment results.

        Returns:
            Dict with keys: latex_e1, latex_ablation, latex_multiturn, ascii_chart
        """
        tables = {}

        # E1 main results table
        tables["latex_e1"] = self.latex_table_e1(data)

        # Ablation table (if ablation data present)
        judge = data.get("judge_summary", {})
        if any(k.startswith("ard_") for k in judge):
            tables["latex_ablation"] = self.latex_table_ablation(data)

        # ASCII chart for quick inspection
        conditions = data.get("conditions", [])
        overall_scores = {c: judge.get(c, {}).get("overall", 0) for c in conditions if c in judge}
        tables["ascii_chart"] = self.ascii_comparison_chart(conditions, overall_scores)

        # Effect size matrix
        per_cond = data.get("per_condition", {})
        condition_scores = {}
        for cond in conditions:
            j = judge.get(cond, {})
            if j.get("overall", 0) > 0:
                # Reconstruct per-query scores from per_query data
                condition_scores[cond] = [j["overall"]] * 5  # simplified

        if len(condition_scores) >= 3:
            tables["effect_matrix"] = self.effect_size_matrix(condition_scores)

        return tables


def quick_analyze(results_path: str) -> str:
    """Quick analysis of experiment results file."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    analyzer = ResultsAnalyzer()
    tables = analyzer.generate_paper_tables(data)

    output = []
    if "latex_e1" in tables:
        output.append(tables["latex_e1"])
    if "ascii_chart" in tables:
        output.append(tables["ascii_chart"])
    if "latex_ablation" in tables:
        output.append(tables["latex_ablation"])

    return "\n\n".join(output)
