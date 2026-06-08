"""Statistical analysis for rigorous experimental comparison.

Provides:
- Paired t-test (per-query pairing across conditions)
- Independent t-test with pooled variance
- Bonferroni correction for multiple comparisons
- Power analysis (verify n is sufficient)
- Effect size decomposition by query category/difficulty/domain
- Non-inferiority test for H3
"""

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Effect Sizes
# ═══════════════════════════════════════════════════════════════════

def cohens_d(group1: list[float], group2: list[float], paired: bool = False) -> float:
    """Cohen's d effect size.

    Args:
        group1, group2: The two groups to compare.
        paired: If True, compute paired Cohen's d (dz = mean_diff / std_diff).

    Returns:
        Cohen's d (|d|>0.2=small, |d|>0.5=medium, |d|>0.8=large).
    """
    if len(group1) < 2 or len(group2) < 2:
        return 0.0

    if paired and len(group1) == len(group2):
        diffs = np.array(group1) - np.array(group2)
        sd = float(np.std(diffs, ddof=1))
        if sd == 0:
            return 0.0
        return float(np.mean(diffs)) / sd
    else:
        m1, m2 = np.mean(group1), np.mean(group2)
        v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        pooled_sd = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        if pooled_sd == 0:
            return 0.0
        return (m1 - m2) / pooled_sd


def hedges_g(group1: list[float], group2: list[float]) -> float:
    """Hedges' g (bias-corrected Cohen's d for small samples)."""
    d = cohens_d(group1, group2)
    n1, n2 = len(group1), len(group2)
    df = n1 + n2 - 2
    # Correction factor (Hedges & Olkin, 1985)
    correction = 1 - 3 / (4 * df - 1) if df > 1 else 1.0
    return d * correction


# ═══════════════════════════════════════════════════════════════════
# Hypothesis Tests
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    effect_size_label: str
    ci_95: tuple[float, float]
    n: int
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        sig = "***" if self.p_value < 0.001 else ("**" if self.p_value < 0.01 else ("*" if self.p_value < 0.05 else "ns"))
        return (f"{self.test_name}: {self.statistic:.3f}, p={self.p_value:.4f} {sig}, "
                f"d={self.effect_size:.3f} [{self.effect_size_label}], "
                f"95%CI=[{self.ci_95[0]:.3f}, {self.ci_95[1]:.3f}], n={self.n}")


def paired_ttest(group1: list[float], group2: list[float],
                 name: str = "paired_t") -> TestResult:
    """Paired t-test between two groups.

    Assumes group1[i] and group2[i] are paired (same query, different condition).
    """
    if len(group1) != len(group2):
        raise ValueError(f"Paired t-test requires equal group sizes: {len(group1)} vs {len(group2)}")

    n = len(group1)
    if n < 2:
        return TestResult(name, 0, 1.0, False, 0, "none", (0, 0), n)

    diffs = np.array(group1) - np.array(group2)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    se = std_diff / math.sqrt(n)

    if se == 0:
        return TestResult(name, float('inf'), 0.0, True, 0, "none", (mean_diff, mean_diff), n)

    t_stat = mean_diff / se
    df = n - 1

    # Two-tailed p-value from t-distribution
    p = _t_pvalue(abs(t_stat), df)

    d = cohens_d(group1, group2, paired=True)
    d_label = _effect_label(abs(d))

    # 95% CI for the mean difference
    t_crit = _t_critical(0.05, df)
    ci = (mean_diff - t_crit * se, mean_diff + t_crit * se)

    return TestResult(
        test_name=name, statistic=round(t_stat, 3),
        p_value=round(p, 4), significant=p < 0.05,
        effect_size=round(d, 3), effect_size_label=d_label,
        ci_95=(round(ci[0], 4), round(ci[1], 4)), n=n,
    )


def independent_ttest(group1: list[float], group2: list[float],
                      name: str = "independent_t") -> TestResult:
    """Independent (unpaired) t-test with pooled variance."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return TestResult(name, 0, 1.0, False, 0, "none", (0, 0), n1 + n2)

    m1, m2 = np.mean(group1), np.mean(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    pooled_se = math.sqrt(v1 / n1 + v2 / n2)
    if pooled_se == 0:
        return TestResult(name, 0, 1.0, False, 0, "none", (m1 - m2, m1 - m2), n1 + n2)

    t_stat = (m1 - m2) / pooled_se
    # Welch-Satterthwaite degrees of freedom
    df = ((v1 / n1 + v2 / n2) ** 2) / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    )

    p = _t_pvalue(abs(t_stat), int(df))
    d = cohens_d(group1, group2, paired=False)
    d_label = _effect_label(abs(d))

    return TestResult(
        test_name=name, statistic=round(t_stat, 3),
        p_value=round(p, 4), significant=p < 0.05,
        effect_size=round(d, 3), effect_size_label=d_label,
        ci_95=(0, 0), n=n1 + n2,
    )


def non_inferiority_test(group1: list[float], group2: list[float],
                         margin: float = 0.5, name: str = "non_inferiority") -> TestResult:
    """Test if group1 is non-inferior to group2 (for H3).

    H0: group1 is worse than group2 by at least `margin`.
    H1: group1 is NOT worse by margin (i.e., g1 >= g2 - margin).

    This is used for H3: 8K+State (group1) vs 32K+No State (group2).
    If significant, we can claim that state management allows smaller context.

    Args:
        group1: The "smaller" condition (e.g., 8K + State)
        group2: The "larger" condition (e.g., 32K + No State)
        margin: The non-inferiority margin (how much worse we allow group1 to be)

    Returns:
        TestResult. If significant, group1 is non-inferior to group2.
    """
    if len(group1) != len(group2):
        raise ValueError("Non-inferiority test requires paired groups")

    n = len(group1)
    if n < 2:
        return TestResult(name, 0, 1.0, False, 0, "none", (0, 0), n)

    diffs = np.array(group1) - np.array(group2)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    se = std_diff / math.sqrt(n)

    if se == 0:
        return TestResult(name, float('inf'), 0.0, True, 0, "none", (mean_diff, mean_diff), n)

    # H0: mean_diff <= -margin → t = (mean_diff - (-margin)) / se
    t_stat = (mean_diff + margin) / se
    df = n - 1

    # One-tailed p-value (we only care if g1 is NOT worse)
    p = _t_pvalue_one_sided(t_stat, df)

    d = cohens_d(group1, group2, paired=True)

    return TestResult(
        test_name=name, statistic=round(t_stat, 3),
        p_value=round(p, 4), significant=p < 0.05,
        effect_size=round(d, 3), effect_size_label=_effect_label(abs(d)),
        ci_95=(0, 0), n=n,
    )


# ═══════════════════════════════════════════════════════════════════
# Multiple Comparison Correction
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BonferroniResult:
    """Result of multiple comparison correction."""
    original_tests: list[TestResult]
    corrected_alpha: float
    n_comparisons: int
    significant_after_correction: list[str]

    def summary(self) -> str:
        lines = [
            f"Bonferroni correction: {self.n_comparisons} comparisons, "
            f"α_corrected = {self.corrected_alpha:.5f}",
            f"Significant after correction: {self.significant_after_correction}",
        ]
        return "\n".join(lines)


def bonferroni_correct(tests: list[TestResult],
                       family_alpha: float = 0.05) -> BonferroniResult:
    """Apply Bonferroni correction to a family of tests.

    Args:
        tests: List of test results from pairwise comparisons.
        family_alpha: Family-wise error rate (default 0.05).

    Returns:
        BonferroniResult with corrected significance.
    """
    n = max(len(tests), 1)
    corrected_alpha = family_alpha / n

    significant_after = [
        t.test_name for t in tests
        if t.p_value < corrected_alpha
    ]

    return BonferroniResult(
        original_tests=tests,
        corrected_alpha=corrected_alpha,
        n_comparisons=n,
        significant_after_correction=significant_after,
    )


# ═══════════════════════════════════════════════════════════════════
# Power Analysis
# ═══════════════════════════════════════════════════════════════════

def power_analysis_paired(n: int, effect_size: float, alpha: float = 0.05) -> float:
    """Compute statistical power for a paired t-test.

    Args:
        n: Sample size (number of pairs).
        effect_size: Expected Cohen's d (paired).
        alpha: Significance level.

    Returns:
        Statistical power (probability of detecting the effect if it exists).
    """
    from scipy import stats as scipy_stats

    if n < 2:
        return 0.0

    df = n - 1
    # Non-centrality parameter
    ncp = effect_size * math.sqrt(n)
    # Critical t-value
    t_crit = _t_critical(alpha, df)
    # Power = P(|t| > t_crit) under non-central t
    power = 1 - scipy_stats.nct.cdf(t_crit, df, ncp) + scipy_stats.nct.cdf(-t_crit, df, ncp)
    return float(power)


def required_sample_size(effect_size: float, target_power: float = 0.8,
                         alpha: float = 0.05) -> int:
    """Estimate required sample size for a paired t-test.

    Args:
        effect_size: Expected Cohen's d.
        target_power: Desired power (default 0.8).
        alpha: Significance level.

    Returns:
        Minimum sample size needed.
    """
    for n in range(5, 500):
        power = power_analysis_paired(n, effect_size, alpha)
        if power >= target_power:
            return n
    return 500


# ═══════════════════════════════════════════════════════════════════
# Effect Size Decomposition
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SubgroupEffect:
    """Effect size for a specific subgroup."""
    subgroup: str
    n: int
    cohens_d: float
    mean_group1: float
    mean_group2: float
    p_value: float

    def to_dict(self) -> dict:
        return {
            "subgroup": self.subgroup,
            "n": self.n,
            "cohens_d": round(self.cohens_d, 3),
            "mean_g1": round(self.mean_group1, 3),
            "mean_g2": round(self.mean_group2, 3),
            "p_value": round(self.p_value, 4),
        }


def decompose_by_subgroup(
    values_g1: list[float],
    values_g2: list[float],
    subgroup_labels: list[str],
) -> list[SubgroupEffect]:
    """Compute effect sizes for each subgroup.

    Args:
        values_g1: Metric values for condition 1.
        values_g2: Metric values for condition 2 (must be same length).
        subgroup_labels: Labels for each data point (e.g., category, difficulty).

    Returns:
        List of SubgroupEffect, one per unique subgroup label.
    """
    if len(values_g1) != len(values_g2) or len(values_g1) != len(subgroup_labels):
        raise ValueError("All input lists must have the same length")

    unique_groups = sorted(set(subgroup_labels))
    effects = []

    for group in unique_groups:
        indices = [i for i, label in enumerate(subgroup_labels) if label == group]
        g1_sub = [values_g1[i] for i in indices]
        g2_sub = [values_g2[i] for i in indices]

        d = cohens_d(g1_sub, g2_sub, paired=True)
        t = paired_ttest(g1_sub, g2_sub, name=group)

        effects.append(SubgroupEffect(
            subgroup=group,
            n=len(indices),
            cohens_d=d,
            mean_group1=float(np.mean(g1_sub)) if g1_sub else 0.0,
            mean_group2=float(np.mean(g2_sub)) if g2_sub else 0.0,
            p_value=t.p_value,
        ))

    return effects


# ═══════════════════════════════════════════════════════════════════
# Report Generator
# ═══════════════════════════════════════════════════════════════════

def generate_full_report(
    condition_scores: dict[str, list[float]],
    metric_name: str = "score",
    subgroup_labels: list[str] | None = None,
    paired: bool = True,
) -> str:
    """Generate a comprehensive statistical report.

    Args:
        condition_scores: Dict mapping condition name → list of metric values.
        metric_name: Name of the metric being compared.
        subgroup_labels: Optional subgroup labels for decomposition.
        paired: Whether to use paired tests.

    Returns:
        Formatted report string.
    """
    lines = [
        f"\n{'='*70}",
        f"STATISTICAL REPORT: {metric_name}",
        f"{'='*70}",
        "",
    ]

    # Descriptive statistics
    lines.append("DESCRIPTIVE STATISTICS")
    lines.append("-" * 40)
    for cond, values in condition_scores.items():
        arr = np.array(values)
        lines.append(
            f"  {cond:20s}: n={len(values):3d}, "
            f"M={np.mean(arr):.3f}, SD={np.std(arr):.3f}, "
            f"95%CI=[{np.percentile(arr, 2.5):.3f}, {np.percentile(arr, 97.5):.3f}]"
        )
    lines.append("")

    # Pairwise comparisons
    conditions = list(condition_scores.keys())
    all_tests = []

    lines.append("PAIRWISE COMPARISONS")
    lines.append("-" * 40)

    for c1, c2 in combinations(conditions, 2):
        if paired and len(condition_scores[c1]) == len(condition_scores[c2]):
            test = paired_ttest(condition_scores[c1], condition_scores[c2],
                                name=f"{c1} vs {c2}")
        else:
            test = independent_ttest(condition_scores[c1], condition_scores[c2],
                                     name=f"{c1} vs {c2}")
        all_tests.append(test)
        lines.append(f"  {test.summary()}")

    lines.append("")

    # Bonferroni correction
    bonf = bonferroni_correct(all_tests)
    lines.append("MULTIPLE COMPARISON CORRECTION")
    lines.append("-" * 40)
    lines.append(f"  {bonf.summary()}")
    lines.append("")

    # Effect size classification
    lines.append("EFFECT SIZE SUMMARY")
    lines.append("-" * 40)
    large = [t for t in all_tests if abs(t.effect_size) >= 0.8]
    medium = [t for t in all_tests if 0.5 <= abs(t.effect_size) < 0.8]
    small = [t for t in all_tests if 0.2 <= abs(t.effect_size) < 0.5]
    negligible = [t for t in all_tests if abs(t.effect_size) < 0.2]

    lines.append(f"  Large (|d|>=0.8):     {len(large)} tests")
    for t in large:
        lines.append(f"    - {t.test_name}: d={t.effect_size:.3f}")
    lines.append(f"  Medium (0.5<=|d|<0.8): {len(medium)} tests")
    for t in medium:
        lines.append(f"    - {t.test_name}: d={t.effect_size:.3f}")
    lines.append(f"  Small (0.2<=|d|<0.5):  {len(small)} tests")
    for t in small:
        lines.append(f"    - {t.test_name}: d={t.effect_size:.3f}")
    lines.append(f"  Negligible (|d|<0.2):   {len(negligible)} tests")

    lines.append("")

    # Subgroup decomposition (if requested)
    if subgroup_labels:
        lines.append("SUBGROUP EFFECT SIZE DECOMPOSITION")
        lines.append("-" * 40)
        for c1, c2 in combinations(conditions, 2):
            if paired and len(condition_scores[c1]) == len(condition_scores[c2]):
                effects = decompose_by_subgroup(
                    condition_scores[c1], condition_scores[c2], subgroup_labels
                )
                lines.append(f"  {c1} vs {c2}:")
                for eff in effects:
                    lines.append(
                        f"    {eff.subgroup:15s}: d={eff.cohens_d:+.3f}, "
                        f"n={eff.n}, p={eff.p_value:.4f}, "
                        f"g1={eff.mean_group1:.3f}, g2={eff.mean_group2:.3f}"
                    )

    lines.append(f"\n{'='*70}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Internal Helpers
# ═══════════════════════════════════════════════════════════════════

def _effect_label(d: float) -> str:
    if d >= 0.8:
        return "large"
    elif d >= 0.5:
        return "medium"
    elif d >= 0.2:
        return "small"
    else:
        return "negligible"


def _t_critical(alpha: float, df: int) -> float:
    """Two-tailed critical t-value."""
    from scipy import stats as scipy_stats
    return float(scipy_stats.t.ppf(1 - alpha / 2, df))


def _t_pvalue(abs_t: float, df: int) -> float:
    """Two-tailed p-value from t-distribution."""
    if df < 1:
        return 1.0
    from scipy import stats as scipy_stats
    return float(2 * scipy_stats.t.sf(abs_t, df))


def hedges_g(group1: list[float], group2: list[float]) -> float:
    """Hedges' g (bias-corrected Cohen's d for small samples)."""
    d = cohens_d(group1, group2)
    n1, n2 = len(group1), len(group2)
    df = n1 + n2 - 2
    correction = 1 - 3 / (4 * df - 1) if df > 1 else 1.0
    return d * correction


def full_pairwise_report(condition_scores: dict[str, list[float]],
                         metric_name: str = "score",
                         paired: bool = True) -> str:
    """Generate comprehensive pairwise comparison report.

    Each comparison reports: n, M, SD, Δ, 95% CI, Cohen's d, Hedges' g,
    raw p, Bonferroni-corrected significance.
    """
    from itertools import combinations

    conditions = sorted(condition_scores.keys())
    all_tests = []
    lines = [
        f"\n{'='*70}",
        f"FULL PAIRWISE COMPARISON REPORT: {metric_name}",
        f"{'='*70}",
        "",
        f"{'Comparison':35s} | {'n':>4s} | {'Δ':>7s} | {'95% CI':>15s} | {'d':>6s} | {'g':>6s} | {'p_raw':>7s} | {'Sig':>4s}",
        "-" * 100,
    ]

    for c1, c2 in combinations(conditions, 2):
        s1 = condition_scores[c1]
        s2 = condition_scores[c2]
        n = min(len(s1), len(s2))

        if paired and len(s1) == len(s2):
            t = paired_ttest(s1[:n], s2[:n], f"{c1} vs {c2}")
        else:
            t = independent_ttest(s1[:n], s2[:n], f"{c1} vs {c2}")

        g = hedges_g(s1[:n], s2[:n])
        all_tests.append(t)

        sig = "***" if t.p_value < 0.001 else ("**" if t.p_value < 0.01 else ("*" if t.p_value < 0.05 else ""))
        lines.append(
            f"{t.test_name:35s} | {n:4d} | {t.ci_95[1]-t.ci_95[0]:+7.3f} | "
            f"[{t.ci_95[0]:.3f},{t.ci_95[1]:.3f}] | {t.effect_size:+6.3f} | "
            f"{g:+6.3f} | {t.p_value:7.4f} | {sig:>4s}"
        )

    # Bonferroni correction
    bonf = bonferroni_correct(all_tests)
    lines.append(f"\nBonferroni correction: {bonf.n_comparisons} comparisons, α_corrected={bonf.corrected_alpha:.5f}")
    lines.append(f"Significant after correction: {len(bonf.significant_after_correction)}/{bonf.n_comparisons}")
    lines.append(f"  {bonf.significant_after_correction}")

    # Effect size summary
    large = [t for t in all_tests if abs(t.effect_size) >= 0.8]
    medium = [t for t in all_tests if 0.5 <= abs(t.effect_size) < 0.8]
    lines.append(f"\nEffect size: {len(large)} large, {len(medium)} medium")

    return "\n".join(lines)


def _t_pvalue_one_sided(t_stat: float, df: int) -> float:
    """One-tailed p-value from t-distribution."""
    if df < 1:
        return 1.0
    from scipy import stats as scipy_stats
    return float(scipy_stats.t.sf(t_stat, df))
