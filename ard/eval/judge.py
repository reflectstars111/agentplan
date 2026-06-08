"""LLM-as-Judge: 5-dimension answer quality evaluation.

Evaluates LLM responses across:
  - correctness:        factual accuracy vs ground truth
  - completeness:       did it answer all parts of the question?
  - conciseness:        no unnecessary verbosity, efficient use of tokens
  - citation_accuracy:  are source references actually relevant?
  - hallucination:      absence of fabricated facts (5 = no hallucination)
  - overall:            holistic quality score

Design: Uses a second LLM (GPT-4 recommended) to judge responses.
Output: Structured JSON scores 0-5 with explanations.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ard.infra.logging import log


@dataclass
class JudgeScores:
    """5-dimension quality scores for a single answer."""
    query_id: str
    condition: str
    correctness: float = 0.0      # 0-5
    completeness: float = 0.0     # 0-5
    conciseness: float = 0.0      # 0-5
    citation_accuracy: float = 0.0  # 0-5
    groundedness: float = 0.0     # 0-5 (5 = fully grounded, no hallucination)
    overall: float = 0.0          # 0-5
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "condition": self.condition,
            "correctness": self.correctness,
            "completeness": self.completeness,
            "conciseness": self.conciseness,
            "citation_accuracy": self.citation_accuracy,
            "groundedness": self.groundedness,
            "overall": self.overall,
            "explanation": self.explanation,
        }

    def mean_score(self) -> float:
        return (self.correctness + self.completeness + self.conciseness
                + self.citation_accuracy + self.groundedness) / 5.0


JUDGE_PROMPT = """You are an expert evaluator for AI-generated answers. Judge the following answer across 5 dimensions on a scale of 0-5.

## Question
{query}

## Ground Truth (reference answer)
{ground_truth}

## Retrieved Context Sources
{context_summary}

## Answer to Evaluate
{answer}

## Evaluation Criteria

1. **correctness** (0-5): How factually accurate is the answer compared to the ground truth?
   - 5: All facts correct, matches ground truth
   - 3: Some correct facts, some errors or omissions
   - 0: Completely wrong or contradictory

2. **completeness** (0-5): Does the answer address ALL parts of the question?
   - 5: Fully addresses all aspects
   - 3: Addresses main points but misses some details
   - 0: Does not answer the question

3. **conciseness** (0-5): Is the answer efficient with tokens, avoiding unnecessary verbosity?
   - 5: Perfectly concise, every sentence adds value
   - 3: Some redundancy or unnecessary detail
   - 0: Extremely verbose with little useful content

4. **citation_accuracy** (0-5): If the answer cites sources, are the citations relevant and correct?
   - 5: All cited sources are relevant and correctly interpreted
   - 3: Some citations are relevant, others are not or incorrectly interpreted
   - 0: Citations are irrelevant or fabricated
   - (If no citations given, score 3 as neutral)

5. **groundedness** (0-5): Degree to which the answer is grounded in provided context (absence of hallucination).
   - 5: All claims are directly supported by context; no fabrication
   - 3: Minor unsupported claims or extrapolations
   - 0: Major fabricated content not in context

6. **overall** (0-5): Holistic quality considering all above dimensions.

## Output Format
Return ONLY a JSON object with these exact keys (no other text):
```json
{{
  "correctness": <0-5>,
  "completeness": <0-5>,
  "conciseness": <0-5>,
  "citation_accuracy": <0-5>,
  "groundedness": <0-5>,
  "overall": <0-5>,
  "explanation": "<brief reasoning for scores>"
}}
```"""


class LLMJudge:
    """Evaluates answer quality using an LLM as judge.

    The judge LLM should be powerful and unbiased — GPT-4 or Claude recommended.
    Uses a different LLM than the one being evaluated to avoid self-bias.
    """

    def __init__(self, llm_fn: Callable | None = None, model_name: str = "gpt-4o"):
        """Args:
            llm_fn: callable(prompt: str) -> str. If None, uses mock.
            model_name: Name of the judge model (for logging).
        """
        self.llm_fn = llm_fn or self._mock_judge
        self.model_name = model_name

    def evaluate(
        self,
        query: str,
        answer: str,
        ground_truth: str = "",
        context_sources: list[str] | None = None,
        condition: str = "unknown",
    ) -> JudgeScores:
        """Evaluate a single answer.

        Args:
            query: The original question.
            answer: The LLM's answer to evaluate.
            ground_truth: Reference answer (optional, for comparison).
            context_sources: List of source references provided to the LLM.
            condition: Experiment condition label.

        Returns:
            JudgeScores with 0-5 ratings on 5 dimensions.
        """
        context_summary = "\n".join(context_sources[:10]) if context_sources else "No context provided"

        prompt = JUDGE_PROMPT.format(
            query=query,
            ground_truth=ground_truth or "No ground truth available",
            context_summary=context_summary,
            answer=answer,
        )

        try:
            raw = self.llm_fn(prompt)
            scores = self._parse_json(raw)
        except Exception as e:
            log.warn("judge_error", query_id=query[:50], error=str(e))
            scores = self._fallback_score(query, answer, ground_truth)

        return JudgeScores(
            query_id=query[:80],
            condition=condition,
            correctness=scores.get("correctness", 0),
            completeness=scores.get("completeness", 0),
            conciseness=scores.get("conciseness", 0),
            citation_accuracy=scores.get("citation_accuracy", 0),
            groundedness=scores.get("groundedness", 0),
            overall=scores.get("overall", 0),
            explanation=scores.get("explanation", ""),
        )

    def evaluate_batch(
        self,
        queries: list[str],
        answers: list[str],
        ground_truths: list[str] | None = None,
        conditions: list[str] | None = None,
    ) -> list[JudgeScores]:
        """Evaluate a batch of answers.

        Returns list of JudgeScores, one per answer.
        """
        if ground_truths is None:
            ground_truths = [""] * len(queries)
        if conditions is None:
            conditions = ["unknown"] * len(queries)

        results = []
        for i, (q, a, gt, cond) in enumerate(zip(queries, answers, ground_truths, conditions)):
            log.debug("judge_progress", idx=i, total=len(queries))
            scores = self.evaluate(query=q, answer=a, ground_truth=gt, condition=cond)
            results.append(scores)
        return results

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Extract JSON from LLM output (may have markdown code fences)."""
        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try to extract from ```json ... ``` block
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any {...} in the text
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {}

    @staticmethod
    def _fallback_score(query: str, answer: str, ground_truth: str) -> dict:
        """Heuristic fallback when LLM judge fails."""
        # Simple keyword overlap as fallback
        gt_words = set(re.findall(r'\w+', ground_truth.lower())) if ground_truth else set()
        ans_words = set(re.findall(r'\w+', answer.lower()))

        if gt_words:
            overlap = len(gt_words & ans_words) / max(len(gt_words), 1)
            score = min(5, round(overlap * 6))
        else:
            score = 3  # neutral

        return {
            "correctness": score,
            "completeness": score - 1 if score > 1 else score,
            "conciseness": min(5, round(len(answer) / 200)) if len(answer) < 800 else 3,
            "citation_accuracy": 3,
            "groundedness": 4,
            "overall": score,
            "explanation": "Fallback heuristic score (LLM judge unavailable)",
        }

    @staticmethod
    def _mock_judge(prompt: str) -> str:
        """Mock judge for testing — returns plausible scores."""
        # Estimate quality from answer length and context presence
        has_context = "No context provided" not in prompt
        return json.dumps({
            "correctness": 4.0 if has_context else 2.5,
            "completeness": 3.5,
            "conciseness": 3.0,
            "citation_accuracy": 3.5 if has_context else 2.0,
            "groundedness": 4.0 if has_context else 3.0,
            "overall": 3.5 if has_context else 2.5,
            "explanation": "Mock evaluation for testing",
        })


def compute_inter_annotator_agreement(
    scores_a: list[JudgeScores],
    scores_b: list[JudgeScores],
) -> dict:
    """Compute inter-annotator agreement metrics between two judges.

    Returns:
        Dict with: overall_kappa, per_dimension_kappa, correlation, agreement_rate
    """
    import numpy as np

    dims = ["correctness", "completeness", "conciseness", "citation_accuracy", "hallucination", "overall"]
    result = {"per_dimension": {}}

    for dim in dims:
        a_vals = np.array([getattr(s, dim) for s in scores_a])
        b_vals = np.array([getattr(s, dim) for s in scores_b])

        # Pearson correlation
        corr = float(np.corrcoef(a_vals, b_vals)[0, 1]) if len(a_vals) > 1 else 1.0

        # Mean absolute difference
        mad = float(np.mean(np.abs(a_vals - b_vals)))

        # Agreement within 1 point
        agree_1pt = float(np.mean(np.abs(a_vals - b_vals) <= 1.0))

        result["per_dimension"][dim] = {
            "pearson_r": round(corr, 4),
            "mean_abs_diff": round(mad, 4),
            "agreement_1pt": round(agree_1pt, 4),
        }

    # Overall metrics
    all_a = np.array([s.mean_score() for s in scores_a])
    all_b = np.array([s.mean_score() for s in scores_b])
    result["overall_pearson_r"] = float(np.corrcoef(all_a, all_b)[0, 1]) if len(all_a) > 1 else 1.0
    result["overall_mean_abs_diff"] = float(np.mean(np.abs(all_a - all_b)))

    return result
