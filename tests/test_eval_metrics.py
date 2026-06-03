"""Tests for evaluation metrics."""

import pytest
from eval.metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
    hit_at_k,
)


class TestPrecisionAtK:
    def test_all_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_half_relevant(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "b", "x", "y"}
        assert precision_at_k(retrieved, relevant, k=4) == 0.5

    def test_none_relevant(self):
        assert precision_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_k_limits_results(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=2) == 1.0  # first 2 are both relevant

    def test_empty_retrieved(self):
        assert precision_at_k([], {"a"}, k=5) == 0.0


class TestRecallAtK:
    def test_all_found(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_half_found(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c", "d"}
        assert recall_at_k(retrieved, relevant, k=2) == 0.5

    def test_none_found(self):
        assert recall_at_k(["x"], {"a", "b"}, k=1) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["a"], set(), k=1) == 0.0


class TestMRR:
    def test_first_position(self):
        results = [["a", "b"], ["c"]]
        relevant_sets = [{"a"}, {"c"}]
        assert mrr(results, relevant_sets) == 1.0

    def test_second_position(self):
        results = [["x", "a"], ["y", "b"]]
        relevant_sets = [{"a"}, {"b"}]
        assert mrr(results, relevant_sets) == 0.5

    def test_not_found(self):
        results = [["x", "y"]]
        relevant_sets = [{"a"}]
        assert mrr(results, relevant_sets) == 0.0

    def test_mixed(self):
        results = [["a"], ["x", "b"], ["y", "z"]]
        relevant_sets = [{"a"}, {"b"}, {"none"}]
        # Q1: rank 1 → 1.0; Q2: rank 2 → 0.5; Q3: not found → 0.0
        # MRR = (1.0 + 0.5 + 0.0) / 3 = 0.5
        assert abs(mrr(results, relevant_sets) - 0.5) < 0.01


class TestNDCG:
    def test_perfect_ranking(self):
        retrieved = ["a", "b", "c"]
        relevance = {"a": 3, "b": 2, "c": 1}
        score = ndcg_at_k(retrieved, relevance, k=3)
        assert score == 1.0

    def test_imperfect_ranking(self):
        retrieved = ["c", "a", "b"]  # lowest-relevance first
        relevance = {"a": 3, "b": 2, "c": 1}
        score = ndcg_at_k(retrieved, relevance, k=3)
        assert score < 1.0

    def test_no_relevance(self):
        retrieved = ["a", "b"]
        relevance = {}
        assert ndcg_at_k(retrieved, relevance, k=2) == 0.0


class TestHitAtK:
    def test_hit(self):
        assert hit_at_k(["a", "b"], {"b", "c"}, k=2) is True

    def test_miss(self):
        assert hit_at_k(["x", "y"], {"a"}, k=2) is False

    def test_hit_within_k(self):
        assert hit_at_k(["x", "y", "z", "a"], {"a"}, k=4) is True
