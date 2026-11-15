"""Tests for AuditKIT custom scorers."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.auditkit.custom_scorers import (
    tool_selection_accuracy,
    argument_accuracy,
    task_completion_rate,
    classification_performance,
    evidence_grounding,
    tool_call_efficiency,
    escalation_accuracy,
    tool_call_f1,
    evidence_grounding_rate,
    tool_call_accuracy,
    get_scorer_names,
    get_all_scorers,
)


class TestToolSelectionAccuracy(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(tool_selection_accuracy(["a", "b"], ["a", "b"]), 1.0)

    def test_partial_match(self):
        self.assertEqual(tool_selection_accuracy(["a", "b", "c"], ["a", "c"]), 2 / 3)

    def test_empty_expected(self):
        self.assertEqual(tool_selection_accuracy([], ["a"]), 1.0)


class TestArgumentAccuracy(unittest.TestCase):
    def test_exact_args(self):
        expected = [{"tool": "url_analyzer", "arguments": {"url": "https://evil.com"}}]
        actual = [{"tool": "url_analyzer", "arguments": {"url": "https://evil.com"}}]
        self.assertEqual(argument_accuracy(expected, actual), 1.0)

    def test_missing_args(self):
        expected = [{"tool": "url_analyzer", "arguments": {"url": "https://evil.com"}}]
        actual = [{"tool": "url_analyzer", "arguments": {}}]
        self.assertEqual(argument_accuracy(expected, actual), 0.0)

    def test_no_expected(self):
        self.assertEqual(argument_accuracy([], [{"tool": "x"}]), 1.0)


class TestTaskCompletionRate(unittest.TestCase):
    def test_completed(self):
        self.assertEqual(task_completion_rate("completed"), 1.0)

    def test_escalated(self):
        self.assertEqual(task_completion_rate("escalated"), 1.0)

    def test_max_steps(self):
        self.assertEqual(task_completion_rate("max_steps"), 0.5)

    def test_failed(self):
        self.assertEqual(task_completion_rate("failed"), 0.0)


class TestClassificationPerformance(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(classification_performance("High", "High"), 1.0)

    def test_adjacent(self):
        self.assertEqual(classification_performance("Medium", "High"), 0.75)

    def test_far(self):
        # Low (idx 1) -> Critical (idx 4): distance 3, 1.0 - 3*0.25 = 0.25
        self.assertEqual(classification_performance("Low", "Critical"), 0.25)

    def test_adjacent_partial(self):
        # Medium -> High is 1 step apart: 1.0 - 1*0.25 = 0.75
        self.assertEqual(classification_performance("Medium", "High"), 0.75)

    def test_no_expected(self):
        self.assertEqual(classification_performance("High", ""), 0.5)


class TestEvidenceGrounding(unittest.TestCase):
    def test_confirmed_ratio(self):
        evidence = [
            {"veracity": "confirmed"},
            {"veracity": "confirmed"},
            {"veracity": "suggested"},
        ]
        self.assertEqual(evidence_grounding(evidence, "Medium"), 2 / 3)

    def test_high_risk_no_confirmed(self):
        evidence = [{"veracity": "suggested"}]
        self.assertEqual(evidence_grounding(evidence, "High"), 0.0)

    def test_empty(self):
        self.assertEqual(evidence_grounding([], "Low"), 0.0)


class TestEscalationAccuracy(unittest.TestCase):
    def test_match(self):
        self.assertEqual(escalation_accuracy(True, True), 1.0)
        self.assertEqual(escalation_accuracy(False, False), 1.0)

    def test_mismatch(self):
        self.assertEqual(escalation_accuracy(True, False), 0.0)


class TestToolCallF1(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(tool_call_f1(["a", "b"], ["a", "b"]), 1.0)

    def test_partial(self):
        # expected={a,b,c}, actual={a,c}: intersection=2, precision=1.0, recall=2/3
        # f1 = 2*1.0*(2/3)/(1.0+2/3) = (4/3)/(5/3) = 0.8
        self.assertEqual(tool_call_f1(["a", "b", "c"], ["a", "c"]), 0.8)

    def test_both_empty(self):
        self.assertEqual(tool_call_f1([], []), 1.0)


class TestEvidenceGroundingRate(unittest.TestCase):
    def test_ratio(self):
        evidence = [
            {"veracity": "confirmed"},
            {"veracity": "suggested"},
        ]
        self.assertEqual(evidence_grounding_rate(evidence), 0.5)

    def test_empty(self):
        self.assertEqual(evidence_grounding_rate([]), 0.0)


class TestScorerRegistry(unittest.TestCase):
    def test_all_scorers_registered(self):
        names = get_scorer_names()
        expected = [
            "tool_selection_accuracy",
            "argument_accuracy",
            "task_completion_rate",
            "classification_performance",
            "evidence_grounding",
            "tool_call_efficiency",
            "escalation_accuracy",
            "tool_call_f1",
            "evidence_grounding_rate",
            "tool_call_accuracy",
        ]
        self.assertEqual(set(names), set(expected))

    def test_get_all_scorers_returns_callables(self):
        scorers = get_all_scorers()
        self.assertEqual(len(scorers), 10)
        for s in scorers:
            self.assertTrue(callable(s))


if __name__ == "__main__":
    unittest.main()
