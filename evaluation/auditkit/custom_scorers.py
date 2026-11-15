"""Custom scorers for AuditKIT - implements ThreatLens-specific evaluation metrics."""

import json
import re
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from pathlib import Path

try:
    from auditkit import scorer, Score, ScoreKind
    from auditkit.scorers import Scorer
    AUDITKIT_AVAILABLE = True
except ImportError:
    AUDITKIT_AVAILABLE = False

# ===== ThreatLens Custom Scorers =====

def tool_selection_accuracy(
    expected: List[str],
    actual: List[str]
) -> float:
    """
    Tool Selection Accuracy: Whether the correct tools were selected.
    Score: intersection / expected (if expected > 0)
    """
    if not expected:
        return 1.0

    expected_set = set(expected)
    actual_set = set(actual)

    intersection = expected_set & actual_set
    return len(intersection) / len(expected) if expected else 1.0


def argument_accuracy(
    expected_calls: List[Dict],
    actual_calls: List[Dict]
) -> float:
    """
    Argument Accuracy: Whether correct arguments were provided for tool calls.
    Score: average argument match ratio across matching tool calls.
    """
    if not expected_calls:
        return 1.0

    matches = 0
    total = 0

    for exp_call in expected_calls:
        exp_tool = exp_call.get("tool")
        exp_args = exp_call.get("arguments", {})

        # Find matching actual call
        for act_call in actual_calls:
            if act_call.get("tool") == exp_call.get("tool"):
                act_args = act_call.get("arguments", {})
                # Check if all expected args are present with correct values
                exp_arg_items = set(exp_args.items())
                act_arg_items = set(act_args.items()) if isinstance(act_call.get("arguments"), dict) else set()

                if exp_arg_items.issubset(act_arg_items):
                    matches += 1
                total = max(total, 1)
                break
        else:
            # No matching tool call found
            pass
        total += 1

    return matches / total if total > 0 else 0.0


def task_completion_rate(
    completion_status: str
) -> float:
    """
    Task Completion Rate: Whether the investigation was completed.
    1.0 for completed/escalated, 0.5 for max_steps, 0.0 for failed.
    """
    if completion_status in ["completed", "escalated"]:
        return 1.0
    elif completion_status == "max_steps":
        return 0.5
    return 0.0


def classification_performance(
    predicted: str,
    expected: str
) -> float:
    """
    Classification Performance: Whether predicted risk class matches reference.
    Supports partial credit for adjacent risk levels.
    """
    if not expected:
        return 0.5  # Neutral if no ground truth

    risk_order = ["Inconclusive", "Low", "Medium", "High", "Critical"]

    if predicted == expected:
        return 1.0

    try:
        pred_idx = ["Inconclusive", "Low", "Medium", "High", "Critical"].index(predicted)
        exp_idx = ["Inconclusive", "Low", "Medium", "High", "Critical"].index(expected)
        distance = abs(pred_idx - exp_idx)
        return max(0.0, 1.0 - distance * 0.25)
    except ValueError:
        return 0.0


def evidence_grounding(
    evidence: List[Dict],
    risk_level: str
) -> float:
    """
    Evidence Grounding: Whether conclusions are supported by evidence.
    High risk requires confirmed evidence.
    """
    if not evidence:
        return 0.0

    confirmed = sum(1 for e in evidence if e.get("veracity") == "confirmed")
    total = len(evidence)

    if total == 0:
        return 0.0

    if evidence and evidence[-1].get("risk_level") in ["High", "Critical"]:
        # High risk requires confirmed evidence
        if confirmed == 0:
            return 0.0

    return confirmed / total


def tool_call_efficiency(
    tool_calls: List[Dict],
    expected_calls: List[Dict]
) -> float:
    """
    Tool-Call Efficiency: Number of tool calls relative to expected.
    Fewer unnecessary calls is better.
    """
    if not expected_calls:
        return 1.0

    actual_count = len([c for c in tool_calls if c.get("success", False)])
    expected_min = len([c for c in expected_calls if c.get("required", True)])

    if expected_min == 0:
        return 1.0

    # Penalize excessive calls
    ratio = len(tool_calls) / max(len(expected_calls), 1)
    if ratio <= 1.0:
        return 1.0
    elif ratio <= 1.5:
        return 0.8
    elif ratio <= 2.0:
        return 0.6
    else:
        return 0.3


def escalation_accuracy(
    predicted: bool,
    expected: bool
) -> float:
    """
    Escalation Accuracy: Whether human escalation was appropriate.
    """
    return 1.0 if predicted == expected else 0.0


def tool_call_f1(
    expected_calls: List[str],
    actual_calls: List[str]
) -> float:
    """
    Tool Call F1: F1 score for tool call selection.
    """
    expected_set = set(expected_calls)
    actual_set = set(actual_calls)

    if not expected_set and not actual_set:
        return 1.0

    intersection = expected_set & actual_set
    precision = len(intersection) / len(actual_set) if actual_set else 0
    recall = len(intersection) / len(expected_set) if expected_set else 0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def evidence_grounding_rate(
    evidence: List[Dict]
) -> float:
    """
    Evidence Grounding Rate: Proportion of evidence that is confirmed.
    """
    if not evidence:
        return 0.0

    confirmed = sum(1 for e in evidence if e.get("veracity") == "confirmed")
    total = len(evidence)

    return confirmed / total if total > 0 else 0.0


def tool_call_accuracy(
    expected_calls: List[Dict],
    actual_calls: List[Dict]
) -> float:
    """
    Tool Call Accuracy: Combined metric for tool selection and arguments.
    """
    tool_acc = tool_selection_accuracy(
        [c.get("tool") for c in expected_calls],
        [c.get("tool") for c in actual_calls]
    )

    arg_acc = argument_accuracy(expected_calls, actual_calls)

    return (tool_acc + arg_acc) / 2


# ===== AuditKIT Scorer Wrappers =====

def create_scorers():
    """Create AuditKIT scorer functions."""
    scorers = {}

    if AUDITKIT_AVAILABLE:
        try:
            from auditkit import scorer
            from auditkit.scorers import Scorer
            from auditkit import Score, ScoreKind

            # Define custom scorers as AuditKIT scorers
            @scorer(name="tool_selection_accuracy", kind=ScoreKind.FLOAT)
            def ts_accuracy(*, expected: List[str], actual: List[str]) -> float:
                return tool_selection_accuracy(expected, actual)

            @scorer(name="argument_accuracy", kind=ScoreKind.FLOAT)
            def arg_accuracy(*, expected_calls: List[Dict], actual_calls: List[Dict]) -> float:
                return argument_accuracy(expected_calls, actual_calls)

            @scorer(name="task_completion_rate", kind=ScoreKind.FLOAT)
            def tc_rate(completion_status: str) -> float:
                return task_completion_rate(completion_status)

            @scorer(name="classification_performance", kind=ScoreKind.FLOAT)
            def cls_perf(predicted: str, expected: str) -> float:
                return classification_performance(predicted, expected)

            @scorer(name="evidence_grounding", kind=ScoreKind.FLOAT)
            def ev_grounding(evidence: List[Dict], risk_level: str) -> float:
                return evidence_grounding(evidence, risk_level)

            @scorer(name="tool_call_efficiency", kind=ScoreKind.FLOAT)
            def tc_efficiency(tool_calls: List[Dict], expected_calls: List[Dict]) -> float:
                return tool_call_efficiency(tool_calls, expected_calls)

            @scorer(name="escalation_accuracy", kind=ScoreKind.FLOAT)
            def esc_accuracy(predicted: bool, expected: bool) -> float:
                return escalation_accuracy(predicted, expected)

            @scorer(name="tool_call_f1", kind=ScoreKind.FLOAT)
            def tc_f1(expected_calls: List[str], actual_calls: List[str]) -> float:
                return tool_call_f1(expected_calls, actual_calls)

            @scorer(name="evidence_grounding_rate", kind=ScoreKind.FLOAT)
            def ev_gr_rate(evidence: List[Dict]) -> float:
                return evidence_grounding_rate(evidence)

            @scorer(name="tool_call_accuracy", kind=ScoreKind.FLOAT)
            def tc_accuracy(expected_calls: List[Dict], actual_calls: List[Dict]) -> float:
                return tool_call_accuracy(expected_calls, actual_calls)

            # Register scorers
            scorers = {
                "tool_selection_accuracy": tool_selection_accuracy,
                "argument_accuracy": argument_accuracy,
                "task_completion_rate": task_completion_rate,
                "classification_performance": classification_performance,
                "evidence_grounding": evidence_grounding,
                "tool_call_efficiency": tool_call_efficiency,
                "escalation_accuracy": escalation_accuracy,
                "tool_call_f1": tool_call_f1,
                "evidence_grounding_rate": evidence_grounding_rate,
                "tool_call_accuracy": tool_call_accuracy,
            }

            return scorers

    # Fallback: return function references
    return {
        "tool_selection_accuracy": tool_selection_accuracy,
        "argument_accuracy": argument_accuracy,
        "task_completion_rate": task_completion_rate,
        "classification_performance": classification_performance,
        "evidence_grounding": evidence_grounding,
        "tool_call_efficiency": tool_call_efficiency,
        "escalation_accuracy": escalation_accuracy,
        "tool_call_f1": tool_call_f1,
        "evidence_grounding_rate": evidence_grounding_rate,
        "tool_call_accuracy": tool_call_accuracy,
    }


# ===== Custom Scorer Classes for AuditKIT =====

if AUDITKIT_AVAILABLE:
    try:
        from auditkit.scorers import Scorer
        from auditkit import Score, ScoreKind
        from auditkit.scorers import scorer

        class ToolSelectionAccuracyScorer(Scorer):
            """AuditKIT scorer for tool selection accuracy."""

            @property
            def name(self) -> str:
                return "tool_selection_accuracy"

            @property
            def kind(self) -> ScoreKind:
                from auditkit import ScoreKind
                return ScoreKind.FLOAT

            def score(self, *, expected: List[str], actual: List[str]) -> float:
                return tool_selection_accuracy(expected, actual)


        class ArgumentAccuracyScorer(Scorer):
            @property
            def name(self) -> str:
                return "argument_accuracy"

            @property
            def kind(self):
                from auditkit import ScoreKind
                return ScoreKind.FLOAT

            def score(self, *, expected_calls: List[Dict], actual_calls: List[Dict]) -> float:
                return argument_accuracy(expected_calls, actual_calls)


        class TaskCompletionRateScorer(Scorer):
            @property
            def name(self) -> str:
                return "task_completion_rate"

            @property
            def kind(self):
                from auditkit import ScoreKind
                return ScoreKind.FLOAT

            def score(self, *, completion_status: str) -> float:
                return task_completion_rate(completion_status)


        class ClassificationPerformanceScorer(Scorer):
            @property
            def name(self) -> str:
                return "classification_performance"

            @property
            def kind(self):
                from auditkit import ScoreKind
                return ScoreKind.FLOAT

            def score(self, *, predicted: str, expected: str) -> float:
                return classification_performance(predicted, expected)


        class EvidenceGroundingScorer(Scorer):
            @property
            def name(self) -> str:
                return "evidence_grounding"

            @property
            def kind(self):
                from auditkit import ScoreKind
                return ScoreKind.FLOAT

            def score(self, *, evidence: List[Dict], risk_level: str) -> float:
                return evidence_grounding(evidence, risk_level)


        class ToolCallEfficiencyScorer(Scorer):
            @property
            def name(self) -> str:
                return "tool_call_efficiency"

            @property
            def kind(self):
                from auditkit import ScoreKind
                return ScoreKind.FLOAT

            def score(self, *, tool_calls: List[Dict], expected_calls: List[Dict]) -> float:
                return tool_call_efficiency(tool_calls, expected_calls)


        class EscalationAccuracyScorer(Scorer):
            @property
            def name(self) -> str:
                return "escalation_accuracy"

            @property
            def kind(self):
                from auditkit import ScoreKind
                return ScoreKind.FLOAT

            def score(self, *, predicted: bool, expected: bool) -> float:
                return escalation_accuracy(predicted, expected)


        class ToolCallF1Scorer(Scorer):
            @property
            def name(self) -> str:
                return "tool_call_f1"

            @property
            def kind(self):
                from auditkit import ScoreKind
                return ScoreKind.FLOAT

            def score(self, *, expected_calls: List[str], actual_calls: List[str]) -> float:
                return tool_call_f1(expected_calls, actual_calls)


        class EvidenceGroundingRateScorer(Scorer):
            @property
            def name(self) -> str:
                return "evidence_grounding_rate"

            @property
            def kind(self):
                from auditkit import ScoreKind
                return ScoreKind.FLOAT

            def score(self, *, evidence: List[Dict]) -> float:
                return evidence_grounding_rate(evidence)


        class ToolCallAccuracyScorer(Scorer):
            @property
            def name(self) -> str:
                return "tool_call_accuracy"

            @property
            def kind(self):
                from auditkit import ScoreKind
                return ScoreKind.FLOAT

            def score(self, *, expected_calls: List[Dict], actual_calls: List[Dict]) -> float:
                return tool_call_accuracy(expected_calls, actual_calls)


        # Registry of all custom scorers
        CUSTOM_SCORERS = {
            "tool_selection_accuracy": ToolSelectionAccuracyScorer(),
            "argument_accuracy": ArgumentAccuracyScorer(),
            "task_completion_rate": TaskCompletionRateScorer(),
            "classification_performance": ClassificationPerformanceScorer(),
            "evidence_grounding": EvidenceGroundingScorer(),
            "tool_call_efficiency": ToolCallEfficiencyScorer(),
            "escalation_accuracy": EscalationAccuracyScorer(),
            "tool_call_f1": ToolCallF1Scorer(),
            "evidence_grounding_rate": EvidenceGroundingRateScorer(),
            "tool_call_accuracy": ToolCallAccuracyScorer(),
        }

        def get_scorer(name: str):
            """Get scorer by name."""
            return CUSTOM_SCORERS.get(name)

        def get_all_scorers():
            return list(CUSTOM_SCORERS.values())

        def get_scorer_names():
            return list(CUSTOM_SCORERS.keys())

    else:
        # Fallback
        CUSTOM_SCORERS = {}
        def get_scorer(name: str):
            return None
        def get_all_scorers():
            return []
        def get_scorer_names():
            return []


def get_all_scorers():
    """Get all custom scorers."""
    if AUDITKIT_AVAILABLE:
        try:
            return list(CUSTOM_SCORERS.values())
        except:
            pass
    return []


def get_scorer(name: str):
    """Get scorer by name."""
    if AUDITKIT_AVAILABLE:
        try:
            return CUSTOM_SCORERS.get(name)
        except:
            pass
    return None


def get_scorer_names() -> List[str]:
    """Get all scorer names."""
    return [
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


def register_scorers():
    """Register all custom scorers with AuditKIT."""
    if AUDITKIT_AVAILABLE:
        try:
            from auditkit import registry
            for name, scorer in CUSTOM_SCORERS.items():
                registry.register(scorer)
            return True
        except Exception as e:
            print(f"Failed to register scorers: {e}")
            return False
    return False


if __name__ == "__main__":
    print("Custom Scorers module ready")
    print("Available scorers:", get_scorer_names())

    # Test basic functions
    print("\nTesting basic functions:")
    print(f"tool_selection_accuracy(['a','b'], ['a','c']): {tool_selection_accuracy(['a','b'], ['a','c'])}")
    print(f"classification_performance('High', 'High'): {classification_performance('High', 'High')}")
    print(f"classification_performance('High', 'Medium'): {classification_performance('High', 'Medium')}")
    print(f"escalation_accuracy(True, True): {escalation_accuracy(True, True)}")