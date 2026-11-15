"""Reward function for AgentTune training - transparent, auditable reward components."""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path
import json


class RewardComponent(str, Enum):
    """Individual reward components."""
    TOOL_SELECTION = "tool_selection"
    ARGUMENT_ACCURACY = "argument_accuracy"
    TOOL_ORDERING = "tool_ordering"
    EXECUTION_SUCCESS = "execution_success"
    EVIDENCE_GROUNDING = "evidence_grounding"
    RISK_CLASSIFICATION = "risk_classification"
    ESCALATION_DECISION = "escalation_decision"
    TASK_COMPLETION = "task_completion"
    STEP_EFFICIENCY = "step_efficiency"
    NO_DUPLICATE_CALLS = "no_duplicate_calls"
    NO_REPEATED_FAILURES = "no_repeated_failures"
    NO_UNSUPPORTED_CONCLUSIONS = "no_unsupported_conclusions"
    CORRECT_ESCALATION = "correct_escalation"


class PenaltyComponent(str, Enum):
    """Penalty components."""
    INVALID_TOOL = "invalid_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    DUPLICATE_CALL = "duplicate_call"
    REPEATED_FAILURE = "repeated_failure"
    UNSUPPORTED_CONCLUSION = "unsupported_conclusion"
    INCORRECT_ESCALATION = "incorrect_escalation"
    EXCESSIVE_STEPS = "excessive_steps"
    FALSE_POSITIVE_HIGH_RISK = "false_positive_high_risk"


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components."""
    # Positive rewards
    tool_selection: float = 0.0
    argument_accuracy: float = 0.0
    tool_ordering: float = 0.0
    execution_success: float = 0.0
    evidence_grounding: float = 0.0
    risk_classification: float = 0.0
    escalation_decision: float = 0.0
    task_completion: float = 0.0
    step_efficiency: float = 0.0
    no_duplicate_calls: float = 0.0
    no_repeated_failures: float = 0.0
    no_unsupported_conclusions: float = 0.0
    correct_escalation: float = 0.0

    # Penalties (negative values)
    invalid_tool: float = 0.0
    invalid_arguments: float = 0.0
    duplicate_call: float = 0.0
    repeated_failure: float = 0.0
    unsupported_conclusion: float = 0.0
    incorrect_escalation: float = 0.0
    excessive_steps: float = 0.0
    false_positive_high_risk: float = 0.0

    def total_reward(self) -> float:
        """Calculate total reward (positive - penalties)."""
        positive = sum([
            self.tool_selection,
            self.argument_accuracy,
            self.tool_ordering,
            self.execution_success,
            self.evidence_grounding,
            self.risk_classification,
            self.escalation_decision,
            self.task_completion,
            self.step_efficiency,
            self.no_duplicate_calls,
            self.no_repeated_failures,
            self.no_unsupported_conclusions,
            self.correct_escalation,
        ])
        penalties = sum([
            self.invalid_tool,
            self.invalid_arguments,
            self.duplicate_call,
            self.repeated_failure,
            self.unsupported_conclusion,
            self.incorrect_escalation,
            self.excessive_steps,
            self.false_positive_high_risk,
        ])
        return positive + penalties  # penalties are negative

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "positive_rewards": {
                "tool_selection": self.tool_selection,
                "argument_accuracy": self.argument_accuracy,
                "tool_ordering": self.tool_ordering,
                "execution_success": self.execution_success,
                "evidence_grounding": self.evidence_grounding,
                "risk_classification": self.risk_classification,
                "escalation_decision": self.escalation_decision,
                "task_completion": self.task_completion,
                "step_efficiency": self.step_efficiency,
                "no_duplicate_calls": self.no_duplicate_calls,
                "no_repeated_failures": self.no_repeated_failures,
                "no_unsupported_conclusions": self.no_unsupported_conclusions,
                "correct_escalation": self.correct_escalation,
            },
            "penalties": {
                "invalid_tool": self.invalid_tool,
                "invalid_arguments": self.invalid_arguments,
                "duplicate_call": self.duplicate_call,
                "repeated_failure": self.repeated_failure,
                "unsupported_conclusion": self.unsupported_conclusion,
                "incorrect_escalation": self.incorrect_escalation,
                "excessive_steps": self.excessive_steps,
                "false_positive_high_risk": self.false_positive_high_risk,
            },
            "total_reward": self.total_reward()
        }


class RewardFunction:
    """
    Transparent, auditable reward function for AgentTune training.

    Rewards correct behavior, penalizes errors.
    All components are recorded separately for auditability.
    """

    # Weight configuration (tunable)
    DEFAULT_WEIGHTS = {
        # Positive rewards
        "tool_selection": 1.0,
        "argument_accuracy": 1.0,
        "tool_ordering": 0.5,
        "execution_success": 1.0,
        "evidence_grounding": 1.5,
        "risk_classification": 2.0,
        "escalation_decision": 1.0,
        "task_completion": 1.5,
        "step_efficiency": 0.5,
        "no_duplicate_calls": 0.5,
        "no_repeated_failures": 0.5,
        "no_unsupported_conclusions": 1.0,
        "correct_escalation": 1.5,

        # Penalties (negative weights)
        "invalid_tool": -2.0,
        "invalid_arguments": -1.5,
        "duplicate_call": -1.0,
        "repeated_failure": -1.0,
        "unsupported_conclusion": -2.0,
        "incorrect_escalation": -2.0,
        "excessive_steps": -0.5,
        "false_positive_high_risk": -3.0,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def compute_reward(
        self,
        trajectory: "InvestigationTrajectory",
        ground_truth: Optional[Dict] = None,
        tool_schemas: Optional[Dict] = None
    ) -> RewardBreakdown:
        """
        Compute reward breakdown for a trajectory.

        Args:
            trajectory: The investigation trajectory
            ground_truth: Ground truth labels (from evaluation dataset)
            tool_schemas: Tool schemas for argument validation

        Returns:
            RewardBreakdown with all components
        """
        breakdown = RewardBreakdown()

        # ===== POSITIVE REWARDS =====

        # 1. Tool Selection Accuracy
        breakdown.tool_selection = self._compute_tool_selection_reward(trajectory)

        # 2. Argument Accuracy
        breakdown.argument_accuracy = self._compute_argument_accuracy(trajectory, tool_schemas)

        # 3. Tool Ordering
        breakdown.tool_ordering = self._compute_tool_ordering_reward(trajectory)

        # 4. Execution Success
        breakdown.execution_success = self._compute_execution_success(trajectory)

        # 5. Evidence Grounding
        breakdown.evidence_grounding = self._compute_evidence_grounding(trajectory)

        # 6. Risk Classification Accuracy
        breakdown.risk_classification = self._compute_risk_classification(trajectory, ground_truth)

        # 7. Escalation Decision
        breakdown.escalation_decision = self._compute_escalation_reward(trajectory, ground_truth)

        # 8. Task Completion
        breakdown.task_completion = self._compute_task_completion(trajectory)

        # 8. Step Efficiency
        breakdown.step_efficiency = self._compute_step_efficiency(trajectory)

        # 8. No Duplicate Calls
        breakdown.no_duplicate_calls = self._check_no_duplicates(trajectory)

        # 9. No Repeated Failures
        breakdown.no_repeated_failures = self._check_no_repeated_failures(trajectory)

        # 10. No Unsupported Conclusions
        breakdown.no_unsupported_conclusions = self._check_no_unsupported_conclusions(trajectory, ground_truth)

        # 11. Correct Escalation
        breakdown.correct_escalation = self._check_correct_escalation(trajectory, ground_truth)

        # ===== PENALTIES =====

        # 1. Invalid Tool
        breakdown.invalid_tool = self._check_invalid_tools(trajectory)

        # 2. Invalid Arguments
        breakdown.invalid_arguments = self._check_invalid_arguments(trajectory)

        # 2. Duplicate Calls
        breakdown.duplicate_call = self._check_duplicate_calls(trajectory)

        # 3. Repeated Failures
        breakdown.repeated_failure = self._check_repeated_failures(trajectory)

        # 4. Unsupported Conclusions
        breakdown.unsupported_conclusion = self._check_unsupported_conclusions(trajectory, ground_truth)

        # 4. Incorrect Escalation
        breakdown.incorrect_escalation = self._check_incorrect_escalation(trajectory, ground_truth)

        # 5. Excessive Steps
        breakdown.excessive_steps = self._check_excessive_steps(trajectory)

        # 5. False Positive High Risk
        breakdown.false_positive_high_risk = self._check_false_positive_high_risk(trajectory, ground_truth)

        return breakdown

    def calculate_total_reward(self, breakdown: RewardBreakdown) -> float:
        """Calculate weighted total reward."""
        total = 0.0
        for component, value in breakdown.to_dict().items():
            if isinstance(value, dict):
                continue
            weight = self.weights.get(component, 0.0)
            total += value * weight
        return total

    # ===== REWARD COMPONENT CALCULATIONS =====

    def _compute_tool_selection_reward(self, trajectory) -> float:
        """Reward for selecting the correct tools."""
        if not hasattr(trajectory, "planned_tool_sequence") or not trajectory.planned_tool_sequence:
            return 0.0

        planned = set(trajectory.planned_tool_sequence)
        actual = set(tc.tool_name for tc in trajectory.tool_calls)

        if not planned:
            return 0.0

        intersection = planned & actual
        return len(intersection) / len(planned)

    def _compute_argument_accuracy(self, trajectory, tool_schemas) -> float:
        """Reward for providing correct arguments to tools."""
        if not tool_schemas or not hasattr(tool_schemas, "properties"):
            return 0.5  # Neutral if no schemas available

        correct = 0
        total = 0

        for tc in trajectory.tool_calls:
            tool_name = tc.tool_name
            args = tc.arguments

            if tool_name in tool_schemas.get("properties", {}):
                schema = tool_schemas["properties"][tool_name]
                required = schema.get("required", [])

                # Check required args present
                required_present = all(req in tc.arguments for req in required)
                if required_present:
                    total += 1
                    correct += 1
                else:
                    total += 1

        return correct / total if total > 0 else 0.5

    def _compute_tool_ordering_reward(self, trajectory) -> float:
        """Reward for calling tools in logical order."""
        if not hasattr(trajectory, "planned_tool_sequence"):
            return 0.5

        planned = trajectory.planned_tool_sequence
        actual = [tc.tool_name for tc in trajectory.tool_calls]

        if not planned or not actual:
            return 0.5

        # Check if actual follows planned order (subsequence)
        planned_idx = 0
        for tool in actual:
            if planned_idx < len(planned) and tool == planned[planned_idx]:
                planned_idx += 1

        return planned_idx / len(planned) if planned else 0.5

    def _compute_execution_success(self, trajectory) -> float:
        """Reward for successful tool execution."""
        if not trajectory.tool_calls:
            return 0.0

        successful = sum(1 for tc in trajectory.tool_calls if tc.success)
        return sum(1 for tc in trajectory.tool_calls if tc.success) / len(trajectory.tool_calls)

    def _compute_evidence_grounding(self, trajectory) -> float:
        """Reward for conclusions grounded in evidence."""
        if not trajectory.evidence_collected:
            return 0.0

        # Check if conclusions are supported by evidence
        # Simplified: proportion of conclusions with confirmed evidence
        confirmed = sum(1 for e in trajectory.evidence_collected if e.veracity == "confirmed")
        total = len(trajectory.evidence_collected)

        if total == 0:
            return 0.0

        return confirmed / total

    def _compute_risk_classification(self, trajectory, ground_truth) -> float:
        """Reward for correct risk classification."""
        if not ground_truth or "expected_risk" not in ground_truth:
            return 0.5  # Neutral if no ground truth

        predicted = trajectory.risk_assessment.get("risk_level", "")
        expected = ground_truth.get("expected_risk", "")

        if not expected:
            return 0.5

        if predicted == expected:
            return 1.0
        else:
            # Partial credit for adjacent risk levels
            risk_order = ["Inconclusive", "Low", "Medium", "High", "Critical"]
            try:
                pred_idx = risk_order.index(predicted)
                exp_idx = risk_order.index(expected)
                distance = abs(pred_idx - exp_idx)
                return max(0.0, 1.0 - distance * 0.25)
            except ValueError:
                return 0.0

    def _compute_escalation_reward(self, trajectory, ground_truth) -> float:
        """Reward for correct escalation decision."""
        if "escalation_decision" not in ground_truth:
            return 0.5

        predicted = trajectory.escalation_decision
        expected = ground_truth.get("escalation_decision", False)

        return 1.0 if predicted == expected else 0.0

    def _compute_task_completion(self, trajectory) -> float:
        """Reward for completing the investigation."""
        status = trajectory.completion_status
        if status in ["completed", "escalated"]:
            return 1.0
        elif status == "max_steps":
            return 0.5
        elif status == "failed":
            return 0.0
        return 0.0

    def _check_no_duplicates(self, trajectory) -> float:
        """Check for duplicate tool calls."""
        seen = set()
        for tc in trajectory.tool_calls:
            key = (tc.tool_name, json.dumps(tc.arguments, sort_keys=True))
            if key in seen:
                return 0.0  # Duplicate found
            seen.add(key)
        return 1.0

    def _check_no_repeated_failures(self, trajectory) -> float:
        """Check for repeated failed calls to same tool."""
        failures = {}
        for tc in trajectory.tool_calls:
            if not tc.success:
                key = (tc.tool_name, json.dumps(tc.arguments, sort_keys=True))
                failures[key] = failures.get(key, 0) + 1

        # Penalty for any tool failing more than once with same args
        for count in failures.values():
            if count > 1:
                return 0.0
        return 1.0

    def _check_unsupported_conclusions(self, trajectory, ground_truth) -> float:
        """Check if conclusions are supported by evidence."""
        if not trajectory.risk_assessment:
            return 0.0

        risk = trajectory.risk_assessment.get("risk_level", "")
        if risk in ["High", "Critical"]:
            # High risk should have confirmed evidence
            confirmed = sum(1 for e in trajectory.evidence_collected if e.veracity == "confirmed")
            if confirmed == 0:
                return 0.0  # High risk without confirmed evidence = unsupported
        return 1.0

    def _check_correct_escalation(self, trajectory, ground_truth) -> float:
        """Check if escalation decision is correct."""
        if "escalation_decision" not in ground_truth:
            return 0.5

        predicted = trajectory.escalation_decision
        expected = ground_truth.get("escalation_decision", False)

        return 1.0 if predicted == expected else 0.0

    # ===== PENALTIES =====

    def _check_invalid_tools(self, trajectory) -> float:
        """Check for invalid tool calls."""
        invalid = 0
        for tc in trajectory.tool_calls:
            if tc.tool_name not in ["email_analyzer", "url_analyzer", "ip_analyzer",
                                     "network_analyzer", "threat_intel_lookup", "correlate", "risk_assess"]:
                return -1.0
        return 0.0

    def _check_invalid_arguments(self, trajectory) -> float:
        """Check for invalid tool arguments."""
        # Simplified: check if any tool failed due to invalid args
        for tc in trajectory.tool_calls:
            if not tc.success and tc.error and "argument" in tc.error.lower():
                return -1.0
        return 0.0

    def _check_duplicate_calls(self, trajectory) -> float:
        """Check for duplicate tool calls."""
        seen = set()
        for tc in trajectory.tool_calls:
            key = (tc.tool_name, json.dumps(tc.arguments, sort_keys=True))
            if key in seen:
                return -1.0
            seen.add(key)
        return 0.0

    def _check_repeated_failures(self, trajectory) -> float:
        """Check for repeated failures with same args."""
        failures = {}
        for tc in trajectory.tool_calls:
            if not tc.success:
                key = (tc.tool_name, json.dumps(tc.arguments, sort_keys=True))
                failures[key] = failures.get(key, 0) + 1

        for count in failures.values():
            if count > 1:
                return -1.0
        return 0.0

    def _check_unsupported_conclusions(self, trajectory, ground_truth) -> float:
        """Check for unsupported high-risk conclusions."""
        risk = trajectory.risk_assessment.get("risk_level", "")
        if risk in ["High", "Critical"]:
            confirmed = sum(1 for e in trajectory.evidence_collected if e.veracity == "confirmed")
            if confirmed == 0:
                return -2.0  # High risk without confirmed evidence
        return 0.0

    def _check_incorrect_escalation(self, trajectory, ground_truth) -> float:
        """Check for incorrect escalation."""
        if "escalation_decision" not in ground_truth:
            return 0.0

        predicted = trajectory.escalation_decision
        expected = ground_truth.get("escalation_decision", False)

        if predicted != expected:
            return -2.0
        return 0.0

    def _check_excessive_steps(self, trajectory) -> float:
        """Check for excessive investigation steps."""
        if len(trajectory.tool_calls) > 10:  # Arbitrary threshold
            return -0.5
        return 0.0

    def _check_false_positive_high_risk(self, trajectory, ground_truth) -> float:
        """Penalize false positive high-risk classifications."""
        if "expected_risk" not in ground_truth:
            return 0.0

        predicted = trajectory.risk_assessment.get("risk_level", "")
        expected = ground_truth.get("expected_risk", "")

        # False positive: predicted High/Critical but expected Low/Inconclusive
        if predicted in ["High", "Critical"] and expected in ["Low", "Inconclusive"]:
            return -3.0
        return 0.0


def compute_trajectory_reward(
    trajectory,
    ground_truth: Optional[Dict] = None,
    tool_schemas: Optional[Dict] = None,
    weights: Optional[Dict[str, float]] = None
) -> Dict:
    """Compute total reward for a trajectory."""
    reward_fn = RewardFunction()
    breakdown = reward_fn.compute_reward(trajectory, ground_truth)
    total = reward_fn.calculate_total_reward(breakdown)

    return {
        "total_reward": total,
        "breakdown": breakdown.to_dict()
    }


if __name__ == "__main__":
    print("Reward Function module ready")
    print("Classes: RewardFunction, RewardBreakdown")
    print("Weights:", list(RewardFunction.DEFAULT_WEIGHTS.keys()))