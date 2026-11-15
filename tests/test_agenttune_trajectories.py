"""Tests for AgentTune trajectory collection, validation, and reward functions."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.agenttune.agenttune_rewards import (
    AGENTTUNE_REWARD_FUNCS,
    reward_tool_selection,
    reward_argument_accuracy,
    reward_evidence_grounding,
    reward_task_completion,
    reward_classification_accuracy,
    reward_escalation_accuracy,
    reward_no_duplicate_calls,
    reward_no_unsupported_conclusions,
)
from evaluation.agenttune.trajectory_validator import validate_trajectory
from evaluation.agenttune.trajectory_collector import (
    InvestigationTrajectory,
    TrajectoryTier,
    ToolCallRecord,
    EvidenceRecord,
    CorrelationRecord,
)


def _make_trajectory(**overrides):
    """Build a valid InvestigationTrajectory with overrides."""
    base = dict(
        trajectory_id="traj_test",
        tier=TrajectoryTier.FULL,
        scenario_id="scenario_001",
        investigation_input="Check https://evil.com",
        extracted_indicators=[{"type": "url", "value": "https://evil.com", "confidence": 0.9}],
        planned_tool_sequence=["url_analyzer", "threat_intel_lookup"],
        tool_calls=[
            ToolCallRecord(
                step=0,
                tool_name="url_analyzer",
                arguments={"url": "https://evil.com"},
                tool_call_id="tc_1",
                response={"combined_verdict": "malicious", "url": "https://evil.com"},
                success=True,
                duration_ms=50,
                timestamp="2026-01-01T00:00:00",
            ),
            ToolCallRecord(
                step=1,
                tool_name="threat_intel_lookup",
                arguments={"indicator": "evil.com", "indicator_type": "domain"},
                tool_call_id="tc_2",
                response={"overall_verdict": "malicious", "results": []},
                success=True,
                duration_ms=100,
                timestamp="2026-01-01T00:00:01",
            ),
        ],
        evidence_collected=[
            EvidenceRecord(
                indicator_type="url",
                indicator_value="https://evil.com",
                source_tool="url_analyzer",
                veracity="confirmed",
                confidence=0.9,
            ),
            EvidenceRecord(
                indicator_type="domain",
                indicator_value="evil.com",
                source_tool="threat_intel_lookup",
                veracity="confirmed",
                confidence=0.9,
            ),
        ],
        correlations=[],
        risk_assessment={"risk_level": "High", "confidence": 0.9, "summary": "Malicious"},
        escalation_decision=True,
        completion_status="completed",
        total_steps=2,
        total_duration_ms=150,
        ground_truth={
            "expected_risk": "High",
            "escalation_decision": True,
            "expected_tools": ["url_analyzer", "threat_intel_lookup"],
        },
        metadata={},
    )
    base.update(overrides)
    return InvestigationTrajectory(**base)


class TestTrajectoryValidation(unittest.TestCase):
    """Test trajectory validation."""

    def test_valid_trajectory_passes(self):
        """A well-formed trajectory passes validation."""
        traj = _make_trajectory()
        is_valid, issues = validate_trajectory(traj)
        self.assertTrue(is_valid, f"Expected valid, got issues: {issues}")

    def test_missing_trajectory_id_fails(self):
        """Missing trajectory_id fails validation."""
        traj = _make_trajectory(trajectory_id="")
        is_valid, issues = validate_trajectory(traj)
        self.assertFalse(is_valid)
        self.assertTrue(any("trajectory_id" in i.get("message", "") for i in issues))

    def test_invalid_completion_status_fails(self):
        """Invalid completion_status fails validation."""
        traj = _make_trajectory(completion_status="weird_status")
        is_valid, issues = validate_trajectory(traj)
        self.assertFalse(is_valid)


class TestRewardFunctions(unittest.TestCase):
    """Test AgentTune reward functions."""

    def test_reward_tool_selection(self):
        """Tool selection reward returns correct score."""
        completions = [
            {
                "tool_calls": [
                    {"tool": "url_analyzer", "arguments": {"url": "https://evil.com"}},
                    {"tool": "threat_intel_lookup", "arguments": {"indicator": "evil.com"}},
                ]
            }
        ]
        scores = reward_tool_selection(
            completions, expected_tools=[["url_analyzer", "threat_intel_lookup"]]
        )
        self.assertEqual(scores, [1.0])

    def test_reward_tool_selection_partial(self):
        """Partial tool selection gets partial credit."""
        completions = [
            {
                "tool_calls": [
                    {"tool": "url_analyzer", "arguments": {"url": "https://evil.com"}},
                ]
            }
        ]
        scores = reward_tool_selection(
            completions, expected_tools=[["url_analyzer", "threat_intel_lookup"]]
        )
        self.assertEqual(scores, [0.5])

    def test_reward_argument_accuracy(self):
        """Argument accuracy reward checks argument correctness."""
        completions = [
            {
                "tool_calls": [
                    {
                        "tool": "url_analyzer",
                        "arguments": {"url": "https://evil.com"},
                    }
                ]
            }
        ]
        scores = reward_argument_accuracy(
            completions,
            expected_calls=[[{"tool": "url_analyzer", "arguments": {"url": "https://evil.com"}}]],
        )
        self.assertEqual(scores, [1.0])

    def test_reward_evidence_grounding(self):
        """Evidence grounding reward requires confirmed evidence."""
        completions = [
            {
                "evidence": [
                    {"veracity": "confirmed"},
                    {"veracity": "confirmed"},
                    {"veracity": "suggested"},
                ],
                "conclusion": {"risk_level": "High"},
            }
        ]
        scores = reward_evidence_grounding(completions)
        self.assertEqual(scores, [2.0 / 3.0])

    def test_reward_evidence_grounding_high_risk_no_confirmed(self):
        """High risk without confirmed evidence gets 0."""
        completions = [
            {
                "evidence": [{"veracity": "suggested"}],
                "conclusion": {"risk_level": "High"},
            }
        ]
        scores = reward_evidence_grounding(completions)
        self.assertEqual(scores, [0.0])

    def test_reward_task_completion(self):
        """Task completion reward for completed/escalated."""
        completions = [{"completion_status": "completed"}]
        scores = reward_task_completion(completions)
        self.assertEqual(scores, [1.0])

        completions = [{"completion_status": "max_steps"}]
        scores = reward_task_completion(completions)
        self.assertEqual(scores, [0.5])

        completions = [{"completion_status": "failed"}]
        scores = reward_task_completion(completions)
        self.assertEqual(scores, [0.0])

    def test_reward_classification_accuracy(self):
        """Classification accuracy reward."""
        completions = [{"conclusion": {"risk_level": "High"}}]
        scores = reward_classification_accuracy(completions, expected_risk=["High"])
        self.assertEqual(scores, [1.0])

        completions = [{"conclusion": {"risk_level": "Medium"}}]
        scores = reward_classification_accuracy(completions, expected_risk=["High"])
        self.assertEqual(scores, [0.75])  # adjacent

    def test_reward_escalation_accuracy(self):
        """Escalation accuracy reward."""
        completions = [{"conclusion": {"escalate": True}}]
        scores = reward_escalation_accuracy(completions, escalation_decision=[True])
        self.assertEqual(scores, [1.0])

        completions = [{"conclusion": {"escalate": False}}]
        scores = reward_escalation_accuracy(completions, escalation_decision=[True])
        self.assertEqual(scores, [0.0])

    def test_reward_no_duplicate_calls(self):
        """No duplicate calls reward."""
        completions = [
            {
                "tool_calls": [
                    {"tool": "url_analyzer", "arguments": {"url": "https://evil.com"}},
                    {"tool": "url_analyzer", "arguments": {"url": "https://evil.com"}},
                ]
            }
        ]
        scores = reward_no_duplicate_calls(completions)
        self.assertEqual(scores, [0.0])

        completions = [
            {
                "tool_calls": [
                    {"tool": "url_analyzer", "arguments": {"url": "https://evil.com"}},
                    {"tool": "ip_analyzer", "arguments": {"indicator": "1.2.3.4"}},
                ]
            }
        ]
        scores = reward_no_duplicate_calls(completions)
        self.assertEqual(scores, [1.0])

    def test_reward_no_unsupported_conclusions(self):
        """No unsupported conclusions reward."""
        completions = [
            {
                "evidence": [{"veracity": "suggested"}],
                "conclusion": {"risk_level": "High"},
            }
        ]
        scores = reward_no_unsupported_conclusions(completions)
        self.assertEqual(scores, [0.0])

        completions = [
            {
                "evidence": [{"veracity": "confirmed"}],
                "conclusion": {"risk_level": "High"},
            }
        ]
        scores = reward_no_unsupported_conclusions(completions)
        self.assertEqual(scores, [1.0])

    def test_all_reward_functions_registered(self):
        """All expected reward functions are registered."""
        expected = {
            "tool_selection",
            "argument_accuracy",
            "evidence_grounding",
            "task_completion",
            "classification_accuracy",
            "escalation_accuracy",
            "no_duplicate_calls",
            "no_unsupported_conclusions",
        }
        self.assertEqual(set(AGENTTUNE_REWARD_FUNCS.keys()), expected)


if __name__ == "__main__":
    unittest.main()
