"""Tests for agent execution: step counting, MAX_STEPS, evidence termination, conclusion processing."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.investigation_agent import run_investigation, MAX_STEPS
from agents.llm_client import TinyAyaClient, TinyAyaResponse, TextBlock
from tools import url_analyzer, email_analyzer, ip_analyzer  # noqa: F401


class TestStepCountingAndMaxSteps(unittest.TestCase):
    """Test MAX_STEPS enforcement and step counting."""

    @patch("agents.investigation_agent.get_llm_client")
    def test_agent_respects_max_steps(self, mock_get_client):
        """Agent stops after MAX_STEPS even without conclusion."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        with patch("agents.llm_client.HF_TOKEN", "test-token"):
            real_client = TinyAyaClient()
        mock_client.extract_json_action = real_client.extract_json_action

        # Always return invalid JSON to force stepping
        mock_client.messages_create.return_value = TinyAyaResponse(
            {"choices": [{"message": {"content": "not valid json"}, "finish_reason": "stop"}]},
            100,
        )

        result = run_investigation("Test investigation")

        self.assertEqual(result["steps_taken"], MAX_STEPS)
        self.assertEqual(result["risk_level"], "Inconclusive")
        self.assertTrue(result["escalation_required"])

    @patch("agents.investigation_agent.get_llm_client")
    def test_duplicate_tool_call_blocked(self, mock_get_client):
        """Agent blocks duplicate tool calls with same arguments."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        with patch("agents.llm_client.HF_TOKEN", "test-token"):
            real_client = TinyAyaClient()
        mock_client.extract_json_action = real_client.extract_json_action

        # First: call_tool, then repeat same call_tool, then final
        mock_client.messages_create.side_effect = [
            TinyAyaResponse(
                {"choices": [{"message": {"content": json.dumps({
                    "action": "call_tool",
                    "tool_name": "url_analyzer",
                    "arguments": {"url": "https://example.com"},
                    "reason": "Check URL"
                })}, "finish_reason": "stop"}]},
                100,
            ),
            TinyAyaResponse(
                {"choices": [{"message": {"content": json.dumps({
                    "action": "call_tool",
                    "tool_name": "url_analyzer",
                    "arguments": {"url": "https://example.com"},
                    "reason": "Check URL again"
                })}, "finish_reason": "stop"}]},
                100,
            ),
            TinyAyaResponse(
                {"choices": [{"message": {"content": json.dumps({
                    "action": "final",
                    "conclusion": {
                        "risk_level": "Low",
                        "confidence": 0.9,
                        "summary": "URL is clean",
                        "escalate": False
                    }
                })}, "finish_reason": "stop"}]},
                100,
            ),
        ]

        with patch("agents.investigation_agent.execute_tool") as mock_exec:
            mock_exec.return_value = {
                "success": True,
                "data": {"combined_verdict": "structurally_unremarkable", "url": "https://example.com"},
                "duration_ms": 50,
            }

            result = run_investigation("Check https://example.com")

        # Should not have executed the duplicate call twice
        self.assertEqual(mock_exec.call_count, 1)
        self.assertEqual(result["risk_level"], "Low")


class TestEvidenceSufficientTermination(unittest.TestCase):
    """Test evidence-sufficient termination."""

    @patch("agents.investigation_agent.get_llm_client")
    def test_evidence_sufficient_prompts_conclusion(self, mock_get_client):
        """When evidence is sufficient, agent is prompted to conclude."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        with patch("agents.llm_client.HF_TOKEN", "test-token"):
            real_client = TinyAyaClient()
        mock_client.extract_json_action = real_client.extract_json_action

        # First: call_tool (url_analyzer returns malicious), then final
        mock_client.messages_create.side_effect = [
            TinyAyaResponse(
                {"choices": [{"message": {"content": json.dumps({
                    "action": "call_tool",
                    "tool_name": "url_analyzer",
                    "arguments": {"url": "https://evil.com"},
                    "reason": "Check URL"
                })}, "finish_reason": "stop"}]},
                100,
            ),
            TinyAyaResponse(
                {"choices": [{"message": {"content": json.dumps({
                    "action": "final",
                    "conclusion": {
                        "risk_level": "High",
                        "confidence": 0.9,
                        "summary": "Malicious URL",
                        "escalate": True
                    }
                })}, "finish_reason": "stop"}]},
                100,
            ),
        ]

        with patch("agents.investigation_agent.execute_tool") as mock_exec:
            mock_exec.return_value = {
                "success": True,
                "data": {"combined_verdict": "malicious", "url": "https://evil.com"},
                "duration_ms": 50,
            }

            result = run_investigation("Check https://evil.com")

        self.assertEqual(result["risk_level"], "High")
        self.assertTrue(result["escalation_required"])


class TestFinalConclusionProcessing(unittest.TestCase):
    """Test final conclusion processing and validation."""

    @patch("agents.investigation_agent.get_llm_client")
    def test_agent_produces_final_conclusion(self, mock_get_client):
        """Agent returns a final conclusion when model returns final action."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        with patch("agents.llm_client.HF_TOKEN", "test-token"):
            real_client = TinyAyaClient()
        mock_client.extract_json_action = real_client.extract_json_action

        mock_client.messages_create.side_effect = [
            TinyAyaResponse(
                {"choices": [{"message": {"content": json.dumps({
                    "action": "call_tool",
                    "tool_name": "url_analyzer",
                    "arguments": {"url": "https://example.com"},
                    "reason": "Check URL"
                })}, "finish_reason": "stop"}]},
                100,
            ),
            TinyAyaResponse(
                {"choices": [{"message": {"content": json.dumps({
                    "action": "final",
                    "conclusion": {
                        "risk_level": "Low",
                        "confidence": 0.9,
                        "summary": "URL is clean",
                        "escalate": False
                    }
                })}, "finish_reason": "stop"}]},
                100,
            ),
        ]

        with patch("agents.investigation_agent.execute_tool") as mock_exec:
            mock_exec.return_value = {
                "success": True,
                "data": {"combined_verdict": "structurally_unremarkable", "url": "https://example.com"},
                "duration_ms": 50,
            }

            result = run_investigation("Check https://example.com")

        self.assertEqual(result["risk_level"], "Low")
        self.assertEqual(result["confidence"], 0.9)
        self.assertEqual(result["summary"], "URL is clean")
        self.assertFalse(result["escalation_required"])

    @patch("agents.investigation_agent.get_llm_client")
    def test_invalid_risk_level_sanitized(self, mock_get_client):
        """Invalid risk level is sanitized to Inconclusive."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        with patch("agents.llm_client.HF_TOKEN", "test-token"):
            real_client = TinyAyaClient()
        mock_client.extract_json_action = real_client.extract_json_action

        mock_client.messages_create.side_effect = [
            TinyAyaResponse(
                {"choices": [{"message": {"content": json.dumps({
                    "action": "final",
                    "conclusion": {
                        "risk_level": "NotARiskLevel",
                        "confidence": 0.5,
                        "summary": "Test",
                        "escalate": False
                    }
                })}, "finish_reason": "stop"}]},
                100,
            ),
        ]

        result = run_investigation("Test")

        self.assertEqual(result["risk_level"], "Inconclusive")


class TestApiFailureEscalation(unittest.TestCase):
    """Test that API failures result in escalation rather than fabricated findings."""

    @patch("agents.investigation_agent.get_llm_client")
    def test_authentication_error_escalates(self, mock_get_client):
        """AuthenticationError results in escalation."""
        from agents.llm_client import AuthenticationError

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages_create.side_effect = AuthenticationError("Invalid API key")

        result = run_investigation("Test")

        self.assertEqual(result["risk_level"], "Inconclusive")
        self.assertTrue(result["escalation_required"])
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
