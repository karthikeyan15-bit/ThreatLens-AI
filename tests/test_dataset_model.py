"""Tests for dataset integrity and model loading."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_loader import load_model, format_model_status


class TestDatasetIntegrity(unittest.TestCase):
    """Test evaluation dataset integrity and consistency."""

    SCENARIOS_PATH = Path("datasets/evaluation/scenarios.json")

    def test_scenarios_file_exists(self):
        self.assertTrue(self.SCENARIOS_PATH.exists())

    def test_scenarios_have_required_fields(self):
        scenarios = json.loads(self.SCENARIOS_PATH.read_text(encoding="utf-8"))
        self.assertGreater(len(scenarios), 0)
        for s in scenarios:
            self.assertIn("id", s)
            self.assertIn("request", s)
            self.assertIn("reference", s)
            ref = s["reference"]
            self.assertIn("expected_risk", ref)
            self.assertIn("expected_tools", ref)
            self.assertIn("escalation_decision", ref)
            self.assertIn("min_expected_evidence", ref)

    def test_risk_levels_valid(self):
        valid = {"Low", "Medium", "High", "Critical", "Inconclusive"}
        scenarios = json.loads(self.SCENARIOS_PATH.read_text(encoding="utf-8"))
        for s in scenarios:
            risk = s["reference"].get("expected_risk")
            self.assertIn(risk, valid, f"Scenario {s['id']} has invalid risk {risk}")

    def test_no_duplicate_scenario_ids(self):
        scenarios = json.loads(self.SCENARIOS_PATH.read_text(encoding="utf-8"))
        ids = [s["id"] for s in scenarios]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate scenario IDs found")


class TestModelLoading(unittest.TestCase):
    """Test trained classifier artifacts load correctly."""

    def test_email_classifier_loads(self):
        result = load_model("email")
        self.assertEqual(result["source"], "trained_ml")
        self.assertIsNotNone(result["pipeline"])
        self.assertIsNotNone(result["metrics"])

    def test_url_classifier_loads(self):
        result = load_model("url")
        self.assertEqual(result["source"], "trained_ml")
        self.assertIsNotNone(result["pipeline"])
        self.assertIsNotNone(result["metrics"])

    def test_model_metrics_real(self):
        """Model metrics are real values, not fabricated."""
        email_result = load_model("email")
        url_result = load_model("url")

        email_acc = email_result["metrics"].get("test_accuracy")
        url_acc = url_result["metrics"].get("test_accuracy")

        self.assertIsNotNone(email_acc)
        self.assertIsNotNone(url_acc)
        self.assertGreater(email_acc, 0.0)
        self.assertGreater(url_acc, 0.0)
        self.assertLessEqual(email_acc, 1.0)
        self.assertLessEqual(url_acc, 1.0)

    def test_model_status_format(self):
        email_result = load_model("email")
        status = format_model_status("email", email_result)
        self.assertIn("Trained ML model loaded", status)


if __name__ == "__main__":
    unittest.main()
