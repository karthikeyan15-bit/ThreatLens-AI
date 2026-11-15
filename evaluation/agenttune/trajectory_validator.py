"""Trajectory validator for AgentTune - validates trajectories for training compatibility."""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from enum import Enum as EnumClass

from evaluation.agenttune.trajectory_collector import (
    InvestigationTrajectory,
    TrajectoryTier,
    ToolCallRecord,
    EvidenceRecord,
    CorrelationRecord,
)


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    step: Optional[int] = None


class TrajectoryValidator:
    """Validates trajectories for AgentTune training compatibility."""

    # Required fields for full-tier trajectories
    REQUIRED_FULL_TIER_FIELDS = [
        "trajectory_id",
        "scenario_id",
        "investigation_input",
        "tier",
        "tool_calls",
        "evidence_collected",
        "risk_assessment",
        "completion_status",
    ]

    # Required fields for all tiers
    REQUIRED_BASE_FIELDS = [
        "trajectory_id",
        "scenario_id",
        "investigation_input",
        "tier",
        "completion_status",
    ]

    VALID_COMPLETION_STATUSES = {"completed", "escalated", "failed", "max_steps", "in_progress"}
    VALID_VERACITY_VALUES = {"confirmed", "suggested", "unknown"}
    VALID_RISK_LEVELS = {"Low", "Medium", "High", "Critical", "Inconclusive"}
    VALID_TOOLS = {
        "email_analyzer", "url_analyzer", "ip_analyzer",
        "network_analyzer", "threat_intel_lookup", "correlate", "risk_assess"
    }

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.issues: List[Dict] = []
        self._tool_call_ids: set = set()

    def validate(self, trajectory: "InvestigationTrajectory") -> Tuple[bool, List[Dict]]:
        """Validate trajectory and return (is_valid, issues)."""
        self.issues = []

        # Run all validations
        self._validate_required_fields(trajectory)
        self._validate_tier_specific(trajectory)
        self._validate_tool_calls(trajectory)
        self._validate_evidence(trajectory)
        self._validate_correlations(trajectory)
        self._validate_risk_assessment(trajectory)
        self._validate_completion_status(trajectory)
        self._validate_consistency(trajectory)

        # Return is_valid (no errors) and issues
        has_errors = any(issue["severity"] == "error" for issue in self.issues)
        return not self.strict_mode or not any(i["severity"] == "error" for i in self.issues), self.issues

    def validate_file(self, filepath: str) -> Tuple[bool, List[Dict]]:
        """Validate a trajectory JSONL file."""
        errors = []
        try:
            with open(filepath, "r") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        # Validate each trajectory
                        # We'd need to reconstruct the object
                        # For now, basic JSON structure check
                        if "trajectory_id" not in data:
                            self._add_issue("error", f"Line {line}: missing trajectory_id")
                        if "scenario_id" not in data:
                            self._add_issue("error", f"Line {line}: missing scenario_id")
                        if "tier" not in data:
                            self._add_issue("warning", f"Line {line}: missing tier")
                    except json.JSONDecodeError as e:
                        self._add_issue("error", f"Line {line_num}: JSON decode error: {e}")
        except Exception as e:
            return False, [{"severity": "error", "message": f"File read error: {e}"}]

        has_errors = any(issue["severity"] == "error" for issue in self.issues)
        return len(self.issues) == 0, self.issues

    def _add_issue(self, severity: str, message: str, field: Optional[str] = None, step: Optional[int] = None):
        self.issues.append({
            "severity": severity,
            "message": message,
            "field": field,
            "step": step
        })

    def _validate_required_fields(self, trajectory: "InvestigationTrajectory"):
        """Validate required fields are present."""
        base_attrs = [
            "trajectory_id", "scenario_id", "investigation_input",
            "tier", "completion_status"
        ]

        for field_name in self.REQUIRED_BASE_FIELDS:
            value = getattr(trajectory, field_name, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                self._add_issue("error", f"Missing required field: {field_name}", field=field_name)

        # Check tier
        if hasattr(trajectory, "tier"):
            tier_val = trajectory.tier.value if hasattr(trajectory.tier, 'value') else trajectory.tier
            if tier_val not in ["light", "full"]:
                self._add_issue("error", f"Invalid tier: {trajectory.tier}", field="tier")

    def _validate_tier_specific(self, trajectory: "InvestigationTrajectory"):
        """Validate tier-specific requirements."""
        tier_val = trajectory.tier.value if hasattr(trajectory.tier, 'value') else trajectory.tier

        if tier_val == "full":
            # Full tier requires more fields
            for field_name in self.REQUIRED_FULL_TIER_FIELDS:
                if not hasattr(trajectory, field_name) or getattr(trajectory, field_name) is None:
                    self._add_issue("error", f"Full tier requires field: {field_name}", field=field)

            # Must have at least one tool call for full tier
            if not hasattr(trajectory, "tool_calls") or not trajectory.tool_calls:
                self._add_issue("error", "Full tier requires at least one tool call", field="tool_calls")

    def _validate_tool_calls(self, trajectory: "InvestigationTrajectory"):
        """Validate tool calls."""
        if not hasattr(trajectory, "tool_calls"):
            return

        seen_tool_call_ids = set()
        for i, tc in enumerate(trajectory.tool_calls):
            if not isinstance(tc, ToolCallRecord):
                # Might be dict
                if not isinstance(tc, dict):
                    self._add_issue("error", f"Tool call {i} is not a ToolCallRecord or dict", step=i)
                    continue
                tc_dict = tc
            else:
                tc_dict = tc.__dict__ if hasattr(tc, '__dict__') else tc

            # Check required fields
            required = ["tool_name", "arguments", "tool_call_id", "success", "duration_ms"]
            for field in required:
                if field not in tc_dict or tc_dict[field] is None:
                    self._add_issue("error", f"Tool call {i} missing field: {field}", step=i)

            # Check tool_call_id uniqueness
            tool_call_id = tc.get("tool_call_id") if isinstance(tc, dict) else getattr(tc, "tool_call_id", None)
            if tool_call_id:
                if tool_call_id in self._tool_call_ids:
                    self._add_issue("error", f"Duplicate tool_call_id: {tool_call_id}", step=i)
                self._tool_call_ids.add(tool_call_id)

            # Validate tool name
            tool_name = tc.get("tool_name") if isinstance(tc, dict) else getattr(tc, "tool_name", "")
            if tool_name not in self.VALID_TOOLS:
                self._add_issue("warning", f"Unknown tool: {tool_name}", step=i)

            # Check duration
            duration = tc.get("duration_ms") if isinstance(tc, dict) else getattr(tc, "duration_ms", None)
            if duration is not None and duration < 0:
                self._add_issue("warning", f"Negative duration for tool call {i}", step=i)

    def _validate_evidence(self, trajectory: "InvestigationTrajectory"):
        """Validate evidence items."""
        if not hasattr(trajectory, "evidence_collected"):
            return

        for i, evidence in enumerate(trajectory.evidence_collected):
            if isinstance(evidence, dict):
                ev = evidence
            elif isinstance(evidence, EvidenceRecord):
                ev = evidence.__dict__
            else:
                self._add_issue("error", f"Evidence {i} is not a dict or EvidenceRecord", step=i)
                continue

            required = ["indicator_type", "indicator_value", "source_tool", "veracity", "confidence"]
            for field in required:
                if field not in ev or ev[field] is None:
                    self._add_issue("error", f"Evidence {i} missing field: {field}", step=i)

            # Validate veracity
            veracity = ev.get("veracity", "").lower()
            if veracity not in self.VALID_VERACITY_VALUES:
                self._add_issue("warning", f"Evidence {i} has invalid veracity: {veracity}")

            # Validate confidence
            conf = ev.get("confidence")
            if conf is not None and (conf < 0 or conf > 1):
                self._add_issue("warning", f"Evidence {i} confidence out of range: {conf}")

    def _validate_correlations(self, trajectory: "InvestigationTrajectory"):
        """Validate correlations."""
        if not hasattr(trajectory, "correlations"):
            return

        for i, corr in enumerate(trajectory.correlations):
            if isinstance(corr, dict):
                corr_dict = corr
            elif isinstance(corr, CorrelationRecord):
                corr_dict = corr.__dict__
            else:
                self._add_issue("error", f"Correlation {i} is not a dict or CorrelationRecord", step=i)
                continue

            required = ["indicator_a", "indicator_b", "relationship", "strength"]
            for field in required:
                if field not in corr_dict or corr_dict[field] is None:
                    self._add_issue("error", f"Correlation {i} missing field: {field}", step=i)

            # Validate strength
            strength = corr_dict.get("strength")
            if strength is not None and (strength < 0 or strength > 1):
                self._add_issue("warning", f"Correlation {i} strength out of range: {strength}")

    def _validate_risk_assessment(self, trajectory: "InvestigationTrajectory"):
        """Validate risk assessment."""
        if not hasattr(trajectory, "risk_assessment") or not trajectory.risk_assessment:
            self._add_issue("warning", "Missing risk_assessment")
            return

        risk = trajectory.risk_assessment

        # Check risk_level
        risk_level = risk.get("risk_level")
        if risk_level:
            if risk_level not in self.VALID_RISK_LEVELS:
                self._add_issue("warning", f"Invalid risk_level: {risk_level}")

        # Check confidence
        conf = risk.get("confidence")
        if conf is not None and (conf < 0 or conf > 1):
            self._add_issue("warning", f"Confidence out of range: {conf}")

    def _validate_completion_status(self, trajectory: "InvestigationTrajectory"):
        """Validate completion status."""
        status = getattr(trajectory, "completion_status", None)
        if status not in self.VALID_COMPLETION_STATUSES:
            self._add_issue("error", f"Invalid completion_status: {status}", field="completion_status")

    def _validate_consistency(self, trajectory: "InvestigationTrajectory"):
        """Check internal consistency."""
        # Check if escalation matches completion status
        escalate = getattr(trajectory, "escalation_decision", False)
        status = getattr(trajectory, "completion_status", "")

        if escalate and trajectory.completion_status not in ["escalated", "completed"]:
            self._add_issue("warning", "Escalation true but completion status not 'escalated' or 'completed'")

        # Check if evidence matches tool calls
        # (Could add more consistency checks here)


def validate_trajectory_file(filepath: str) -> Tuple[bool, List[Dict]]:
    """Validate a trajectory JSONL file."""
    validator = TrajectoryValidator()
    return validator.validate_file(filepath)


def validate_trajectory(trajectory: "InvestigationTrajectory") -> Tuple[bool, List[Dict]]:
    """Validate a trajectory object."""
    validator = TrajectoryValidator()
    return validator.validate(trajectory)


if __name__ == "__main__":
    print("Trajectory Validator module ready")
    print("Classes: TrajectoryValidator")
    print("Functions: validate_trajectory, validate_trajectory_file")