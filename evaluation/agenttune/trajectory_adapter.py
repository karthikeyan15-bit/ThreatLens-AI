"""Trajectory adapter for AgentTune - converts ThreatLens trajectories to AgentTune format."""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from evaluation.agenttune.trajectory_collector import (
    InvestigationTrajectory,
    TrajectoryTier,
    ToolCallRecord,
    EvidenceRecord,
    CorrelationRecord,
)


class AgentTuneTrajectoryAdapter:
    """Adapts ThreatLens trajectories to AgentTune training format."""

    @staticmethod
    def to_agenttune_format(trajectory: "InvestigationTrajectory") -> Dict:
        """Convert ThreatLens trajectory to AgentTune training format."""
        # AgentTune expects specific format for training
        return {
            "trajectory_id": trajectory.trajectory_id,
            "scenario_id": trajectory.scenario_id,
            "input": trajectory.investigation_input,
            "tier": trajectory.tier.value,
            "steps": [
                {
                    "step": tc.step,
                    "tool": tc.tool_name,
                    "arguments": tc.arguments,
                    "tool_call_id": tc.tool_call_id,
                    "result": tc.response,
                    "success": tc.success,
                    "duration_ms": tc.duration_ms,
                }
                for tc in trajectory.tool_calls
            ],
            "evidence": [
                {
                    "type": e.indicator_type,
                    "value": e.indicator_value,
                    "source": e.source_tool,
                    "veracity": e.veracity,
                    "confidence": e.confidence
                }
                for e in trajectory.evidence_collected
            ],
            "correlations": [
                {
                    "a": c.indicator_a,
                    "b": c.indicator_b,
                    "relationship": c.relationship,
                    "strength": c.strength
                }
                for c in trajectory.correlations
            ],
            "outcome": {
                "risk_level": trajectory.risk_assessment.get("risk_level"),
                "confidence": trajectory.risk_assessment.get("confidence"),
                "summary": trajectory.risk_assessment.get("summary"),
                "escalate": trajectory.escalation_decision,
                "status": trajectory.completion_status
            },
            "ground_truth": trajectory.ground_truth,
            "metadata": trajectory.metadata
        }

    @staticmethod
    def to_agenttune_training_format(trajectory: "InvestigationTrajectory") -> Dict:
        """Convert to AgentTune training format (messages + tools + rewards)."""
        # This format is for training with agenttune.train_agentic
        # Returns a dict with messages, tools, and rewards ready for AgentTune

        # Build conversation history
        messages = []
        # System prompt would be first
        # Then alternating user/assistant/tool messages

        messages = []
        # We'll need the system prompt from the agent
        system_prompt = """You are a cybersecurity investigation analyst. Your job is to investigate suspicious indicators using available tools.

RULES:
- Only use the tools provided; never invent results.
- If a tool reports an error or "not configured", note it but continue with other tools.
- Correlate findings across tools — link shared domains, IPs, and email senders.
- Be conservative: NEVER declare something safe just because no threat intel match was found. If all threat intel returns inconclusive, conclude "Inconclusive".
- When uncertain, recommend human escalation.
- Always call your tools one at a time and wait for results before deciding the next step.
- Validate your tool arguments before calling.

OUTPUT FORMAT:
When you have gathered enough evidence to conclude, respond with a JSON block like:
{
  "conclusion": {
    "risk_level": "Low|Medium|High|Critical|Inconclusive",
    "confidence": 0.0-1.0,
    "summary": "Brief explanation of findings",
    "escalate": true|false
  }
}

Start by extracting indicators from the request, then call the appropriate tools."""

        messages.append({"role": "system", "content": system_prompt})

        # Add user message with investigation request
        # This would need the original request from the trajectory
        # For now, return a template

        return {
            "messages": messages,
            "tools": [],  # Would need to serialize tools
            "rewards": {},  # Would be computed by reward function
            "metadata": {
                "trajectory_id": "",
                "scenario_id": "",
                "ground_truth": {}
            }
        }

    @staticmethod
    def to_training_sample(trajectory: "InvestigationTrajectory") -> Dict:
        """Convert trajectory to a training sample for AgentTune."""
        return {
            "input": trajectory.investigation_input,
            "scenario_id": trajectory.scenario_id,
            "trajectory_id": trajectory.trajectory_id,
            "tier": trajectory.tier.value,
            "expected_tools": trajectory.planned_tool_sequence,
            "actual_tools": [tc.tool_name for tc in trajectory.tool_calls],
            "tool_calls": [
                {
                    "tool": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": tc.response,
                    "success": tc.success,
                    "duration_ms": tc.duration_ms,
                }
                for tc in trajectory.tool_calls
            ],
            "evidence": [
                {
                    "type": e.indicator_type,
                    "value": e.indicator_value,
                    "source": e.source_tool,
                    "veracity": e.veracity,
                    "confidence": e.confidence
                }
                for e in trajectory.evidence_collected
            ],
            "correlations": [
                {
                    "a": c.indicator_a,
                    "b": c.indicator_b,
                    "relationship": c.relationship,
                    "strength": c.strength
                }
                for c in trajectory.correlations
            ],
            "expected_outcome": trajectory.ground_truth or {},
            "actual_outcome": {
                "risk_level": trajectory.risk_assessment.get("risk_level"),
                "confidence": trajectory.risk_assessment.get("confidence"),
                "summary": trajectory.risk_assessment.get("summary"),
                "escalate": trajectory.escalation_decision,
                "status": trajectory.completion_status
            }
        }


class AuditKitTrajectoryAdapter:
    """Adapts ThreatLens trajectories to AuditKIT evaluation format."""

    @staticmethod
    def to_auditkit_format(trajectory: "InvestigationTrajectory") -> Dict:
        """Convert ThreatLens trajectory to AuditKIT evaluation format."""
        return {
            "scenario_id": trajectory.scenario_id,
            "input": trajectory.investigation_input,
            "expected_tool_calls": trajectory.planned_tool_sequence,
            "actual_tool_calls": [tc.tool_name for tc in trajectory.tool_calls],
            "evidence": [
                {
                    "indicator_type": e.indicator_type,
                    "indicator_value": e.indicator_value,
                    "veracity": e.veracity
                }
                for e in trajectory.evidence_collected
            ],
            "expected_risk": trajectory.ground_truth.get("expected_risk") if trajectory.ground_truth else None,
            "expected_escalation": trajectory.ground_truth.get("escalation_decision") if trajectory.ground_truth else None,
            "actual_risk": trajectory.risk_assessment.get("risk_level"),
            "actual_escalation": trajectory.escalation_decision,
            "completion_status": trajectory.completion_status
        }

    @staticmethod
    def to_auditkit_samples(trajectories: List) -> List[Dict]:
        """Convert multiple trajectories to AuditKIT format."""
        return [TrajectoryAdapter.to_auditkit_format(t) for t in trajectories]


def export_trajectories_jsonl(
    trajectories: List,
    output_path: str,
    format: str = "threatlens"
) -> Path:
    """Export trajectories to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for traj in trajectories:
            if format == "agenttune":
                data = TrajectoryAdapter.to_agenttune_format(traj)
            elif format == "auditkit":
                data = AuditKitTrajectoryAdapter.to_auditkit_format(traj)
            else:
                # Default ThreatLens format
                data = traj.__dict__ if hasattr(traj, '__dict__') else traj

            f.write(json.dumps(data, default=str) + "\n")

    return Path(output_path)


def load_trajectories_jsonl(filepath: str) -> List[Dict]:
    """Load trajectories from JSONL file."""
    trajectories = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    return trajectories


def load_trajectory_objects(filepath: str) -> List:
    """Load trajectory objects from JSONL file."""
    from evaluation.agenttune.trajectory_collector import InvestigationTrajectory

    trajectories = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Reconstruct trajectory object
                # This would need proper deserialization
                trajectories.append(data)
    return trajectories


if __name__ == "__main__":
    print("Trajectory Adapter module ready")
    print("Available classes:")
    print("  - AgentTuneTrajectoryAdapter")
    print("  - AuditKitTrajectoryAdapter")
    print("Functions:")
    print("  - export_trajectories_jsonl")
    print("  - load_trajectories_jsonl")