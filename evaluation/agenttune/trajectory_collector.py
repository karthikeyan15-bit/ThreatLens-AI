"""Trajectory collector for AgentTune - captures investigation trajectories."""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from agents.investigation_agent import run_investigation, AgentState
from agents.llm_client import get_llm_client
from tools.tool_registry import serialize_tools, get_tool_by_name, execute_tool
from agents.planner import plan_investigation, InvestigationPlan
from database.db import log_audit_event, get_audit_trail, init_db


class TrajectoryTier(str, Enum):
    """Trajectory tier for AgentTune."""
    LIGHT = "light"       # Observational - for evaluation/failure analysis
    FULL = "full"         # Trainable - full token spans, log probs


@dataclass
class ToolCallRecord:
    """Single tool call within a trajectory."""
    step: int
    tool_name: str
    arguments: Dict[str, Any]
    tool_call_id: str
    response: Dict[str, Any]
    success: bool
    duration_ms: int
    timestamp: str
    error: Optional[str] = None


@dataclass
class EvidenceRecord:
    """Evidence item collected during investigation."""
    indicator_type: str
    indicator_value: str
    source_tool: str
    veracity: str  # confirmed, suggested, unknown
    confidence: float
    source_record: Optional[Dict] = None


@dataclass
class CorrelationRecord:
    """Correlation between indicators."""
    indicator_a: str
    indicator_b: str
    relationship: str
    strength: float


@dataclass
class InvestigationTrajectory:
    """Complete investigation trajectory for AgentTune."""
    trajectory_id: str
    tier: TrajectoryTier
    scenario_id: str
    investigation_input: str
    extracted_indicators: List[Dict]
    planned_tool_sequence: List[str]
    tool_calls: List[ToolCallRecord]
    evidence_collected: List[EvidenceRecord]
    correlations: List[CorrelationRecord]
    risk_assessment: Dict
    escalation_decision: bool
    completion_status: str  # completed, escalated, failed, max_steps
    total_steps: int
    total_duration_ms: int
    ground_truth: Optional[Dict] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict = field(default_factory=dict)


class TrajectoryCollector:
    """Collects investigation trajectories for AgentTune training."""

    def __init__(self, output_dir: str = "artifacts/agenttune/trajectories"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_trajectory: Optional[InvestigationTrajectory] = None
        self.step_counter = 0

    def start_trajectory(
        self,
        request: str,
        scenario_id: str,
        tier: TrajectoryTier = TrajectoryTier.FULL,
        ground_truth: Optional[Dict] = None
    ) -> str:
        """Start a new trajectory collection."""
        # Create plan
        plan = plan_investigation(request)
        indicators = plan.indicators

        trajectory_id = f"traj_{uuid.uuid4().hex[:12]}"

        self.current_trajectory = InvestigationTrajectory(
            trajectory_id=trajectory_id,
            tier=tier,
            scenario_id=scenario_id,
            investigation_input=request,
            extracted_indicators=[{"type": i.type, "value": i.value, "confidence": i.confidence} for i in indicators],
            planned_tool_sequence=plan.tool_sequence,
            tool_calls=[],
            evidence_collected=[],
            correlations=[],
            risk_assessment={},
            escalation_decision=False,
            completion_status="in_progress",
            total_steps=0,
            total_duration_ms=0,
            ground_truth=ground_truth,
            metadata={"planner_notes": plan.notes}
        )
        self.step_counter = 0
        return trajectory_id

    def record_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_call_id: str,
        response: Dict[str, Any],
        success: bool,
        duration_ms: int,
        error: Optional[str] = None
    ):
        """Record a tool call in the current trajectory."""
        if not self.current_trajectory:
            return

        record = ToolCallRecord(
            step=self.step_counter,
            tool_name=tool_name,
            arguments=arguments,
            tool_call_id=tool_call_id,
            response=response,
            success=success,
            duration_ms=duration_ms,
            timestamp=datetime.utcnow().isoformat(),
            error=error
        )
        self.current_trajectory.tool_calls.append(record)
        self.step_counter += 1

    def record_evidence(
        self,
        indicator_type: str,
        indicator_value: str,
        source_tool: str,
        veracity: str,
        confidence: float,
        source_record: Optional[Dict] = None
    ):
        """Record evidence collected."""
        if not self.current_trajectory:
            return

        evidence = EvidenceRecord(
            indicator_type=indicator_type,
            indicator_value=indicator_value,
            source_tool=source_tool,
            veracity=veracity,
            confidence=confidence,
            source_record=source_record
        )
        self.current_trajectory.evidence_collected.append(evidence)

    def record_correlation(
        self,
        indicator_a: str,
        indicator_b: str,
        relationship: str,
        strength: float
    ):
        """Record a correlation between indicators."""
        if not self.current_trajectory:
            return

        corr = CorrelationRecord(
            indicator_a=indicator_a,
            indicator_b=indicator_b,
            relationship=relationship,
            strength=strength
        )
        self.current_trajectory.correlations.append(corr)

    def finalize_trajectory(
        self,
        risk_assessment: Dict,
        escalation_decision: bool,
        completion_status: str,
        total_duration_ms: int
    ) -> InvestigationTrajectory:
        """Finalize the trajectory."""
        if not self.current_trajectory:
            raise ValueError("No active trajectory to finalize")

        self.current_trajectory.risk_assessment = risk_assessment
        self.current_trajectory.escalation_decision = escalation_decision
        self.current_trajectory.completion_status = completion_status
        self.current_trajectory.total_steps = len(self.current_trajectory.tool_calls)
        self.current_trajectory.total_duration_ms = total_duration_ms

        trajectory = self.current_trajectory
        self.current_trajectory = None
        return trajectory

    def save_trajectory(self, trajectory: InvestigationTrajectory) -> Path:
        """Save trajectory to disk as JSONL."""
        filename = f"{trajectory.trajectory_id}.jsonl"
        filepath = self.output_dir / filename

        # Convert to dict
        data = asdict(trajectory)
        # Convert enums
        data["tier"] = trajectory.tier.value

        with open(filepath, "w") as f:
            f.write(json.dumps(data, default=str) + "\n")

        return filepath


class TrajectoryValidator:
    """Validates trajectories for AgentTune compatibility."""

    @staticmethod
    def validate(trajectory: InvestigationTrajectory) -> List[str]:
        """Validate trajectory and return list of errors (empty if valid)."""
        errors = []

        if not trajectory.trajectory_id:
            errors.append("Missing trajectory_id")

        if not trajectory.scenario_id:
            errors.append("Missing scenario_id")

        if not trajectory.investigation_input:
            errors.append("Missing investigation_input")

        if trajectory.total_steps == 0:
            errors.append("No tool calls recorded")

        if trajectory.total_duration_ms <= 0:
            errors.append("Invalid total_duration_ms")

        if trajectory.completion_status not in ["completed", "escalated", "failed", "max_steps"]:
            errors.append(f"Invalid completion_status: {trajectory.completion_status}")

        if trajectory.tier == "full":
            # Full tier requires more validation
            if not trajectory.tool_calls:
                errors.append("Full tier requires tool calls")

            for tc in trajectory.tool_calls:
                if not tc.tool_call_id:
                    errors.append("Tool call missing tool_call_id")

        return errors

    @staticmethod
    def validate_file(filepath: Path) -> List[str]:
        """Validate a trajectory JSONL file."""
        errors = []
        try:
            with open(filepath, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)
                        # Basic schema validation
                        if "trajectory_id" not in data:
                            errors.append(f"Line {line_num}: missing trajectory_id")
                        if "scenario_id" not in data:
                            errors.append(f"Line {line_num}: missing scenario_id")
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {line_num}: JSON decode error: {e}")
        except Exception as e:
            errors.append(f"File read error: {e}")
        return errors


class TrajectoryAdapter:
    """Adapts ThreatLens trajectories to AgentTune format."""

    @staticmethod
    def to_agenttune_format(trajectory: InvestigationTrajectory) -> Dict:
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
    def to_auditkit_format(trajectory: InvestigationTrajectory) -> Dict:
        """Convert to AuditKIT evaluation format."""
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


def collect_trajectory(
    request: str,
    scenario_id: str,
    api_key: Optional[str] = None,
    tier: TrajectoryTier = TrajectoryTier.FULL,
    ground_truth: Optional[Dict] = None
) -> InvestigationTrajectory:
    """
    Run a full investigation and collect trajectory.
    This is the main entry point for trajectory collection.
    """
    # Initialize DB
    init_db()

    # Create collector
    collector = TrajectoryCollector()
    trajectory_id = collector.start_trajectory(request, scenario_id, tier, ground_truth)

    # Get LLM client
    client = get_llm_client()

    # Run investigation with trajectory collection
    inv_id = create_investigation("multi", request)
    update_investigation(inv_id, status="running")

    plan = plan_investigation(request)
    state = AgentState(investigation_id=inv_id)

    # Import the client
    from agents.investigation_agent import _build_system_prompt, _assess_risk, _validate_tool_args, _should_continue_investigation, _extract_evidence_from_result, _finalize_result, correlate_evidence

    system_prompt = _build_system_prompt()
    state.history.append({"role": "system", "content": system_prompt})

    # Include plan in initial message
    initial_msg = f"""Investigate this cybersecurity incident:

{request}

PLAN (use as guidance):
- Request type: {plan.request_type}
- Extracted indicators: {[f"{i.type}:{i.value}" for i in plan.indicators]}
- Recommended tool sequence: {plan.tool_sequence}
- Notes: {plan.notes}

Extract all indicators and determine the appropriate tools to use. Start with the first recommended tool if applicable."""
    state.history.append({
        "role": "user",
        "content": initial_msg,
    })

    start_time = time.time()
    trajectory_id = f"traj_{uuid.uuid4().hex[:12]}"
    trajectory = InvestigationTrajectory(
        trajectory_id=trajectory_id,
        tier=TrajectoryTier.FULL,
        scenario_id=scenario_id,
        investigation_input=request,
        extracted_indicators=[{"type": i.type, "value": i.value, "confidence": i.confidence} for i in plan.indicators],
        planned_tool_sequence=plan.tool_sequence,
        tool_calls=[],
        evidence_collected=[],
        correlations=[],
        risk_assessment={},
        escalation_decision=False,
        completion_status="in_progress",
        total_steps=0,
        total_duration_ms=0,
        ground_truth=ground_truth,
        metadata={"planner_notes": plan.notes}
    )

    step_counter = 0
    max_steps = 12

    try:
        while len(trajectory.tool_calls) < max_steps:
            start_time = time.time()

            # Get response from LLM
            response = client.messages_create(
                model="tiny-aya-global",
                max_tokens=1024,
                messages=state.history,
                tools=serialize_tools(),
            )

            duration_ms = int((time.time() - start_time) * 1000)

            for block in response.content:
                if block.type == "text":
                    state.history.append({"role": "assistant", "content": block.text})

                    # Check for conclusion
                    if "{" in block.text and "}" in block.text:
                        try:
                            conclusion_match = block.text[block.text.rfind("{"):block.text.rfind("}") + 1]
                            conclusion = json.loads(conclusion_match)
                            if "conclusion" in conclusion:
                                result = conclusion["conclusion"]
                                risk, confidence = _assess_risk(trajectory.evidence_collected)
                                final_risk = result.get("risk_level", risk)
                                final_confidence = result.get("confidence", confidence)

                                trajectory.risk_assessment = result
                                trajectory.escalation_decision = result.get("escalate", False)
                                trajectory.completion_status = "completed" if not result.get("escalate") else "escalated"
                                trajectory.total_steps = len(trajectory.tool_calls)
                                trajectory.total_duration_ms = int((time.time() - time.time()) * 1000)  # Will be updated

                                # Save trajectory
                                collector = TrajectoryCollector()
                                collector.current_trajectory = trajectory
                                collector.finalize_trajectory(
                                    risk_assessment=trajectory.risk_assessment,
                                    escalation_decision=trajectory.escalation_decision,
                                    completion_status=trajectory.completion_status,
                                    total_duration_ms=int((time.time() - time.time()) * 1000)
                                )
                                return trajectory
                        except (json.JSONDecodeError, IndexError):
                            pass

                elif block.type == "tool_use":
                    tool_name = block.name
                    tool_args = block.input or {}
                    tool_call_id = block.id

                    # Validate
                    valid, error = _validate_tool_args(tool_name, tool_args)
                    if not valid:
                        tool_result = {"success": False, "error": error, "data": {}}
                        trajectory.tool_calls.append(ToolCallRecord(
                            step=step_counter,
                            tool_name=tool_name,
                            arguments=tool_args,
                            tool_call_id=tool_call_id,
                            response=tool_result,
                            success=False,
                            duration_ms=0,
                            timestamp=datetime.utcnow().isoformat(),
                            error=error
                        ))
                        continue

                    # Execute tool
                    start = time.time()
                    tool_result = execute_tool(tool_name, tool_args)
                    duration_ms = int((time.time() - start) * 1000)

                    # Record tool call
                    trajectory.tool_calls.append(ToolCallRecord(
                        step=step_counter,
                        tool_name=tool_name,
                        arguments=tool_args,
                        tool_call_id=tool_call_id,
                        response=tool_result,
                        success=tool_result.get("success", False),
                        duration_ms=duration_ms,
                        timestamp=datetime.utcnow().isoformat()
                    ))
                    step_counter += 1

                    # Extract evidence
                    _extract_evidence_from_result(inv_id="", tool_name=tool_name, result=tool_result, state=type('State', (), {'evidence': []})())

                    # Add to history for next iteration
                    state.history.append({
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": tool_call_id, "name": tool_name, "input": tool_args}],
                    })
                    state.history.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": json.dumps(tool_result),
                            "is_error": not tool_result.get("success", False),
                        }],
                    })

                    # Check if should continue
                    should_continue, reason = _should_continue_investigation(
                        trajectory.evidence_collected, trajectory.tool_calls, 12, step_counter
                    )
                    if not should_continue:
                        # Force conclusion
                        state.history.append({
                            "role": "user",
                            "content": f"Evidence sufficient to conclude ({reason}). Provide your conclusion in the JSON format."
                        })
                        continue

            step_counter += 1

    except Exception as exc:
        trajectory.completion_status = "failed"
        trajectory.risk_assessment = {"error": str(exc)}

    return trajectory


if __name__ == "__main__":
    # Test trajectory collection
    import os
    os.environ["TINY_AYA_API_KEY"] = "test"

    test_request = "Investigate this suspicious email: From: security@payroll-services.xyz Subject: URGENT: Account suspended! Click http://payroll-verify.xyz/login to verify now!"

    traj = collect_trajectory(
        request=test_request,
        scenario_id="test_scenario_001",
        api_key="test"
    )

    print(f"Trajectory collected: {trajectory.trajectory_id}")
    print(f"Steps: {trajectory.total_steps}")
    print(f"Status: {trajectory.completion_status}")

    # Validate
    validator = TrajectoryValidator()
    errors = validator.validate(trajectory)
    print(f"Validation errors: {errors}")

    # Save
    collector = TrajectoryCollector()
    path = collector.save_trajectory(trajectory)
    print(f"Saved to: {path}")