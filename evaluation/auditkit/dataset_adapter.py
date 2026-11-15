"""Dataset adapter for AuditKIT - converts ThreatLens scenarios to AuditKIT evaluation format."""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator
from dataclasses import dataclass, field
from datetime import datetime

try:
    from auditkit import Scenario, Sample, Adapter
    AUDITKIT_AVAILABLE = True
except ImportError:
    AUDITKIT_AVAILABLE = False

from evaluation.agenttune.trajectory_collector import InvestigationTrajectory
from evaluation.agenttune.trajectory_adapter import AuditKitTrajectoryAdapter


@dataclass
class ThreatLensScenario:
    """ThreatLens evaluation scenario."""
    id: str
    name: str
    description: str
    request: str
    indicators: Dict
    reference: Dict
    category: str = "general"


class ThreatLensScenarioLoader:
    """Loads ThreatLens evaluation scenarios."""

    @staticmethod
    def load_scenarios(filepath: str = "datasets/evaluation/scenarios.json") -> List[Dict]:
        """Load scenarios from JSON file."""
        with open(filepath, "r") as f:
            return json.load(f)

    @staticmethod
    def get_scenario(scenario_id: str) -> Optional[Dict]:
        """Get a specific scenario by ID."""
        scenarios = ThreatLensScenarioLoader.load_scenarios()
        for s in scenarios:
            if s["id"] == scenario_id:
                return s
        return None

    @staticmethod
    def get_all_scenario_ids() -> List[str]:
        """Get all scenario IDs."""
        scenarios = ThreatLensScenarioLoader.load_scenarios()
        return [s["id"] for s in scenarios]


class ThreatLensToAuditKitAdapter:
    """Adapts ThreatLens scenarios and trajectories to AuditKIT format."""

    @staticmethod
    def scenario_to_auditkit_scenario(scenario: Dict) -> Dict:
        """Convert ThreatLens scenario to AuditKIT scenario format."""
        return {
            "id": scenario["id"],
            "name": scenario.get("name", scenario["id"]),
            "description": scenario.get("description", ""),
            "input": scenario["request"],
            "metadata": {
                "indicators": scenario.get("indicators", {}),
                "category": scenario.get("category", "cybersecurity")
            },
            "expected_output": {
                "risk_level": scenario["reference"].get("expected_risk"),
                "escalation": scenario["reference"].get("escalate"),
                "expected_tools": scenario["reference"].get("expected_tools", []),
                "expected_calls": scenario["reference"].get("expected_calls", []),
                "min_evidence": scenario["reference"].get("min_expected_evidence", 1),
                "ground_truth_labels": scenario["reference"].get("ground_truth_labels", {})
            }
        }

    @staticmethod
    def trajectory_to_auditkit_sample(trajectory, ground_truth: Dict = None) -> Dict:
        """Convert trajectory to AuditKIT evaluation sample."""
        return {
            "id": trajectory.trajectory_id,
            "input": trajectory.investigation_input,
            "expected_output": {
                "risk_level": trajectory.ground_truth.get("expected_risk") if trajectory.ground_truth else None,
                "escalation": trajectory.ground_truth.get("escalation_decision") if trajectory.ground_truth else None,
                "expected_tools": trajectory.planned_tool_sequence,
                "expected_calls": trajectory.ground_truth.get("expected_calls", []) if trajectory.ground_truth else [],
                "min_evidence": trajectory.ground_truth.get("min_expected_evidence", 1) if trajectory.ground_truth else 1,
                "expected_risk": trajectory.ground_truth.get("expected_risk") if trajectory.ground_truth else None,
                "escalation_decision": trajectory.ground_truth.get("escalation_decision") if trajectory.ground_truth else None,
                "ground_truth_labels": trajectory.ground_truth.get("ground_truth_labels", {}) if trajectory.ground_truth else {}
            },
            "actual_output": {
                "risk_level": trajectory.risk_assessment.get("risk_level"),
                "confidence": trajectory.risk_assessment.get("confidence"),
                "summary": trajectory.risk_assessment.get("summary"),
                "escalation_decision": trajectory.escalation_decision,
                "completion_status": trajectory.completion_status,
                "tool_calls": [
                    {
                        "tool": tc.tool_name,
                        "arguments": tc.arguments,
                        "result": tc.response,
                        "success": tc.success,
                        "duration_ms": tc.duration_ms
                    }
                    for tc in trajectory.tool_calls
                ],
                "evidence": [
                    {
                        "indicator_type": e.indicator_type,
                        "indicator_value": e.indicator_value,
                        "veracity": e.veracity,
                        "confidence": e.confidence,
                        "source_tool": e.source_tool
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
                "tool_calls_count": len(trajectory.tool_calls),
                "completion_status": trajectory.completion_status,
                "escalation_decision": trajectory.escalation_decision,
                "evidence_count": len(trajectory.evidence_collected)
            }
        }

    @staticmethod
    def trajectory_to_auditkit_sample_v2(trajectory, ground_truth: Dict = None) -> Dict:
        """Convert trajectory to AuditKIT v2 sample format (AgentCase format)."""
        return {
            "id": trajectory.trajectory_id,
            "input": trajectory.investigation_input,
            "expected": {
                "risk_level": trajectory.ground_truth.get("expected_risk") if trajectory.ground_truth else None,
                "escalation": trajectory.ground_truth.get("escalation_decision") if trajectory.ground_truth else None,
                "tools": trajectory.planned_tool_sequence,
                "evidence_min": trajectory.ground_truth.get("min_expected_evidence", 1) if trajectory.ground_truth else 1,
            },
            "metadata": {
                "scenario_id": trajectory.scenario_id,
                "planned_tools": trajectory.planned_tool_sequence,
                "ground_truth": trajectory.ground_truth or {}
            }
        }


class ThreatLensAuditKitDataset:
    """AuditKIT-compatible dataset for ThreatLens evaluation."""

    def __init__(self, scenarios_path: str = "datasets/evaluation/scenarios.json"):
        self.scenarios_path = Path(scenarios_path)
        self.scenarios = self._load_scenarios()

    def _load_scenarios(self) -> List[Dict]:
        """Load scenarios from file."""
        with open(self.scenarios_path, "r") as f:
            return json.load(f)

    def to_auditkit_samples(self, trajectories: List = None) -> List[Dict]:
        """Convert trajectories to AuditKIT samples."""
        samples = []

        if trajectories is None:
            # Load from trajectories directory
            trajectories_dir = Path("artifacts/agenttune/trajectories")
            if trajectories_dir.exists():
                for jsonl_file in trajectories_dir.glob("*.jsonl"):
                    with open(jsonl_file, "r") as f:
                        for line in f:
                            if line.strip():
                                traj_data = json.loads(line)
                                samples.append(self._trajectory_to_sample(traj_data))
        else:
            for traj in trajectories:
                samples.append(self._trajectory_to_sample(traj))

        return samples

    def _trajectory_to_sample(self, trajectory: Dict) -> Dict:
        """Convert trajectory dict to AuditKIT sample."""
        return {
            "id": trajectory.get("trajectory_id", str(uuid.uuid4())),
            "input": trajectory.get("investigation_input", ""),
            "expected": {
                "risk_level": trajectory.get("ground_truth", {}).get("expected_risk"),
                "escalation": trajectory.get("ground_truth", {}).get("escalation_decision"),
                "tools": trajectory.get("planned_tool_sequence", []),
                "evidence_min": trajectory.get("ground_truth", {}).get("min_expected_evidence", 1),
            },
            "actual": {
                "risk_level": trajectory.get("risk_assessment", {}).get("risk_level"),
                "confidence": trajectory.get("risk_assessment", {}).get("confidence"),
                "summary": trajectory.get("risk_assessment", {}).get("summary"),
                "escalation": trajectory.get("escalation_decision"),
                "completion": trajectory.get("completion_status"),
                "tools_used": [tc.get("tool_name") for tc in trajectory.get("tool_calls", [])],
                "evidence": trajectory.get("evidence_collected", []),
            },
            "metadata": {
                "trajectory_id": trajectory.get("trajectory_id"),
                "scenario_id": trajectory.get("scenario_id"),
                "steps": len(trajectory.get("tool_calls", [])),
                "duration_ms": trajectory.get("total_duration_ms", 0),
            }
        }

    def to_auditkit_dataset(self, output_path: str = "artifacts/auditkit/dataset.jsonl"):
        """Export dataset in AuditKIT JSONL format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        samples = []
        if AUDITKIT_AVAILABLE:
            # Use AuditKIT's native format if available
            pass
        else:
            # Fallback to JSONL
            with open(output_path, "w") as f:
                for sample in self._generate_samples():
                    f.write(json.dumps(sample) + "\n")
            return output_path

    def _generate_samples(self) -> List[Dict]:
        """Generate samples from trajectory files."""
        samples = []
        trajectories_dir = Path("artifacts/agenttune/trajectories")
        if not trajectories_dir.exists():
            return []

        for jsonl_file in trajectories_dir.glob("*.jsonl"):
            with open(jsonl_file, "r") as f:
                for line in f:
                    if line.strip():
                        traj = json.loads(line)
                        samples.append(self._trajectory_to_sample(traj))
        return samples


class AuditKitAdapter:
    """AuditKIT adapter for ThreatLens agent evaluation."""

    def __init__(self, model_name: str = "tiny-aya-global"):
        self.model_name = model_name
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize the LLM client."""
        from agents.llm_client import get_llm_client
        self.client = get_llm_client()

    def __call__(self, inputs: List[str]) -> List[str]:
        """AuditKIT adapter call interface."""
        results = []
        for inp in inputs:
            # Run investigation
            from agents.investigation_agent import run_investigation
            result = run_investigation(inp)
            results.append(result.get("risk_summary", {}).get("summary", "No summary"))
        return results

    def run_single(self, input_text: str) -> Dict:
        """Run single investigation and return structured result."""
        from agents.investigation_agent import run_investigation
        result = run_investigation(input_text)
        return result


def create_auditkit_dataset(
    trajectories_dir: str = "artifacts/agenttune/trajectories",
    output_path: str = "artifacts/auditkit/dataset.jsonl"
) -> Path:
    """Create AuditKIT-compatible dataset from trajectories."""
    dataset = ThreatLensAuditKitDataset()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    trajectories_dir = Path(trajectories_dir)

    if not trajectories_dir.exists():
        return Path(output_path)

    for jsonl_file in trajectories_dir.glob("*.jsonl"):
        with open(jsonl_file, "r") as f:
            for line in f:
                if line.strip():
                    traj = json.loads(line)
                    sample = {
                        "id": traj.get("trajectory_id", str(uuid.uuid4())),
                        "input": traj.get("investigation_input", ""),
                        "expected": {
                            "risk_level": traj.get("ground_truth", {}).get("expected_risk"),
                            "escalation": traj.get("ground_truth", {}).get("escalation_decision"),
                            "tools": traj.get("planned_tool_sequence", []),
                            "evidence_min": traj.get("ground_truth", {}).get("min_expected_evidence", 1),
                        },
                        "metadata": {
                            "trajectory_id": traj.get("trajectory_id"),
                            "scenario_id": traj.get("scenario_id"),
                        }
                    }
                    output_path.write_text(json.dumps(sample) + "\n")

    return output_path


def load_auditkit_dataset(filepath: str) -> List[Dict]:
    """Load AuditKIT dataset from JSONL file."""
    samples = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


if __name__ == "__main__":
    print("AuditKIT Dataset Adapter ready")
    print("Classes:")
    print("  - ThreatLensAuditKitDataset")
    print("  - AuditKitAdapter")
    print("Functions:")
    print("  - create_auditkit_dataset")
    print("  - load_auditkit_dataset")