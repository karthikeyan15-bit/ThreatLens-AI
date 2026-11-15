"""Benchmark runner for ThreatLens AI evaluation."""

import json
import uuid
from pathlib import Path
from typing import Any

from evaluation.metrics import compute_metrics
from database.db import save_evaluation_run
from agents.investigation_agent import run_investigation


SCENARIOS_DIR = Path(__file__).parent.parent / "datasets" / "evaluation"


def load_scenarios() -> list[dict]:
    """Load evaluation scenarios from JSON file."""
    scenario_file = SCENARIOS_DIR / "scenarios.json"
    if not scenario_file.exists():
        return []
    return json.loads(scenario_file.read_text())


def run_single_scenario(scenario: dict, agent_version: str = "baseline", api_key: str | None = None) -> dict:
    """Run one evaluation scenario and return metrics."""
    request = scenario["request"]
    reference = scenario["reference"]

    actual_result = run_investigation(request, api_key=api_key)

    metrics = compute_metrics(actual_result, reference)
    metrics["agent_version"] = agent_version
    metrics["scenario_id"] = scenario["id"]

    save_evaluation_run({
        "agent_version": agent_version,
        "scenario_id": scenario["id"],
        **metrics,
        "actual_output": actual_result,
        "reference_output": reference,
    })

    return {
        "scenario_id": scenario["id"],
        "metrics": metrics,
        "investigation_id": actual_result.get("investigation_id"),
    }


def run_all_scenarios(agent_version: str = "baseline", api_key: str | None = None) -> dict:
    """Run all evaluation scenarios and return aggregated results."""
    scenarios = load_scenarios()
    if not scenarios:
        return {"error": "No scenarios found. Create datasets/evaluation/scenarios.json"}

    results = []
    for scenario in scenarios:
        result = run_single_scenario(scenario, agent_version, api_key)
        results.append(result)

    from .metrics import aggregate_metrics
    aggregated = aggregate_metrics([r["metrics"] for r in results])
    aggregated["runs"] = results

    return {
        "agent_version": agent_version,
        "total_scenarios": len(scenarios),
        "aggregated_metrics": aggregated,
        "per_scenario": results,
    }
