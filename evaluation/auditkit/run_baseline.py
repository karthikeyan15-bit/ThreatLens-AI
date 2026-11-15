"""Run baseline evaluation with ThreatLens AI agent."""

import json
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from agents.investigation_agent import run_investigation
from evaluation.auditkit.dataset_adapter import ThreatLensAuditKitDataset, create_auditkit_dataset
from evaluation.auditkit.custom_scorers import get_all_scorers, get_scorer_names

try:
    from auditkit import evaluate, evaluate_many, RunConfig
    AUDITKIT_AVAILABLE = True
except ImportError:
    AUDITKIT_AVAILABLE = False


def run_baseline_evaluation(
    scenario_ids: List[int] = None,
    api_key: Optional[str] = None,
    output_dir: str = "artifacts/auditkit/baseline",
    experiment_name: str = "threatlens-baseline"
) -> Dict:
    """
    Run baseline evaluation using the current ThreatLens AI agent.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scenarios
    dataset = ThreatLensAuditKitDataset()
    scenarios = dataset._load_scenarios()

    if scenario_ids is None:
        scenario_ids = list(range(1, 9))  # 1-8

    scenario_objs = []
    for sid in scenario_ids:
        scenario = next((s for s in dataset._load_scenarios() if s["id"] == f"scenario_{sid:03d}"), None)
        if scenario:
            scenario_objs.append(scenario)

    print(f"Running baseline evaluation on {len(scenario_objs)} scenarios...")

    results = []
    total_start = time.time()

    for scenario in scenario_objs:
        scenario_id = scenario["id"]
        request = scenario["request"]

        print(f"\nRunning scenario: {scenario_id}")
        print(f"  Request: {request[:100]}...")

        try:
            start = time.time()
            result = run_investigation(scenario["request"])
            duration = time.time() - start

            result["scenario_id"] = scenario["id"]
            result["duration_seconds"] = duration
            result["reference"] = scenario.get("reference", {})

            print(f"  Completed in {duration:.1f}s - Risk: {result.get('risk_level', 'N/A')}")

            results.append(result)

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "scenario_id": scenario["id"],
                "error": str(e),
                "failed": True
            })

    # Save raw results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = Path(output_dir) / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nBaseline results saved to {results_path}")
    print(f"Total time: {time.time() - total_start:.1f}s")

    return {
        "results": results,
        "output_dir": str(output_dir),
        "total_scenarios": len(results)
    }


def run_auditkit_evaluation(
    results: List[Dict],
    output_dir: str = "artifacts/auditkit/baseline",
    experiment_name: str = "threatlens-baseline"
) -> Dict:
    """
    Run AuditKIT evaluation on collected results.
    """
    if not AUDITKIT_AVAILABLE:
        print("AuditKIT not available, skipping formal evaluation")
        return {"status": "skipped", "reason": "AuditKIT not available"}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare samples for AuditKIT
    samples = []
    for result in results:
        if result.get("failed"):
            continue

        scenario_id = result.get("scenario_id", "unknown")
        reference = result.get("reference", {})

        sample = {
            "id": result.get("investigation_id", str(uuid.uuid4())),
            "input": result.get("input", ""),
            "expected": {
                "risk_level": reference.get("expected_risk"),
                "escalation": reference.get("escalate"),
                "tools": reference.get("expected_tools", []),
                "evidence_min": reference.get("min_expected_evidence", 1),
            },
            "actual": {
                "risk_level": result.get("risk_level"),
                "confidence": result.get("confidence"),
                "summary": result.get("summary"),
                "escalation": result.get("escalation_required", False),
                "tools_used": [tc.get("tool", "") for tc in result.get("tool_calls", [])],
                "evidence": result.get("evidence", []),
            }
        }

    # Run AuditKIT evaluation
    try:
        scorers = [
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

        config = RunConfig(
            experiment_name="threatlens-baseline",
            tags=["baseline", "threatlens"]
        )

        # Note: This would need the actual AuditKIT evaluate function
        # For now, we compute metrics manually
        return compute_manual_metrics(results)

    except Exception as e:
        print(f"AuditKIT evaluation error: {e}")
        return {"status": "error", "error": str(e)}


def compute_manual_metrics(results: List[Dict]) -> Dict:
    """Compute metrics manually without AuditKIT."""
    from evaluation.auditkit.custom_scorers import (
        tool_selection_accuracy, argument_accuracy, task_completion_rate,
        classification_performance, evidence_grounding, tool_call_efficiency,
        escalation_accuracy, tool_call_f1, evidence_grounding_rate, tool_call_accuracy
    )

    metrics = {
        "tool_selection_accuracy": [],
        "argument_accuracy": [],
        "task_completion_rate": [],
        "classification_performance": [],
        "evidence_grounding": [],
        "tool_call_efficiency": [],
        "escalation_accuracy": [],
        "tool_call_f1": [],
        "evidence_grounding_rate": [],
        "tool_call_accuracy": [],
    }

    for result in results:
        if result.get("failed"):
            continue

        reference = result.get("reference", {})

        # Tool selection
        expected_tools = reference.get("expected_tools", [])
        actual_tools = [tc.get("tool", "") for tc in result.get("tool_calls", [])]
        metrics["tool_selection_accuracy"].append(
            tool_selection_accuracy(expected_tools, [t for t in actual_tools if t])
        )

        # Task completion
        status = result.get("status", "unknown")
        metrics["task_completion_rate"].append(
            1.0 if status in ["completed", "escalated"] else (0.5 if status == "max_steps" else 0.0)
        )

        # Classification
        predicted = result.get("risk_level", "")
        expected = reference.get("expected_risk", "")
        metrics["classification_performance"].append(
            classification_performance(predicted, expected)
        )

        # Escalation
        predicted_esc = result.get("escalation_required", False)
        expected_esc = reference.get("escalate", False)
        metrics["escalation_accuracy"].append(
            1.0 if predicted_esc == expected_esc else 0.0
        )

        # Evidence grounding (simplified)
        evidence = result.get("evidence", [])
        risk = result.get("risk_level", "")
        confirmed = sum(1 for e in evidence if e.get("veracity") == "confirmed")
        total = len(evidence)
        metrics["evidence_grounding"].append(
            (sum(1 for e in evidence if e.get("veracity") == "confirmed") / len(evidence)) if evidence else 0.0
        )

        # Tool call F1
        expected_tools_set = set(reference.get("expected_tools", []))
        actual_tools_set = set([tc.get("tool", "") for tc in result.get("tool_calls", []) if tc.get("tool")])
        intersection = len(set(reference.get("expected_tools", [])) & set(actual_tools))
        precision = len(intersection) / len(actual_tools_set) if actual_tools_set else 0
        recall = len(intersection) / len(set(reference.get("expected_tools", []))) if reference.get("expected_tools") else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics["tool_call_f1"].append(f1)

    # Aggregate
    summary = {}
    for key, values in metrics.items():
        if values:
            metrics[key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
        else:
            metrics[key] = {"mean": 0, "min": 0, "max": 0, "count": 0}

    # Overall summary
    summary = {
        "tool_selection_accuracy": metrics["tool_selection_accuracy"]["mean"],
        "argument_accuracy": 0.0,  # Not computed
        "task_completion_rate": metrics["task_completion_rate"]["mean"],
        "classification_accuracy": metrics["classification_performance"]["mean"],
        "evidence_grounding": metrics["evidence_grounding"]["mean"],
        "tool_call_efficiency": 0.0,  # Not computed
        "escalation_accuracy": metrics["escalation_accuracy"]["mean"],
        "tool_call_f1": metrics["tool_call_f1"]["mean"],
        "evidence_grounding_rate": 0.0,  # Not computed
        "tool_call_accuracy": 0.0,  # Not computed
        "per_scenario": [],
        "total_scenarios": len([r for r in results if not r.get("failed")]),
        "failed_scenarios": len([r for r in results if r.get("failed")])
    }

    # Add per-scenario details
    for result in results:
        if result.get("failed"):
            continue
        reference = result.get("reference", {})
        summary["per_scenario"].append({
            "scenario_id": result.get("scenario_id"),
            "risk_level": result.get("risk_level"),
            "expected_risk": reference.get("expected_risk"),
            "escalation_required": result.get("escalation_required"),
            "expected_escalation": reference.get("escalate"),
            "tool_calls": len(result.get("tool_calls", [])),
            "duration": result.get("duration_seconds", 0)
        })

    return summary


def run_baseline_experiment(
    scenario_ids: List[int] = None,
    output_dir: str = "artifacts/auditkit/baseline",
    api_key: Optional[str] = None
) -> Dict:
    """Run complete baseline experiment."""
    print("=" * 60)
    print("Running Baseline Experiment")
    print("=" * 60)

    # Step 1: Run investigations
    results = run_baseline_evaluation(scenario_ids=scenario_ids, api_key=api_key, output_dir=output_dir)

    # Step 2: Compute metrics
    print("\nComputing metrics...")
    metrics = compute_manual_metrics(results["results"])

    # Save metrics
    output_dir = Path(output_dir)
    metrics_path = Path(output_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {metrics_path}")
    print(f"\nSummary:")
    print(f"  Tool Selection Accuracy: {metrics['tool_selection_accuracy']:.2%}")
    print(f"  Task Completion Rate: {metrics['task_completion_rate']:.2%}")
    print(f"  Classification Accuracy: {metrics['classification_accuracy']:.2%}")
    print(f"  Evidence Grounding: {metrics['evidence_grounding']:.2%}")
    print(f"  Escalation Accuracy: {metrics['escalation_accuracy']:.2%}")
    print(f"  Tool Call F1: {metrics['tool_call_f1']:.2%}")

    return {
        "results": results["results"],
        "metrics": metrics,
        "output_dir": str(output_dir)
    }


if __name__ == "__main__":
    print("Running Baseline Experiment")
    print("=" * 50)

    # Run with test API key
    import os
    os.environ["TINY_AYA_API_KEY"] = os.environ.get("TINY_AYA_API_KEY", "test")

    results = run_baseline_experiment(
        scenario_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        output_dir="artifacts/auditkit/baseline"
    )

    print("\nBaseline experiment completed!")
    print(f"Results saved to artifacts/auditkit/baseline/")