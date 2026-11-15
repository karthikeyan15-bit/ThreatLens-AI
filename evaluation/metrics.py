"""Evaluation metrics for ThreatLens AI agent performance."""

from typing import Any


def compute_metrics(actual: dict, reference: dict) -> dict:
    """Compute evaluation metrics comparing actual agent output to reference."""
    return {
        "tool_selection_correct": _check_tool_selection(actual, reference),
        "argument_accuracy": _check_argument_accuracy(actual, reference),
        "task_completed": actual.get("status") in ("completed", "escalated"),
        "classification_correct": _check_classification(actual, reference),
        "evidence_grounding": _check_evidence_grounding(actual, reference),
        "tool_call_count": len(actual.get("tool_calls", [])),
        "escalation_appropriate": actual.get("escalation_required") == reference.get("escalate", False),
    }


def _check_tool_selection(actual: dict, reference: dict) -> bool:
    actual_tools = {tc["tool"] for tc in actual.get("tool_calls", [])}
    expected_tools = set(reference.get("expected_tools", []))
    if not expected_tools:
        return True
    return expected_tools.issubset(actual_tools) or actual_tools.intersection(expected_tools)


def _check_argument_accuracy(actual: dict, reference: dict) -> float:
    calls = actual.get("tool_calls", [])
    expected = reference.get("expected_calls", [])
    if not expected:
        return 1.0
    matches = 0
    for exp in expected:
        for act in calls:
            if act["tool"] == exp.get("tool"):
                act_args = set(act.get("arguments", {}).keys())
                exp_args = set(exp.get("arguments", {}).keys())
                if exp_args and act_args & exp_args:
                    matches += 1
                    break
    return round(matches / len(expected), 3) if expected else 1.0


def _check_classification(actual: dict, reference: dict) -> bool:
    actual_risk = actual.get("risk_level", "")
    expected_risk = reference.get("expected_risk", "")
    if not expected_risk:
        return True
    return actual_risk == expected_risk


def _check_evidence_grounding(actual: dict, reference: dict) -> float:
    actual_evidence_count = actual.get("evidence_count", 0)
    expected_evidence_count = reference.get("min_expected_evidence", 0)
    if expected_evidence_count == 0:
        return 1.0
    return min(1.0, actual_evidence_count / expected_evidence_count)


def aggregate_metrics(run_results: list[dict]) -> dict:
    """Aggregate metrics across multiple evaluation runs."""
    if not run_results:
        return {"count": 0}

    n = len(run_results)
    return {
        "total_runs": n,
        "tool_selection_accuracy": sum(1 for r in run_results if r["tool_selection_correct"]) / n,
        "avg_argument_accuracy": sum(r["argument_accuracy"] for r in run_results) / n,
        "completion_rate": sum(1 for r in run_results if r["task_completed"]) / n,
        "classification_accuracy": sum(1 for r in run_results if r["classification_correct"]) / n,
        "avg_evidence_grounding": sum(r["evidence_grounding"] for r in run_results) / n,
        "avg_tool_calls": sum(r["tool_call_count"] for r in run_results) / n,
        "escalation_accuracy": sum(1 for r in run_results if r["escalation_appropriate"]) / n,
    }
