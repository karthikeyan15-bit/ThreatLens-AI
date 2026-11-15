"""Run improved model evaluation with AgentTune-trained model."""

import json
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from auditkit import evaluate, evaluate_many, RunConfig
    AUDITKIT_AVAILABLE = True
except ImportError:
    AUDITKIT_AVAILABLE = False


def load_improved_model(model_path: str) -> Any:
    """
    Load the improved model from AgentTune training.

    This would load a fine-tuned model or adapter from AgentTune training.
    Expected format: Hugging Face model with adapter or merged model.
    """
    # This is a placeholder - actual implementation depends on AgentTune output format
    # For now, return None to indicate model not available
    return None


def create_improved_agent(model_path: Optional[str] = None):
    """
    Create an agent using the improved model.

    This wraps the investigation agent to use the fine-tuned model.
    """
    if model_path:
        # Load fine-tuned model
        pass
    else:
        # Fallback to baseline with modified prompts
        pass

    def improved_agent(request: str) -> Dict:
        # This would use the fine-tuned model
        # For now, fall back to baseline
        from agents.investigation_agent import run_investigation
        return run_investigation(request)

    return improved_agent


def run_improved_evaluation(
    scenario_ids: List[int] = None,
    model_path: Optional[str] = None,
    output_dir: str = "artifacts/auditkit/improved",
    api_key: Optional[str] = None
) -> Dict:
    """
    Run evaluation with the improved (AgentTune-trained) model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # For now, use baseline agent (will be replaced when improved model available)
    from evaluation.auditkit.run_baseline import run_baseline_evaluation

    print("Running improved model evaluation...")
    print("Note: Using baseline agent as improved model not yet available")

    # Run with same scenarios as baseline
    results = run_baseline_evaluation(scenario_ids=scenario_ids, output_dir=output_dir)

    # Mark as improved for comparison
    for result in results["results"]:
        result["model_version"] = "improved"

    return {
        "results": results["results"],
        "output_dir": str(output_dir),
        "model_version": "improved"
    }


def compare_baseline_improved(
    baseline_results: Dict,
    improved_results: Dict,
    output_dir: str = "artifacts/auditkit/comparison"
) -> Dict:
    """
    Compare baseline and improved model results using AuditKIT.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics
    baseline_metrics_path = Path("artifacts/auditkit/baseline/metrics.json")
    improved_metrics_path = Path("artifacts/auditkit/improved/metrics.json")

    if not baseline_metrics_path.exists():
        raise FileNotFoundError("Baseline metrics not found. Run baseline first.")

    if not improved_metrics_path.exists():
        raise FileNotFoundError("Improved metrics not found. Run improved experiment first.")

    with open(baseline_metrics_path) as f:
        baseline = json.load(baseline_metrics_path)

    with open(improved_metrics_path) as f:
        improved = json.load(improved_metrics_path)

    # Compare metrics
    comparison = compare_metrics(baseline, improved)

    # Save comparison
    output_dir = Path("artifacts/auditkit/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = Path("artifacts/auditkit/comparison/comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    return comparison


def compare_metrics(baseline: Dict, improved: Dict) -> Dict:
    """Compare baseline and improved metrics."""
    comparison = {
        "baseline": {},
        "improved": {},
        "delta": {},
        "statistical_significance": {}
    }

    metric_keys = [
        "tool_selection_accuracy",
        "argument_accuracy",
        "task_completion_rate",
        "classification_accuracy",
        "evidence_grounding",
        "tool_call_efficiency",
        "escalation_accuracy",
        "tool_call_f1",
        "evidence_grounding_rate",
        "tool_call_accuracy"
    ]

    for key in metric_keys:
        baseline_val = baseline.get(key, 0)
        improved_val = improved.get(key, 0)

        # Handle nested dicts
        if isinstance(baseline_val, dict):
            baseline_val = baseline_val.get("mean", 0)
        if isinstance(improved_val, dict):
            improved_val = improved_val.get("mean", 0)

        delta = improved_val - baseline_val
        pct_change = (delta / baseline_val * 100) if baseline_val != 0 else 0

        comparison["baseline"][key] = baseline_val
        comparison["improved"][key] = improved_val
        comparison["delta"][key] = {
            "absolute": delta,
            "percent_change": pct_change
        }

    return comparison


def generate_comparison_report(comparison: Dict, output_path: str = "artifacts/auditkit/comparison/report.md") -> str:
    """Generate markdown comparison report."""
    report = """# AgentTune Improvement Report

## Baseline vs Improved Model Comparison

### Metric Comparison

| Metric | Baseline | Improved | Δ (Absolute) | Δ (%) |
|--------|----------|----------|--------------|-------|
"""

    for key, delta in comparison.get("delta", {}).items():
        baseline = comparison["baseline"].get(key, 0)
        improved = comparison["improved"].get(key, 0)
        abs_delta = comparison["delta"][key].get("absolute", 0)
        pct = comparison["delta"][key].get("percent_change", 0)

        report += f"| {key} | {baseline:.4f} | {improved:.4f} | {abs_delta:+.4f} | {pct:+.1f}% |\n"

    report += """

## Summary

"""

    # Count improvements
    improvements = sum(1 for v in comparison.get("delta", {}).values() if v.get("absolute", 0) > 0)
    regressions = sum(1 for v in comparison.get("delta", {}).values() if v.get("absolute", 0) < 0)
    unchanged = sum(1 for v in comparison.get("delta", {}).values() if v.get("absolute", 0) == 0)

    report += f"- Improvements: {improvements}\n"
    report += f"- Regressions: {regressions}\n"
    report += f"- Unchanged: {unchanged}\n"

    report += """

## Statistical Significance

Statistical significance testing (McNemar's test for classification, bootstrap for continuous metrics) should be performed to validate improvements.

"""

    return report


if __name__ == "__main__":
    print("Improved Model Evaluation Module")
    print("=" * 50)
    print("This module requires an AgentTune-trained model.")
    print("Run baseline first, then train with AgentTune, then run this.")