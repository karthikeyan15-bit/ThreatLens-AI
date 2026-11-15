"""Compare baseline and improved model results using AuditKIT."""

import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from typing import List, Optional


def load_metrics(filepath: str) -> dict:
    """Load metrics from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


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


def bootstrap_confidence_interval(
    baseline_samples: List[float],
    improved_samples: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> Dict:
    """Compute bootstrap confidence interval for difference in means."""
    import numpy as np

    baseline = np.array(baseline_samples)
    improved = np.array(improved_samples)

    # Bootstrap
    n_baseline = len(baseline)
    n_improved = len(improved)

    diffs = []
    for _ in range(n_bootstrap):
        b_sample = np.random.choice(baseline, size=n_baseline, replace=True)
        i_sample = np.random.choice(improved, size=n_improved, replace=True)
        diffs.append(i_sample.mean() - baseline.mean())

    diffs = np.array(diffs)
    alpha = (1 - 0.95) / 2
    ci_lower = np.percentile(diffs, alpha * 100)
    ci_upper = np.percentile(diffs, (1 - alpha) * 100)

    return {
        "mean_diff": np.mean(diffs),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "significant": ci_lower > 0 or ci_upper < 0
    }


def mcnemar_test(baseline_preds: List, improved_preds: List, labels: List) -> Dict:
    """McNemar's test for paired nominal data."""
    # For classification tasks
    # b = baseline wrong, improved correct
    # c = baseline correct, improved wrong
    b = sum(1 for b, i, l in zip(baseline_preds, improved_preds, labels) if b != l and i == l)
    c = sum(1 for b, i, l in zip(baseline_preds, improved_preds, labels) if b == l and i != l)

    if b + c == 0:
        return {"statistic": 0, "p_value": 1.0, "significant": False}

    # Chi-square approximation with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    from scipy import stats
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "statistic": chi2,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "b": b,
        "c": c
    }


def run_comparison(
    baseline_path: str = "artifacts/auditkit/baseline/metrics.json",
    improved_path: str = "artifacts/auditkit/improved/metrics.json",
    output_dir: str = "artifacts/auditkit/comparison"
) -> Dict:
    """Run full comparison between baseline and improved models."""
    output_dir = Path("artifacts/auditkit/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics
    baseline = load_metrics(baseline_path) if Path(baseline_path).exists() else {}
    improved = load_metrics(improved_path) if Path(improved_path).exists() else {}

    if not baseline:
        raise FileNotFoundError(f"Baseline metrics not found at {baseline_path}")
    if not improved:
        raise FileNotFoundError(f"Improved metrics not found at {improved_path}")

    # Compare
    comparison = compare_metrics(baseline, improved)

    # Add statistical significance if we have sample data
    # (would need per-sample predictions)

    # Save comparison
    output_dir = Path("artifacts/auditkit/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = Path("artifacts/auditkit/comparison/comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # Generate markdown report
    report = generate_comparison_report(comparison)
    report_path = Path("artifacts/auditkit/comparison/report.md")
    with open(report_path, "w") as f:
        f.write(report)

    return comparison


def generate_comparison_report(comparison: Dict) -> str:
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare baseline and improved model results")
    parser.add_argument("--baseline", default="artifacts/auditkit/baseline/metrics.json")
    parser.add_argument("--improved", default="artifacts/auditkit/improved/metrics.json")
    parser.add_argument("--output", default="artifacts/auditkit/comparison")

    args = parser.parse_args()

    comparison = run_comparison(args.baseline, args.improved, args.output)

    print("Comparison completed!")
    print(f"Results saved to {args.output}/")

    # Print summary
    improvements = sum(1 for v in comparison.get("delta", {}).values() if v.get("absolute", 0) > 0)
    regressions = sum(1 for v in comparison.get("delta", {}).values() if v.get("absolute", 0) < 0)

    print(f"Improvements: {improvements}")
    print(f"Regressions: {regressions}")


if __name__ == "__main__":
    import sys
    main()