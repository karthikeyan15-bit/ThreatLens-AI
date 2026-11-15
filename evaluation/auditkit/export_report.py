"""Export AuditKIT evaluation reports in multiple formats."""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass, asdict


def export_json(results: Dict, filepath: str):
    """Export results as JSON."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)


def export_csv(results: Dict, filepath: str):
    """Export results as CSV."""
    # Flatten results for CSV
    rows = []

    # Add summary metrics
    for key, value in results.get("baseline", {}).items():
        rows.append({
            "model": "baseline",
            "metric": key,
            "value": results["baseline"].get(key, 0)
        })

    for key, value in results.get("improved", {}).items():
        rows.append({
            "model": "improved",
            "metric": key,
            "value": results["improved"].get(key, 0)
        })

    # Delta
    for key, delta in results.get("delta", {}).items():
        rows.append({
            "model": "delta",
            "metric": key,
            "value": delta.get("absolute", 0),
            "percent_change": delta.get("percent_change", 0)
        })

    if results.get("per_scenario"):
        for scenario in results.get("per_scenario", []):
            row = {
                "model": "scenario",
                "scenario_id": scenario.get("scenario_id", ""),
                "risk_level": scenario.get("risk_level", ""),
                "expected_risk": scenario.get("expected_risk", ""),
                "escalation_required": scenario.get("escalation_required", False),
                "expected_escalation": scenario.get("expected_escalation", False),
                "tool_calls": scenario.get("tool_calls", 0),
                "duration": scenario.get("duration", 0)
            }
            rows.append(row)

    if not rows:
        return

    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def export_markdown(results: Dict, filepath: str):
    """Export results as Markdown report."""
    now = datetime.utcnow().isoformat()

    with open(filepath, "w") as f:
        f.write("# ThreatLens AI — AuditKIT Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.utcnow().isoformat()} UTC\n\n")

        # Baseline vs Improved
        if "baseline" in results and "improved" in results:
            f.write("## Baseline vs Improved Model Comparison\n\n")
            f.write("| Metric | Baseline | Improved | Δ (Absolute) | Δ (%) |\n")
            f.write("|--------|----------|----------|--------------|-------|\n")

            for key in results.get("delta", {}).keys():
                baseline = results["baseline"].get(key, 0)
                improved = results["improved"].get(key, 0)
                delta = results["delta"].get(key, {})

                if isinstance(baseline, dict):
                    baseline = baseline.get("mean", 0)
                if isinstance(results["improved"].get(key), dict):
                    improved = results["improved"][key].get("mean", 0)

                delta_abs = results["delta"][key].get("absolute", 0)
                delta_pct = results["delta"][key].get("percent_change", 0)

                f.write(f"| {key} | {baseline:.4f} | {improved:.4f} | {results['delta'][key].get('absolute', 0):+.4f} | {results['delta'][key].get('percent_change', 0):+.1f}% |\n")

            f.write("\n### Summary\n\n")
            improvements = sum(1 for v in results.get("delta", {}).values() if v.get("absolute", 0) > 0)
            regressions = sum(1 for v in results.get("delta", {}).values() if v.get("absolute", 0) < 0)
            unchanged = sum(1 for v in results.get("delta", {}).values() if v.get("absolute", 0) == 0)

            f.write(f"- Improvements: {improvements}\n")
            f.write(f"- Regressions: {regressions}\n")
            f.write(f"- Unchanged: {unchanged}\n\n")

        # Per-scenario details
        if "per_scenario" in results:
            f.write("## Per-Scenario Results\n\n")
            f.write("| Scenario | Risk Level | Expected Risk | Escalation | Expected Escalation | Tool Calls | Duration (s) |\n")
            f.write("|----------|------------|---------------|------------|---------------------|------------|--------------|\n")

            for scenario in results.get("per_scenario", []):
                f.write(f"| {scenario.get('scenario_id', '')} | "
                        f"{scenario.get('risk_level', '')} | "
                        f"{scenario.get('expected_risk', '')} | "
                        f"{scenario.get('escalation_required', '')} | "
                        f"{scenario.get('expected_escalation', '')} | "
                        f"{scenario.get('tool_calls', 0)} | "
                        f"{scenario.get('duration', 0):.1f} |\n")

        # Baseline only
        elif "tool_selection_accuracy" in results:
            f.write("## Baseline Evaluation Results\n\n")
            f.write("| Metric | Mean | Min | Max | Count |\n")
            f.write("|--------|------|-----|-----|-------|\n")

            for key, value in results.items():
                if isinstance(results[key], dict) and "mean" in results[key]:
                    f.write(f"| {key} | {results[key]['mean']:.4f} | {results[key]['min']:.4f} | {results[key]['max']:.4f} | {results[key]['count']} |\n")

            f.write(f"\n**Total Scenarios:** {results.get('total_scenarios', 0)}\n")
            f.write(f"**Failed Scenarios:** {results.get('failed_scenarios', 0)}\n")

        f.write(f"\n---\n*Report generated at {datetime.utcnow().isoformat()} UTC*\n")


def export_html(results: Dict, filepath: str):
    """Export results as HTML report."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>ThreatLens AI - AuditKIT Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .improvement {{ color: green; }}
        .regression {{ color: red; }}
        .neutral {{ color: gray; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
    </style>
</head>
<body>
    <h1>ThreatLens AI - AuditKIT Evaluation Report</h1>
    <p>Generated: {datetime.utcnow().isoformat()} UTC</p>
"""

    # Add comparison table
    if "delta" in results:
        html += """
    <h2>Baseline vs Improved Model Comparison</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Baseline</th>
            <th>Improved</th>
            <th>Δ (Absolute)</th>
            <th>Δ (%)</th>
        </tr>
"""
        for key, delta in results.get("delta", {}).items():
            baseline = results["baseline"].get(key, 0)
            improved = results["improved"].get(key, 0)

            if isinstance(baseline, dict):
                baseline = baseline.get("mean", 0)
            if isinstance(results["improved"].get(key), dict):
                improved = results["improved"][key].get("mean", 0)

            delta_abs = results["delta"][key].get("absolute", 0)
            delta_pct = results["delta"][key].get("percent_change", 0)

            css_class = "improvement" if delta.get("absolute", 0) > 0 else ("regression" if delta.get("absolute", 0) < 0 else "neutral")
            html += f"""
        <tr>
            <td>{key}</td>
            <td>{baseline:.4f}</td>
            <td>{improved:.4f}</td>
            <td class="{css_class}">{results['delta'][key].get('absolute', 0):+.4f}</td>
            <td class="{css_class}">{results['delta'][key].get('percent_change', 0):+.1f}%</td>
        </tr>
"""

        html += "    </table>\n"

    html += f"""
    <hr>
    <p><small>Report generated at {datetime.utcnow().isoformat()} UTC</small></p>
</body>
</html>"""

    with open(filepath, "w") as f:
        f.write(html)


def export_all_formats(results: Dict, output_dir: str = "artifacts/auditkit/comparison"):
    """Export results in all formats."""
    output_dir = Path(results)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_json(results, Path(results) / "comparison.json")
    export_csv(results, Path(results) / "comparison.csv")
    export_markdown(results, Path(results) / "report.md")
    export_html(results, Path(results) / "report.html")

    print(f"Exported all formats to {output_dir}")


def generate_auditkit_report(
    baseline_path: str = "artifacts/auditkit/baseline/metrics.json",
    improved_path: str = "artifacts/auditkit/improved/metrics.json",
    output_dir: str = "artifacts/auditkit/comparison"
) -> Dict:
    """Generate complete AuditKIT comparison report."""
    from evaluation.auditkit.compare_results import run_comparison

    comparison = run_comparison(baseline_path, improved_path, "artifacts/auditkit/comparison")

    output_dir = Path("artifacts/auditkit/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export all formats
    export_json(comparison, "artifacts/auditkit/comparison/comparison.json")
    export_csv(comparison, "artifacts/auditkit/comparison/comparison.csv")
    export_markdown(comparison, "artifacts/auditkit/comparison/report.md")
    export_html(comparison, "artifacts/auditkit/comparison/report.html")

    return {
        "json": "artifacts/auditkit/comparison/comparison.json",
        "csv": "artifacts/auditkit/comparison/comparison.csv",
        "markdown": "artifacts/auditkit/comparison/report.md",
        "html": "artifacts/auditkit/comparison/report.html"
    }


if __name__ == "__main__":
    print("Export Report Module")
    print("Functions: export_json, export_csv, export_markdown, export_html, export_all_formats, generate_auditkit_report")