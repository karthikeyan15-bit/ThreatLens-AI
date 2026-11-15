"""Generate PDF incident reports from investigation results."""

import io
import json
from datetime import datetime, timezone
from typing import Any


def generate_html_report(result: dict, investigation_id: str) -> str:
    """Create an HTML report suitable for PDF conversion."""
    risk = result.get("risk_level", "Unknown")
    risk_color = {
        "Low": "#4CAF50",
        "Medium": "#FF9800",
        "High": "#F44336",
        "Critical": "#B71C1C",
        "Inconclusive": "#9E9E9E",
    }.get(risk, "#607D8B")

    evidence_rows = ""
    for e in result.get("evidence_items", [])[:15]:
        verbatim_badge = "CONFIRMED" if e.get("veracity") == "confirmed" else "SUGGESTED"
        verbatim_color = "#4CAF50" if e.get("veracity") == "confirmed" else "#FF9800"
        evidence_rows += f"""
        <tr>
            <td>{e.get('indicator_type', '')}</td>
            <td><code>{e.get('indicator_value', '')}</code></td>
            <td><span style="color:{verbatim_color};font-weight:bold">{verbatim_badge}</span></td>
            <td>{e.get('source_tool', 'N/A')}</td>
            <td>{round((e.get('confidence') or 0) * 100)}%</td>
        </tr>"""

    tool_rows = ""
    for tc in result.get("tool_calls", [])[:10]:
        args_str = json.dumps(tc.get("arguments", {}), default=str)[:60]
        status = "OK" if tc.get("duration_ms") else "FAILED"
        tool_rows += f"""
        <tr>
            <td><code>{tc.get('tool', '')}</code></td>
            <td><small>{args_str}</small></td>
            <td>{tc.get('duration_ms', 0)}ms</td>
            <td>{status}</td>
        </tr>"""

    correlations_rows = ""
    for c in result.get("correlations", []):
        correlations_rows += f"""
        <tr>
            <td><code>{c.get('indicator_a', '')}</code></td>
            <td><code>{c.get('indicator_b', '')}</code></td>
            <td>{c.get('relationship', '')}</td>
            <td>{round((c.get('strength') or 0) * 100)}%</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #333; }}
        .header {{ border-bottom: 3px solid {risk_color}; padding-bottom: 10px; margin-bottom: 20px; }}
        .risk-badge {{ display: inline-block; background: {risk_color}; color: white;
                       padding: 4px 16px; border-radius: 4px; font-size: 18px; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 13px; }}
        th {{ background: #f5f5f5; text-align: left; padding: 8px; border: 1px solid #ddd; }}
        td {{ padding: 6px 8px; border: 1px solid #ddd; }}
        code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
        .section-title {{ background: #263238; color: white; padding: 8px 12px;
                          margin: 20px 0 10px; font-size: 14px; }}
        .footer {{ margin-top: 30px; font-size: 11px; color: #888; border-top: 1px solid #ddd; padding-top: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ThreatLens AI — Incident Report</h1>
        <p>Investigation ID: <strong>{investigation_id}</strong></p>
        <p>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        <p>Risk Level: <span class="risk-badge">{risk}</span></p>
        <p>Confidence: {round((result.get('confidence') or 0) * 100)}%</p>
    </div>

    <div class="section-title">Investigation Summary</div>
    <p>{result.get('summary', 'No summary available.')}</p>

    <div class="section-title">Evidence Collected ({result.get('evidence_count', 0)} items)</div>
    <table>
        <thead><tr><th>Type</th><th>Indicator</th><th>Veracity</th><th>Source Tool</th><th>Confidence</th></tr></thead>
        <tbody>{evidence_rows or '<tr><td colspan="5">No evidence items recorded.</td></tr>'}</tbody>
    </table>

    <div class="section-title">Tool Calls Executed ({result.get('tool_calls_made', 0)})</div>
    <table>
        <thead><tr><th>Tool</th><th>Arguments</th><th>Duration</th><th>Status</th></tr></thead>
        <tbody>{tool_rows or '<tr><td colspan="4">No tool calls recorded.</td></tr>'}</tbody>
    </table>

    <div class="section-title">Indicator Correlations ({result.get('correlation_count', 0)})</div>
    <table>
        <thead><tr><th>Indicator A</th><th>Indicator B</th><th>Relationship</th><th>Strength</th></tr></thead>
        <tbody>{correlations_rows or '<tr><td colspan="4">No correlations found.</td></tr>'}</tbody>
    </table>

    <div class="footer">
        <p>This report was generated automatically by ThreatLens AI. Human review is recommended before taking action.</p>
        <p>Evidence-based analysis &mdash; all conclusions are supported by tool-extracted evidence.</p>
    </div>
</body>
</html>"""
    return html


def save_pdf(html_content: str, output_path: str):
    """Convert HTML report to PDF using WeasyPrint."""
    try:
        from weasyprint import HTML
        HTML(string=html_content).write_pdf(output_path)
        return {"success": True, "path": output_path}
    except ImportError:
        # Fallback: save as HTML
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return {"success": True, "path": output_path, "note": "PDF not available, saved as HTML"}
    except Exception as e:
        return {"success": False, "error": str(e)}
