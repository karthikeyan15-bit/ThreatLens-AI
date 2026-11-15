"""Generate human-readable investigation reports."""

import json
from datetime import datetime, timezone
from typing import Any


def generate_risk_summary(
    risk_level: str,
    confidence: float,
    evidence_items: list[dict],
    correlations: list[dict],
    tool_calls: list[dict],
) -> dict:
    """Build a structured summary for human review."""
    confirmed = [e for e in evidence_items if e.get("veracity") == "confirmed"]
    suggested = [e for e in evidence_items if e.get("veracity") == "suggested"]

    return {
        "risk_level": risk_level,
        "confidence": round(confidence, 3),
        "summary": _build_text_summary(risk_level, evidence_items, correlations),
        "evidence_count": len(evidence_items),
        "confirmed_evidence": len(confirmed),
        "suggested_evidence": len(suggested),
        "correlations_found": len(correlations),
        "tool_calls_made": len(tool_calls),
        "evidence_items": evidence_items[:20],
        "correlations": correlations[:10],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_text_summary(risk: str, evidence: list[dict], correlations: list[dict]) -> str:
    lines = [f"Investigation Assessment: {risk} Risk"]
    lines.append(f"Evidence items: {len(evidence)} total")

    threats = [e for e in evidence if e.get("veracity") == "confirmed"]
    if threats:
        lines.append("\nConfirmed Threat Indicators:")
        for t in threats[:5]:
            lines.append(f"  - [{t.get('indicator_type', '?')}] {t.get('indicator_value', '')} (source: {t.get('source_tool', 'unknown')})")

    corr = correlations
    if corr:
        lines.append("\nIndicator Correlations:")
        for c in corr[:5]:
            lines.append(f"  - {c.get('indicator_a', '')} ↔ {c.get('indicator_b', '')} ({c.get('relationship', '')}, strength: {c.get('strength', '?')})")

    return "\n".join(lines)


def format_investigation_result(result: dict, inv_id: str) -> str:
    """Format the final result as a readable string for the UI."""
    summary = result.get("summary", {})
    lines = [
        f"# Investigation Report — {inv_id}",
        f"**Risk Level:** {summary.get('risk_level', 'N/A')}",
        f"**Confidence:** {summary.get('confidence', 'N/A')}",
        "",
        summary.get("summary", ""),
        "",
        "---",
        "## Evidence Collected",
    ]

    for e in summary.get("evidence_items", [])[:10]:
        verbatim = "[CONFIRMED]" if e.get("veracity") == "confirmed" else "[SUGGESTED]"
        lines.append(f"- {verbatim} {e.get('indicator_type', '?')}: `{e.get('indicator_value', '')}` — {e.get('source_tool', '?')}")

    lines += [
        "",
        "---",
        "## Tool Calls Executed",
    ]
    for tc in result.get("tool_calls", [])[:10]:
        args_str = json.dumps(tc.get("arguments", {}), default=str)[:80]
        lines.append(f"- `{tc.get('tool_name', '?')}`({args_str}) → {'OK' if tc.get('duration_ms') else 'FAILED'}")

    lines += ["", "---", f"*Report generated at {datetime.now(timezone.utc).isoformat()} UTC*"]
    return "\n".join(lines)
