"""Evidence collection service.

Parses tool results into structured evidence items and persists them
to SQLite. Veracity is one of:

    confirmed   - tool returned a definitive malicious finding
    suggested   - tool returned a suspicious or partial finding
    unknown     - tool returned no clear signal
"""

import json
from typing import Any

from database.db import add_evidence


def collect_from_result(inv_id: str, tool_name: str, result: dict) -> list[dict]:
    """Parse a tool result into evidence items and persist them.

    Returns the list of evidence items created.
    """
    data = result.get("data", {}) or result
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return []

    created = []

    # ── Email analyzer ────────────────────────────────────────────────
    if tool_name == "email_analyzer":
        if "indicators" in data:
            add_evidence(
                inv_id, "email", "email_analysis", "email_analyzer",
                data, "confirmed", data.get("confidence", 0.5),
            )
            created.append({"type": "email", "value": "email_analysis", "veracity": "confirmed"})
        for url in data.get("extracted_urls", []):
            add_evidence(
                inv_id, "url", url, "email_analyzer",
                {"context": "extracted_from_email"}, "suggested", 0.6,
            )
            created.append({"type": "url", "value": url, "veracity": "suggested"})

    # ── URL analyzer ──────────────────────────────────────────────────
    if tool_name == "url_analyzer" and "combined_verdict" in data:
        verdict = data.get("combined_verdict", "")
        veracity = "confirmed" if verdict == "malicious" else "suggested"
        add_evidence(
            inv_id, "url", data.get("url", ""), "url_analyzer",
            data, veracity, data.get("structural_risk_score", 0.5),
        )
        created.append({"type": "url", "value": data.get("url", ""), "veracity": veracity})

    # ── Threat intel ──────────────────────────────────────────────────
    if tool_name == "threat_intel_lookup" and "results" in data:
        for svc_result in data.get("results", []):
            verdict = svc_result.get("verdict", "")
            veracity = (
                "confirmed" if verdict == "malicious"
                else "suggested" if verdict == "inconclusive"
                else "unknown"
            )
            add_evidence(
                inv_id,
                data.get("indicator_type", "url"),
                data.get("indicator", ""),
                svc_result.get("service", "unknown"),
                svc_result,
                veracity,
                0.9 if verdict == "malicious" else 0.5,
            )
            created.append({
                "type": data.get("indicator_type", "url"),
                "value": data.get("indicator", ""),
                "veracity": veracity,
                "source": svc_result.get("service", "unknown"),
            })

    # ── Network analyzer ──────────────────────────────────────────────
    if tool_name == "network_analyzer" and "overall_risk" in data:
        add_evidence(
            inv_id, "network", "network_log_analyzed", "network_analyzer",
            data, "confirmed", 0.8,
        )
        created.append({"type": "network", "value": "network_log_analyzed", "veracity": "confirmed"})

    # ── IP analyzer ───────────────────────────────────────────────────
    if tool_name == "ip_analyzer":
        if "domain" in data:
            add_evidence(
                inv_id, "domain", data.get("domain", ""), "ip_analyzer",
                data, "suggested", 0.5,
            )
            created.append({"type": "domain", "value": data.get("domain", ""), "veracity": "suggested"})
        elif "ip" in data:
            add_evidence(
                inv_id, "ip", data.get("ip", ""), "ip_analyzer",
                data, "suggested", 0.5,
            )
            created.append({"type": "ip", "value": data.get("ip", ""), "veracity": "suggested"})

    return created


def summarize_evidence(evidence_items: list[dict]) -> dict:
    """Summarize evidence by veracity."""
    confirmed = [e for e in evidence_items if e.get("veracity") == "confirmed"]
    suggested = [e for e in evidence_items if e.get("veracity") == "suggested"]
    unknown = [e for e in evidence_items if e.get("veracity") == "unknown"]
    return {
        "total": len(evidence_items),
        "confirmed": len(confirmed),
        "suggested": len(suggested),
        "unknown": len(unknown),
    }