"""Threat intelligence lookup tool — queries external APIs with safety enforcement."""

from integrations.threat_intelligence import search_all
from integrations.threat_intelligence.base import Verdict
from .tool_registry import ToolDefinition, register_tool


def run(args: dict) -> dict:
    indicator = args.get("indicator", "").strip()
    indicator_type = args.get("indicator_type", "url")

    if not indicator:
        return {"success": False, "error": "Indicator is required"}

    results = search_all(indicator, indicator_type)

    # Check for definitive malicious verdicts from any source
    any_malicious = any(
        r.get("verdict") == Verdict.MALICIOUS for r in results
    )
    any_error = any(
        r.get("verdict") == Verdict.ERROR for r in results
    )
    any_not_configured = any(
        r.get("verdict") == Verdict.NOT_CONFIGURED for r in results
    )

    # Count sources by status
    sources_queried = len([r for r in results if r.get("verdict") not in (Verdict.NOT_CONFIGURED, Verdict.ERROR)])
    sources_unavailable = len([r for r in results if r.get("verdict") in (Verdict.NOT_CONFIGURED, Verdict.ERROR)])

    # Safety enforcement: if all configured sources return INCONCLUSIVE, we must NOT declare "safe"
    configured_results = [r for r in results if r.get("verdict") not in (Verdict.NOT_CONFIGURED, Verdict.ERROR)]
    all_inconclusive = all(r.get("verdict") == Verdict.INCONCLUSIVE for r in configured_results) if configured_results else False

    # Overall verdict
    if any_malicious:
        overall_verdict = Verdict.MALICIOUS
    elif any_error:
        overall_verdict = Verdict.ERROR
    elif all_inconclusive:
        overall_verdict = Verdict.INCONCLUSIVE
    else:
        overall_verdict = Verdict.INCONCLUSIVE

    return {
        "indicator": indicator,
        "indicator_type": indicator_type,
        "results": results,
        "overall_verdict": overall_verdict,
        "any_malicious": any_malicious,
        "all_inconclusive": all_inconclusive,
        "sources_queried": sources_queried,
        "sources_unavailable": sources_unavailable,
        "safety_note": "No source returned definitive malicious verdict; result is inconclusive. NEVER declare safe based on lack of threat intel matches." if overall_verdict == Verdict.INCONCLUSIVE else None,
    }


THREAT_INTEL_TOOL = ToolDefinition(
    name="threat_intel_lookup",
    description=(
        "Queries multiple external threat intelligence services (URLhaus, PhishTank, "
        "AbuseIPDB, VirusTotal) for a given indicator (URL, IP, or domain). "
        "Returns results from each service with standardized verdicts (malicious/inconclusive/error). "
        "Enforces safety rule: NEVER declare an indicator safe just because no threat intel match is found. "
        "Unconfigured services are noted but do not block the investigation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "indicator": {
                "type": "string",
                "description": "The URL, IP address, or domain to look up",
            },
            "indicator_type": {
                "type": "string",
                "enum": ["url", "ip", "domain"],
                "description": "Type of indicator being looked up",
            },
        },
        "required": ["indicator", "indicator_type"],
    },
    category="intelligence",
    func=run,
)


register_tool(THREAT_INTEL_TOOL)