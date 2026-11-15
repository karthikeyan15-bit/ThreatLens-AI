"""Investigation agent — orchestrates multi-step threat investigations."""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Use absolute imports to work correctly on Hugging Face Spaces
from agents.llm_client import get_llm_client, TinyAyaClient
from agents.planner import Indicator, InvestigationPlan, extract_indicators, plan_investigation
from agents.response_generator import generate_risk_summary
from database.db import (
    add_correlation,
    add_evidence,
    create_investigation,
    get_tool_calls,
    log_tool_call,
    update_investigation,
    log_audit_event,
)
from tools.tool_registry import execute_tool, serialize_tools, get_tool_by_name
from config.risk_thresholds import get_risk_level, get_all_thresholds

MAX_STEPS = 12


@dataclass
class AgentState:
    investigation_id: str
    history: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    tool_calls_log: list[dict] = field(default_factory=list)
    correlations: list[dict] = field(default_factory=list)
    steps_taken: int = 0
    done: bool = False


SYSTEM_PROMPT = """You are a cybersecurity investigation analyst. Your job is to investigate suspicious indicators using available tools.

RULES:
- Only use the tools provided; never invent results.
- If a tool reports an error or "not configured", note it but continue with other tools.
- Correlate findings across tools — link shared domains, IPs, and email senders.
- Be conservative: NEVER declare something safe just because no threat intel match was found. If all threat intel returns inconclusive, conclude "Inconclusive".
- When uncertain, recommend human escalation.
- Always call your tools one at a time and wait for results before deciding the next step.
- Validate your tool arguments before calling.

OUTPUT FORMAT:
When you have gathered enough evidence to conclude, respond with a JSON block like:
{
  "conclusion": {
    "risk_level": "Low|Medium|High|Critical|Inconclusive",
    "confidence": 0.0-1.0,
    "summary": "Brief explanation of findings",
    "escalate": true|false
  }
}

Start by extracting indicators from the request, then call the appropriate tools."""


def _build_system_prompt() -> str:
    tools_json = json.dumps(serialize_tools(), indent=2)
    thresholds = get_all_thresholds()
    thresholds_str = ", ".join(f"{k}={v}" for k, v in thresholds.items())
    return f"{SYSTEM_PROMPT}\n\nAVAILABLE TOOLS:\n{tools_json}\n\nRISK THRESHOLDS: {thresholds_str}"


def _assess_risk(evidence_items: list[dict]) -> tuple[str, float]:
    score = 0.0
    weights = {
        "confirmed": 1.0,
        "suggested": 0.5,
        "unknown": 0.2,
    }
    for item in evidence_items:
        veracity = item.get("veracity", "unknown")
        conf = item.get("confidence", 0.5)
        score += weights.get(veracity, 0.2) * conf

    max_score = len(evidence_items) if evidence_items else 1
    normalized = min(1.0, score / max(max_score * 0.7, 1))

    risk = get_risk_level(normalized)
    return risk, round(normalized, 3)


def _validate_tool_args(tool_name: str, args: dict) -> tuple[bool, str]:
    """Validate tool arguments against tool schema."""
    tool = get_tool_by_name(tool_name)
    if not tool:
        return False, f"Tool '{tool_name}' not found"

    schema = tool.parameters
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    for req in required:
        if req not in args:
            return False, f"Missing required argument: {req}"

    # Could add more validation here (type checking, enum values, etc.)
    return True, ""


def _should_continue_investigation(evidence: list[dict], tool_calls: list[dict], max_steps: int, steps_taken: int) -> tuple[bool, str]:
    """
    Determine if investigation should continue based on evidence sufficiency.
    Returns (should_continue, reason).
    """
    if steps_taken >= max_steps:
        return False, "Maximum steps reached"

    # Check if we have enough evidence to conclude
    confirmed_count = sum(1 for e in evidence if e.get("veracity") == "confirmed")
    total_evidence = len(evidence)

    # If we have multiple confirmed indicators from different sources, we can conclude
    if confirmed_count >= 2 and total_evidence >= 3:
        return False, "Sufficient confirmed evidence collected"

    # If we have high-confidence ML classifications, we can conclude
    ml_confident = sum(1 for e in evidence if e.get("confidence", 0) > 0.9 and e.get("source", "").startswith("ml_"))
    if ml_confident >= 1 and total_evidence >= 2:
        return False, "High-confidence ML classification obtained"

    return True, ""


def correlate_evidence(evidence: list[dict], llm_client: TinyAyaClient) -> list[dict]:
    """Find relationships between evidence items using LLM reasoning."""
    if len(evidence) < 2:
        return []

    items_text = "\n".join(
        f"- [{e['indicator_type']}] {e['indicator_value']} (veracity: {e['veracity']})"
        for e in evidence
    )

    prompt = f"""Given these investigation evidence items, identify any correlations or relationships between them.
Return a JSON array like: [{{"indicator_a": "...", "indicator_b": "...", "relationship": "...", "strength": 0.0-1.0}}]

Evidence:
{items_text}

If no clear correlations exist, return an empty array."""

    response = llm_client.messages_create(
        model="tiny-aya-global",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text if response.content else "[]"
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return []


def run_investigation(request: str, api_key: str | None = None, model: str = "tiny-aya-global") -> dict:
    """Run a full agentic investigation and return structured results."""
    client = get_llm_client()

    inv_id = create_investigation("multi", request)
    update_investigation(inv_id, status="running")

    # Log initial request
    log_audit_event(inv_id, 0, "investigation_started", {"request": request})

    state = AgentState(investigation_id=inv_id)
    plan: InvestigationPlan = plan_investigation(request)

    # Log plan
    log_audit_event(inv_id, 0, "plan_created", {
        "request_type": plan.request_type,
        "indicators": [{"type": i.type, "value": i.value} for i in plan.indicators],
        "tool_sequence": plan.tool_sequence
    })

    system_prompt = _build_system_prompt()
    state.history.append({"role": "system", "content": system_prompt})

    # Include plan in initial user message to seed the agent
    initial_msg = f"""Investigate this cybersecurity incident:

{request}

PLAN (use as guidance):
- Request type: {plan.request_type}
- Extracted indicators: {[f"{i.type}:{i.value}" for i in plan.indicators]}
- Recommended tool sequence: {plan.tool_sequence}
- Notes: {plan.notes}

Extract all indicators and determine the appropriate tools to use. Start with the first recommended tool if applicable."""
    state.history.append({
        "role": "user",
        "content": initial_msg,
    })

    try:
        while state.steps_taken < MAX_STEPS and not state.done:
            start_time = time.time()

            response = client.messages_create(
                model=model,
                max_tokens=1024,
                messages=state.history,
                tools=serialize_tools(),
            )

            duration_ms = int((time.time() - start_time) * 1000)

            for block in response.content:
                if block.type == "text":
                    state.history.append({"role": "assistant", "content": block.text})

                    # Log LLM decision
                    log_audit_event(inv_id, state.steps_taken, "llm_decision", {"text": block.text})

                    # Check for conclusion JSON
                    if "{" in block.text and "}" in block.text:
                        try:
                            conclusion_match = block.text[block.text.rfind("{"):block.text.rfind("}") + 1]
                            conclusion = json.loads(conclusion_match)
                            if "conclusion" in conclusion:
                                result = conclusion["conclusion"]
                                risk, confidence = _assess_risk(state.evidence)
                                final_risk = result.get("risk_level", risk)
                                final_confidence = result.get("confidence", confidence)
                                return _finalize_result(
                                    inv_id, state, final_risk, final_confidence,
                                    result.get("summary", ""), result.get("escalate", False),
                                    duration_ms,
                                )
                        except (json.JSONDecodeError, IndexError):
                            pass

                elif block.type == "tool_use":
                    tool_name = block.name
                    tool_args = block.input or {}

                    # Validate tool arguments
                    valid, error = _validate_tool_args(tool_name, tool_args)
                    if not valid:
                        tool_result = {"success": False, "error": error, "data": {}}
                        state.history.append({
                            "role": "assistant",
                            "content": [{"type": "tool_use", "id": block.id, "name": tool_name, "input": tool_args}],
                        })
                        state.history.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(tool_result),
                                "is_error": True,
                            }],
                        })
                        log_audit_event(inv_id, state.steps_taken, "tool_argument_error", {"tool": tool_name, "error": error})
                        continue

                    # Log tool call decision
                    log_audit_event(inv_id, state.steps_taken, "tool_call", {"tool": tool_name, "arguments": tool_args})

                    tool_result = execute_tool(tool_name, tool_args)

                    state.history.append({
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": block.id, "name": tool_name, "input": tool_args}],
                    })
                    state.history.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(tool_result),
                            "is_error": not tool_result.get("success", False),
                        }],
                    })

                    # Log tool result
                    log_audit_event(inv_id, state.steps_taken, "tool_result", {"tool": tool_name, "result": tool_result})

                    log_tool_call(inv_id, state.steps_taken, tool_name, tool_args, tool_result, duration_ms)
                    state.tool_calls_log.append({
                        "step": state.steps_taken,
                        "tool": tool_name,
                        "arguments": tool_args,
                        "result": tool_result,
                        "duration_ms": duration_ms,
                    })

                    # Collect evidence from tool results
                    _extract_evidence_from_result(inv_id, tool_name, tool_result, state)

                    # Check if we should continue after each tool
                    should_continue, reason = _should_continue_investigation(state.evidence, state.tool_calls_log, MAX_STEPS, state.steps_taken)
                    if not should_continue:
                        log_audit_event(inv_id, state.steps_taken, "evidence_sufficient", {"reason": reason})
                        state.done = True
                        # Force conclusion on next iteration by adding a prompt
                        state.history.append({
                            "role": "user",
                            "content": f"Evidence sufficient to conclude ({reason}). Provide your conclusion in the JSON format."
                        })
                        continue

                    # Run correlation mid-loop every 3 steps
                    if state.steps_taken % 3 == 0 and state.steps_taken > 0:
                        corr_results = correlate_evidence(state.evidence, client)
                        for c in corr_results:
                            add_correlation(inv_id, c["indicator_a"], c["indicator_b"], c["relationship"], c["strength"])
                        state.correlations.extend(corr_results)
                        log_audit_event(inv_id, state.steps_taken, "mid_loop_correlation", {"correlations": corr_results})

            state.steps_taken += 1

    except Exception as exc:
        update_investigation(inv_id, status="escalated")
        log_audit_event(inv_id, state.steps_taken, "investigation_error", {"error": str(exc)})
        return {
            "investigation_id": inv_id,
            "error": str(exc),
            "steps_taken": state.steps_taken,
        }

    return _finalize_result(inv_id, state, "Inconclusive", 0.1, "Investigation completed without sufficient evidence.", True, 0)


def _extract_evidence_from_result(inv_id: str, tool_name: str, result: dict, state: AgentState):
    """Parse tool results into structured evidence items."""
    data = result.get("data", {}) or result
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return

    # Email analyzer
    if tool_name == "email_analyzer" and "indicators" in data:
        add_evidence(inv_id, "email", "email_analysis", "email_analyzer", data, "confirmed", data.get("confidence", 0.5))
        for url in data.get("extracted_urls", []):
            add_evidence(inv_id, "url", url, "email_analyzer", {"context": "extracted_from_email"}, "suggested", 0.6)

    # URL analyzer
    if tool_name == "url_analyzer" and "combined_verdict" in data:
        verdict = data.get("combined_verdict", "")
        veracity = "confirmed" if verdict == "malicious" else "suggested"
        add_evidence(inv_id, "url", data.get("url", ""), "url_analyzer", data, veracity, data.get("structural_risk_score", 0.5))

    # Threat intel
    if tool_name == "threat_intel_lookup" and "results" in data:
        for svc_result in data.get("results", []):
            verdict = svc_result.get("verdict", "")
            veracity = "confirmed" if verdict == "malicious" else "suggested" if verdict == "inconclusive" else "unknown"
            add_evidence(
                inv_id, data.get("indicator_type", "url"), data.get("indicator", ""),
                svc_result.get("service", "unknown"), svc_result, veracity,
                0.9 if verdict == "malicious" else 0.5
            )

    # Network analyzer
    if tool_name == "network_analyzer" and "overall_risk" in data:
        add_evidence(inv_id, "network", "network_log_analyzed", "network_analyzer", data, "confirmed", 0.8)

    # IP analyzer
    if tool_name == "ip_analyzer" and "domain" in data:
        add_evidence(inv_id, "domain", data.get("domain", ""), "ip_analyzer", data, "suggested", 0.5)
    if tool_name == "ip_analyzer" and "ip" in data and "domain" not in data:
        add_evidence(inv_id, "ip", data.get("ip", ""), "ip_analyzer", data, "suggested", 0.5)


def _finalize_result(
    inv_id: str,
    state: AgentState,
    risk_level: str,
    confidence: float,
    summary: str,
    escalate: bool,
    total_duration_ms: int,
) -> dict:
    status = "escalated" if escalate else "completed"
    update_investigation(inv_id, risk_level=risk_level, confidence=confidence,
                         summary=summary, status=status)

    # Final correlation pass
    # Note: correlations already added mid-loop, but we can add any remaining
    risk_summary = generate_risk_summary(risk_level, confidence, state.evidence, state.correlations, state.tool_calls_log)

    log_audit_event(inv_id, state.steps_taken, "investigation_concluded", {
        "risk_level": risk_level,
        "confidence": confidence,
        "escalated": escalate,
        "total_steps": state.steps_taken
    })

    return {
        "investigation_id": inv_id,
        "status": status,
        "risk_level": risk_level,
        "confidence": confidence,
        "summary": summary,
        "evidence_count": len(state.evidence),
        "correlation_count": len(state.correlations),
        "tool_calls": state.tool_calls_log,
        "risk_summary": risk_summary,
        "escalation_required": escalate,
    }