"""Indicator extractor and investigation planner."""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Indicator:
    type: str  # "url", "ip", "domain", "email"
    value: str
    confidence: float = 0.5


@dataclass
class InvestigationPlan:
    request_type: str
    indicators: list[Indicator] = field(default_factory=list)
    tool_sequence: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b")
URL_RE = re.compile(r"https?://[^\s\'\")\]>]+")
DOMAIN_RE = re.compile(r"\b([a-zA-Z0-9][\w.-]*\.[a-zA-Z]{2,})\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def extract_indicators(text: str) -> list[Indicator]:
    """Pull out structured indicators from raw text."""
    results = []
    seen = set()

    for match in URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;:")
        if url not in seen:
            seen.add(url)
            results.append(Indicator("url", url, 0.9))

    for match in IPV4_RE.finditer(text):
        ip = match.group(0)
        parts = [int(p) for p in ip.split(".")]
        if parts[0] in (10, 127) or (parts[0] == 172 and 16 <= parts[1] <= 31) or (parts[0] == 192 and parts[1] == 168):
            continue
        if ip not in seen:
            seen.add(ip)
            results.append(Indicator("ip", ip, 0.7))

    for match in EMAIL_RE.finditer(text):
        email = match.group(0)
        if email not in seen:
            seen.add(email)
            results.append(Indicator("email", email, 0.85))

    for match in DOMAIN_RE.finditer(text):
        domain = match.group(1)
        if domain not in seen:
            seen.add(domain)
            results.append(Indicator("domain", domain, 0.6))

    return results


def determine_request_type(text: str, indicators: list[Indicator]) -> str:
    """Classify what kind of investigation this is."""
    types_found = {ind.type for ind in indicators}
    if len(indicators) == 1 and indicators[0].type == "email":
        return "email"
    if len(indicators) == 1 and indicators[0].type == "url":
        return "url"
    if len(indicators) == 1 and indicators[0].type in ("ip", "domain"):
        return "ip"
    text_lower = text.lower()
    if "csv" in text_lower or "network" in text_lower or "flow" in text_lower:
        return "network"
    if len(types_found) > 1:
        return "multi"
    return "unknown"


def plan_investigation(text: str) -> InvestigationPlan:
    """Create a full investigation plan from raw input text."""
    indicators = extract_indicators(text)
    req_type = determine_request_type(text, indicators)

    tool_sequence = []
    notes = []

    if req_type == "email" or "email" in {i.type for i in indicators}:
        tool_sequence.extend(["email_analyzer"])
        notes.append("Running email phishing analysis")

    if req_type == "url" or any(i.type == "url" for i in indicators):
        tool_sequence.extend(["threat_intel_lookup"])
        notes.append("Querying threat intelligence for URLs")

    if req_type == "ip" or any(i.type in ("ip", "domain") for i in indicators):
        tool_sequence.extend(["ip_analyzer", "threat_intel_lookup"])
        notes.append("Checking IP/domain reputation")

    if "csv" in text.lower() or req_type == "network":
        tool_sequence.append("network_analyzer")
        notes.append("Analyzing network flow data")

    if req_type == "multi":
        tool_sequence = list(dict.fromkeys(tool_sequence))
        notes.append("Multi-type incident — running all relevant analyses")

    if not tool_sequence:
        tool_sequence = ["threat_intel_lookup"]
        notes.append("No clear indicators found; falling back to broad intel lookup")

    return InvestigationPlan(
        request_type=req_type,
        indicators=indicators,
        tool_sequence=tool_sequence,
        notes=notes,
    )
